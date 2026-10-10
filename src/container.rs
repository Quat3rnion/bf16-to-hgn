use crate::{
    codec::{payload_layout, Encoding},
    rows_cols, ContainerPlan,
};
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs::{File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
};

pub const HEADER_SIZE: u64 = 104;
pub const ENTRY_SIZE: u64 = 160;
const HASH_CHUNK: usize = 8 * 1024 * 1024;

pub fn align(value: u64, alignment: u64) -> Result<u64> {
    ensure!(alignment.is_power_of_two(), "invalid alignment");
    Ok(value
        .checked_add(alignment - 1)
        .context("alignment overflow")?
        & !(alignment - 1))
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Entry {
    pub name: String,
    pub encoding: Encoding,
    pub shape: Vec<u64>,
    pub offset: u64,
    pub size: u64,
    pub checksum: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Layout {
    pub identity: String,
    pub data_offset: u64,
    pub file_size: u64,
    pub entries: Vec<Entry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Fingerprint {
    pub size: u64,
    pub sha256: String,
    pub xor32: u32,
}

pub fn layout(plan: &ContainerPlan) -> Result<Layout> {
    ensure!(
        !plan.tensors.is_empty() && plan.tensors.len() <= 1 << 20,
        "invalid HGN tensor count"
    );
    check_ascii(&plan.identity, 63, "identity")?;
    let table_end = HEADER_SIZE
        .checked_add(
            ENTRY_SIZE
                .checked_mul(plan.tensors.len() as u64)
                .context("table overflow")?,
        )
        .context("table overflow")?;
    let data_offset = align(table_end, 64)?;
    let mut offset = data_offset;
    let mut names = BTreeSet::new();
    let mut entries = Vec::with_capacity(plan.tensors.len());
    for tensor in &plan.tensors {
        check_ascii(&tensor.name, 95, "tensor name")?;
        ensure!(
            names.insert(&tensor.name),
            "duplicate HGN tensor {}",
            tensor.name
        );
        ensure!(
            tensor.shape.len() <= 4,
            "HGN rank exceeds four: {}",
            tensor.name
        );
        ensure!(
            tensor.shape.iter().all(|&n| n > 0 && n <= i64::MAX as u64),
            "invalid dimensions: {}",
            tensor.name
        );
        let (rows, cols) = rows_cols(&tensor.shape)?;
        if tensor.encoding.ht_bits().is_some() {
            validate_ht_shape(&tensor.shape)?;
        }
        if tensor.encoding == Encoding::I4r {
            ensure!(
                tensor.shape.len() == 3 && tensor.shape[1] % 128 == 0 && cols % 128 == 0,
                "I4R needs expert matrices aligned to 128"
            );
        }
        if tensor.encoding == Encoding::Fp8G128 {
            ensure!(
                tensor.shape.len() == 3 && cols == 160,
                "FP8G needs a stacked 160-column table"
            );
        }
        let size = payload_layout(tensor.encoding, rows, cols)?.len;
        ensure!(size > 0, "empty HGN payload");
        entries.push(Entry {
            name: tensor.name.clone(),
            encoding: tensor.encoding,
            shape: tensor.shape.clone(),
            offset,
            size,
            checksum: 0,
        });
        offset = align(offset.checked_add(size).context("payload overflow")?, 64)?;
    }
    let last = entries.last().expect("nonempty entries");
    let file_size = last
        .offset
        .checked_add(last.size)
        .context("file size overflow")?;
    Ok(Layout {
        identity: plan.identity.clone(),
        data_offset,
        file_size,
        entries,
    })
}

fn validate_ht_shape(shape: &[u64]) -> Result<()> {
    ensure!(
        (shape.len() == 2
            && shape
                .iter()
                .all(|&n| n >= 128 && n <= 1 << 24 && n % 128 == 0))
            || shape == [512, 1280, 2560]
            || shape == [512, 2560, 640],
        "HT requires a 128-aligned 2-D matrix or a Flash-Next fused expert shape"
    );
    Ok(())
}

fn check_ascii(text: &str, limit: usize, kind: &str) -> Result<()> {
    ensure!(
        !text.is_empty() && text.len() <= limit && text.is_ascii() && !text.as_bytes().contains(&0),
        "invalid HGN {kind}"
    );
    Ok(())
}

pub struct Writer {
    file: File,
    pub layout: Layout,
    dirty_bytes: u64,
}

impl Writer {
    pub fn create(path: &Path, plan: &ContainerPlan) -> Result<Self> {
        let layout = layout(plan)?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(path)
            .with_context(|| format!("create {}", path.display()))?;
        file.set_len(layout.file_size)?;
        Ok(Self {
            file,
            layout,
            dirty_bytes: 0,
        })
    }

    pub fn write_at(&mut self, index: usize, relative_offset: u64, bytes: &[u8]) -> Result<()> {
        let entry = self
            .layout
            .entries
            .get(index)
            .context("invalid output tensor index")?;
        ensure!(
            relative_offset
                .checked_add(bytes.len() as u64)
                .context("write size overflow")?
                <= entry.size,
            "write exceeds HGN payload {}",
            entry.name
        );
        self.file
            .seek(SeekFrom::Start(entry.offset + relative_offset))?;
        self.file.write_all(bytes)?;
        self.dirty_bytes += bytes.len() as u64;
        if self.dirty_bytes >= 64 * 1024 * 1024 {
            self.file.sync_data()?;
            advise_drop(&self.file, 0, self.layout.file_size);
            self.dirty_bytes = 0;
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<(Layout, Vec<Fingerprint>)> {
        self.file.sync_data()?;
        advise_drop(&self.file, 0, self.layout.file_size);
        let mut fingerprints = Vec::with_capacity(self.layout.entries.len());
        for entry in &mut self.layout.entries {
            let digest = fingerprint(&mut self.file, entry.offset, entry.size)?;
            entry.checksum = digest.xor32;
            fingerprints.push(digest);
        }
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&header_bytes(&self.layout)?)?;
        for entry in &self.layout.entries {
            self.file.write_all(&entry_bytes(entry)?)?;
        }
        self.file.sync_all()?;
        advise_drop(&self.file, 0, self.layout.file_size);
        Ok((self.layout, fingerprints))
    }
}

pub fn header_bytes(layout: &Layout) -> Result<[u8; 104]> {
    check_ascii(&layout.identity, 63, "identity")?;
    let mut result = [0u8; 104];
    result[0..4].copy_from_slice(&0x314e4748u32.to_le_bytes());
    result[4..8].copy_from_slice(&2u32.to_le_bytes());
    result[8..16].copy_from_slice(&(layout.entries.len() as u64).to_le_bytes());
    result[16..24].copy_from_slice(&HEADER_SIZE.to_le_bytes());
    result[24..32].copy_from_slice(&layout.data_offset.to_le_bytes());
    result[32..40].copy_from_slice(&layout.file_size.to_le_bytes());
    result[40..40 + layout.identity.len()].copy_from_slice(layout.identity.as_bytes());
    Ok(result)
}

pub fn entry_bytes(entry: &Entry) -> Result<[u8; 160]> {
    check_ascii(&entry.name, 95, "tensor name")?;
    let mut result = [0u8; 160];
    result[..entry.name.len()].copy_from_slice(entry.name.as_bytes());
    result[96..100].copy_from_slice(&entry.encoding.code().to_le_bytes());
    result[100..104].copy_from_slice(&(entry.shape.len() as u32).to_le_bytes());
    for (i, &dimension) in entry.shape.iter().enumerate() {
        result[104 + i * 8..112 + i * 8]
            .copy_from_slice(&(i64::try_from(dimension)?).to_le_bytes());
    }
    result[136..144].copy_from_slice(&entry.offset.to_le_bytes());
    result[144..152].copy_from_slice(&entry.size.to_le_bytes());
    result[152..156].copy_from_slice(&entry.checksum.to_le_bytes());
    result[156..160].copy_from_slice(&entry.encoding.variant().to_le_bytes());
    Ok(result)
}

pub struct Reader {
    file: File,
    pub layout: Layout,
    pub version: u32,
}

impl Reader {
    pub fn open(path: &Path) -> Result<Self> {
        ensure!(
            !std::fs::symlink_metadata(path)?.file_type().is_symlink(),
            "HGN must not be a symlink"
        );
        let mut file = File::open(path)?;
        let file_size = file.metadata()?.len();
        ensure!(file_size >= HEADER_SIZE, "truncated HGN header");
        let mut header = [0u8; 104];
        file.read_exact(&mut header)?;
        ensure!(u32le(&header, 0) == 0x314e4748, "invalid HGN magic");
        let version = u32le(&header, 4);
        ensure!(version == 1 || version == 2, "unsupported HGN version");
        let count = u64le(&header, 8);
        ensure!(count > 0 && count <= 1 << 20, "invalid HGN tensor count");
        ensure!(
            u64le(&header, 16) == HEADER_SIZE,
            "noncanonical HGN table offset"
        );
        ensure!(u64le(&header, 32) == file_size, "HGN file size mismatch");
        let table_end = HEADER_SIZE
            .checked_add(count.checked_mul(ENTRY_SIZE).context("table overflow")?)
            .context("table overflow")?;
        ensure!(table_end <= file_size, "HGN table exceeds file");
        let data_offset = align(table_end, 64)?;
        ensure!(
            u64le(&header, 24) == data_offset,
            "HGN data offset mismatch"
        );
        let identity = ascii_field(&header[40..104], "identity")?;
        let mut entries = Vec::with_capacity(usize::try_from(count)?);
        let mut names = BTreeSet::new();
        let mut next_offset = data_offset;
        for _ in 0..count {
            let mut bytes = [0u8; 160];
            file.read_exact(&mut bytes)?;
            let name = ascii_field(&bytes[..96], "tensor name")?;
            ensure!(names.insert(name.clone()), "duplicate HGN tensor {name}");
            let rank = u32le(&bytes, 100) as usize;
            ensure!((1..=4).contains(&rank), "invalid HGN rank");
            let mut shape = Vec::with_capacity(rank);
            for i in 0..4 {
                let dimension = i64::from_le_bytes(
                    bytes[104 + i * 8..112 + i * 8]
                        .try_into()
                        .expect("eight bytes"),
                );
                if i < rank {
                    ensure!(dimension > 0, "invalid HGN dimension");
                    shape.push(dimension as u64);
                } else {
                    ensure!(dimension == 0, "nonzero unused HGN shape slot");
                }
            }
            let encoding = encoding_from_code(u32le(&bytes, 96), u32le(&bytes, 156))?;
            let offset = u64le(&bytes, 136);
            let size = u64le(&bytes, 144);
            ensure!(
                offset == next_offset && offset % 64 == 0,
                "noncanonical/overlapping HGN payload {name}"
            );
            ensure!(
                offset.checked_add(size).context("payload overflow")? <= file_size,
                "HGN payload exceeds file"
            );
            let (rows, cols) = rows_cols(&shape)?;
            if encoding.ht_bits().is_some() {
                validate_ht_shape(&shape)?;
            }
            if encoding == Encoding::I4r {
                ensure!(
                    shape.len() == 3 && shape[1] % 128 == 0 && cols % 128 == 0,
                    "invalid I4R expert shape"
                );
            }
            if encoding == Encoding::Fp8G128 {
                ensure!(shape.len() == 3 && cols == 160, "invalid FP8G table shape");
            }
            ensure!(
                size == payload_layout(encoding, rows, cols)?.len,
                "HGN payload size mismatch: {name}"
            );
            next_offset = align(offset + size, 64)?;
            entries.push(Entry {
                name,
                encoding,
                shape,
                offset,
                size,
                checksum: u32le(&bytes, 152),
            });
        }
        let last = entries.last().expect("nonempty");
        let end = last.offset + last.size;
        ensure!(
            file_size == end || file_size == align(end, 64)?,
            "unexpected trailing HGN bytes"
        );
        let layout = Layout {
            identity,
            data_offset,
            file_size,
            entries,
        };
        check_zero_range(&mut file, table_end, data_offset - table_end)?;
        for pair in layout.entries.windows(2) {
            let start = pair[0].offset + pair[0].size;
            check_zero_range(&mut file, start, pair[1].offset - start)?;
        }
        check_zero_range(&mut file, end, file_size - end)?;
        Ok(Self {
            file,
            layout,
            version,
        })
    }

    pub fn read_at(&mut self, index: usize, offset: u64, len: usize) -> Result<Vec<u8>> {
        let entry = self
            .layout
            .entries
            .get(index)
            .context("invalid tensor index")?;
        ensure!(
            offset.checked_add(len as u64).context("read overflow")? <= entry.size,
            "read exceeds HGN payload"
        );
        self.file.seek(SeekFrom::Start(entry.offset + offset))?;
        let mut bytes = vec![0u8; len];
        self.file.read_exact(&mut bytes)?;
        advise_drop(&self.file, entry.offset + offset, len as u64);
        Ok(bytes)
    }

    pub fn fingerprint(&mut self, index: usize) -> Result<Fingerprint> {
        let entry = self
            .layout
            .entries
            .get(index)
            .context("invalid tensor index")?;
        fingerprint(&mut self.file, entry.offset, entry.size)
    }
}

fn encoding_from_code(code: u32, variant: u32) -> Result<Encoding> {
    let encoding = match code {
        0 => Encoding::Bf16,
        1 => Encoding::F32,
        2 => Encoding::F16,
        4 => Encoding::I64,
        5 => Encoding::Q4Cp,
        7 => Encoding::Q8G64,
        10 => Encoding::Fp8G128,
        16 if variant == 0x1206 => Encoding::Ht3,
        16 => Encoding::Ht,
        23 => Encoding::I4r,
        24 => Encoding::Q6G64,
        _ => bail!("unimplemented HGN store {code}"),
    };
    ensure!(
        variant == encoding.variant(),
        "unsupported HGN store variant"
    );
    Ok(encoding)
}

fn ascii_field(bytes: &[u8], kind: &str) -> Result<String> {
    let end = bytes
        .iter()
        .position(|&b| b == 0)
        .context("unterminated HGN text field")?;
    ensure!(
        end > 0 && bytes[..end].is_ascii() && bytes[end..].iter().all(|&b| b == 0),
        "invalid HGN {kind} padding"
    );
    Ok(std::str::from_utf8(&bytes[..end])?.to_owned())
}

fn u32le(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("four bytes"))
}
fn u64le(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().expect("eight bytes"))
}

fn check_zero_range(file: &mut File, offset: u64, len: u64) -> Result<()> {
    ensure!(len < 64, "excessive HGN padding");
    file.seek(SeekFrom::Start(offset))?;
    let mut bytes = [0u8; 64];
    file.read_exact(&mut bytes[..len as usize])?;
    ensure!(
        bytes[..len as usize].iter().all(|&b| b == 0),
        "nonzero HGN padding"
    );
    Ok(())
}

pub fn fingerprint(file: &mut File, offset: u64, len: u64) -> Result<Fingerprint> {
    file.seek(SeekFrom::Start(offset))?;
    let mut hash = Sha256::new();
    let mut xor = 0u32;
    let mut bytes = vec![0u8; HASH_CHUNK.min(usize::try_from(len).unwrap_or(HASH_CHUNK))];
    let mut done = 0u64;
    while done < len {
        let take = (len - done).min(bytes.len() as u64) as usize;
        file.read_exact(&mut bytes[..take])?;
        hash.update(&bytes[..take]);
        for word in bytes[..take].chunks(4) {
            let mut padded = [0u8; 4];
            padded[..word.len()].copy_from_slice(word);
            xor ^= u32::from_le_bytes(padded);
        }
        advise_drop(file, offset + done, take as u64);
        done += take as u64;
    }
    Ok(Fingerprint {
        size: len,
        sha256: format!("{:x}", hash.finalize()),
        xor32: xor,
    })
}

pub fn advise_drop(file: &File, offset: u64, len: u64) {
    #[cfg(unix)]
    {
        use std::os::fd::AsRawFd;
        if let (Ok(offset), Ok(len)) = (libc::off_t::try_from(offset), libc::off_t::try_from(len)) {
            // Advisory only; failure does not alter contents or verification.
            unsafe {
                libc::posix_fadvise(file.as_raw_fd(), offset, len, libc::POSIX_FADV_DONTNEED);
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = (file, offset, len);
    }
}
