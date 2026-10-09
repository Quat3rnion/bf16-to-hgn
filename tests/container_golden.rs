use anyhow::Result;
use bf16_to_hgn::{
    codec::Encoding,
    container::{self, Reader},
    ContainerPlan, Origin, TensorPlan,
};
use sha2::{Digest, Sha256};
use std::{fs, io::Write};

#[test]
fn complete_specification_file_matches_pinned_sha256() -> Result<()> {
    let plan = ContainerPlan {
        filename: "example.hgn".into(),
        identity: "example".into(),
        tensors: vec![
            tensor("example.bf16.weight", &[2, 32], Encoding::Bf16),
            tensor("example.q4c.weight", &[1, 64], Encoding::Q4Cp),
            tensor("example.i64", &[3], Encoding::I64),
        ],
    };
    let mut layout = container::layout(&plan)?;
    assert_eq!(layout.data_offset, 640);
    assert_eq!(
        layout
            .entries
            .iter()
            .map(|e| (e.offset, e.size))
            .collect::<Vec<_>>(),
        [(640, 128), (768, 144), (960, 24)]
    );
    layout.file_size = container::align(layout.file_size, 64)?;
    for (entry, checksum) in layout.entries.iter_mut().zip([0x01600160, 0xf5003800, 1]) {
        entry.checksum = checksum;
    }
    let mut bytes = vec![0u8; layout.file_size as usize];
    bytes[..104].copy_from_slice(&container::header_bytes(&layout)?);
    for (i, entry) in layout.entries.iter().enumerate() {
        bytes[104 + i * 160..104 + (i + 1) * 160].copy_from_slice(&container::entry_bytes(entry)?);
    }
    let bf16 = hex(concat!(
        "80bf78bf70bf68bf5fbf57bf4fbf47bf3fbf37bf2fbf27bf1ebf16bf0ebf06bf",
        "fcb eecbedbbecbb ebbbeabbe9abe8abe74be53be33be12bee4bda3bd43bd82bc",
        "823c433da33de43d123e333e533e743e8a3e9a3eab3ebb3ecb3edb3eec3efc3e",
        "063f0e3f163f1e3f273f2f3f373f3f3f473f4f3f573f5f3f683f703f783f803f"
    ));
    assert_eq!(bf16.len(), 128);
    bytes[640..768].copy_from_slice(&bf16);
    let q4c = hex(concat!(
        "000000c10000e0c00000c0c00000a0c0000080c0000040c0000000c0000080bf",
        "000000000000803f0000004000004040000080400000a0400000c0400000e040",
        "832dc7610ba54fe9832dc7610ba54fe9832dc7610ba54fe9832dc7610ba54fe9",
        "0000000000000000000000000000000000000000000000000000000000000000",
        "003800b4000000000000000000000000"
    ));
    assert_eq!(q4c.len(), 144);
    bytes[768..912].copy_from_slice(&q4c);
    for (i, value) in [3i64, 5, 7].iter().enumerate() {
        bytes[960 + i * 8..968 + i * 8].copy_from_slice(&value.to_le_bytes());
    }
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        "f88df782c300afa52494f572bd9425d6bc7a7296d378bef5fc5e582181c2c54e"
    );
    let temporary = tempfile::tempdir()?;
    let path = temporary.path().join("example.hgn");
    fs::write(&path, bytes)?;
    let mut reader = Reader::open(&path)?;
    assert_eq!(reader.layout, layout);
    for i in 0..3 {
        assert_eq!(reader.fingerprint(i)?.xor32, layout.entries[i].checksum);
    }
    Ok(())
}

#[test]
fn short_final_xor_word_is_zero_extended() -> Result<()> {
    let mut file = tempfile::tempfile()?;
    file.write_all(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])?;
    assert_eq!(container::fingerprint(&mut file, 0, 10)?.xor32, 0x0c040e0d);
    Ok(())
}

fn tensor(name: &str, shape: &[u64], encoding: Encoding) -> TensorPlan {
    TensorPlan {
        name: name.into(),
        shape: shape.into(),
        encoding,
        origin: Origin::Tensor { name: name.into() },
        rotation: None,
    }
}
fn hex(text: &str) -> Vec<u8> {
    let text = text
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect::<String>();
    assert_eq!(text.len() % 2, 0);
    text.as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            u8::from_str_radix(std::str::from_utf8(pair).expect("hex ASCII"), 16).expect("hex byte")
        })
        .collect()
}
