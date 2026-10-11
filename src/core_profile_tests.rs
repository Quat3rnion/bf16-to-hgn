use super::*;
use crate::{hf::Snapshot, manifest::OutputProfile, ContainerPlan, Rotation};
use serde_json::json;

fn expert_source(root: &Path, shape: &[u64], input: &[f32]) -> Snapshot {
    fs::create_dir(root).unwrap();
    fs::write(root.join("config.json"), b"{}").unwrap();
    let raw: Vec<u8> = input
        .iter()
        .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
        .collect();
    let mut header = serde_json::to_vec(&json!({
        "expert": {"dtype": "BF16", "shape": shape, "data_offsets": [0, raw.len()]}
    }))
    .unwrap();
    header.resize(header.len() + (8 - header.len() % 8) % 8, b' ');
    let mut file = File::create(root.join("model.safetensors")).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    file.write_all(&raw).unwrap();
    drop(file);
    fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::to_vec(&json!({
            "metadata": {"total_size": raw.len()}, "weight_map": {"expert": "model.safetensors"}
        }))
        .unwrap(),
    )
    .unwrap();
    Snapshot::open(root).unwrap()
}

#[test]
fn paired_stream_encode_and_independent_verify_cover_canonical_source_order() {
    let directory = tempfile::tempdir().unwrap();
    let (experts, expert_rows, cols) = (2, 512, 128);
    let values: Vec<f32> = (0..experts * expert_rows * cols)
        .map(|index| (index % 29) as f32 / 32.0 + (index / (128 * cols)) as f32 / 8.0)
        .collect();
    let snapshot = expert_source(&directory.path().join("source"), &[2, 512, 128], &values);
    for encoding in [Encoding::Ht, Encoding::Ht3] {
        let logical = TensorPlan {
            name: "layers.1.mlp.experts.gate_up_proj.weight".into(),
            shape: vec![2, 512, 128],
            encoding,
            origin: Origin::Tensor {
                name: "expert".into(),
            },
            rotation: Some(Rotation {
                su: "experts.su".into(),
                sv: "experts.gate_up_proj.svh".into(),
            }),
        };
        let vectors = BTreeMap::from([
            ("experts.su".into(), vec![1.0; cols]),
            (
                "experts.gate_up_proj.svh".into(),
                (0..experts * expert_rows)
                    .map(|row| if row / expert_rows == 0 { 1.0 } else { -1.0 })
                    .collect(),
            ),
        ]);
        // A canonical 2-D fixture container is only the byte sink for this
        // private streaming test. Public output validation still refuses tiny
        // non-Flash-Next rank-3 experts; there is no architecture escape hatch.
        let mut storage = logical.clone();
        storage.name = "paired-stream-fixture".into();
        storage.shape = vec![(experts * expert_rows) as u64, cols as u64];
        storage.rotation = None;
        let plan = ContainerPlan {
            filename: "fixture.hgn".into(),
            identity: "paired-stream-fixture".into(),
            tensors: vec![storage],
        };
        let path = directory.path().join(format!("{encoding:?}.hgn"));
        let mut writer = Writer::create(&path, &plan).unwrap();
        let options = Options {
            profile: OutputProfile::Ht43,
            ht_beam: 1,
            ..Options::default()
        };
        let hash = encode_tensor(
            &snapshot,
            &snapshot.tensors["expert"],
            &logical,
            0,
            &mut writer,
            &vectors,
            &options,
            &crate::progress::Progress::disabled().stage(
                "fixture",
                0,
                crate::progress::Unit::Values,
            ),
        )
        .unwrap();
        assert_eq!(hash, snapshot.hash_tensor("expert").unwrap());
        writer.finish().unwrap();
        let mut reader = container::Reader::open(&path).unwrap();
        let tile_bytes = 32 * encoding.ht_bits().unwrap();
        let pair_bytes = 256 * cols * encoding.ht_bits().unwrap() / 8;
        let mut expected_errors = crate::manifest::ErrorTotals::default();
        for expert in 0..experts {
            for band in 0..2 {
                let payload = reader
                    .read_at(0, ((expert * 2 + band) * pair_bytes) as u64, pair_bytes)
                    .unwrap();
                for up in [false, true] {
                    let row = expert * expert_rows + band * 128 + if up { 256 } else { 0 };
                    let input = snapshot
                        .read_f32("expert", (row * cols) as u64, 128 * cols)
                        .unwrap();
                    let (su, sv) = rotation_slice(&logical, &vectors, row, 128).unwrap();
                    let meta = QuantMeta {
                        su,
                        sv,
                        ht_beam: 1,
                        ..QuantMeta::default()
                    };
                    let encoded = codec::encode_rows(encoding, &input, 128, cols, &meta).unwrap();
                    let decoded = codec::decode_rows(encoding, &encoded, 128, cols, &meta).unwrap();
                    for (&value, &decoded) in input.iter().zip(&decoded) {
                        let value = f64::from(value);
                        let error = f64::from(decoded) - value;
                        expected_errors.source_squared += value * value;
                        expected_errors.error_squared += error * error;
                    }
                    expected_errors.values += input.len() as u64;
                    for column in 0..cols / 16 {
                        let target = (column * 16 + if up { 8 } else { 0 }) * tile_bytes;
                        let source = column * 8 * tile_bytes;
                        assert_eq!(
                            &payload[target..target + 8 * tile_bytes],
                            &encoded.codes[source..source + 8 * tile_bytes]
                        );
                    }
                }
            }
        }
        let actual = crate::verify::verify_tensor_fixture(
            &snapshot,
            &logical,
            &mut reader,
            &options,
            vectors,
            hash,
        )
        .unwrap();
        assert_eq!(actual.values, (experts * expert_rows * cols) as u64);
        assert_eq!(actual.values, expected_errors.values);
        assert_eq!(actual.source_squared, expected_errors.source_squared);
        assert_eq!(actual.error_squared, expected_errors.error_squared);
    }
}

#[test]
fn rotations_select_per_expert_ht_vectors_but_shared_i4r_vectors() {
    let mut tensor = TensorPlan {
        name: "layers.1.mlp.experts.down_proj.weight".into(),
        shape: vec![2, 256, 128],
        encoding: Encoding::Ht3,
        origin: Origin::Tensor {
            name: "expert".into(),
        },
        rotation: Some(Rotation {
            su: "su".into(),
            sv: "sv".into(),
        }),
    };
    let vectors = BTreeMap::from([
        ("su".into(), (0..256).map(|i| i as f32).collect()),
        ("sv".into(), (0..512).map(|i| i as f32).collect()),
    ]);
    let (su, sv) = rotation_slice(&tensor, &vectors, 384, 128).unwrap();
    assert_eq!(
        (su[0], su[127], sv[0], sv[127]),
        (128.0, 255.0, 384.0, 511.0)
    );
    assert!(rotation_slice(&tensor, &vectors, 128, 256).is_err());
    assert!(rotation_slice(&tensor, &vectors, 512, 128).is_err());
    tensor.encoding = Encoding::I4r;
    let (su, sv) = rotation_slice(&tensor, &vectors, 384, 128).unwrap();
    assert_eq!((su[0], su[127], sv[0], sv[127]), (0.0, 127.0, 128.0, 255.0));
}

#[test]
fn canonical_container_rejects_tiny_ht_experts_and_unknown_ht_variant() {
    let mut tensor = TensorPlan {
        name: "fixture.weight".into(),
        shape: vec![2, 256, 128],
        encoding: Encoding::Ht3,
        origin: Origin::Tensor {
            name: "source".into(),
        },
        rotation: None,
    };
    let mut plan = ContainerPlan {
        filename: "fixture.hgn".into(),
        identity: "fixture".into(),
        tensors: vec![tensor.clone()],
    };
    assert!(container::layout(&plan).is_err());
    tensor.shape = vec![128, 128];
    plan.tensors[0] = tensor;
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("fixture.hgn");
    Writer::create(&path, &plan).unwrap().finish().unwrap();
    let reader = container::Reader::open(&path).unwrap();
    assert_eq!(reader.layout.entries[0].encoding, Encoding::Ht3);
    assert_eq!(reader.layout.entries[0].size, 6144);
    let mut file = OpenOptions::new().write(true).open(&path).unwrap();
    use std::io::{Seek, SeekFrom};
    file.seek(SeekFrom::Start(container::HEADER_SIZE + 156))
        .unwrap();
    file.write_all(&0x1207u32.to_le_bytes()).unwrap();
    drop(file);
    assert!(container::Reader::open(&path).is_err());
}
