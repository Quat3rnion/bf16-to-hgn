use super::*;
use crate::{hf::Snapshot, manifest::OutputProfile, ContainerPlan, Rotation};
use serde_json::json;

fn expert_source(root: &Path, shape: &[u64], input: &[f32]) -> Snapshot {
    tensor_source(root, "expert", shape, input)
}

fn tensor_source(root: &Path, name: &str, shape: &[u64], input: &[f32]) -> Snapshot {
    fs::create_dir(root).unwrap();
    fs::write(root.join("config.json"), b"{}").unwrap();
    let raw: Vec<u8> = input
        .iter()
        .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
        .collect();
    let mut header = serde_json::to_vec(&json!({
        (name): {"dtype": "BF16", "shape": shape, "data_offsets": [0, raw.len()]}
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
            "metadata": {"total_size": raw.len()}, "weight_map": {(name): "model.safetensors"}
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

#[test]
fn conversion_work_counts_ngram_passes_and_rotation_writes_not_preparation() {
    let directory = tempfile::tempdir().unwrap();
    let name = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight";
    let snapshot = tensor_source(
        &directory.path().join("source"),
        name,
        &[2, 160],
        &[0.0; 320],
    );
    let plan = ConversionPlan {
        architecture: "work-pass-fixture".into(),
        files: vec![
            ContainerPlan {
                filename: "model.hgn".into(),
                identity: "work-pass-fixture".into(),
                tensors: vec![
                    TensorPlan {
                        name: "fixture.su".into(),
                        shape: vec![128],
                        encoding: Encoding::F16,
                        origin: Origin::Signs {
                            key: "fixture.su".into(),
                        },
                        rotation: None,
                    },
                    TensorPlan {
                        name: "fixture.sv".into(),
                        shape: vec![2],
                        encoding: Encoding::F16,
                        origin: Origin::HtScales {
                            tensor: name.into(),
                            signs_key: "fixture.sv".into(),
                        },
                        rotation: None,
                    },
                ],
            },
            ContainerPlan {
                filename: "qwen38-flash-next-ngram.hgn".into(),
                identity: "work-pass-fixture".into(),
                tensors: vec![TensorPlan {
                    name: "layers.0.ple.ngram_embedding.weight".into(),
                    shape: vec![1, 2, 160],
                    encoding: Encoding::Fp8G128,
                    origin: Origin::Ngram {
                        shards: vec![name.into()],
                    },
                    rotation: None,
                }],
            },
        ],
        assets: Vec::new(),
    };
    validate_plan(&snapshot, &plan).unwrap();
    for backend in [
        codec::HtBackend::Cpu,
        codec::HtBackend::Hip,
        codec::HtBackend::Cuda,
    ] {
        let options = Options {
            ht_backend: backend,
            ..Options::default()
        };
        let estimate = conversion_work(&snapshot, &plan, &options).unwrap();
        // 320 values in each table pass, plus 128 signs and two scale values
        // written once. The earlier 320-value HT scale scan is not counted here.
        assert_eq!(estimate.total(), 770);
        assert_eq!(estimate.planned_values(WorkClass::ScanFp8Scale), 320);
        assert_eq!(
            estimate.planned_values(WorkClass::encode(Encoding::Fp8G128, backend)),
            320
        );
        assert_eq!(estimate.planned_values(WorkClass::WriteRotation), 130);
        for encoding in [Encoding::Ht, Encoding::Ht3] {
            assert_eq!(
                estimate.planned_values(WorkClass::encode(encoding, backend)),
                0
            );
        }
        let mut overflow = plan.clone();
        overflow.files[0].tensors[0].shape = vec![u64::MAX];
        assert!(conversion_work(&snapshot, &overflow, &options).is_err());
    }
}

#[test]
fn conversion_work_classifies_copy_reencoding_and_actual_backend() {
    for backend in [
        codec::HtBackend::Cpu,
        codec::HtBackend::Hip,
        codec::HtBackend::Cuda,
    ] {
        let options = Options {
            ht_backend: backend,
            ..Options::default()
        };
        for (encoding, dtype) in [
            (Encoding::Bf16, Dtype::Bf16),
            (Encoding::F32, Dtype::F32),
            (Encoding::I64, Dtype::I64),
        ] {
            let class = source_work(encoding, dtype, &options);
            assert_eq!(class, WorkClass::Copy(encoding));
            assert_eq!(class.backend_label(), "CPU");
        }
        // Imported F32 values converted back to BF16 are encoding work, not copies.
        assert_eq!(
            source_work(Encoding::Bf16, Dtype::F32, &options),
            WorkClass::encode(Encoding::Bf16, codec::HtBackend::Cpu)
        );
        for encoding in [Encoding::Ht, Encoding::Ht3] {
            let class = source_work(encoding, Dtype::Bf16, &options);
            assert_eq!(class, WorkClass::Encode(encoding, backend));
            assert_eq!(class.backend_label(), backend.label());
        }
        for encoding in [
            Encoding::F16,
            Encoding::F32,
            Encoding::I64,
            Encoding::Fp8G128,
            Encoding::I4r,
            Encoding::Q4Cp,
            Encoding::Q8G64,
            Encoding::Q6G64,
        ] {
            let class = source_work(encoding, Dtype::Bf16, &options);
            assert_eq!(class, WorkClass::Encode(encoding, codec::HtBackend::Cpu));
            assert_eq!(class.backend_label(), "CPU");
        }
    }
    assert_eq!(WorkClass::WriteRotation.backend_label(), "CPU");
    assert_eq!(WorkClass::ScanFp8Scale.backend_label(), "CPU");
}

#[test]
fn conversion_work_slots_keep_cpu_hip_cuda_and_nonencoding_classes_distinct() {
    let mut classes = Vec::new();
    for encoding in [
        Encoding::Bf16,
        Encoding::F16,
        Encoding::F32,
        Encoding::I64,
        Encoding::Q4Cp,
        Encoding::Q8G64,
        Encoding::Q6G64,
        Encoding::Fp8G128,
        Encoding::Ht,
        Encoding::Ht3,
        Encoding::I4r,
    ] {
        classes.push(WorkClass::encode(encoding, codec::HtBackend::Cpu));
    }
    for backend in [codec::HtBackend::Hip, codec::HtBackend::Cuda] {
        for encoding in [Encoding::Ht, Encoding::Ht3] {
            classes.push(WorkClass::encode(encoding, backend));
        }
    }
    for encoding in [Encoding::Bf16, Encoding::F32, Encoding::I64] {
        classes.push(WorkClass::Copy(encoding));
    }
    classes.extend([WorkClass::WriteRotation, WorkClass::ScanFp8Scale]);
    let mut estimate = WorkEstimate::default();
    for (index, &class) in classes.iter().enumerate() {
        estimate.add(class, index as u64 + 1).unwrap();
    }
    assert_eq!(estimate.total(), 210);
    for (index, &class) in classes.iter().enumerate() {
        assert_eq!(
            estimate.planned_values(class),
            index as u64 + 1,
            "{class:?}"
        );
    }
}
