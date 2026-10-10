use anyhow::Result;
use bf16_to_hgn::{
    codec::Encoding, hf::Snapshot, manifest::OutputProfile, source::TensorSource, ContainerPlan,
    ConversionPlan, Origin, Rotation, TensorPlan,
};
use serde_json::json;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

/// Actual small safetensors inputs for every v2 codec; not an engine-compatible model.
pub fn fixture(parent: &Path) -> Result<(Snapshot, ConversionPlan)> {
    let source = parent.join("source");
    fs::create_dir(&source)?;
    fs::write(
        source.join("config.json"),
        b"{\"model_type\":\"structural_fixture\",\"architectures\":[\"StructuralFixture\"]}\n",
    )?;
    fs::write(
        source.join("chat_template.jinja"),
        b"{% for message in messages %}{{ message.content }}{% endfor %}\n",
    )?;
    let mut tensors = BTreeMap::<String, (String, Vec<u64>, Vec<u8>)>::new();
    add_bf16(&mut tensors, "fixture.raw.weight", &[4], false);
    tensors.insert(
        "fixture.raw_f32".into(),
        (
            "F32".into(),
            vec![4],
            [0.0f32, -1.0, 1.25, 0.125]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        ),
    );
    tensors.insert(
        "fixture.i64".into(),
        (
            "I64".into(),
            vec![3],
            [-1i64, 123, (1i64 << 55) + 3]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        ),
    );
    add_bf16(&mut tensors, "fixture.embed.weight", &[4, 64], false);
    add_bf16(&mut tensors, "fixture.affine8.weight", &[4, 64], false);
    add_bf16(&mut tensors, "fixture.affine6.weight", &[4, 320], false);
    add_bf16(&mut tensors, "fixture.dense.weight", &[128, 128], true);
    add_bf16(
        &mut tensors,
        "fixture.experts.gate_up",
        &[2, 256, 128],
        false,
    );
    add_bf16(&mut tensors, "fixture.experts.down", &[2, 128, 128], false);
    add_bf16(&mut tensors, "mtp.fc_hidden.weight", &[4, 64], false);
    add_bf16(
        &mut tensors,
        "model.visual.patch_embed.proj.weight",
        &[2, 1, 2, 2, 2],
        false,
    );
    let ngram_shards = (0..2)
        .map(|i| {
            format!(
                "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{i}.weight"
            )
        })
        .collect::<Vec<_>>();
    for (i, name) in ngram_shards.iter().enumerate() {
        let raw = (0..3 * 160)
            .flat_map(|k| {
                let value = if i == 0 {
                    0.0
                } else {
                    ((k * 19 % 63) as f32 - 31.0) / 128.0
                };
                half::bf16::from_f32(value).to_bits().to_le_bytes()
            })
            .collect();
        tensors.insert(name.clone(), ("BF16".into(), vec![3, 160], raw));
    }
    write_snapshot(&source, &tensors)?;

    let mut main = vec![
        direct(
            "fixture.raw.weight",
            "fixture.raw.weight",
            &[4],
            Encoding::Bf16,
        ),
        direct("fixture.raw_f32", "fixture.raw_f32", &[4], Encoding::F32),
        direct("fixture.i64", "fixture.i64", &[3], Encoding::I64),
        direct(
            "fixture.embed.weight",
            "fixture.embed.weight",
            &[4, 64],
            Encoding::Q4Cp,
        ),
        direct(
            "fixture.affine8.weight",
            "fixture.affine8.weight",
            &[4, 64],
            Encoding::Q8G64,
        ),
        direct(
            "fixture.affine6.weight",
            "fixture.affine6.weight",
            &[4, 320],
            Encoding::Q6G64,
        ),
        direct(
            "mtp.fc_hidden.weight",
            "mtp.fc_hidden.weight",
            &[4, 64],
            Encoding::Q8G64,
        ),
    ];
    main.push(signs("fixture.dense.suh", 128));
    main.push(TensorPlan {
        name: "fixture.dense.svh".into(),
        shape: vec![128],
        encoding: Encoding::F16,
        origin: Origin::HtScales {
            tensor: "fixture.dense.weight".into(),
            signs_key: "fixture.dense.svh".into(),
        },
        rotation: None,
    });
    let mut ht = direct(
        "fixture.dense.weight",
        "fixture.dense.weight",
        &[128, 128],
        Encoding::Ht,
    );
    ht.rotation = Some(Rotation {
        su: "fixture.dense.suh".into(),
        sv: "fixture.dense.svh".into(),
    });
    main.push(ht);
    for (name, size) in [
        ("fixture.experts.su", 128),
        ("fixture.experts.gate_up.svh", 256),
        ("fixture.experts.down.suh", 128),
        ("fixture.experts.down.svh", 128),
    ] {
        main.push(signs(name, size));
    }
    let mut gate = direct(
        "fixture.experts.gate_up.weight",
        "fixture.experts.gate_up",
        &[2, 256, 128],
        Encoding::I4r,
    );
    gate.rotation = Some(Rotation {
        su: "fixture.experts.su".into(),
        sv: "fixture.experts.gate_up.svh".into(),
    });
    main.push(gate);
    let mut down = direct(
        "fixture.experts.down.weight",
        "fixture.experts.down",
        &[2, 128, 128],
        Encoding::I4r,
    );
    down.rotation = Some(Rotation {
        su: "fixture.experts.down.suh".into(),
        sv: "fixture.experts.down.svh".into(),
    });
    main.push(down);
    main.sort_by(|a, b| a.name.cmp(&b.name));
    let plan = ConversionPlan {
        architecture: "structural-fixture-not-engine-checkpoint".into(),
        files: vec![
            ContainerPlan {
                filename: "model.hgn".into(),
                identity: "bf16-hgn-structural-fixture".into(),
                tensors: main,
            },
            ContainerPlan {
                filename: "qwen38-flash-next-ngram.hgn".into(),
                identity: "bf16-hgn-fixture-ngram".into(),
                tensors: vec![TensorPlan {
                    name: "layers.1.ple.ngram_embedding.weight".into(),
                    shape: vec![2, 3, 160],
                    encoding: Encoding::Fp8G128,
                    origin: Origin::Ngram {
                        shards: ngram_shards,
                    },
                    rotation: None,
                }],
            },
            ContainerPlan {
                filename: "qwen38-flash-next-vision.hgn".into(),
                identity: "bf16-hgn-fixture-vision".into(),
                tensors: vec![direct(
                    "visual.patch_embed.proj.weight",
                    "model.visual.patch_embed.proj.weight",
                    &[2, 8],
                    Encoding::Bf16,
                )],
            },
        ],
        assets: vec!["config.json".into(), "chat_template.jinja".into()],
    };
    Ok((Snapshot::open(&source)?, plan))
}

/// Small canonical matrices exercising HT4/HT3 and Q8G64 without pretending to
/// be an engine-compatible Flash-Next checkpoint or bypassing its planner.
pub fn matrix_fixture(parent: &Path, profile: OutputProfile) -> Result<(Snapshot, ConversionPlan)> {
    let source = parent.join("source");
    fs::create_dir(&source)?;
    fs::write(
        source.join("config.json"),
        b"{\"model_type\":\"structural_fixture\"}\n",
    )?;
    fs::write(source.join("chat_template.jinja"), b"{{ messages }}\n")?;
    let mut tensors = BTreeMap::new();
    let mut main = Vec::new();
    for layer in 0..2 {
        let target = format!("layers.{layer}.self_attn.q_proj.weight");
        let input = format!("model.language_model.{target}");
        add_bf16(&mut tensors, &input, &[128, 128], layer == 0);
        let stem = target.strip_suffix(".weight").expect("weight suffix");
        let su = format!("{stem}.suh");
        let sv = format!("{stem}.svh");
        main.push(signs(&su, 128));
        main.push(TensorPlan {
            name: sv.clone(),
            shape: vec![128],
            encoding: Encoding::F16,
            origin: Origin::HtScales {
                tensor: input.clone(),
                signs_key: sv.clone(),
            },
            rotation: None,
        });
        let encoding = if profile == OutputProfile::Ht43 && layer == 1 {
            Encoding::Ht3
        } else {
            Encoding::Ht
        };
        let mut weight = direct(&target, &input, &[128, 128], encoding);
        weight.rotation = Some(Rotation { su, sv });
        main.push(weight);
    }
    let hyper = "model.language_model.layers.0.hyper_connection_mixer.weight";
    add_bf16(&mut tensors, hyper, &[4, 320], false);
    main.push(direct(
        "layers.0.hyper_connection_mixer.weight",
        hyper,
        &[4, 320],
        if profile == OutputProfile::Ht43 {
            Encoding::Q8G64
        } else {
            Encoding::Q6G64
        },
    ));
    let log = "model.language_model.layers.0.linear_attn.A_log";
    tensors.insert(
        log.into(),
        (
            "F32".into(),
            vec![4],
            [0.0f32, -1.001, 1.234567, 0.00001234]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        ),
    );
    main.push(direct(
        "layers.0.linear_attn.A_log",
        log,
        &[4],
        Encoding::F32,
    ));
    write_snapshot(&source, &tensors)?;
    main.sort_by(|a, b| a.name.cmp(&b.name));
    let plan = ConversionPlan {
        architecture: "matrix-fixture-not-engine-checkpoint".into(),
        files: vec![ContainerPlan {
            filename: "model.hgn".into(),
            identity: match profile {
                OutputProfile::V2 => "bf16-hgn-v2-matrix-fixture",
                OutputProfile::Ht43 => "bf16-hgn-ht43-matrix-fixture",
            }
            .into(),
            tensors: main,
        }],
        assets: vec!["config.json".into(), "chat_template.jinja".into()],
    };
    Ok((Snapshot::open(&source)?, plan))
}

/// Give a library-only importer fixture matching real on-disk asset bytes.
/// The production architecture planner still rejects the structural config.
#[derive(Debug)]
pub struct FixtureAssets<'a> {
    pub tensors: &'a dyn TensorSource,
    pub assets: &'a Snapshot,
}
impl TensorSource for FixtureAssets<'_> {
    fn root(&self) -> &Path {
        self.tensors.root()
    }
    fn tensors(&self) -> &BTreeMap<String, bf16_to_hgn::hf::TensorInfo> {
        self.tensors.tensors()
    }
    fn config(&self) -> &serde_json::Value {
        &self.assets.config
    }
    fn kind(&self) -> bf16_to_hgn::source::SourceKind {
        self.tensors.kind()
    }
    fn asset_path(&self, name: &str) -> Result<PathBuf> {
        self.assets.asset_path(name)
    }
    fn read_bytes(&self, name: &str, offset: u64, len: usize) -> Result<Vec<u8>> {
        self.tensors.read_bytes(name, offset, len)
    }
    fn read_f32(&self, name: &str, offset: u64, len: usize) -> Result<Vec<f32>> {
        self.tensors.read_f32(name, offset, len)
    }
    fn hash_tensor(&self, name: &str) -> Result<String> {
        self.tensors.hash_tensor(name)
    }
    fn check_unchanged(&self) -> Result<()> {
        self.tensors.check_unchanged()?;
        self.assets.check_unchanged()
    }
    fn provenance(&self) -> Result<serde_json::Value> {
        self.check_unchanged()?;
        let (_, config_sha256) =
            bf16_to_hgn::conversion::hash_file(&self.asset_path("config.json")?)?;
        Ok(
            json!({ "tensor_input": self.tensors.provenance()?, "asset_config_sha256": config_sha256 }),
        )
    }
    fn input_paths(&self) -> Vec<PathBuf> {
        let mut paths = self.tensors.input_paths();
        paths.extend(self.assets.input_paths());
        paths
    }
}

pub fn imported_norm_fixture(
    parent: &Path,
    values: &[f32],
) -> Result<(bf16_to_hgn::hgn_source::HgnSidecar, ConversionPlan)> {
    let name = "layers.0.input_layernorm.weight";
    let shape = vec![u64::try_from(values.len())?];
    let input = ContainerPlan {
        filename: "input.hgn".into(),
        identity: "bf16-hgn-f32-norm-fixture".into(),
        tensors: vec![direct(name, name, &shape, Encoding::F32)],
    };
    let mut writer = bf16_to_hgn::container::Writer::create(&parent.join("input.hgn"), &input)?;
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    writer.write_at(0, 0, &bytes)?;
    writer.finish()?;
    let source = bf16_to_hgn::hgn_source::HgnSidecar::open(&parent.join("input.hgn"))?;
    let plan = ConversionPlan {
        architecture: "norm-fixture-not-engine-checkpoint".into(),
        files: vec![ContainerPlan {
            filename: "model.hgn".into(),
            identity: "bf16-hgn-bf16-norm-fixture".into(),
            tensors: vec![direct(
                name,
                &format!("model.language_model.{name}"),
                &shape,
                Encoding::Bf16,
            )],
        }],
        assets: vec!["config.json".into(), "chat_template.jinja".into()],
    };
    Ok((source, plan))
}

/// Independently serialized GGUF v3 bytes: raw F32, Q8_0, integer PLE metadata,
/// and a tiny embedded table. The importer still applies its real name mapping.
pub fn gguf_matrix_fixture(
    parent: &Path,
) -> Result<(bf16_to_hgn::gguf_source::GgufSidecar, ConversionPlan)> {
    let raw = (0..128 * 128)
        .flat_map(|i| (((i * 17 % 67) as f32 - 33.0) / 37.0).to_le_bytes())
        .collect::<Vec<_>>();
    let mut q8 = Vec::new();
    for block in 0..128 * 128 / 32 {
        q8.extend_from_slice(&0x2800u16.to_le_bytes()); // exact f16 scale 1/32
        q8.extend(
            (0..32)
                .map(|i| ((block * 32 + i) % 63) as i8 - 31)
                .map(|v| v as u8),
        );
    }
    let table = (0..128 * 160)
        .flat_map(|i| (((i * 7 % 101) as f32 - 50.0) / 128.0).to_le_bytes())
        .collect::<Vec<_>>();
    let tensors = [
        ("blk.0.attn_q.weight", vec![128u64, 128], 0u32, raw),
        ("blk.1.attn_q.weight", vec![128u64, 128], 8u32, q8),
        (
            "per_layer_token_embd.weight",
            vec![160u64, 128],
            0u32,
            table,
        ),
    ];
    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&10u64.to_le_bytes());
    gguf_string(&mut bytes, "general.architecture");
    bytes.extend_from_slice(&8u32.to_le_bytes());
    gguf_string(&mut bytes, "qwen4exp");
    for (key, value) in [
        ("ssm.group_count", 16u32),
        ("ssm.time_step_rank", 48),
        ("ssm.state_size", 128),
        ("block_count", 48),
        ("expert_count", 512),
        ("embedding_length", 2560),
    ] {
        gguf_string(&mut bytes, &format!("qwen4exp.{key}"));
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    for (key, values) in [
        ("layer_multipliers", vec![1i64, 3, 5]),
        ("head_offsets", vec![0i64; 16]),
        ("head_vocab_sizes", vec![1i64; 16]),
    ] {
        gguf_string(&mut bytes, &format!("qwen4exp.ple.{key}"));
        bytes.extend_from_slice(&9u32.to_le_bytes());
        bytes.extend_from_slice(&11u32.to_le_bytes());
        bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    let mut payload = Vec::new();
    for (name, dimensions, kind, data) in tensors {
        while payload.len() % 32 != 0 {
            payload.push(0);
        }
        gguf_string(&mut bytes, name);
        bytes.extend_from_slice(&(dimensions.len() as u32).to_le_bytes());
        for dimension in dimensions {
            bytes.extend_from_slice(&dimension.to_le_bytes());
        }
        bytes.extend_from_slice(&kind.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        payload.extend(data);
    }
    while bytes.len() % 32 != 0 {
        bytes.push(0);
    }
    bytes.extend(payload);
    let path = parent.join("input.gguf");
    fs::write(&path, bytes)?;
    let source = bf16_to_hgn::gguf_source::GgufSidecar::open(&path)?;
    let mut main = Vec::new();
    let mut shards = Vec::new();
    for (input, info) in source.tensors() {
        if input.contains(".ngram_embedding.shard_") {
            shards.push(input.clone());
        } else if info.dtype == bf16_to_hgn::hf::Dtype::I64 {
            let target = input.replace(
                "model.language_model.layers.1.ple.ple_embedding.",
                "layers.1.ple.",
            );
            main.push(direct(&target, input, &info.shape, Encoding::I64));
        } else {
            let target = input
                .strip_prefix("model.language_model.")
                .expect("canonical GGUF tensor");
            let stem = target.strip_suffix(".weight").expect("matrix weight");
            let su = format!("{stem}.suh");
            let sv = format!("{stem}.svh");
            main.push(signs(&su, 128));
            main.push(TensorPlan {
                name: sv.clone(),
                shape: vec![128],
                encoding: Encoding::F16,
                origin: Origin::HtScales {
                    tensor: input.clone(),
                    signs_key: sv.clone(),
                },
                rotation: None,
            });
            let mut weight = direct(
                target,
                input,
                &info.shape,
                if target.starts_with("layers.1.") {
                    Encoding::Ht3
                } else {
                    Encoding::Ht
                },
            );
            weight.rotation = Some(Rotation { su, sv });
            main.push(weight);
        }
    }
    shards.sort_by_key(|name| {
        name.rsplit_once("shard_")
            .expect("shard name")
            .1
            .strip_suffix(".weight")
            .expect("shard suffix")
            .parse::<u64>()
            .expect("shard number")
    });
    main.sort_by(|a, b| a.name.cmp(&b.name));
    let plan = ConversionPlan {
        architecture: "gguf-matrix-fixture-not-engine-checkpoint".into(),
        files: vec![
            ContainerPlan {
                filename: "model.hgn".into(),
                identity: "gguf-hgn-ht43-matrix-fixture".into(),
                tensors: main,
            },
            ContainerPlan {
                filename: "ngram.hgn".into(),
                identity: "gguf-hgn-ht43-ngram-fixture".into(),
                tensors: vec![TensorPlan {
                    name: "layers.1.ple.ngram_embedding.weight".into(),
                    shape: vec![128, 1, 160],
                    encoding: Encoding::Fp8G128,
                    origin: Origin::Ngram { shards },
                    rotation: None,
                }],
            },
        ],
        assets: vec!["config.json".into(), "chat_template.jinja".into()],
    };
    Ok((source, plan))
}

fn gguf_string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn write_snapshot(
    source: &Path,
    tensors: &BTreeMap<String, (String, Vec<u64>, Vec<u8>)>,
) -> Result<()> {
    let mut header = serde_json::Map::new();
    let mut payload = Vec::new();
    let mut weight_map = BTreeMap::new();
    for (name, (dtype, shape, bytes)) in tensors {
        let start = payload.len();
        payload.extend_from_slice(bytes);
        header.insert(
            name.clone(),
            json!({ "dtype": dtype, "shape": shape, "data_offsets": [start, payload.len()] }),
        );
        weight_map.insert(name.clone(), "model.safetensors");
    }
    let mut encoded_header = serde_json::to_vec(&header)?;
    while encoded_header.len() % 8 != 0 {
        encoded_header.push(b' ');
    }
    let mut safetensors = (encoded_header.len() as u64).to_le_bytes().to_vec();
    safetensors.extend(encoded_header);
    safetensors.extend_from_slice(&payload);
    fs::write(source.join("model.safetensors"), safetensors)?;
    fs::write(
        source.join("model.safetensors.index.json"),
        serde_json::to_vec(&json!({
            "metadata": { "total_size": payload.len() }, "weight_map": weight_map
        }))?,
    )?;
    Ok(())
}

fn direct(name: &str, source: &str, shape: &[u64], encoding: Encoding) -> TensorPlan {
    TensorPlan {
        name: name.into(),
        shape: shape.into(),
        encoding,
        origin: Origin::Tensor {
            name: source.into(),
        },
        rotation: None,
    }
}
fn signs(name: &str, count: u64) -> TensorPlan {
    TensorPlan {
        name: name.into(),
        shape: vec![count],
        encoding: Encoding::F16,
        origin: Origin::Signs { key: name.into() },
        rotation: None,
    }
}
fn add_bf16(
    tensors: &mut BTreeMap<String, (String, Vec<u64>, Vec<u8>)>,
    name: &str,
    shape: &[u64],
    zero_first_row: bool,
) {
    let elements = shape.iter().product::<u64>() as usize;
    let columns = *shape.last().expect("nonempty shape") as usize;
    let raw = (0..elements)
        .flat_map(|i| {
            let value = if zero_first_row && i < columns {
                0.0
            } else {
                ((i * 37 % 127) as f32 - 63.0) / 80.0
            };
            half::bf16::from_f32(value).to_bits().to_le_bytes()
        })
        .collect();
    tensors.insert(name.into(), ("BF16".into(), shape.into(), raw));
}
