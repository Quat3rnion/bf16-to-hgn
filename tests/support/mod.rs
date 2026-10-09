use anyhow::Result;
use bf16_to_hgn::{
    codec::Encoding, hf::Snapshot, ContainerPlan, ConversionPlan, Origin, Rotation, TensorPlan,
};
use serde_json::json;
use std::{collections::BTreeMap, fs, path::Path};

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
    let mut header = serde_json::Map::new();
    let mut payload = Vec::new();
    let mut weight_map = BTreeMap::new();
    for (name, (dtype, shape, bytes)) in &tensors {
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
        serde_json::to_vec(
            &json!({ "metadata": { "total_size": payload.len() }, "weight_map": weight_map }),
        )?,
    )?;

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
