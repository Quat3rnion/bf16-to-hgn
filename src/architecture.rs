use std::collections::BTreeMap;
use std::fs;
use std::io::ErrorKind;
use std::path::Path;

use anyhow::{ensure, Context, Result};
use serde_json::Value;

use crate::codec::Encoding;
use crate::hf::{Dtype, Snapshot, TensorInfo};
use crate::{ContainerPlan, ConversionPlan, Origin, Rotation, TensorPlan};

// hgn-spec 1.1.0, 79e75fd4ae7642478823525b0a3212951f584af5:
// TENSORS.md and GGUF-IMPORT.md (the latter supplies ordering, not HF transforms).
const LAYERS: u64 = 48;
const HIDDEN: u64 = 2560;
const STREAMS: u64 = 4;
const HC_WIDTH: u64 = STREAMS * HIDDEN;
const HC_RANK: u64 = 320;
const VOCAB: u64 = 248320;
const EXPERTS: u64 = 512;
const EXPERT_WIDTH: u64 = 640;
const NGRAM_PARTS: u64 = 128;
const NGRAM_ROWS: u64 = 2500012;
const NGRAM_WIDTH: u64 = 160;
const VISION_BLOCKS: u64 = 27;
const VISION_WIDTH: u64 = 1152;
const VISION_INTERMEDIATE: u64 = 4304;
const VISION_MERGED: u64 = 4608;
const SOURCE_COUNT: usize = 1658;
const MAIN_SOURCE_COUNT: usize = 1197;
const MAIN_COUNT: usize = 2015;
const VISION_COUNT: usize = 333;
const HF_LANGUAGE_PREFIX: &str = "model.language_model.";
const HF_NGRAM_PREFIX: &str =
    "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_";

const REQUIRED_ASSETS: &[&str] = &[
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "merges.txt",
    "generation_config.json",
];
const OPTIONAL_ASSETS: &[&str] = &[
    "vocab.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
];

// These are HF config values, including its one-based ple_layer_ids. In
// particular, [2] here means HGN layers.1, not layers.2. Provenance such as
// transformers_version is not a model constant and is deliberately not gated.
const INTEGER_CONFIG: &[(&str, u64)] = &[
    ("/image_token_id", 248056),
    ("/video_token_id", 248057),
    ("/vision_end_token_id", 248054),
    ("/vision_start_token_id", 248053),
    ("/text_config/bos_token_id", 248044),
    ("/text_config/eos_token_id", 248044),
    ("/text_config/full_attention_interval", 4),
    ("/text_config/hc_count", STREAMS),
    ("/text_config/hc_lowrank", HC_RANK),
    ("/text_config/head_dim", 256),
    ("/text_config/heads_per_ngram", 8),
    ("/text_config/hidden_size", HIDDEN),
    ("/text_config/indexer_budget", 2048),
    ("/text_config/indexer_compress_ratio", 4),
    ("/text_config/indexer_head_dim", 128),
    ("/text_config/indexer_kv_heads", 1),
    ("/text_config/indexer_n_heads", 4),
    ("/text_config/linear_conv_kernel_dim", 4),
    ("/text_config/linear_key_head_dim", 128),
    ("/text_config/linear_num_key_heads", 16),
    ("/text_config/linear_num_value_heads", 48),
    ("/text_config/linear_value_head_dim", 128),
    (
        "/text_config/make_ngram_vocab_size_divisible_by",
        NGRAM_PARTS,
    ),
    ("/text_config/max_position_embeddings", 262144),
    ("/text_config/moe_intermediate_size", EXPERT_WIDTH),
    ("/text_config/mtp/num_hidden_layers", 1),
    ("/text_config/mtp/rope_theta", 10000000),
    ("/text_config/mtp_num_hidden_layers", 1),
    ("/text_config/ngram_size", 3),
    ("/text_config/ngram_vocab_size_base", 20000000),
    ("/text_config/num_attention_heads", 24),
    ("/text_config/num_experts", EXPERTS),
    ("/text_config/num_experts_per_tok", 10),
    ("/text_config/num_hidden_layers", LAYERS),
    ("/text_config/num_key_value_heads", 2),
    ("/text_config/ple_conv_kernel_size", 4),
    ("/text_config/ple_embed_dim", HIDDEN),
    ("/text_config/rope_parameters/rope_theta", 10000000),
    ("/text_config/shared_expert_intermediate_size", EXPERT_WIDTH),
    ("/text_config/split_ngram_parts", NGRAM_PARTS),
    ("/text_config/vocab_size", VOCAB),
    ("/vision_config/depth", VISION_BLOCKS),
    ("/vision_config/hidden_size", VISION_WIDTH),
    ("/vision_config/in_channels", 3),
    ("/vision_config/intermediate_size", VISION_INTERMEDIATE),
    ("/vision_config/num_heads", 16),
    ("/vision_config/num_position_embeddings", 2304),
    ("/vision_config/out_hidden_size", HIDDEN),
    ("/vision_config/patch_size", 16),
    ("/vision_config/spatial_merge_size", 2),
    ("/vision_config/temporal_patch_size", 2),
];
const STRING_CONFIG: &[(&str, &str)] = &[
    ("/model_type", "qwen4_exp"),
    ("/text_config/dtype", "bfloat16"),
    ("/text_config/hidden_act", "silu"),
    ("/text_config/mamba_ssm_dtype", "float32"),
    ("/text_config/model_type", "qwen4_exp_text"),
    ("/text_config/output_gate_type", "sigmoid"),
    ("/text_config/rope_parameters/rope_type", "default"),
    ("/vision_config/hidden_act", "gelu_pytorch_tanh"),
    ("/vision_config/model_type", "qwen4_exp"),
];
const BOOL_CONFIG: &[(&str, bool)] = &[
    ("/language_model_only", false),
    ("/tie_word_embeddings", false),
    ("/text_config/attention_bias", false),
    ("/text_config/mtp/hybrid", true),
    ("/text_config/mtp_use_dedicated_embeddings", false),
    ("/text_config/output_router_logits", false),
    ("/text_config/rope_parameters/mrope_interleaved", true),
    ("/text_config/tie_word_embeddings", false),
    ("/text_config/use_cache", true),
];
const FLOAT_CONFIG: &[(&str, f64)] = &[
    ("/text_config/attention_dropout", 0.0),
    ("/text_config/initializer_range", 0.02),
    ("/text_config/partial_rotary_factor", 0.25),
    ("/text_config/rms_norm_eps", 1e-6),
    ("/text_config/rope_parameters/partial_rotary_factor", 0.25),
    ("/text_config/router_aux_loss_coef", 0.001),
    ("/vision_config/initializer_range", 0.02),
];

pub fn plan(snapshot: &Snapshot) -> Result<ConversionPlan> {
    let mut result = plan_contract(&snapshot.config, &snapshot.tensors)?;
    result.assets = plan_assets(&snapshot.root)?;
    Ok(result)
}

fn validate_config(config: &Value) -> Result<()> {
    require_strings(
        config,
        "/architectures",
        &["Qwen4ExpForConditionalGeneration"],
    )?;
    for &(path, expected) in INTEGER_CONFIG {
        ensure!(
            config.pointer(path).and_then(Value::as_u64) == Some(expected),
            "config {path} must be the integer {expected}"
        );
    }
    for &(path, expected) in STRING_CONFIG {
        ensure!(
            config.pointer(path).and_then(Value::as_str) == Some(expected),
            "config {path} must be {expected:?}"
        );
    }
    for &(path, expected) in BOOL_CONFIG {
        ensure!(
            config.pointer(path).and_then(Value::as_bool) == Some(expected),
            "config {path} must be {expected}"
        );
    }
    for &(path, expected) in FLOAT_CONFIG {
        ensure!(
            config.pointer(path).and_then(Value::as_f64) == Some(expected),
            "config {path} must be {expected}"
        );
    }
    for path in [
        "/text_config/pad_token_id",
        "/text_config/mtp/mtp_use_hidden_state_from_layer",
    ] {
        ensure!(
            config.pointer(path) == Some(&Value::Null),
            "config {path} must be null"
        );
    }
    require_integers(config, "/text_config/ple_layer_ids", &[2])?;
    require_integers(
        config,
        "/text_config/rope_parameters/mrope_section",
        &[11, 11, 10],
    )?;
    require_integers(config, "/vision_config/deepstack_visual_indexes", &[])?;
    require_strings(config, "/text_config/mtp/layer_types", &["full_attention"])?;
    let layer_types = config
        .pointer("/text_config/layer_types")
        .and_then(Value::as_array)
        .context("config /text_config/layer_types must be an array")?;
    ensure!(
        layer_types.len() == LAYERS as usize,
        "config /text_config/layer_types must have {LAYERS} entries"
    );
    for (layer, actual) in layer_types.iter().enumerate() {
        let expected = if layer % 4 == 3 {
            "full_attention"
        } else {
            "linear_attention"
        };
        ensure!(
            actual.as_str() == Some(expected),
            "config /text_config/layer_types/{layer} must be {expected:?}"
        );
    }
    for path in [
        "/quantization_config",
        "/text_config/quantization_config",
        "/text_config/rope_scaling",
    ] {
        if let Some(value) = config.pointer(path) {
            ensure!(value.is_null(), "unsupported config override {path}");
        }
    }
    Ok(())
}

fn require_strings(config: &Value, path: &str, expected: &[&str]) -> Result<()> {
    let values = config.pointer(path).and_then(Value::as_array);
    ensure!(
        values.is_some_and(|values| {
            values.len() == expected.len()
                && values
                    .iter()
                    .zip(expected)
                    .all(|(value, expected)| value.as_str() == Some(*expected))
        }),
        "config {path} must be {expected:?}"
    );
    Ok(())
}

fn require_integers(config: &Value, path: &str, expected: &[u64]) -> Result<()> {
    let values = config.pointer(path).and_then(Value::as_array);
    ensure!(
        values.is_some_and(|values| {
            values.len() == expected.len()
                && values
                    .iter()
                    .zip(expected)
                    .all(|(value, expected)| value.as_u64() == Some(*expected))
        }),
        "config {path} must be {expected:?}"
    );
    Ok(())
}

fn plan_assets(root: &Path) -> Result<Vec<String>> {
    let mut assets = Vec::with_capacity(REQUIRED_ASSETS.len() + OPTIONAL_ASSETS.len());
    for (names, required) in [(REQUIRED_ASSETS, true), (OPTIONAL_ASSETS, false)] {
        for &name in names {
            match fs::symlink_metadata(root.join(name)) {
                Ok(metadata) => {
                    ensure!(
                        metadata.file_type().is_file(),
                        "asset {name} must be a regular file, not a symlink or directory"
                    );
                    assets.push(name.to_owned());
                }
                Err(error) if error.kind() == ErrorKind::NotFound && !required => {}
                Err(error) => return Err(error).with_context(|| format!("required asset {name}")),
            }
        }
    }
    assets.sort_unstable();
    Ok(assets)
}

#[derive(Clone, Copy)]
enum SourceDtype {
    Bf16,
    Bf16OrF32,
    I64,
}

struct SourceRequirement {
    shape: Vec<u64>,
    dtype: SourceDtype,
}

struct Contract {
    sources: BTreeMap<String, SourceRequirement>,
    main: Vec<TensorPlan>,
    vision: Vec<TensorPlan>,
    ngram_shards: Vec<String>,
}

impl Contract {
    fn add_source(&mut self, name: String, shape: &[u64], dtype: SourceDtype) {
        let previous = self.sources.insert(
            name,
            SourceRequirement {
                shape: shape.to_vec(),
                dtype,
            },
        );
        assert!(
            previous.is_none(),
            "duplicate name in fixed source contract"
        );
    }

    fn add_main(&mut self, name: String, shape: &[u64], encoding: Encoding) {
        let source_suffix = if name.ends_with(".mlp.experts.gate_up_proj.weight")
            || name.ends_with(".mlp.experts.down_proj.weight")
        {
            name.strip_suffix(".weight").unwrap()
        } else {
            &name
        };
        let source = if name.starts_with("mtp.") || name == "lm_head.weight" {
            source_suffix.to_owned()
        } else {
            format!("{HF_LANGUAGE_PREFIX}{source_suffix}")
        };
        let dtype = if encoding == Encoding::I64 {
            SourceDtype::I64
        } else if name.ends_with(".linear_attn.A_log") || name.ends_with(".linear_attn.dt_bias") {
            SourceDtype::Bf16OrF32
        } else {
            SourceDtype::Bf16
        };
        self.add_source(source.clone(), shape, dtype);
        let rotation = match encoding {
            Encoding::Ht | Encoding::I4r => {
                let base = name.strip_suffix(".weight").unwrap();
                let su = if encoding == Encoding::I4r {
                    match base.strip_suffix(".gate_up_proj") {
                        Some(experts) => format!("{experts}.su"),
                        None => format!("{base}.suh"),
                    }
                } else {
                    format!("{base}.suh")
                };
                let sv = format!("{base}.svh");
                let rows = shape[shape.len() - 2];
                let cols = shape[shape.len() - 1];
                self.main.push(signs(&su, cols));
                self.main.push(TensorPlan {
                    name: sv.clone(),
                    shape: vec![rows],
                    encoding: Encoding::F16,
                    origin: if encoding == Encoding::Ht {
                        Origin::HtScales {
                            tensor: source.clone(),
                            signs_key: sv.clone(),
                        }
                    } else {
                        Origin::Signs { key: sv.clone() }
                    },
                    rotation: None,
                });
                Some(Rotation { su, sv })
            }
            _ => None,
        };
        self.main.push(TensorPlan {
            name,
            shape: shape.to_vec(),
            encoding,
            origin: Origin::Tensor { name: source },
            rotation,
        });
    }

    fn add_main_group(&mut self, prefix: &str, specs: &[(&str, &[u64], Encoding)]) {
        for &(suffix, shape, encoding) in specs {
            self.add_main(format!("{prefix}{suffix}"), shape, encoding);
        }
    }

    fn add_mixer(&mut self, prefix: &str, encoding: Encoding, block_inject: bool) {
        self.add_main_group(
            prefix,
            &[
                ("hc_norm.weight", &[HC_WIDTH], Encoding::Bf16),
                (
                    "input_mix_weight_down.weight",
                    &[HC_RANK, HC_WIDTH],
                    encoding,
                ),
                ("input_mix_weight_up.weight", &[HC_WIDTH, HC_RANK], encoding),
            ],
        );
        if block_inject {
            self.add_main(
                format!("{prefix}block_inject_weight.weight"),
                &[STREAMS, HC_WIDTH],
                encoding,
            );
        }
    }

    fn add_layer(&mut self, prefix: &str, full_attention: bool, mtp: bool) {
        let dense = if mtp { Encoding::Q8G64 } else { Encoding::Ht };
        let hyper = if mtp {
            Encoding::Q8G64
        } else {
            Encoding::Q6G64
        };
        self.add_mixer(&format!("{prefix}attn_hyper_connection."), hyper, true);
        self.add_mixer(&format!("{prefix}mlp_hyper_connection."), hyper, true);
        if full_attention {
            self.add_main_group(
                &format!("{prefix}self_attn."),
                &[
                    ("q_proj.weight", &[12288, HIDDEN], dense),
                    ("k_proj.weight", &[512, HIDDEN], dense),
                    ("v_proj.weight", &[512, HIDDEN], dense),
                    ("o_proj.weight", &[HIDDEN, 6144], dense),
                    ("q_norm.weight", &[256], Encoding::Bf16),
                    ("k_norm.weight", &[256], Encoding::Bf16),
                    ("indexer.index_qk_proj.weight", &[640, HIDDEN], dense),
                    ("indexer.q_layernorm.weight", &[128], Encoding::Bf16),
                    ("indexer.k_layernorm.weight", &[128], Encoding::Bf16),
                ],
            );
        } else {
            self.add_main_group(
                &format!("{prefix}linear_attn."),
                &[
                    ("in_proj_qkv.weight", &[10240, HIDDEN], dense),
                    ("in_proj_z.weight", &[6144, HIDDEN], dense),
                    ("in_proj_a.weight", &[48, HIDDEN], Encoding::Q4Cp),
                    ("in_proj_b.weight", &[48, HIDDEN], Encoding::Q4Cp),
                    ("out_proj.weight", &[HIDDEN, 6144], dense),
                    ("conv1d.weight", &[10240, 1, 4], Encoding::Bf16),
                    ("norm.weight", &[128], Encoding::Bf16),
                    ("A_log", &[48], Encoding::Bf16),
                    ("dt_bias", &[48], Encoding::Bf16),
                ],
            );
        }
        let expert = if mtp { Encoding::Q4Cp } else { Encoding::I4r };
        self.add_main_group(
            &format!("{prefix}mlp."),
            &[
                ("gate.weight", &[EXPERTS, HIDDEN], Encoding::Bf16),
                (
                    "shared_expert.gate_proj.weight",
                    &[EXPERT_WIDTH, HIDDEN],
                    dense,
                ),
                (
                    "shared_expert.up_proj.weight",
                    &[EXPERT_WIDTH, HIDDEN],
                    dense,
                ),
                (
                    "shared_expert.down_proj.weight",
                    &[HIDDEN, EXPERT_WIDTH],
                    dense,
                ),
                ("shared_expert_gate.weight", &[1, HIDDEN], Encoding::Q4Cp),
                (
                    "experts.gate_up_proj.weight",
                    &[EXPERTS, 2 * EXPERT_WIDTH, HIDDEN],
                    expert,
                ),
                (
                    "experts.down_proj.weight",
                    &[EXPERTS, HIDDEN, EXPERT_WIDTH],
                    expert,
                ),
            ],
        );
    }

    fn add_vision(&mut self, name: String, shape: &[u64]) {
        let source = format!("model.{name}");
        let source_shape: &[u64] = if name == "visual.patch_embed.proj.weight" {
            &[VISION_WIDTH, 3, 2, 16, 16]
        } else {
            shape
        };
        self.add_source(source.clone(), source_shape, SourceDtype::Bf16);
        self.vision.push(TensorPlan {
            name,
            shape: shape.to_vec(),
            encoding: Encoding::Bf16,
            origin: Origin::Tensor { name: source },
            rotation: None,
        });
    }
}

fn signs(name: &str, count: u64) -> TensorPlan {
    TensorPlan {
        name: name.to_owned(),
        shape: vec![count],
        encoding: Encoding::F16,
        origin: Origin::Signs {
            key: name.to_owned(),
        },
        rotation: None,
    }
}

// Construct from model constants only, never from the input's names or count.
fn expected_contract() -> Contract {
    let mut contract = Contract {
        sources: BTreeMap::new(),
        main: Vec::with_capacity(MAIN_COUNT),
        vision: Vec::with_capacity(VISION_COUNT),
        ngram_shards: Vec::with_capacity(NGRAM_PARTS as usize),
    };
    contract.add_main(
        "embed_tokens.weight".into(),
        &[VOCAB, HIDDEN],
        Encoding::Q4Cp,
    );
    contract.add_main("lm_head.weight".into(), &[VOCAB, HIDDEN], Encoding::Ht);
    contract.add_mixer("hyper_connection_mixer.", Encoding::Q6G64, false);
    for layer in 0..LAYERS {
        contract.add_layer(&format!("layers.{layer}."), layer % 4 == 3, false);
    }
    contract.add_main_group(
        "layers.1.ple.",
        &[
            ("key_proj.weight", &[HC_WIDTH, HIDDEN], Encoding::Bf16),
            ("value_proj.weight", &[HIDDEN, HIDDEN], Encoding::Bf16),
            ("conv1d.weight", &[HC_WIDTH, 1, 4], Encoding::Bf16),
            ("norm_key.weight", &[HC_WIDTH], Encoding::Bf16),
            ("norm_query.weight", &[HC_WIDTH], Encoding::Bf16),
            ("norm_conv.weight", &[HC_WIDTH], Encoding::Bf16),
            ("ple_embedding.layer_multipliers", &[3], Encoding::I64),
            ("ple_embedding.ngram_heads_offsets", &[16], Encoding::I64),
            (
                "ple_embedding.ngram_heads_vocab_sizes",
                &[16],
                Encoding::I64,
            ),
        ],
    );
    for shard in 0..NGRAM_PARTS {
        let source = format!("{HF_NGRAM_PREFIX}{shard}.weight");
        contract.add_source(
            source.clone(),
            &[NGRAM_ROWS, NGRAM_WIDTH],
            SourceDtype::Bf16,
        );
        contract.ngram_shards.push(source);
    }
    contract.add_main_group(
        "mtp.",
        &[
            ("fc_embedding.weight", &[HIDDEN, HIDDEN], Encoding::Q8G64),
            ("fc_hidden.weight", &[HIDDEN, HIDDEN], Encoding::Q8G64),
            ("pre_fc_norm_embedding.weight", &[HIDDEN], Encoding::Bf16),
            ("pre_fc_norm_hidden.weight", &[HC_WIDTH], Encoding::Bf16),
        ],
    );
    contract.add_mixer("mtp.hyper_connection_mixer.", Encoding::Q8G64, false);
    contract.add_layer("mtp.layers.0.", true, true);
    for (name, shape) in [
        ("visual.patch_embed.proj.weight", &[VISION_WIDTH, 1536][..]),
        ("visual.patch_embed.proj.bias", &[VISION_WIDTH][..]),
        ("visual.pos_embed.weight", &[2304, VISION_WIDTH][..]),
        ("visual.merger.norm.weight", &[VISION_WIDTH][..]),
        ("visual.merger.norm.bias", &[VISION_WIDTH][..]),
        (
            "visual.merger.linear_fc1.weight",
            &[VISION_MERGED, VISION_MERGED][..],
        ),
        ("visual.merger.linear_fc1.bias", &[VISION_MERGED][..]),
        (
            "visual.merger.linear_fc2.weight",
            &[HIDDEN, VISION_MERGED][..],
        ),
        ("visual.merger.linear_fc2.bias", &[HIDDEN][..]),
    ] {
        contract.add_vision(name.into(), shape);
    }
    for block in 0..VISION_BLOCKS {
        for (suffix, shape) in [
            ("norm1.weight", &[VISION_WIDTH][..]),
            ("norm1.bias", &[VISION_WIDTH][..]),
            ("attn.qkv.weight", &[3 * VISION_WIDTH, VISION_WIDTH][..]),
            ("attn.qkv.bias", &[3 * VISION_WIDTH][..]),
            ("attn.proj.weight", &[VISION_WIDTH, VISION_WIDTH][..]),
            ("attn.proj.bias", &[VISION_WIDTH][..]),
            ("norm2.weight", &[VISION_WIDTH][..]),
            ("norm2.bias", &[VISION_WIDTH][..]),
            (
                "mlp.linear_fc1.weight",
                &[VISION_INTERMEDIATE, VISION_WIDTH][..],
            ),
            ("mlp.linear_fc1.bias", &[VISION_INTERMEDIATE][..]),
            (
                "mlp.linear_fc2.weight",
                &[VISION_WIDTH, VISION_INTERMEDIATE][..],
            ),
            ("mlp.linear_fc2.bias", &[VISION_WIDTH][..]),
        ] {
            contract.add_vision(format!("visual.blocks.{block}.{suffix}"), shape);
        }
    }
    contract
}

fn plan_contract(config: &Value, tensors: &BTreeMap<String, TensorInfo>) -> Result<ConversionPlan> {
    validate_config(config)?;
    let mut contract = expected_contract();
    ensure!(
        contract.sources.len() == SOURCE_COUNT
            && contract.main.len() == MAIN_COUNT
            && contract.vision.len() == VISION_COUNT
            && contract.ngram_shards.len() == NGRAM_PARTS as usize
            && contract
                .main
                .iter()
                .filter(|tensor| matches!(&tensor.origin, Origin::Tensor { .. }))
                .count()
                == MAIN_SOURCE_COUNT,
        "internal fixed architecture contract is inconsistent"
    );
    for (name, expected) in &contract.sources {
        let actual = tensors
            .get(name)
            .with_context(|| format!("missing source tensor {name}"))?;
        ensure!(
            actual.name == *name,
            "source tensor name mismatch for {name}"
        );
        ensure!(
            actual.shape == expected.shape,
            "source tensor {name} has shape {:?}; expected {:?}",
            actual.shape,
            expected.shape
        );
        let allowed = match expected.dtype {
            SourceDtype::Bf16 => actual.dtype == Dtype::Bf16,
            SourceDtype::Bf16OrF32 => matches!(actual.dtype, Dtype::Bf16 | Dtype::F32),
            SourceDtype::I64 => actual.dtype == Dtype::I64,
        };
        ensure!(
            allowed,
            "source tensor {name} has unsupported dtype {:?}",
            actual.dtype
        );
    }
    for name in tensors.keys() {
        ensure!(
            contract.sources.contains_key(name),
            "unexpected source tensor {name}"
        );
    }
    // HF already uses zero-centred gains, grouped DeltaNet heads and A_log.
    // Origins deliberately request the original values: no GGUF offset,
    // logarithm or head permutation is applicable. Rank-5 vision patch weights
    // flatten in row-major order without changing the values. PLE i64 contents
    // (including hash bounds) are checked during conversion, not metadata planning.
    for tensor in &mut contract.main {
        if let Origin::Tensor { name } = &tensor.origin {
            if tensors[name].dtype == Dtype::F32 {
                tensor.encoding = Encoding::F32;
            }
        }
    }
    canonical_sort(&mut contract.main);
    canonical_sort(&mut contract.vision);
    Ok(ConversionPlan {
        architecture: "qwen4_exp".into(),
        files: vec![
            ContainerPlan {
                filename: "model.hgn".into(),
                identity: "qwen3.8-flash-next-v2".into(),
                tensors: contract.main,
            },
            ContainerPlan {
                filename: "qwen38-flash-next-ngram.hgn".into(),
                identity: "qwen3.8-flash-next-ngram".into(),
                tensors: vec![TensorPlan {
                    name: "layers.1.ple.ngram_embedding.weight".into(),
                    shape: vec![NGRAM_PARTS, NGRAM_ROWS, NGRAM_WIDTH],
                    encoding: Encoding::Fp8G128,
                    origin: Origin::Ngram {
                        shards: contract.ngram_shards,
                    },
                    rotation: None,
                }],
            },
            ContainerPlan {
                filename: "qwen38-flash-next-vision.hgn".into(),
                identity: "qwen3.8-flash-next-vision".into(),
                tensors: contract.vision,
            },
        ],
        assets: Vec::new(),
    })
}

fn numbered_suffix(name: &str) -> Option<(u64, &str)> {
    let (number, suffix) = name.split_once('.')?;
    Some((number.parse().ok()?, suffix))
}

fn sublayer_index(suffix: &str) -> u8 {
    match suffix.split_once('.').map(|(sublayer, _)| sublayer) {
        Some("ple") => 0,
        Some("attn_hyper_connection") => 1,
        Some("linear_attn") => 2,
        Some("self_attn") => 3,
        Some("mlp_hyper_connection") => 4,
        Some("mlp") => 5,
        _ => 6,
    }
}

fn canonical_key(name: &str) -> (u8, u8, u64, u8, &str) {
    if name.starts_with("embed_tokens.") {
        (0, 0, 0, 0, name)
    } else if let Some((layer, suffix)) = name.strip_prefix("layers.").and_then(numbered_suffix) {
        (1, 0, layer, sublayer_index(suffix), suffix)
    } else if name.starts_with("hyper_connection_mixer.") {
        (2, 0, 0, 0, name)
    } else if name.starts_with("lm_head.") {
        (3, 0, 0, 0, name)
    } else if let Some(mtp) = name.strip_prefix("mtp.") {
        match mtp.strip_prefix("layers.").and_then(numbered_suffix) {
            Some((layer, suffix)) => (4, 1, layer, sublayer_index(suffix), suffix),
            None => (4, 0, 0, 0, mtp),
        }
    } else if let Some((block, suffix)) = name
        .strip_prefix("visual.blocks.")
        .and_then(numbered_suffix)
    {
        (5, 0, block, 0, suffix)
    } else if name.starts_with("visual.") {
        (5, 1, 0, 0, name)
    } else {
        (6, 0, 0, 0, name)
    }
}

pub fn canonical_sort(tensors: &mut [TensorPlan]) {
    tensors.sort_unstable_by(|a, b| {
        canonical_key(&a.name)
            .cmp(&canonical_key(&b.name))
            .then_with(|| a.name.cmp(&b.name))
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::BTreeSet;

    fn config() -> Value {
        let mut value: Value = serde_json::from_str(
            r#"{
                "architectures": ["Qwen4ExpForConditionalGeneration"],
                "model_type": "qwen4_exp", "language_model_only": false,
                "tie_word_embeddings": false, "image_token_id": 248056,
                "video_token_id": 248057, "vision_end_token_id": 248054,
                "vision_start_token_id": 248053,
                "text_config": {
                    "attention_bias": false, "attention_dropout": 0.0,
                    "bos_token_id": 248044, "eos_token_id": 248044, "dtype": "bfloat16",
                    "full_attention_interval": 4, "hc_count": 4, "hc_lowrank": 320,
                    "head_dim": 256, "heads_per_ngram": 8, "hidden_act": "silu",
                    "hidden_size": 2560, "indexer_budget": 2048,
                    "indexer_compress_ratio": 4, "indexer_head_dim": 128,
                    "indexer_kv_heads": 1, "indexer_n_heads": 4, "initializer_range": 0.02,
                    "layer_types": [],
                    "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128,
                    "linear_num_key_heads": 16, "linear_num_value_heads": 48,
                    "linear_value_head_dim": 128, "make_ngram_vocab_size_divisible_by": 128,
                    "mamba_ssm_dtype": "float32", "max_position_embeddings": 262144,
                    "model_type": "qwen4_exp_text", "moe_intermediate_size": 640,
                    "mtp": {
                        "hybrid": true, "layer_types": ["full_attention"],
                        "mtp_use_hidden_state_from_layer": null,
                        "num_hidden_layers": 1, "rope_theta": 10000000
                    },
                    "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": false,
                    "ngram_size": 3, "ngram_vocab_size_base": 20000000,
                    "num_attention_heads": 24, "num_experts": 512,
                    "num_experts_per_tok": 10, "num_hidden_layers": 48,
                    "num_key_value_heads": 2, "output_gate_type": "sigmoid",
                    "output_router_logits": false, "pad_token_id": null,
                    "partial_rotary_factor": 0.25, "ple_conv_kernel_size": 4,
                    "ple_embed_dim": 2560, "ple_layer_ids": [2], "rms_norm_eps": 1e-6,
                    "rope_parameters": {
                        "mrope_interleaved": true, "mrope_section": [11, 11, 10],
                        "partial_rotary_factor": 0.25, "rope_theta": 10000000,
                        "rope_type": "default"
                    },
                    "router_aux_loss_coef": 0.001, "shared_expert_intermediate_size": 640,
                    "split_ngram_parts": 128, "tie_word_embeddings": false,
                    "use_cache": true, "vocab_size": 248320
                },
                "vision_config": {
                    "deepstack_visual_indexes": [], "depth": 27,
                    "hidden_act": "gelu_pytorch_tanh", "hidden_size": 1152,
                    "in_channels": 3, "initializer_range": 0.02, "intermediate_size": 4304,
                    "model_type": "qwen4_exp", "num_heads": 16,
                    "num_position_embeddings": 2304, "out_hidden_size": 2560,
                    "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2
                }
            }"#,
        )
        .unwrap();
        value["text_config"]["layer_types"] = json!((0..48)
            .map(|layer| if layer % 4 == 3 {
                "full_attention"
            } else {
                "linear_attention"
            })
            .collect::<Vec<_>>());
        value
    }

    fn metadata() -> BTreeMap<String, TensorInfo> {
        expected_contract()
            .sources
            .into_iter()
            .map(|(name, expected)| {
                let dtype = match expected.dtype {
                    SourceDtype::I64 => Dtype::I64,
                    SourceDtype::Bf16 | SourceDtype::Bf16OrF32 => Dtype::Bf16,
                };
                let size = expected.shape.iter().product::<u64>() * dtype.byte_width();
                let tensor = TensorInfo {
                    name: name.clone(),
                    shard: "model.safetensors".into(),
                    dtype,
                    shape: expected.shape,
                    offset: 8,
                    size,
                };
                (name, tensor)
            })
            .collect()
    }

    fn count_store(tensors: &[TensorPlan], encoding: Encoding) -> usize {
        tensors
            .iter()
            .filter(|tensor| tensor.encoding == encoding)
            .count()
    }

    fn tensor<'a>(tensors: &'a [TensorPlan], name: &str) -> &'a TensorPlan {
        tensors.iter().find(|tensor| tensor.name == name).unwrap()
    }

    #[test]
    fn exact_source_and_v2_store_contract() {
        let source = metadata();
        assert_eq!(source.len(), 1658);
        let result = plan_contract(&config(), &source).unwrap();
        assert_eq!(result.architecture, "qwen4_exp");
        assert_eq!(result.files.len(), 3);
        let main = &result.files[0];
        assert_eq!(main.filename, "model.hgn");
        assert_eq!(main.identity, "qwen3.8-flash-next-v2");
        assert_eq!(main.tensors.len(), 2015);
        for (encoding, count) in [
            (Encoding::Ht, 313),
            (Encoding::Q6G64, 290),
            (Encoding::I4r, 96),
            (Encoding::Q4Cp, 124),
            (Encoding::Q8G64, 18),
            (Encoding::Bf16, 353),
            (Encoding::F16, 818),
            (Encoding::I64, 3),
            (Encoding::F32, 0),
        ] {
            assert_eq!(count_store(&main.tensors, encoding), count, "{encoding:?}");
        }
        assert_eq!(
            main.tensors
                .iter()
                .filter(|t| t.name.starts_with("mtp."))
                .count(),
            31
        );
        let vision = &result.files[2];
        assert_eq!(vision.filename, "qwen38-flash-next-vision.hgn");
        assert_eq!(vision.identity, "qwen3.8-flash-next-vision");
        assert_eq!(vision.tensors.len(), 333);
        assert_eq!(count_store(&vision.tensors, Encoding::Bf16), 333);
        let patch = tensor(&vision.tensors, "visual.patch_embed.proj.weight");
        assert_eq!(patch.shape, [1152, 1536]);
        assert_eq!(
            patch.origin,
            Origin::Tensor {
                name: "model.visual.patch_embed.proj.weight".into()
            }
        );
        assert_eq!(
            source["model.visual.patch_embed.proj.weight"].shape,
            [1152, 3, 2, 16, 16]
        );
        let table = &result.files[1];
        assert_eq!(table.filename, "qwen38-flash-next-ngram.hgn");
        assert_eq!(table.identity, "qwen3.8-flash-next-ngram");
        assert_eq!(table.tensors.len(), 1);
        assert_eq!(table.tensors[0].shape, [128, 2500012, 160]);
        assert_eq!(table.tensors[0].encoding, Encoding::Fp8G128);
        let Origin::Ngram { shards } = &table.tensors[0].origin else {
            panic!("table must stack HF shards");
        };
        assert_eq!(shards.len(), 128);
        for (index, name) in shards.iter().enumerate() {
            assert_eq!(name, &format!("{HF_NGRAM_PREFIX}{index}.weight"));
        }
        let mut coverage = BTreeSet::new();
        for file in &result.files {
            let mut target_names = BTreeSet::new();
            for planned in &file.tensors {
                assert!(target_names.insert(&planned.name));
                match &planned.origin {
                    Origin::Tensor { name } => assert!(coverage.insert(name.as_str())),
                    Origin::Ngram { shards } => {
                        for name in shards {
                            assert!(coverage.insert(name.as_str()));
                        }
                    }
                    Origin::Signs { .. } | Origin::HtScales { .. } => {}
                }
            }
        }
        assert_eq!(coverage, source.keys().map(String::as_str).collect());
    }

    #[test]
    fn expert_origins_and_rotations_are_exact() {
        let result = plan_contract(&config(), &metadata()).unwrap();
        let main = &result.files[0].tensors;
        for layer in 0..48 {
            let prefix = format!("layers.{layer}.mlp.experts");
            for (projection, shape, su, sv) in [
                (
                    "gate_up_proj",
                    vec![512, 1280, 2560],
                    format!("{prefix}.su"),
                    format!("{prefix}.gate_up_proj.svh"),
                ),
                (
                    "down_proj",
                    vec![512, 2560, 640],
                    format!("{prefix}.down_proj.suh"),
                    format!("{prefix}.down_proj.svh"),
                ),
            ] {
                let weight = tensor(main, &format!("{prefix}.{projection}.weight"));
                assert_eq!(weight.shape, shape);
                assert_eq!(weight.encoding, Encoding::I4r);
                assert_eq!(
                    weight.origin,
                    Origin::Tensor {
                        name: format!("model.language_model.{prefix}.{projection}")
                    }
                );
                assert_eq!(
                    weight.rotation,
                    Some(Rotation {
                        su: su.clone(),
                        sv: sv.clone()
                    })
                );
                for (name, count) in [(&su, shape[2]), (&sv, shape[1])] {
                    let vector = tensor(main, name);
                    assert_eq!(vector.shape, [count]);
                    assert_eq!(vector.encoding, Encoding::F16);
                    assert_eq!(vector.origin, Origin::Signs { key: name.clone() });
                }
            }
        }
        for weight in main.iter().filter(|t| t.encoding == Encoding::Ht) {
            let rotation = weight.rotation.as_ref().unwrap();
            let Origin::Tensor { name } = &weight.origin else {
                panic!("HT input must name an HF matrix");
            };
            let su = tensor(main, &rotation.su);
            let sv = tensor(main, &rotation.sv);
            assert_eq!(su.shape, [weight.shape[1]]);
            assert_eq!(sv.shape, [weight.shape[0]]);
            assert_eq!(
                su.origin,
                Origin::Signs {
                    key: rotation.su.clone()
                }
            );
            assert_eq!(
                sv.origin,
                Origin::HtScales {
                    tensor: name.clone(),
                    signs_key: rotation.sv.clone()
                }
            );
        }
        for projection in ["gate_up_proj", "down_proj"] {
            let name = format!("mtp.layers.0.mlp.experts.{projection}");
            let weight = tensor(main, &format!("{name}.weight"));
            assert_eq!(weight.origin, Origin::Tensor { name });
            assert_eq!(weight.encoding, Encoding::Q4Cp);
            assert!(weight.rotation.is_none());
        }
    }

    #[test]
    fn f32_is_preserved_only_for_deltanet_parameters() {
        let mut source = metadata();
        for (name, info) in &mut source {
            if name.ends_with(".linear_attn.A_log") || name.ends_with(".linear_attn.dt_bias") {
                info.dtype = Dtype::F32;
                info.size *= 2;
            }
        }
        let result = plan_contract(&config(), &source).unwrap();
        assert_eq!(count_store(&result.files[0].tensors, Encoding::F32), 72);
        assert_eq!(count_store(&result.files[0].tensors, Encoding::Bf16), 281);
        for name in [
            "model.language_model.layers.0.linear_attn.norm.weight",
            "model.visual.patch_embed.proj.weight",
            "model.language_model.layers.1.ple.ple_embedding.layer_multipliers",
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
        ] {
            let mut invalid = source.clone();
            invalid.get_mut(name).unwrap().dtype = Dtype::F32;
            assert!(plan_contract(&config(), &invalid).is_err(), "{name}");
        }
    }

    #[test]
    fn missing_unexpected_and_wrong_source_contracts_are_rejected() {
        let source = metadata();
        for name in [
            "lm_head.weight",
            "model.language_model.layers.47.self_attn.indexer.index_qk_proj.weight",
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_127.weight",
            "model.visual.blocks.26.norm1.bias",
            "mtp.layers.0.mlp.experts.gate_up_proj",
        ] {
            let mut invalid = source.clone();
            invalid.remove(name);
            assert!(
                plan_contract(&config(), &invalid).is_err(),
                "missing {name}"
            );
        }
        let expert = "model.language_model.layers.0.mlp.experts.gate_up_proj";
        let mut invalid = source.clone();
        let mut renamed = invalid.remove(expert).unwrap();
        renamed.name = format!("{expert}.weight");
        invalid.insert(renamed.name.clone(), renamed);
        assert_eq!(invalid.len(), 1658);
        assert!(plan_contract(&config(), &invalid).is_err());
        let mut invalid = source.clone();
        let mut extra = invalid["lm_head.weight"].clone();
        extra.name = "unmapped.weight".into();
        invalid.insert(extra.name.clone(), extra);
        assert!(plan_contract(&config(), &invalid).is_err());
        for name in [
            "model.visual.patch_embed.proj.weight",
            "model.language_model.layers.0.linear_attn.conv1d.weight",
            "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets",
            expert,
        ] {
            let mut invalid = source.clone();
            invalid.get_mut(name).unwrap().shape[0] += 1;
            assert!(plan_contract(&config(), &invalid).is_err(), "shape {name}");
        }
        let mut invalid = source.clone();
        invalid
            .get_mut("model.visual.patch_embed.proj.weight")
            .unwrap()
            .shape = vec![1152, 1536];
        assert!(plan_contract(&config(), &invalid).is_err());
        let mut invalid = source;
        invalid.get_mut("lm_head.weight").unwrap().name = "different.weight".into();
        assert!(plan_contract(&config(), &invalid).is_err());
    }

    #[test]
    fn every_fixed_configuration_value_is_required() {
        let valid = config();
        validate_config(&valid).unwrap();
        for &(path, value) in INTEGER_CONFIG {
            let mut invalid = valid.clone();
            *invalid.pointer_mut(path).unwrap() = json!(value + 1);
            assert!(validate_config(&invalid).is_err(), "{path}");
            *invalid.pointer_mut(path).unwrap() = Value::Null;
            assert!(validate_config(&invalid).is_err(), "missing {path}");
        }
        for &(path, _) in STRING_CONFIG {
            let mut invalid = valid.clone();
            *invalid.pointer_mut(path).unwrap() = json!("unsupported");
            assert!(validate_config(&invalid).is_err(), "{path}");
        }
        for &(path, value) in BOOL_CONFIG {
            let mut invalid = valid.clone();
            *invalid.pointer_mut(path).unwrap() = json!(!value);
            assert!(validate_config(&invalid).is_err(), "{path}");
        }
        for &(path, value) in FLOAT_CONFIG {
            let mut invalid = valid.clone();
            *invalid.pointer_mut(path).unwrap() = json!(value + 1.0);
            assert!(validate_config(&invalid).is_err(), "{path}");
        }
        for (path, wrong) in [
            ("/architectures", json!(["Qwen3_5ForConditionalGeneration"])),
            ("/text_config/ple_layer_ids", json!([1])),
            ("/text_config/layer_types/3", json!("linear_attention")),
            ("/text_config/mtp/layer_types", json!(["linear_attention"])),
            (
                "/text_config/rope_parameters/mrope_section",
                json!([10, 11, 11]),
            ),
            ("/vision_config/deepstack_visual_indexes", json!([1])),
            (
                "/text_config/mtp/mtp_use_hidden_state_from_layer",
                json!(47),
            ),
            ("/text_config/pad_token_id", json!(0)),
        ] {
            let mut invalid = valid.clone();
            *invalid.pointer_mut(path).unwrap() = wrong;
            assert!(validate_config(&invalid).is_err(), "{path}");
        }
        let mut invalid = valid.clone();
        invalid["text_config"]["layer_types"]
            .as_array_mut()
            .unwrap()
            .pop();
        assert!(validate_config(&invalid).is_err());
        let mut invalid = valid;
        invalid["quantization_config"] = json!({"quant_method": "awq"});
        assert!(validate_config(&invalid).is_err());
    }

    #[test]
    fn canonical_order_is_numeric_and_sublayer_aware() {
        let expected = [
            "embed_tokens.weight",
            "layers.1.ple.key_proj.weight",
            "layers.1.attn_hyper_connection.hc_norm.weight",
            "layers.1.linear_attn.A_log",
            "layers.1.self_attn.k_proj.weight",
            "layers.1.mlp_hyper_connection.hc_norm.weight",
            "layers.1.mlp.gate.weight",
            "layers.2.mlp.gate.weight",
            "layers.10.mlp.gate.weight",
            "hyper_connection_mixer.hc_norm.weight",
            "lm_head.suh",
            "lm_head.svh",
            "lm_head.weight",
            "mtp.fc_embedding.weight",
            "mtp.pre_fc_norm_hidden.weight",
            "mtp.layers.0.attn_hyper_connection.hc_norm.weight",
            "mtp.layers.0.self_attn.k_proj.weight",
            "mtp.layers.0.mlp.gate.weight",
            "mtp.layers.2.mlp.gate.weight",
            "mtp.layers.10.mlp.gate.weight",
            "visual.blocks.2.attn.proj.weight",
            "visual.blocks.2.norm1.weight",
            "visual.blocks.10.attn.proj.weight",
            "visual.merger.norm.weight",
        ];
        let mut tensors: Vec<_> = expected.iter().rev().map(|name| signs(name, 1)).collect();
        canonical_sort(&mut tensors);
        assert_eq!(
            tensors.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
            expected
        );
    }

    #[test]
    fn assets_are_required_flat_allowlisted_regular_files() {
        let temp = tempfile::tempdir().unwrap();
        for name in REQUIRED_ASSETS {
            fs::write(temp.path().join(name), []).unwrap();
        }
        fs::write(temp.path().join("vocab.json"), []).unwrap();
        fs::write(temp.path().join("auth.json"), b"do not copy").unwrap();
        fs::create_dir(temp.path().join(".cache")).unwrap();
        let assets = plan_assets(temp.path()).unwrap();
        assert_eq!(assets.len(), 7);
        assert!(assets.iter().any(|name| name == "vocab.json"));
        assert!(!assets
            .iter()
            .any(|name| name == "auth.json" || name == ".cache"));
        fs::remove_file(temp.path().join("merges.txt")).unwrap();
        assert!(plan_assets(temp.path()).is_err());
        fs::create_dir(temp.path().join("merges.txt")).unwrap();
        assert!(plan_assets(temp.path()).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn optional_and_required_asset_symlinks_are_rejected() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        for name in REQUIRED_ASSETS {
            fs::write(temp.path().join(name), []).unwrap();
        }
        symlink("not-present", temp.path().join("vocab.json")).unwrap();
        assert!(plan_assets(temp.path()).is_err());
        fs::remove_file(temp.path().join("vocab.json")).unwrap();
        fs::remove_file(temp.path().join("tokenizer.json")).unwrap();
        symlink("config.json", temp.path().join("tokenizer.json")).unwrap();
        assert!(plan_assets(temp.path()).is_err());
    }
}
