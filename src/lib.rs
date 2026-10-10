pub mod architecture;
pub mod codec;
pub mod container;
pub mod conversion;
mod gguf_decode;
pub mod gguf_source;
pub mod hf;
pub mod hgn_source;
pub mod manifest;
pub mod source;
pub mod verify;

use codec::Encoding;
use serde::{Deserialize, Serialize};

pub const SPEC_VERSION: &str = "1.1.0";
pub const SPEC_COMMIT: &str = "79e75fd4ae7642478823525b0a3212951f584af5";
pub const POLICY: &str = "bf16-hgn-v2-rtn-nf4-signed-rms-trellis-v1";
pub const ENCODER_SHA256: &str = env!("BF16_HGN_ENCODER_SHA256");

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Origin {
    Tensor { name: String },
    Signs { key: String },
    HtScales { tensor: String, signs_key: String },
    Ngram { shards: Vec<String> },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Rotation {
    pub su: String,
    pub sv: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TensorPlan {
    pub name: String,
    pub shape: Vec<u64>,
    pub encoding: Encoding,
    pub origin: Origin,
    pub rotation: Option<Rotation>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ContainerPlan {
    pub filename: String,
    pub identity: String,
    pub tensors: Vec<TensorPlan>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConversionPlan {
    pub architecture: String,
    pub files: Vec<ContainerPlan>,
    pub assets: Vec<String>,
}

pub fn numel(shape: &[u64]) -> anyhow::Result<u64> {
    anyhow::ensure!(!shape.is_empty() && shape.len() <= 5, "invalid tensor rank");
    shape.iter().try_fold(1u64, |n, &dimension| {
        anyhow::ensure!(dimension > 0, "zero tensor dimension");
        n.checked_mul(dimension)
            .ok_or_else(|| anyhow::anyhow!("tensor size overflow"))
    })
}

pub fn rows_cols(shape: &[u64]) -> anyhow::Result<(usize, usize)> {
    let elements = numel(shape)?;
    let cols = *shape.last().expect("validated rank");
    Ok((usize::try_from(elements / cols)?, usize::try_from(cols)?))
}
