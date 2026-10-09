use crate::{
    container::{Entry, Fingerprint, Layout},
    hf::TensorInfo,
    ConversionPlan,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Options {
    pub seed: u64,
    pub threads: usize,
    pub ht_beam: usize,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            seed: 0,
            threads: 1,
            ht_beam: 128,
        }
    }
}
impl Options {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            (1..=64).contains(&self.threads),
            "threads must be in 1..=64"
        );
        anyhow::ensure!(
            (1..=256).contains(&self.ht_beam),
            "HT beam must be in 1..=256"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceRecord {
    pub tensor: TensorInfo,
    pub sha256: String,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AssetRecord {
    pub source: String,
    pub destination: String,
    pub size: u64,
    pub sha256: String,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OutputRecord {
    pub entry: Entry,
    pub fingerprint: Fingerprint,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ContainerRecord {
    pub filename: String,
    pub layout: Layout,
    pub tensors: Vec<OutputRecord>,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Manifest {
    pub schema: u32,
    pub tool: String,
    pub tool_version: String,
    pub encoder_sha256: String,
    pub spec_version: String,
    pub spec_commit: String,
    pub policy: String,
    pub options: Options,
    pub plan: ConversionPlan,
    pub config_sha256: String,
    pub index_sha256: String,
    pub sources: Vec<SourceRecord>,
    pub assets: Vec<AssetRecord>,
    pub containers: Vec<ContainerRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorTotals {
    pub values: u64,
    pub source_squared: f64,
    pub error_squared: f64,
}
impl Default for ErrorTotals {
    fn default() -> Self {
        Self {
            values: 0,
            source_squared: 0.0,
            error_squared: 0.0,
        }
    }
}
impl ErrorTotals {
    pub fn relative_squared_error(&self) -> f64 {
        if self.source_squared == 0.0 {
            0.0
        } else {
            self.error_squared / self.source_squared
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyReport {
    pub source_tensors: usize,
    pub output_tensors: usize,
    pub payload_bytes: u64,
    pub errors: std::collections::BTreeMap<String, ErrorTotals>,
}
