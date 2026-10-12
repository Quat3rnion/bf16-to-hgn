use crate::{
    codec::{HtBackend, I4rBackend},
    container::{Entry, Fingerprint, Layout},
    hf::TensorInfo,
    source::SourceKind,
    ConversionPlan,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum OutputProfile {
    #[default]
    V2,
    Ht43,
}
impl OutputProfile {
    pub fn policy(self) -> &'static str {
        match self {
            Self::V2 => crate::POLICY,
            Self::Ht43 => "bf16-hgn-ht43-rtn-nf4-signed-rms-trellis-v1",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Options {
    #[serde(default)]
    pub profile: OutputProfile,
    pub seed: u64,
    pub threads: usize,
    pub ht_beam: usize,
    /// Encoding provenance only; verification selects its decode backend at runtime.
    #[serde(default)]
    pub ht_backend: HtBackend,
    /// Independent I4R encoding provenance; never selects the verification device.
    #[serde(default)]
    pub i4r_backend: I4rBackend,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            profile: OutputProfile::V2,
            seed: 0,
            threads: 1,
            ht_beam: 128,
            ht_backend: HtBackend::Cpu,
            i4r_backend: I4rBackend::Cpu,
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

    /// Check real encoding availability before creating staging/output files.
    /// `validate` deliberately remains device-independent for plans and verification.
    pub fn validate_for_execution(&self) -> anyhow::Result<()> {
        self.validate()?;
        self.ht_backend.preflight()?;
        self.i4r_backend.preflight()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceRecord {
    pub tensor: TensorInfo,
    /// SHA-256 of canonical little-endian dtype bytes, not encoded import bytes.
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
    pub source_kind: SourceKind,
    /// Physical dependency metadata/checksums; never canonical hashes posing as raw hashes.
    pub source_provenance: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index_sha256: Option<String>,
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
    pub manifest_schema: u32,
    /// False for historical schema 1, which never attested physical source provenance.
    pub provenance_attested: bool,
    pub source_tensors: usize,
    pub output_tensors: usize,
    pub payload_bytes: u64,
    pub errors: std::collections::BTreeMap<String, ErrorTotals>,
}
