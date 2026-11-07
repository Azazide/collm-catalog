// SPDX-License-Identifier: AGPL-3.0-only
//! The shape of `catalog.json`, the one file Collm reads.
//!
//! A change to a field name or meaning bumps [`FORMAT`]; the app refuses a file whose `format` it
//! does not know rather than guessing.

use serde::{Deserialize, Serialize};

pub const FORMAT: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Catalog {
    pub format: u32,
    pub models: Vec<Model>,
}

impl Catalog {
    pub fn empty() -> Self {
        Catalog {
            format: FORMAT,
            models: Vec::new(),
        }
    }

    /// The catalog a file holds, or an empty one when the text is not a catalog of this format: a
    /// damaged or foreign file costs a full read of Hugging Face, nothing more.
    pub fn from_text(text: &str) -> Catalog {
        match serde_json::from_str::<Catalog>(text) {
            Ok(catalog) if catalog.format == FORMAT => catalog,
            _ => Catalog::empty(),
        }
    }
}

/// One Hugging Face repository that holds GGUF chat models.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Model {
    /// The repository, `owner/name`: `unsloth/gemma-4-12b-it-GGUF`. Two owners may name a repository the
    /// same, so the name alone does not tell models apart.
    pub model_name: String,
    /// The repository owner: `unsloth`.
    pub developer: String,
    /// The commit the files below were read at. Every `path` points at it, so a later push to the
    /// repository cannot make a download fail its hash.
    pub sha: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    /// Downloads over the last 30 days, as Hugging Face counts them.
    pub downloads: u64,
    pub likes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pipeline_tag: Option<String>,
    pub architecture: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u64>,
    /// What the model can do, from `vision`, `tools`, `code` and `agent`, sorted. Every chat model is
    /// a chat model, so that has no entry; "agent and coding" is `agent` and `code` together. How each
    /// is decided is in `classify.rs` and the README.
    pub capabilities: Vec<String>,
    /// The model's README at the same commit; left out when the repository has none.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub readme: String,
    /// Every single-file weights file, smallest first. Draft files for speculative decoding
    /// (`mtp-…`, `MTP/…`) are listed here too; the app tells them apart by name.
    pub quants: Vec<File>,
    /// Vision projectors (`mmproj-…`).
    pub mmproj_models: Vec<File>,
}

impl Model {
    pub fn repo_id(&self) -> String {
        self.model_name.clone()
    }
}

/// One downloadable file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct File {
    /// The path in the repository without `.gguf`: `Q4_K_M/model-Q4_K_M`.
    pub model_id: String,
    pub path: String,
    /// Bytes.
    pub size: u64,
    /// SHA-256 of the file, hex, when Hugging Face states it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
}

/// The file as text: one model per line, so a day's change reads as a diff and the file stays small.
pub fn render(catalog: &Catalog) -> String {
    let mut out = String::new();
    out.push_str(&format!("{{\"format\":{},\"models\":[\n", catalog.format));
    for (i, model) in catalog.models.iter().enumerate() {
        out.push_str(&serde_json::to_string(model).expect("a model serializes"));
        out.push_str(if i + 1 < catalog.models.len() {
            ",\n"
        } else {
            "\n"
        });
    }
    out.push_str("]}\n");
    out
}

/// A run that would publish a catalog much smaller than the last one is a failed run (Hugging Face
/// answered badly, a filter broke), not news. `Err` says why the file must not be replaced.
pub fn check_shrink(previous: usize, next: usize, keep_ratio: f64) -> Result<(), String> {
    if next == 0 {
        return Err("the new catalog has no models".to_string());
    }
    if previous >= 20 && (next as f64) < (previous as f64) * keep_ratio {
        return Err(format!(
            "the new catalog has {next} models and the last one {previous}; \
             a drop below {:.0}% is taken for a failed run (use --force to publish it)",
            keep_ratio * 100.0
        ));
    }
    Ok(())
}
