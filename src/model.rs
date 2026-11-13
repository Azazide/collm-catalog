// SPDX-License-Identifier: AGPL-3.0-only
//! The shape of `catalog.json`, the one file Collm reads.
//!
//! A change to a field name or meaning bumps [`FORMAT`]; the app refuses a file whose `format` it
//! does not know rather than guessing.

use serde::{Deserialize, Serialize};

pub const FORMAT: u32 = 1;

/// The version of the rules that decide what a repository's entry says (which file is a model, a draft
/// or not a model, what a description is, what a template taking tools is). An entry carried over from
/// a catalog made under other rules is read again, so a change to a rule reaches every model at the
/// next run and not only those whose repository changed. Bump it with any such change. The app ignores
/// it: it reads `format` and `models`.
pub const RULES: u32 = 4;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Catalog {
    pub format: u32,
    /// The rules the models were made under; 0 for a file that names none.
    #[serde(default)]
    pub rules: u32,
    pub models: Vec<Model>,
}

impl Catalog {
    pub fn empty() -> Self {
        Catalog {
            format: FORMAT,
            rules: RULES,
            models: Vec::new(),
        }
    }

    /// What the next run starts from: this catalog's models, or nothing when they are to be read again,
    /// by request (`refresh`) or because they were made under other [`RULES`].
    pub fn carried(&self, refresh: bool) -> Catalog {
        if refresh || self.rules != RULES {
            Catalog::empty()
        } else {
            self.clone()
        }
    }

    /// Takes out the models whose data is not well formed (see [`Model::problems`]) and says which,
    /// and why: one repository's odd data must not stop the catalog, nor go into it.
    pub fn drop_malformed(&mut self) -> Vec<(String, Vec<String>)> {
        let mut dropped = Vec::new();
        let mut seen = std::collections::HashSet::new();
        self.models.retain(|model| {
            let mut problems = model.problems();
            if !seen.insert(model.model_name.to_lowercase()) {
                problems.push("listed twice".to_string());
            }
            if problems.is_empty() {
                true
            } else {
                dropped.push((model.model_name.clone(), problems));
                false
            }
        });
        dropped
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
    /// The repository the model was made from (`describe::base_model`): the app finds a model by its maker with it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_model: Option<String>,
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
    /// Every weights file that runs on its own and is whole, smallest first.
    pub quants: Vec<File>,
    /// The drafts for speculative decoding (`mtp`, `dspark`, `dflash`, `eagle3`) that pair with a
    /// model above: files no one runs alone, so they are not among the `quants`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub drafts: Vec<File>,
    /// Vision projectors (`mmproj-…`).
    pub mmproj_models: Vec<File>,
}

impl Model {
    pub fn repo_id(&self) -> String {
        self.model_name.clone()
    }

    /// What is wrong with the entry, as the app and a download would meet it: a name that is not
    /// `owner/name`, a commit that is not a full hash, no file to download, a file with no size or no
    /// SHA-256, an address that is not on Hugging Face under this repository and commit.
    pub fn problems(&self) -> Vec<String> {
        let mut found = Vec::new();
        if !is_repo_id(&self.model_name) {
            found.push(format!("`{}` is not owner/name", self.model_name));
            return found;
        }
        let owner = self.model_name.split('/').next().unwrap_or_default();
        if !self.developer.eq_ignore_ascii_case(owner) {
            found.push(format!("owner {} is not {owner}", self.developer));
        }
        if self.sha.len() != 40 || !self.sha.bytes().all(|b| b.is_ascii_hexdigit()) {
            found.push("the commit is not a 40-digit hash".to_string());
            return found;
        }
        let at = format!(
            "https://huggingface.co/{}/resolve/{}/",
            self.model_name, self.sha
        );
        if self.quants.is_empty() {
            found.push("no file to download".to_string());
        }
        for (kind, files) in [
            ("file", &self.quants),
            ("draft", &self.drafts),
            ("projector", &self.mmproj_models),
        ] {
            for file in files {
                if let Some(problem) = file.problem(&at) {
                    found.push(format!("{kind} {}: {problem}", file.model_id));
                }
            }
        }
        if !self.readme.is_empty() && self.readme != format!("{at}README.md") {
            found.push("the README is not this repository's at this commit".to_string());
        }
        found
    }
}

/// `owner/name` of a repository: letters, digits, `.`, `_` and `-`, each part not longer than Hugging
/// Face allows (96 characters).
pub fn is_repo_id(text: &str) -> bool {
    let part = |p: &str| {
        !p.is_empty()
            && p.len() <= 96
            && p.bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
    };
    text.split_once('/')
        .is_some_and(|(owner, name)| part(owner) && part(name))
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

impl File {
    /// What is wrong with the file, given the address every file of its model starts with.
    fn problem(&self, at: &str) -> Option<String> {
        if self.model_id.is_empty() {
            return Some("no name".to_string());
        }
        if self.size == 0 {
            return Some("no size".to_string());
        }
        if !self
            .sha256
            .as_deref()
            .is_some_and(|h| h.len() == 64 && h.bytes().all(|b| b.is_ascii_hexdigit()))
        {
            return Some("no SHA-256".to_string());
        }
        let Some(tail) = self.path.strip_prefix(at) else {
            return Some(format!("the address {} is not under {at}", self.path));
        };
        let plain = !tail.is_empty()
            && !tail.contains("//")
            && tail.split('/').all(|s| {
                !s.is_empty() && s != "." && s != ".." && s != "%2e" && s != "%2E" && s != "%2e%2e"
            })
            && !tail
                .chars()
                .any(|c| c.is_control() || c == '\\' || c == ' ');
        (!plain).then(|| format!("the address {} leaves the commit", self.path))
    }
}

/// The file as text: one model per line, so a day's change reads as a diff and the file stays small.
pub fn render(catalog: &Catalog) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{{\"format\":{},\"rules\":{},\"models\":[\n",
        catalog.format, catalog.rules
    ));
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

/// A catalog with fewer models than this is a failed run, whatever the last one held: the shrink guard
/// alone lets a catalog lose a third of itself a night, five nights running.
pub fn check_floor(next: usize, floor: usize) -> Result<(), String> {
    if next < floor {
        return Err(format!(
            "the new catalog has {next} models, fewer than the {floor} a good run gives (use --force to publish it)"
        ));
    }
    Ok(())
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
