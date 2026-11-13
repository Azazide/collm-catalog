// SPDX-License-Identifier: AGPL-3.0-only
//! `mlx.json`: the MLX chat models Collm's MLX build offers, read from Hugging Face.
//!
//! An MLX model is a folder, not one file: its configuration, its weights (one file or several parts),
//! its tokenizer and its chat template. So an entry lists every file of the folder with its size and
//! SHA-256, and the app downloads them all into one model. The ordinary build reads `catalog.json` and
//! never this file; this one is told apart by `"engine":"mlx"`.
//!
//! A repository is taken when it is tagged `mlx` and labelled a text-generation model on Hugging Face,
//! anyone can download it, its architecture is one the engine loads (`model_types` in `mlx.toml`), its
//! weights are not quantized in a way the engine does not read, its folder is whole (the configuration,
//! a tokenizer, every part the weights' index names) and it has a chat template. A file kept in LFS has
//! its SHA-256 from Hugging Face; a small one kept in git (a JSON configuration) is read and hashed here.

use crate::classify::{self, Signals};
use crate::describe;
use crate::hf::{Hf, HfError, Repo, TreeEntry, encode_path};
use crate::model::is_repo_id;
use crate::select::{Reason, split_id};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

pub const FORMAT: u32 = 1;
/// What tells this file from `catalog.json`: the app's MLX build reads only a file that says it.
pub const ENGINE: &str = "mlx";
/// The version of the rules an entry was made under (see `model::RULES`); a change reads every
/// repository again.
pub const RULES: u32 = 4;

/// Whether a README's first statement is only what a conversion tool or a quantizer's card writes about every model it
/// makes ("… was converted to MLX format from … using mlx-lm", "Model creator: … Original model: …"): it says nothing
/// of the model, and each one names its own repository, so the shared-blurb rule never sees two alike.
pub fn is_conversion_blurb(text: &str) -> bool {
    let lower = text.to_lowercase();
    lower.contains("converted to mlx format")
        || lower.starts_with("model creator:")
        || lower.starts_with("compatibility:")
        || (lower.contains("model creator:") && lower.contains("original model:"))
}

/// The repository a conversion blurb names as its source ("… converted to MLX format from `owner/name` …").
fn converted_from(text: &str) -> Option<String> {
    let lower = text.to_lowercase();
    let at = lower.find("mlx format from ")? + "mlx format from ".len();
    let rest = text.get(at..)?;
    let word = rest.split_whitespace().next()?.trim_end_matches(['.', ',', ';', ')']);
    is_repo_id(word).then(|| word.to_string())
}

/// What stands for a conversion blurb: the note of what the model was made from (its `base_model` tags, else the
/// source the blurb names), or nothing.
fn instead_of_blurb(blurb: &str, note: Option<String>) -> String {
    note.or_else(|| {
        converted_from(blurb).and_then(|base| describe::base_model_note(&[format!("base_model:quantized:{base}")]))
    })
    .unwrap_or_default()
}

/// A small file kept in git is read whole to be hashed, never more than this.
const SMALL_FILE_LIMIT: u64 = 32 * 1024 * 1024;
/// A configuration is read to be understood; more than this is not a configuration.
const CONFIG_LIMIT: u64 = 4 * 1024 * 1024;
/// A paragraph that opens the README of this many repositories is a blurb (as in `assemble.rs`).
const SHARED_BLURB: usize = 3;

/// What `mlx.toml` holds. An unknown key is an error.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MlxCurated {
    #[serde(default = "default_max_models")]
    pub max_models: usize,
    #[serde(default = "default_max_candidates")]
    pub max_candidates: usize,
    #[serde(default = "default_max_list_pages")]
    pub max_list_pages: usize,
    /// The largest folder offered, in bytes; none when 0.
    #[serde(default)]
    pub max_bytes: u64,
    /// The architectures (`model_type` of `config.json`) the engine loads for text.
    pub model_types: Vec<String>,
    /// The architectures it loads as models that see images: a folder with a vision tower (`vision_config`) and an
    /// image processor (`preprocessor_config.json` or `processor_config.json`) of one of these sees images.
    #[serde(default)]
    pub vision_model_types: Vec<String>,
    /// The bit widths of quantized weights the engine reads (unquantized weights are always read).
    #[serde(default = "default_bits")]
    pub bits: Vec<u64>,
    /// The other quantizations of the models taken, by their owners, that are added on top of `max_models`, at most.
    #[serde(default = "default_max_siblings")]
    pub max_siblings: usize,
    #[serde(default)]
    pub pinned: Vec<String>,
    #[serde(default)]
    pub exclude_developers: Vec<String>,
    #[serde(default)]
    pub exclude_name_contains: Vec<String>,
}

fn default_max_models() -> usize {
    300
}
fn default_max_candidates() -> usize {
    900
}
fn default_max_list_pages() -> usize {
    3
}
fn default_max_siblings() -> usize {
    1200
}
fn default_bits() -> Vec<u64> {
    vec![2, 3, 4, 5, 6, 8]
}

impl MlxCurated {
    pub fn parse(text: &str) -> Result<MlxCurated, String> {
        toml::from_str(text).map_err(|e| format!("mlx.toml: {e}"))
    }

    fn loads(&self, model_type: &str) -> bool {
        self.model_types.iter().any(|t| t.eq_ignore_ascii_case(model_type)) || self.sees(model_type)
    }

    fn sees(&self, model_type: &str) -> bool {
        self.vision_model_types.iter().any(|t| t.eq_ignore_ascii_case(model_type))
    }
}

/// Whether a part of a repository's name says only how it was quantized or packed (`4bit`, `bf16`, `mlx`, `dwq`,
/// `gs32`, `q4`, `mxfp4`). The app groups a model's repositories by the same words (`variantFamily`).
fn is_quant_word(word: &str) -> bool {
    let w = word.to_ascii_lowercase();
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    let after = |prefix: &str| w.strip_prefix(prefix).is_some_and(digits);
    matches!(w.as_str(), "bf16" | "fp16" | "f16" | "fp32" | "mlx" | "dwq" | "mixed")
        || w.strip_suffix("bits").or_else(|| w.strip_suffix("bit")).is_some_and(digits)
        || after("gs")
        || after("q")
        || after("mxfp")
        || after("nvfp")
}

/// A repository's name without its quantization's words at the end: what its quantizations share.
pub fn family_name(name: &str) -> &str {
    let mut rest = name;
    for _ in 0..6 {
        let Some(at) = rest.rfind(['-', '_']) else { break };
        let (head, word) = (&rest[..at], &rest[at + 1..]);
        if is_quant_word(word) {
            rest = head;
        } else if matches!(word.to_ascii_lowercase().as_str(), "bit" | "bits")
            && let Some(before) = head.rfind(['-', '_'])
            && head[before + 1..].bytes().all(|b| b.is_ascii_digit())
            && before + 1 < head.len()
        {
            rest = &head[..before];
        } else {
            break;
        }
    }
    if rest.is_empty() { name } else { rest }
}

/// The model a repository is a quantization of: its owner and its name without the quantization's words, lowercased.
pub fn variant_family(repo: &str) -> String {
    match split_id(repo) {
        Some((owner, name)) => format!("{owner}/{}", family_name(name)).to_lowercase(),
        None => repo.to_lowercase(),
    }
}

/// The file names of a model folder the app takes (`models/mlx.rs` of Collm says the same): the weights,
/// their index, the configuration and the tokenizer's files. Anything else in a repository (a README,
/// pictures, other formats) is not downloaded.
pub fn wanted(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    // A hidden file (`._model.safetensors`, what a Mac's copy leaves beside a file) is never one of them.
    !lower.starts_with('.') && lower.ends_with(".safetensors")
        || matches!(
            lower.as_str(),
            "config.json"
                | "generation_config.json"
                | "model.safetensors.index.json"
                | "tokenizer.json"
                | "tokenizer_config.json"
                | "special_tokens_map.json"
                | "added_tokens.json"
                | "vocab.json"
                | "merges.txt"
                | "tokenizer.model"
                | "chat_template.jinja"
                | "chat_template.json"
                // The image processor of a model that sees images.
                | "preprocessor_config.json"
                | "processor_config.json"
        )
}

const TOKENIZERS: &[&str] = &["tokenizer.json", "tokenizer.model", "vocab.json"];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MlxCatalog {
    pub format: u32,
    pub engine: String,
    #[serde(default)]
    pub rules: u32,
    pub models: Vec<MlxModel>,
}

/// One repository that holds an MLX chat model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MlxModel {
    pub model_name: String,
    pub developer: String,
    /// The commit every `path` below points at.
    pub sha: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
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
    /// `model_type` of `config.json`.
    pub architecture: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u64>,
    /// The bit width of quantized weights; left out for unquantized ones.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bits: Option<u64>,
    pub capabilities: Vec<String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub readme: String,
    /// The whole folder, bytes.
    pub size: u64,
    /// Every file of the folder, by name.
    pub files: Vec<FolderFile>,
}

/// One file of a model folder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FolderFile {
    /// Its name in the folder (the repository's top level).
    pub name: String,
    pub path: String,
    pub size: u64,
    pub sha256: String,
}

impl MlxCatalog {
    pub fn empty() -> Self {
        MlxCatalog { format: FORMAT, engine: ENGINE.to_string(), rules: RULES, models: Vec::new() }
    }

    /// What the next run starts from (as `Catalog::carried`).
    pub fn carried(&self, refresh: bool) -> MlxCatalog {
        if refresh || self.rules != RULES { MlxCatalog::empty() } else { self.clone() }
    }

    /// The catalog a file holds, or an empty one when it is not an MLX catalog of this format.
    pub fn from_text(text: &str) -> MlxCatalog {
        match serde_json::from_str::<MlxCatalog>(text) {
            Ok(catalog) if catalog.format == FORMAT && catalog.engine == ENGINE => catalog,
            _ => MlxCatalog::empty(),
        }
    }

    /// Takes out the models whose data is not well formed, and says which and why.
    pub fn drop_malformed(&mut self) -> Vec<(String, Vec<String>)> {
        let mut dropped = Vec::new();
        let mut seen = HashSet::new();
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
}

impl MlxModel {
    /// What is wrong with the entry, as the app and a download would meet it.
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
        let at = format!("https://huggingface.co/{}/resolve/{}/", self.model_name, self.sha);
        let names: HashSet<String> = self.files.iter().map(|f| f.name.to_ascii_lowercase()).collect();
        if !names.contains("config.json") {
            found.push("no config.json".to_string());
        }
        if !names.iter().any(|n| n.ends_with(".safetensors")) {
            found.push("no weights".to_string());
        }
        if !TOKENIZERS.iter().any(|t| names.contains(*t)) {
            found.push("no tokenizer".to_string());
        }
        if names.len() != self.files.len() {
            found.push("a file is listed twice".to_string());
        }
        for file in &self.files {
            let plain = !file.name.is_empty()
                && !file.name.contains(['/', '\\'])
                && !file.name.starts_with('.')
                && !file.name.chars().any(char::is_control);
            if !plain || !wanted(&file.name) {
                found.push(format!("`{}` is not a file of a model folder", file.name));
                continue;
            }
            if file.size == 0 {
                found.push(format!("{}: no size", file.name));
            }
            if file.sha256.len() != 64 || !file.sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
                found.push(format!("{}: no SHA-256", file.name));
            }
            if file.path != format!("{at}{}", encode_path(&file.name)) {
                found.push(format!("{}: the address {} is not the file at this commit", file.name, file.path));
            }
        }
        if self.size != self.files.iter().map(|f| f.size).sum::<u64>() {
            found.push("the size is not the sum of the files".to_string());
        }
        if !self.readme.is_empty() && self.readme != format!("{at}README.md") {
            found.push("the README is not this repository's at this commit".to_string());
        }
        found
    }
}

/// The file as text: one model per line.
pub fn render(catalog: &MlxCatalog) -> String {
    let mut out = format!(
        "{{\"format\":{},\"engine\":{},\"rules\":{},\"models\":[\n",
        catalog.format,
        serde_json::to_string(&catalog.engine).expect("a string serializes"),
        catalog.rules
    );
    for (i, model) in catalog.models.iter().enumerate() {
        out.push_str(&serde_json::to_string(model).expect("a model serializes"));
        out.push_str(if i + 1 < catalog.models.len() { ",\n" } else { "\n" });
    }
    out.push_str("]}\n");
    out
}

#[derive(Debug, Default, Clone)]
pub struct Stats {
    pub listed: usize,
    pub unreadable: usize,
    pub examined: usize,
    pub reused: usize,
    pub fetched: usize,
    pub kept_after_error: usize,
    pub failed: usize,
    pub rejected: BTreeMap<Reason, usize>,
}

enum Outcome {
    Reused(MlxModel),
    Fetched { model: MlxModel, note: Option<String> },
    Rejected(Reason),
}

/// What the list entry alone rules out.
fn prefilter(curated: &MlxCurated, repo: &Repo) -> Result<(), Reason> {
    let Some((owner, _)) = split_id(&repo.id) else {
        return Err(Reason::ExcludedName);
    };
    if !repo.gated.is_open() {
        return Err(Reason::Gated);
    }
    if repo.disabled {
        return Err(Reason::Disabled);
    }
    if repo.private {
        return Err(Reason::Private);
    }
    // A chat model, or one that sees images and answers in text; an unlabelled one is judged by its template.
    if let Some(tag) = repo.pipeline_tag.as_deref()
        && !matches!(tag, "text-generation" | "image-text-to-text")
    {
        return Err(Reason::NotAChatPipeline);
    }
    if curated.exclude_developers.iter().any(|d| d.eq_ignore_ascii_case(owner)) {
        return Err(Reason::ExcludedDeveloper);
    }
    let name = repo.id.to_lowercase();
    if curated.exclude_name_contains.iter().any(|part| name.contains(&part.to_lowercase())) {
        return Err(Reason::ExcludedName);
    }
    // What the list read from `config.json`; the file itself decides when the list says nothing.
    if let Some(model_type) = repo.config.as_ref().and_then(|c| c.model_type.as_deref())
        && !curated.loads(model_type)
    {
        return Err(Reason::ExcludedArchitecture);
    }
    Ok(())
}

pub fn build(
    hf: &dyn Hf,
    curated: &MlxCurated,
    previous: &MlxCatalog,
    log: &mut dyn FnMut(String),
) -> Result<(MlxCatalog, Stats), HfError> {
    let before: HashMap<String, &MlxModel> =
        previous.models.iter().map(|m| (m.model_name.to_lowercase(), m)).collect();
    let pinned: HashSet<String> = curated.pinned.iter().map(|id| id.to_lowercase()).collect();
    let mut stats = Stats::default();
    let mut seen: HashSet<String> = HashSet::new();
    let mut models: Vec<MlxModel> = Vec::new();
    let mut notes: HashMap<String, String> = HashMap::new();
    let mut queue: VecDeque<Repo> = VecDeque::new();
    let (mut taken, mut looked_into) = (0usize, 0usize);

    for id in &curated.pinned {
        match hf.mlx_repo(id) {
            Ok(repo) => queue.push_back(repo),
            Err(e) => {
                log(format!("pinned {id}: {e}"));
                match before.get(&id.to_lowercase()) {
                    Some(old) if seen.insert(id.to_lowercase()) => {
                        stats.kept_after_error += 1;
                        models.push((*old).clone());
                    }
                    Some(_) => {}
                    None => stats.failed += 1,
                }
            }
        }
    }

    let (mut cursor, mut pages, mut list_done): (Option<String>, usize, bool) = (None, 0, false);
    loop {
        if taken >= curated.max_models || looked_into >= curated.max_candidates {
            break;
        }
        let repo = match queue.pop_front() {
            Some(repo) => repo,
            None => {
                if list_done || pages >= curated.max_list_pages {
                    break;
                }
                let page = hf.mlx_list_page(cursor.as_deref())?;
                pages += 1;
                stats.listed += page.repos.len();
                stats.unreadable += page.skipped;
                list_done = page.next.is_none();
                cursor = page.next;
                queue.extend(page.repos);
                continue;
            }
        };
        let key = repo.id.to_lowercase();
        if seen.contains(&key) {
            continue;
        }
        if let Err(reason) = prefilter(curated, &repo) {
            *stats.rejected.entry(reason).or_default() += 1;
            continue;
        }
        seen.insert(key.clone());
        stats.examined += 1;
        let is_pinned = pinned.contains(&key);
        if !is_pinned {
            looked_into += 1;
        }
        match process(hf, curated, &repo, before.get(&key).copied()) {
            Ok(Outcome::Reused(model)) => {
                stats.reused += 1;
                taken += usize::from(!is_pinned);
                models.push(model);
            }
            Ok(Outcome::Fetched { model, note }) => {
                stats.fetched += 1;
                if let Some(note) = note {
                    notes.insert(key.clone(), note);
                }
                taken += usize::from(!is_pinned);
                let place = if is_pinned { "pinned".to_string() } else { format!("{taken}/{}", curated.max_models) };
                log(format!("[{place}] read {}: {} files, {} bytes", repo.id, model.files.len(), model.size));
                models.push(model);
            }
            Ok(Outcome::Rejected(reason)) => *stats.rejected.entry(reason).or_default() += 1,
            Err(e) => {
                log(format!("{}: {e}", repo.id));
                match before.get(&key) {
                    Some(old) => {
                        stats.kept_after_error += 1;
                        taken += usize::from(!is_pinned);
                        models.push((*old).clone());
                    }
                    None => stats.failed += 1,
                }
            }
        }
    }

    // A model is offered with all its quantizations by its owner, not only those downloads put in the list: one search
    // per model for the owner's repositories of the same name, each judged as any other, on top of the limit.
    let mut families: Vec<(String, String, String)> = Vec::new();
    for model in &models {
        let family = variant_family(&model.model_name);
        if !families.iter().any(|(key, _, _)| key == &family) {
            if let Some((owner, name)) = split_id(&model.model_name) {
                families.push((family, owner.to_string(), family_name(name).to_string()));
            }
        }
    }
    let mut siblings = 0usize;
    for (family, owner, name) in families {
        if siblings >= curated.max_siblings {
            break;
        }
        let found = match hf.mlx_search(&owner, &name) {
            Ok(found) => found,
            Err(e) => {
                log(format!("the other quantizations of {owner}/{name}: {e}"));
                continue;
            }
        };
        for repo in found {
            let key = repo.id.to_lowercase();
            if siblings >= curated.max_siblings || seen.contains(&key) || variant_family(&repo.id) != family {
                continue;
            }
            seen.insert(key.clone());
            if let Err(reason) = prefilter(curated, &repo) {
                *stats.rejected.entry(reason).or_default() += 1;
                continue;
            }
            stats.examined += 1;
            match process(hf, curated, &repo, before.get(&key).copied()) {
                Ok(Outcome::Reused(model)) => {
                    stats.reused += 1;
                    siblings += 1;
                    models.push(model);
                }
                Ok(Outcome::Fetched { model, note }) => {
                    stats.fetched += 1;
                    siblings += 1;
                    if let Some(note) = note {
                        notes.insert(key, note);
                    }
                    log(format!("[variant {siblings}] read {}: {} files, {} bytes", repo.id, model.files.len(), model.size));
                    models.push(model);
                }
                Ok(Outcome::Rejected(reason)) => *stats.rejected.entry(reason).or_default() += 1,
                Err(e) => {
                    log(format!("{}: {e}", repo.id));
                    match before.get(&key) {
                        Some(old) => {
                            stats.kept_after_error += 1;
                            models.push((*old).clone());
                        }
                        None => stats.failed += 1,
                    }
                }
            }
        }
    }

    drop_shared_blurbs(&mut models, &notes);
    models.sort_by_key(|m| m.model_name.to_lowercase());
    Ok((MlxCatalog { format: FORMAT, engine: ENGINE.to_string(), rules: RULES, models }, stats))
}

/// The JSON of a small file, or nothing when it is not JSON.
fn json_of(bytes: &[u8]) -> Option<serde_json::Value> {
    serde_json::from_slice(bytes).ok()
}

/// The chat template a tokenizer configuration holds: a text, or the `default` of a named list.
fn template_in(value: &serde_json::Value) -> Option<String> {
    match value.get("chat_template")? {
        serde_json::Value::String(text) if !text.trim().is_empty() => Some(text.clone()),
        serde_json::Value::Array(list) => list
            .iter()
            .find(|t| t.get("name").and_then(|n| n.as_str()) == Some("default"))
            .or_else(|| list.first())
            .and_then(|t| t.get("template"))
            .and_then(|t| t.as_str())
            .filter(|t| !t.trim().is_empty())
            .map(str::to_string),
        _ => None,
    }
}

/// The window a configuration states, for a model or the text part of one.
fn context_of(config: &serde_json::Value) -> Option<u64> {
    let read = |v: &serde_json::Value| v.get("max_position_embeddings").and_then(|n| n.as_u64()).filter(|n| *n > 0);
    read(config).or_else(|| config.get("text_config").and_then(read))
}

fn hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

fn process(hf: &dyn Hf, curated: &MlxCurated, repo: &Repo, old: Option<&MlxModel>) -> Result<Outcome, HfError> {
    let Some(rev) = repo.sha.clone() else {
        return Ok(Outcome::Rejected(Reason::NoCommit));
    };
    if let Some(old) = old.filter(|o| o.sha == rev) {
        if !curated.loads(&old.architecture) {
            return Ok(Outcome::Rejected(Reason::ExcludedArchitecture));
        }
        let mut model = old.clone();
        model.downloads = repo.downloads;
        model.likes = repo.likes;
        return Ok(Outcome::Reused(model));
    }
    let Some((developer, _)) = split_id(&repo.id) else {
        return Ok(Outcome::Rejected(Reason::ExcludedName));
    };
    let id = repo.id.as_str();

    // The folder: the files of the top level a model folder holds.
    let tree = hf.tree(id, &rev)?;
    let entries: Vec<&TreeEntry> =
        tree.iter().filter(|e| e.kind == "file" && !e.path.contains('/') && wanted(&e.path)).collect();
    let has = |name: &str| entries.iter().any(|e| e.path.eq_ignore_ascii_case(name));
    if !entries.iter().any(|e| e.path.to_ascii_lowercase().ends_with(".safetensors")) {
        return Ok(Outcome::Rejected(Reason::NoWholeFile));
    }
    if !has("config.json") || !TOKENIZERS.iter().any(|t| has(t)) {
        return Ok(Outcome::Rejected(Reason::MissingFiles));
    }

    // The small files are read once: to be understood and to be hashed.
    let mut bodies: HashMap<String, Vec<u8>> = HashMap::new();
    let mut read = |name: &str, limit: u64| -> Result<Option<Vec<u8>>, HfError> {
        if let Some(body) = bodies.get(name) {
            return Ok(Some(body.clone()));
        }
        let body = hf.small_file(id, &rev, name, limit)?;
        if let Some(body) = &body {
            bodies.insert(name.to_string(), body.clone());
        }
        Ok(body)
    };

    let Some(config) = read("config.json", CONFIG_LIMIT)?.as_deref().and_then(json_of) else {
        return Ok(Outcome::Rejected(Reason::MissingFiles));
    };
    let Some(architecture) = config.get("model_type").and_then(|t| t.as_str()).map(str::to_string) else {
        return Ok(Outcome::Rejected(Reason::ExcludedArchitecture));
    };
    if !curated.loads(&architecture) {
        return Ok(Outcome::Rejected(Reason::ExcludedArchitecture));
    }
    // A model that sees images, as the engine loads one (`Engine.takesImages`); an architecture the engine loads only
    // that way is no text model without its vision tower and processor.
    let vision = curated.sees(&architecture)
        && config.get("vision_config").is_some_and(|v| v.is_object())
        && (has("preprocessor_config.json") || has("processor_config.json"));
    if !vision && !curated.model_types.iter().any(|t| t.eq_ignore_ascii_case(&architecture)) {
        return Ok(Outcome::Rejected(Reason::ExcludedArchitecture));
    }
    let quantization = config.get("quantization").or_else(|| config.get("quantization_config"));
    let bits = match quantization {
        None => None,
        Some(q) => {
            let mode = q.get("mode").and_then(|m| m.as_str()).unwrap_or("affine");
            match (mode, q.get("bits").and_then(|b| b.as_u64())) {
                ("affine", Some(bits)) if curated.bits.contains(&bits) => Some(bits),
                // The engine's microscaling floats (mlx-swift's `QuantizationMode`): their width is the mode's own.
                ("mxfp4" | "nvfp4", _) => Some(4),
                ("mxfp8", _) => Some(8),
                _ => return Ok(Outcome::Rejected(Reason::UnsupportedQuantization)),
            }
        }
    };

    // Every part the weights' index names is in the folder.
    if has("model.safetensors.index.json") {
        let Some(index) = read("model.safetensors.index.json", SMALL_FILE_LIMIT)?.as_deref().and_then(json_of) else {
            return Ok(Outcome::Rejected(Reason::MissingFiles));
        };
        let parts: HashSet<&str> = index
            .get("weight_map")
            .and_then(|m| m.as_object())
            .map(|m| m.values().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();
        if parts.is_empty() || !parts.iter().all(|part| has(part)) {
            return Ok(Outcome::Rejected(Reason::MissingFiles));
        }
    }

    // The chat template: the tokenizer's configuration, or a file of its own.
    let mut template = None;
    if has("tokenizer_config.json") {
        template = read("tokenizer_config.json", SMALL_FILE_LIMIT)?.as_deref().and_then(json_of).as_ref().and_then(template_in);
    }
    if template.is_none() && has("chat_template.jinja") {
        template = read("chat_template.jinja", SMALL_FILE_LIMIT)?
            .map(|b| String::from_utf8_lossy(&b).into_owned())
            .filter(|t| !t.trim().is_empty());
    }
    if template.is_none() && has("chat_template.json") {
        template = read("chat_template.json", SMALL_FILE_LIMIT)?.as_deref().and_then(json_of).as_ref().and_then(template_in);
    }
    let Some(template) = template else {
        return Ok(Outcome::Rejected(Reason::NoChatTemplate));
    };

    // Each file with its size and SHA-256: Hugging Face's for a file in LFS, this program's for one in git.
    let mut files = Vec::with_capacity(entries.len());
    for entry in &entries {
        let (size, sha256) = match &entry.lfs {
            Some(lfs) => (if lfs.size > 0 { lfs.size } else { entry.size.unwrap_or(0) }, lfs.oid.to_ascii_lowercase()),
            None => match read(&entry.path, SMALL_FILE_LIMIT)? {
                Some(body) => (body.len() as u64, hex(&body)),
                None => return Ok(Outcome::Rejected(Reason::MissingFiles)),
            },
        };
        files.push(FolderFile {
            name: entry.path.clone(),
            path: format!("https://huggingface.co/{id}/resolve/{rev}/{}", encode_path(&entry.path)),
            size,
            sha256,
        });
    }
    files.sort_by(|a, b| a.name.cmp(&b.name));
    let size: u64 = files.iter().map(|f| f.size).sum();
    if curated.max_bytes > 0 && size > curated.max_bytes {
        return Ok(Outcome::Rejected(Reason::TooLarge));
    }

    let readme = hf.readme(id, &rev)?;
    let note = describe::base_model_note(&repo.tags);
    let first = readme.as_deref().map(describe::description).unwrap_or_default();
    let description = if is_conversion_blurb(&first) {
        instead_of_blurb(&first, note.clone())
    } else {
        Some(first).filter(|text| !text.is_empty()).or_else(|| note.clone()).unwrap_or_default()
    };
    Ok(Outcome::Fetched {
        model: MlxModel {
            model_name: repo.id.clone(),
            developer: developer.to_string(),
            sha: rev.clone(),
            description,
            downloads: repo.downloads,
            likes: repo.likes,
            created_at: repo.created_at.clone(),
            updated_at: repo.last_modified.clone(),
            license: repo.card_data.as_ref().and_then(|c| c.license()).and_then(|l| describe::label(&l)),
            pipeline_tag: repo.pipeline_tag.as_deref().and_then(describe::label),
            architecture,
            context_length: context_of(&config),
            bits,
            capabilities: {
                let mut found = classify::capabilities(&Signals {
                    id,
                    tags: &repo.tags,
                    pipeline_tag: repo.pipeline_tag.as_deref(),
                    tools: Some(classify::template_takes_tools(&template)),
                    // Sound and video are never offered: the engine takes text and pictures. Vision is the folder's.
                    has_projector: false,
                    readme: readme.as_deref(),
                });
                if vision {
                    found.push(classify::VISION.to_string());
                    found.sort();
                }
                found
            },
            readme: readme.map(|_| format!("https://huggingface.co/{id}/resolve/{rev}/README.md")).unwrap_or_default(),
            size,
            files,
        },
        note,
    })
}

/// As `assemble::drop_shared_blurbs`: a description three or more models share is replaced by what the
/// model was made from, or by nothing (a conversion's README is one text for every model it converted).
fn drop_shared_blurbs(models: &mut [MlxModel], notes: &HashMap<String, String>) {
    let counted = |m: &MlxModel| !m.description.is_empty() && !describe::is_note(&m.description);
    let mut count: HashMap<String, usize> = HashMap::new();
    for model in models.iter().filter(|m| counted(m)) {
        *count.entry(model.description.clone()).or_default() += 1;
    }
    for model in models.iter_mut().filter(|m| counted(m)) {
        if count[&model.description] >= SHARED_BLURB {
            model.description = notes.get(&model.model_name.to_lowercase()).cloned().unwrap_or_default();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hf::{Lfs, ListPage, RepoConfig};
    use std::cell::RefCell;

    const SHA: &str = "0123456789abcdef0123456789abcdef01234567";
    const WEIGHTS_SHA: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const TEMPLATE: &str = "{% for m in messages %}{{ m.content }}{% endfor %}{% if tools %}{{ tools }}{% endif %}";

    /// Hugging Face answered from memory: one repository per id, its files by name.
    #[derive(Default)]
    struct Fake {
        repos: Vec<Repo>,
        /// Repositories the list does not give (too few downloads), found only by a search of their owner.
        unlisted: Vec<Repo>,
        files: HashMap<(String, String), Vec<u8>>,
        lfs: HashMap<(String, String), (u64, String)>,
        asked: RefCell<Vec<String>>,
    }

    impl Fake {
        fn with(mut self, id: &str, model_type: &str, files: &[(&str, &str)], weights: &[&str]) -> Self {
            self.repos.push(Repo {
                id: id.to_string(),
                downloads: 10,
                sha: Some(SHA.to_string()),
                pipeline_tag: Some("text-generation".into()),
                config: Some(RepoConfig { model_type: Some(model_type.to_string()) }),
                ..Repo::default()
            });
            for (name, body) in files {
                self.files.insert((id.to_string(), name.to_string()), body.as_bytes().to_vec());
            }
            for name in weights {
                self.lfs.insert((id.to_string(), name.to_string()), (1000, WEIGHTS_SHA.to_string()));
            }
            self
        }
    }

    impl Hf for Fake {
        fn list_page(&self, _: Option<&str>) -> Result<ListPage, HfError> {
            unreachable!("the MLX catalog never reads the GGUF list")
        }
        fn repo(&self, _: &str) -> Result<Repo, HfError> {
            unreachable!()
        }
        fn mlx_list_page(&self, _: Option<&str>) -> Result<ListPage, HfError> {
            let listed = self.repos.iter().filter(|r| !self.unlisted.iter().any(|u| u.id == r.id)).cloned().collect();
            Ok(ListPage { repos: listed, next: None, skipped: 0 })
        }
        fn mlx_search(&self, author: &str, text: &str) -> Result<Vec<Repo>, HfError> {
            self.asked.borrow_mut().push(format!("search {author} {text}"));
            Ok(self
                .repos
                .iter()
                .filter(|r| r.id.starts_with(&format!("{author}/")) && r.id.to_lowercase().contains(&text.to_lowercase()))
                .cloned()
                .collect())
        }
        fn tree(&self, id: &str, _: &str) -> Result<Vec<TreeEntry>, HfError> {
            self.asked.borrow_mut().push(format!("tree {id}"));
            let mut out = Vec::new();
            for ((repo, name), body) in &self.files {
                if repo == id {
                    out.push(TreeEntry { kind: "file".into(), path: name.clone(), size: Some(body.len() as u64), lfs: None });
                }
            }
            for ((repo, name), (size, oid)) in &self.lfs {
                if repo == id {
                    out.push(TreeEntry {
                        kind: "file".into(),
                        path: name.clone(),
                        size: Some(*size),
                        lfs: Some(Lfs { oid: oid.clone(), size: *size }),
                    });
                }
            }
            out.push(TreeEntry { kind: "file".into(), path: "README.md".into(), size: Some(5), lfs: None });
            out.push(TreeEntry { kind: "file".into(), path: "sub/model.safetensors".into(), size: Some(5), lfs: None });
            Ok(out)
        }
        fn readme(&self, _: &str, _: &str) -> Result<Option<String>, HfError> {
            Ok(Some("# A model\n\nA small model that answers questions about the weather in plain words.\n".into()))
        }
        fn small_file(&self, id: &str, _: &str, path: &str, limit: u64) -> Result<Option<Vec<u8>>, HfError> {
            self.asked.borrow_mut().push(format!("file {id} {path}"));
            match self.files.get(&(id.to_string(), path.to_string())) {
                Some(body) if body.len() as u64 > limit => Err(HfError("too large".into())),
                other => Ok(other.cloned()),
            }
        }
    }

    fn curated() -> MlxCurated {
        MlxCurated::parse("model_types = [\"qwen3\", \"llama\"]\nvision_model_types = [\"qwen2_5_vl\", \"qwen3\"]\n").unwrap()
    }

    fn config(extra: &str) -> String {
        format!("{{\"model_type\":\"qwen3\",\"max_position_embeddings\":40960{extra}}}")
    }

    fn tokenizer_config() -> String {
        serde_json::json!({ "chat_template": TEMPLATE }).to_string()
    }

    fn run(fake: &Fake) -> (MlxCatalog, Stats) {
        build(fake, &curated(), &MlxCatalog::empty(), &mut |_| {}).unwrap()
    }

    #[test]
    fn a_whole_folder_becomes_an_entry_with_every_file_s_size_and_sha256() {
        let cfg = config(",\"quantization\":{\"group_size\":64,\"bits\":4}");
        let tok = tokenizer_config();
        let fake = Fake::default().with(
            "mlx-community/Tiny-4bit",
            "qwen3",
            &[("config.json", &cfg), ("tokenizer_config.json", &tok), ("tokenizer.json", "{}")],
            &["model.safetensors"],
        );
        let (mut catalog, stats) = run(&fake);
        assert_eq!(stats.fetched, 1, "{stats:?}");
        assert!(catalog.drop_malformed().is_empty());
        let model = &catalog.models[0];
        assert_eq!(model.architecture, "qwen3");
        assert_eq!(model.bits, Some(4));
        assert_eq!(model.context_length, Some(40960));
        assert!(model.capabilities.contains(&"tools".to_string()), "{:?}", model.capabilities);
        let names: Vec<&str> = model.files.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"], "no README, nothing below the top");
        let config_file = &model.files[0];
        assert_eq!(config_file.sha256, hex(cfg.as_bytes()), "a file in git is hashed here");
        assert_eq!(config_file.size, cfg.len() as u64);
        assert_eq!(config_file.path, format!("https://huggingface.co/mlx-community/Tiny-4bit/resolve/{SHA}/config.json"));
        assert_eq!(model.files[1].sha256, WEIGHTS_SHA, "a file in LFS has Hugging Face's");
        assert_eq!(model.size, model.files.iter().map(|f| f.size).sum::<u64>());
        // Read once, though it is both understood and hashed.
        assert_eq!(fake.asked.borrow().iter().filter(|a| a.ends_with(" config.json")).count(), 1);
        let text = render(&catalog);
        assert!(text.starts_with(&format!("{{\"format\":1,\"engine\":\"mlx\",\"rules\":{RULES},\"models\":[\n")));
        assert_eq!(MlxCatalog::from_text(&text), catalog);
    }

    #[test]
    fn what_the_engine_cannot_run_or_the_folder_lacks_is_left_out_and_said_why() {
        let tok = tokenizer_config();
        let base = [("tokenizer_config.json", tok.as_str()), ("tokenizer.json", "{}")];
        let with = |fake: Fake, id: &str, model_type: &str, cfg: &str, more: &[(&str, &str)], weights: &[&str]| {
            let mut files = vec![("config.json", cfg)];
            files.extend(base.iter().copied());
            files.extend(more.iter().copied());
            fake.with(id, model_type, &files, weights)
        };
        let gemma = "{\"model_type\":\"gemma9\"}";
        let one_bit = config(",\"quantization\":{\"bits\":1}");
        // A mode the engine has no kernel for.
        let mx = config(",\"quantization\":{\"bits\":4,\"mode\":\"oq\"}");
        let index = "{\"weight_map\":{\"a\":\"model-00001-of-00002.safetensors\",\"b\":\"model-00002-of-00002.safetensors\"}}";
        let plain = config("");
        let fake = Fake::default();
        let fake = with(fake, "a/unknown-arch", "gemma9", gemma, &[], &["model.safetensors"]);
        let fake = with(fake, "a/one-bit", "qwen3", &one_bit, &[], &["model.safetensors"]);
        let fake = with(fake, "a/mx", "qwen3", &mx, &[], &["model.safetensors"]);
        let fake = with(fake, "a/missing-part", "qwen3", &plain, &[("model.safetensors.index.json", index)], &["model-00001-of-00002.safetensors"]);
        let fake = with(fake, "a/no-weights", "qwen3", &plain, &[], &[]);
        let fake = fake.with("a/no-template", "qwen3", &[("config.json", &plain), ("tokenizer.json", "{}"), ("tokenizer_config.json", "{}")], &["model.safetensors"]);
        let fake = fake.with("a/no-tokenizer", "qwen3", &[("config.json", &plain)], &["model.safetensors"]);
        let (catalog, stats) = run(&fake);
        assert!(catalog.models.is_empty(), "{:?}", catalog.models);
        let count = |r: Reason| stats.rejected.get(&r).copied().unwrap_or(0);
        assert_eq!(count(Reason::ExcludedArchitecture), 1);
        assert_eq!(count(Reason::UnsupportedQuantization), 2);
        assert_eq!(count(Reason::MissingFiles), 2);
        assert_eq!(count(Reason::NoWholeFile), 1);
        assert_eq!(count(Reason::NoChatTemplate), 1);
        // The architecture the list read is enough to pass a repository by without a request.
        assert!(!fake.asked.borrow().iter().any(|a| a.contains("unknown-arch")));
    }

    #[test]
    fn a_folder_that_sees_images_is_offered_with_vision_and_one_without_its_processor_is_not() {
        let tok = tokenizer_config();
        let vl = r#"{"model_type":"qwen2_5_vl","vision_config":{"depth":32},"text_config":{"max_position_embeddings":32768}}"#;
        let mut fake = Fake::default()
            .with("o/Sees-4bit", "qwen2_5_vl", &[("config.json", vl), ("tokenizer.json", "{}"), ("tokenizer_config.json", &tok), ("preprocessor_config.json", "{}")], &["model.safetensors"])
            .with("o/Blind-4bit", "qwen2_5_vl", &[("config.json", vl), ("tokenizer.json", "{}"), ("tokenizer_config.json", &tok)], &["model.safetensors"])
            .with("o/Text-4bit", "qwen3", &[("config.json", &config("")), ("tokenizer.json", "{}"), ("tokenizer_config.json", &tok)], &["model.safetensors"])
            .with("o/Talks-4bit", "qwen3", &[("config.json", &config("")), ("tokenizer.json", "{}"), ("tokenizer_config.json", &tok)], &["model.safetensors"]);
        // Hugging Face labels a model that sees images `image-text-to-text`; one that speaks is no chat model here.
        fake.repos[0].pipeline_tag = Some("image-text-to-text".into());
        fake.repos[3].pipeline_tag = Some("text-to-speech".into());
        let (catalog, stats) = run(&fake);
        let names: Vec<&str> = catalog.models.iter().map(|m| m.model_name.as_str()).collect();
        assert_eq!(names, ["o/Sees-4bit", "o/Text-4bit"], "{stats:?}");
        let sees = &catalog.models[0];
        assert!(sees.capabilities.contains(&"vision".to_string()), "{:?}", sees.capabilities);
        assert!(sees.files.iter().any(|f| f.name == "preprocessor_config.json"));
        assert!(!catalog.models[1].capabilities.contains(&"vision".to_string()));
        assert_eq!(stats.rejected.get(&Reason::ExcludedArchitecture).copied(), Some(1), "a vision architecture without its processor");
        assert_eq!(stats.rejected.get(&Reason::NotAChatPipeline).copied(), Some(1));
    }

    #[test]
    fn a_template_of_its_own_file_counts_and_an_unchanged_repository_costs_no_request() {
        let plain = config("");
        let fake = Fake::default().with(
            "o/Model",
            "qwen3",
            &[("config.json", &plain), ("tokenizer.json", "{}"), ("chat_template.jinja", TEMPLATE)],
            &["model.safetensors"],
        );
        let (first, _) = run(&fake);
        assert_eq!(first.models.len(), 1);
        fake.asked.borrow_mut().clear();
        let (again, stats) = build(&fake, &curated(), &first.carried(false), &mut |_| {}).unwrap();
        assert_eq!(stats.reused, 1);
        // Only the search for the model's other quantizations: nothing of the repository is read again.
        assert!(fake.asked.borrow().iter().all(|a| a.starts_with("search ")), "{:?}", fake.asked.borrow());
        assert_eq!(again.models, first.models);
        // Another rule version reads it again.
        let mut older = first.clone();
        older.rules = 0;
        assert!(older.carried(false).models.is_empty());
    }

    #[test]
    fn an_entry_that_leaves_its_commit_or_names_a_foreign_file_is_malformed() {
        let file = |name: &str| FolderFile {
            name: name.to_string(),
            path: format!("https://huggingface.co/o/m/resolve/{SHA}/{}", encode_path(name)),
            size: 1,
            sha256: WEIGHTS_SHA.to_string(),
        };
        let good = MlxModel {
            model_name: "o/m".into(),
            developer: "o".into(),
            sha: SHA.into(),
            description: String::new(),
            downloads: 0,
            likes: 0,
            created_at: None,
            updated_at: None,
            license: None,
            pipeline_tag: None,
            architecture: "qwen3".into(),
            context_length: None,
            bits: None,
            capabilities: vec![],
            readme: String::new(),
            size: 3,
            files: vec![file("config.json"), file("model.safetensors"), file("tokenizer.json")],
        };
        assert!(good.problems().is_empty(), "{:?}", good.problems());
        let mut elsewhere = good.clone();
        elsewhere.files[1].path = "https://example.com/model.safetensors".into();
        assert!(elsewhere.problems().iter().any(|p| p.contains("not the file at this commit")));
        let mut foreign = good.clone();
        foreign.files.push(file("run.sh"));
        foreign.size += 1;
        assert!(foreign.problems().iter().any(|p| p.contains("not a file of a model folder")));
        let mut climbing = good.clone();
        climbing.files[0].name = "../config.json".into();
        assert!(!climbing.problems().is_empty());
        let mut unhashed = good.clone();
        unhashed.files[2].sha256 = "abc".into();
        assert!(unhashed.problems().iter().any(|p| p.contains("SHA-256")));
        let mut wrong_sum = good;
        wrong_sum.size = 99;
        assert!(wrong_sum.problems().iter().any(|p| p.contains("sum")));
    }

    #[test]
    fn the_shipped_mlx_toml_reads_and_names_the_engine_s_architectures() {
        let shipped = MlxCurated::parse(include_str!("../mlx.toml")).unwrap();
        assert!(shipped.loads("qwen3") && shipped.loads("llama") && !shipped.loads("bert"));
        assert_eq!(shipped.bits, default_bits());
        assert!(MlxCurated::parse("model_types = []\nmax_model = 3\n").is_err(), "an unknown key is an error");
    }

    #[test]
    fn a_conversion_blurb_gives_way_to_what_the_model_was_made_from() {
        let blurb = "This model mlx-community/Qwen3-4B-4bit was converted to MLX format from Qwen/Qwen3-4B using mlx-lm version 0.26.2.";
        assert!(is_conversion_blurb(blurb));
        assert!(is_conversion_blurb("The Model a/b was converted to MLX format from c/d using mlx-lm version 0.20."));
        assert!(is_conversion_blurb("Model creator: Qwen Original model: Qwen3-4B MLX quantization: provided by LM Studio team"));
        assert!(is_conversion_blurb("Compatibility: Apple Silicon Macs Model creator: Qwen Original model: Qwen3-4B"));
        assert!(!is_conversion_blurb("A small model that answers questions about the weather."));
        // The tags' note first; the source the blurb names when there are none; nothing when neither says.
        let tags = describe::base_model_note(&["base_model:finetune:x/Base".to_string()]);
        assert_eq!(instead_of_blurb(blurb, tags.clone()), tags.unwrap());
        let named = instead_of_blurb(blurb, None);
        assert!(named.contains("Qwen/Qwen3-4B") && !named.contains("mlx-lm"), "{named}");
        assert_eq!(instead_of_blurb("Model creator: Qwen Original model: Qwen3-4B", None), "");
    }

    #[test]
    fn a_hidden_file_is_no_file_of_a_model_folder() {
        assert!(wanted("model.safetensors") && !wanted("._model.safetensors") && !wanted(".config.json"));
    }

    #[test]
    fn a_name_loses_only_how_it_was_quantized() {
        for (name, family) in [
            ("Qwen3-Coder-Next-MLX-4bit", "Qwen3-Coder-Next"),
            ("Qwen3-Coder-Next-mxfp4", "Qwen3-Coder-Next"),
            ("LFM2.5-24B-A2B-MLX-bf16", "LFM2.5-24B-A2B"),
            ("Model-DWQ-3bit", "Model"),
            ("Model-6bit-gs32", "Model"),
            ("Model-4-bit", "Model"),
            ("gemma-3-270m-it-qat-4bit", "gemma-3-270m-it-qat"),
            ("Qwen3-Coder-30B-A3B-Instruct", "Qwen3-Coder-30B-A3B-Instruct"),
            ("MLX", "MLX"),
        ] {
            assert_eq!(family_name(name), family, "{name}");
        }
        assert_eq!(variant_family("Owner/Qwen3-4B-8bit"), variant_family("owner/qwen3-4b-bf16"));
    }

    #[test]
    fn a_model_is_offered_with_its_owner_s_other_quantizations_and_the_engine_s_microscaling_floats() {
        let tok = tokenizer_config();
        let affine = |bits: u32| config(&format!(",\"quantization\":{{\"group_size\":64,\"bits\":{bits}}}"));
        let (four, three) = (affine(4), affine(3));
        let mx = config(",\"quantization\":{\"group_size\":32,\"bits\":4,\"mode\":\"mxfp4\"}");
        let odd = config(",\"quantization\":{\"group_size\":32,\"bits\":4,\"mode\":\"oq\"}");
        let files = |cfg: &str| vec![("config.json", cfg.to_string()), ("tokenizer.json", "{}".to_string()), ("tokenizer_config.json", tok.clone())];
        let with = |fake: Fake, id: &str, cfg: &str| {
            let owned = files(cfg);
            let refs: Vec<(&str, &str)> = owned.iter().map(|(a, b)| (*a, b.as_str())).collect();
            fake.with(id, "qwen3", &refs, &["model.safetensors"])
        };
        let mut fake = with(Fake::default(), "o/Big-4bit", &four);
        fake = with(fake, "o/Big-3bit", &three);
        fake = with(fake, "o/Big-mxfp4", &mx);
        fake = with(fake, "o/Big-q4", &odd);
        fake = with(fake, "o/Bigger-4bit", &four);
        fake = with(fake, "x/Big-3bit", &three);
        let unlisted: Vec<Repo> = fake.repos.iter().filter(|r| r.id != "o/Big-4bit").cloned().collect();
        fake.unlisted = unlisted;
        let (catalog, stats) = run(&fake);
        let names: Vec<&str> = catalog.models.iter().map(|m| m.model_name.as_str()).collect();
        // The owner's other quantizations of the model, the float one too; not another model of a longer name, not
        // another owner's, not a mode the engine does not read.
        assert_eq!(names, ["o/Big-3bit", "o/Big-4bit", "o/Big-mxfp4"], "{stats:?}");
        assert_eq!(catalog.models[2].bits, Some(4));
        assert!(fake.asked.borrow().contains(&"search o Big".to_string()), "{:?}", fake.asked.borrow());
        assert_eq!(stats.rejected.get(&Reason::UnsupportedQuantization).copied(), Some(1));
    }

    #[test]
    fn a_gguf_catalog_is_not_read_as_an_mlx_one() {
        assert!(MlxCatalog::from_text("{\"format\":1,\"rules\":3,\"models\":[]}").models.is_empty());
        assert!(MlxCatalog::from_text("{\"format\":1,\"engine\":\"gguf\",\"rules\":1,\"models\":[]}").engine == ENGINE);
    }
}
