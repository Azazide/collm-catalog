// SPDX-License-Identifier: AGPL-3.0-only
//! The run: walk Hugging Face's list by downloads, keep what is a chat model with a whole weights
//! file, and carry over what has not changed since the last catalog.
//!
//! A repository whose commit is the one the last catalog was built from costs no request at all
//! beyond the list; only new and changed ones are read in full. A failure on one repository keeps
//! its old entry (or leaves it out); a failure on the list itself ends the run, so a half-read list
//! never becomes a catalog.

use crate::classify::{self, Signals};
use crate::describe;
use crate::files;
use crate::hf::{Hf, HfError, Repo};
use crate::model::{Catalog, FORMAT, Model, RULES};
use crate::select::{self, Curated, Reason};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

#[derive(Debug, Default, Clone)]
pub struct Stats {
    pub listed: usize,
    /// List entries that could not be read at all.
    pub unreadable: usize,
    /// Repositories that were looked into, that is, that got past the list's own filters.
    pub examined: usize,
    pub reused: usize,
    pub fetched: usize,
    pub kept_after_error: usize,
    pub failed: usize,
    pub rejected: BTreeMap<Reason, usize>,
}

enum Outcome {
    Reused(Model),
    /// `note` is what the tags say the model was made from, kept in case its README's paragraph turns
    /// out to be a blurb the uploader pastes into every repository.
    Fetched {
        model: Model,
        note: Option<String>,
    },
    Rejected(Reason),
}

/// A paragraph that opens the README of this many repositories is a blurb, not a statement about a
/// model.
const SHARED_BLURB: usize = 3;

pub fn build(
    hf: &dyn Hf,
    curated: &Curated,
    previous: &Catalog,
    log: &mut dyn FnMut(String),
) -> Result<(Catalog, Stats), HfError> {
    let before: HashMap<String, &Model> = previous
        .models
        .iter()
        .map(|m| (m.repo_id().to_lowercase(), m))
        .collect();
    let mut stats = Stats::default();
    let mut seen: HashSet<String> = HashSet::new();
    let mut models: Vec<Model> = Vec::new();
    let mut queue: VecDeque<Repo> = VecDeque::new();
    let mut notes: HashMap<String, String> = HashMap::new();
    // Pinned repositories are on top of the limits: `max_models` and `max_candidates` are about what
    // the list gives, so pinning a model never costs the catalog one that is not pinned.
    let pinned: HashSet<String> = curated.pinned.iter().map(|id| id.to_lowercase()).collect();
    let mut taken = 0usize;
    let mut looked_into = 0usize;

    for id in &curated.pinned {
        match hf.repo(id) {
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

    let mut cursor: Option<String> = None;
    let mut pages = 0;
    let mut list_done = false;
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
                let page = hf.list_page(cursor.as_deref())?;
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
        if let Err(reason) = select::prefilter(curated, &repo) {
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
                let place = if is_pinned {
                    "pinned".to_string()
                } else {
                    format!("{taken}/{}", curated.max_models)
                };
                log(format!(
                    "[{place}] read {}: {} files, {} projectors",
                    repo.id,
                    model.quants.len(),
                    model.mmproj_models.len()
                ));
                models.push(model);
            }
            Ok(Outcome::Rejected(reason)) => {
                *stats.rejected.entry(reason).or_default() += 1;
            }
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

    drop_shared_blurbs(&mut models, &notes);
    models.sort_by_key(|m| m.model_name.to_lowercase());
    Ok((
        Catalog {
            format: FORMAT,
            rules: RULES,
            models,
        },
        stats,
    ))
}

fn process(
    hf: &dyn Hf,
    curated: &Curated,
    repo: &Repo,
    old: Option<&Model>,
) -> Result<Outcome, HfError> {
    let Some(listed_sha) = repo.sha.as_deref() else {
        return Ok(Outcome::Rejected(Reason::NoCommit));
    };
    if let Some(old) = old.filter(|o| o.sha == listed_sha) {
        if select::architecture_excluded(curated, &old.architecture) {
            return Ok(Outcome::Rejected(Reason::ExcludedArchitecture));
        }
        let mut model = old.clone();
        model.downloads = repo.downloads;
        model.likes = repo.likes;
        return Ok(Outcome::Reused(model));
    }

    // The list usually carries the GGUF metadata; a repository fetched on its own (a pinned one)
    // always does. Where the list lacks the chat template the repository is asked for it.
    let listed = repo.gguf.clone().unwrap_or_default();
    let listed_projector = listed
        .architecture
        .as_deref()
        .is_some_and(select::is_projector_architecture);
    let detail = if listed.chat_template.is_some() || listed_projector {
        repo.clone()
    } else {
        hf.repo(&repo.id)?
    };
    let gguf = detail.gguf.clone().unwrap_or_default();
    // Hugging Face read a vision projector, not the weights: the metadata is not the model's, so
    // nothing is taken from it, and only the pipeline label can say that this is a chat model.
    let projector = gguf
        .architecture
        .as_deref()
        .is_some_and(select::is_projector_architecture);
    let (template, architecture, context_length) = if projector {
        if detail.pipeline_tag.is_none() {
            return Ok(Outcome::Rejected(Reason::NoChatTemplate));
        }
        (String::new(), String::new(), None)
    } else {
        let template = gguf.chat_template.clone().unwrap_or_default();
        if template.trim().is_empty() {
            return Ok(Outcome::Rejected(Reason::NoChatTemplate));
        }
        let architecture = gguf.architecture.clone().unwrap_or_default();
        if select::architecture_excluded(curated, &architecture) {
            return Ok(Outcome::Rejected(Reason::ExcludedArchitecture));
        }
        (template, architecture, gguf.context_length)
    };
    // Unknown, not false, when the metadata was a projector's.
    let tools = (!projector).then(|| classify::template_takes_tools(&template));
    let rev = detail.sha.clone().unwrap_or_else(|| listed_sha.to_string());
    let Some((developer, _)) = select::split_id(&detail.id) else {
        return Ok(Outcome::Rejected(Reason::ExcludedName));
    };

    let tree = hf.tree(&detail.id, &rev)?;
    let found = files::classify(&detail.id, &rev, &tree);
    if found.quants.is_empty() {
        let reason = if found.drafts.is_empty() {
            Reason::NoWholeFile
        } else {
            Reason::OnlyDrafts
        };
        return Ok(Outcome::Rejected(reason));
    }
    let readme = hf.readme(&detail.id, &rev)?;
    let note = describe::base_model_note(&detail.tags);
    let description = Some(
        readme
            .as_deref()
            .map(describe::description)
            .unwrap_or_default(),
    )
    .filter(|text| !text.is_empty())
    .or_else(|| note.clone())
    .unwrap_or_default();
    Ok(Outcome::Fetched {
        model: Model {
            model_name: detail.id.clone(),
            developer: developer.to_string(),
            sha: rev.clone(),
            description,
            downloads: detail.downloads,
            likes: detail.likes,
            created_at: detail.created_at.clone(),
            updated_at: detail.last_modified.clone(),
            license: detail
                .card_data
                .as_ref()
                .and_then(|c| c.license())
                .and_then(|l| describe::label(&l)),
            pipeline_tag: detail.pipeline_tag.as_deref().and_then(describe::label),
            architecture,
            context_length,
            capabilities: classify::capabilities(&Signals {
                id: &detail.id,
                tags: &detail.tags,
                tools,
                has_projector: !found.mmproj.is_empty(),
                readme: readme.as_deref(),
            }),
            readme: readme
                .map(|_| {
                    format!(
                        "https://huggingface.co/{}/resolve/{rev}/README.md",
                        detail.id
                    )
                })
                .unwrap_or_default(),
            quants: found.quants,
            drafts: found.drafts,
            mmproj_models: found.mmproj,
        },
        note,
    })
}

/// A description that three or more models share is replaced by what the model was made from, or by
/// nothing. A note is never counted: many quantizations of one model rightly say the same.
fn drop_shared_blurbs(models: &mut [Model], notes: &HashMap<String, String>) {
    let counted = |m: &Model| !m.description.is_empty() && !describe::is_note(&m.description);
    let mut count: HashMap<String, usize> = HashMap::new();
    for model in models.iter().filter(|m| counted(m)) {
        *count.entry(model.description.clone()).or_default() += 1;
    }
    for model in models.iter_mut().filter(|m| counted(m)) {
        if count[&model.description] >= SHARED_BLURB {
            model.description = notes
                .get(&model.repo_id().to_lowercase())
                .cloned()
                .unwrap_or_default();
        }
    }
}
