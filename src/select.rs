// SPDX-License-Identifier: AGPL-3.0-only
//! Which repositories the catalog considers, from what the list already says about them.
//!
//! The choices a person makes live in `curated.toml`; what makes a repository a chat model at all
//! is here.

use crate::hf::Repo;
use serde::Deserialize;

/// What `curated.toml` holds. An unknown key is an error: a misspelt filter must not pass silently.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Curated {
    /// The catalog holds at most this many models.
    #[serde(default = "default_max_models")]
    pub max_models: usize,
    /// Repositories that are looked into in one run, at most; each costs requests.
    #[serde(default = "default_max_candidates")]
    pub max_candidates: usize,
    #[serde(default = "default_max_list_pages")]
    pub max_list_pages: usize,
    /// Listed first, whatever their downloads (`owner/name`).
    #[serde(default)]
    pub pinned: Vec<String>,
    /// Owners whose repositories are never listed.
    #[serde(default)]
    pub exclude_developers: Vec<String>,
    /// Repositories whose name holds one of these (any case) are never listed.
    #[serde(default)]
    pub exclude_name_contains: Vec<String>,
    /// GGUF architectures that are not language models.
    #[serde(default)]
    pub exclude_architectures: Vec<String>,
}

fn default_max_models() -> usize {
    300
}
fn default_max_candidates() -> usize {
    900
}
fn default_max_list_pages() -> usize {
    5
}

impl Default for Curated {
    fn default() -> Self {
        Curated {
            max_models: default_max_models(),
            max_candidates: default_max_candidates(),
            max_list_pages: default_max_list_pages(),
            pinned: Vec::new(),
            exclude_developers: Vec::new(),
            exclude_name_contains: Vec::new(),
            exclude_architectures: Vec::new(),
        }
    }
}

impl Curated {
    pub fn parse(text: &str) -> Result<Curated, String> {
        toml::from_str(text).map_err(|e| format!("curated.toml: {e}"))
    }
}

/// Hugging Face's own label for what a model does. A repository with no label is let through and
/// judged by its chat template; one with any of these is a language model.
const CHAT_PIPELINES: &[&str] = &[
    "text-generation",
    "image-text-to-text",
    "any-to-any",
    "text2text-generation",
    "visual-question-answering",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Reason {
    Gated,
    Disabled,
    Private,
    NotAChatPipeline,
    ExcludedName,
    ExcludedDeveloper,
    ExcludedArchitecture,
    NoCommit,
    NoChatTemplate,
    NoWholeFile,
    OnlyDrafts,
}

impl Reason {
    pub fn text(self) -> &'static str {
        match self {
            Reason::Gated => "the files are behind an access form",
            Reason::Disabled => "the repository is disabled",
            Reason::Private => "the repository is private",
            Reason::NotAChatPipeline => {
                "Hugging Face labels it as something other than a chat model"
            }
            Reason::ExcludedName => "its name is on the exclusion list",
            Reason::ExcludedDeveloper => "its owner is on the exclusion list",
            Reason::ExcludedArchitecture => "its architecture is on the exclusion list",
            Reason::NoCommit => "no commit is named",
            Reason::NoChatTemplate => {
                "nothing shows it is a chat model (no chat template, no chat label)"
            }
            Reason::NoWholeFile => "no weights file is whole (all are cut into parts)",
            Reason::OnlyDrafts => "the only whole files are drafts for speculative decoding",
        }
    }
}

fn developer_of(id: &str) -> Option<&str> {
    let (owner, name) = id.split_once('/')?;
    (!owner.is_empty() && !name.is_empty() && !name.contains('/')).then_some(owner)
}

/// What the list entry alone rules out.
pub fn prefilter(curated: &Curated, repo: &Repo) -> Result<(), Reason> {
    let Some(owner) = developer_of(&repo.id) else {
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
    if let Some(tag) = repo.pipeline_tag.as_deref()
        && !CHAT_PIPELINES.contains(&tag)
    {
        return Err(Reason::NotAChatPipeline);
    }
    if curated
        .exclude_developers
        .iter()
        .any(|d| d.eq_ignore_ascii_case(owner))
    {
        return Err(Reason::ExcludedDeveloper);
    }
    let name = repo.id.to_lowercase();
    if curated
        .exclude_name_contains
        .iter()
        .any(|part| name.contains(&part.to_lowercase()))
    {
        return Err(Reason::ExcludedName);
    }
    if let Some(arch) = repo.gguf.as_ref().and_then(|g| g.architecture.as_deref())
        && architecture_excluded(curated, arch)
    {
        return Err(Reason::ExcludedArchitecture);
    }
    Ok(())
}

/// Hugging Face summarizes a repository by its first GGUF file in alphabetical order. When that file
/// is a vision projector (`mmproj-…`, architecture `clip`) the summary describes the projector, not
/// the model, and says nothing of its architecture, template or context.
pub fn is_projector_architecture(architecture: &str) -> bool {
    architecture.eq_ignore_ascii_case("clip")
}

/// Never true for a projector's architecture: that is not what the model is, whatever the list says.
pub fn architecture_excluded(curated: &Curated, architecture: &str) -> bool {
    !is_projector_architecture(architecture)
        && curated
            .exclude_architectures
            .iter()
            .any(|a| a.eq_ignore_ascii_case(architecture))
}

pub fn split_id(id: &str) -> Option<(&str, &str)> {
    developer_of(id)?;
    id.split_once('/')
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hf::{Gated, GgufInfo};

    fn repo(id: &str) -> Repo {
        Repo {
            id: id.into(),
            pipeline_tag: Some("text-generation".into()),
            ..Repo::default()
        }
    }

    #[test]
    fn a_plain_text_generation_repository_passes() {
        assert_eq!(
            prefilter(&Curated::default(), &repo("owner/model-GGUF")),
            Ok(())
        );
    }

    #[test]
    fn what_cannot_be_downloaded_is_out() {
        let c = Curated::default();
        let mut gated = repo("o/m");
        gated.gated = Gated::Mode("auto".into());
        assert_eq!(prefilter(&c, &gated), Err(Reason::Gated));
        let mut disabled = repo("o/m");
        disabled.disabled = true;
        assert_eq!(prefilter(&c, &disabled), Err(Reason::Disabled));
        let mut private = repo("o/m");
        private.private = true;
        assert_eq!(prefilter(&c, &private), Err(Reason::Private));
    }

    #[test]
    fn speech_image_and_other_labels_are_out_and_no_label_or_vision_is_in() {
        let c = Curated::default();
        for tag in [
            "text-to-speech",
            "automatic-speech-recognition",
            "text-to-image",
            "image-to-video",
            "feature-extraction",
        ] {
            let mut r = repo("o/m");
            r.pipeline_tag = Some(tag.into());
            assert_eq!(prefilter(&c, &r), Err(Reason::NotAChatPipeline), "{tag}");
        }
        for tag in ["image-text-to-text", "any-to-any"] {
            let mut r = repo("o/m");
            r.pipeline_tag = Some(tag.into());
            assert_eq!(prefilter(&c, &r), Ok(()), "{tag}");
        }
        let mut unlabelled = repo("o/m");
        unlabelled.pipeline_tag = None;
        assert_eq!(prefilter(&c, &unlabelled), Ok(()));
    }

    #[test]
    fn the_exclusion_lists_ignore_case() {
        let c = Curated::parse(
            "exclude_developers = [\"BadOwner\"]\nexclude_name_contains = [\"Embed\"]\nexclude_architectures = [\"DFlash\"]",
        )
        .unwrap();
        assert_eq!(
            prefilter(&c, &repo("badowner/m")),
            Err(Reason::ExcludedDeveloper)
        );
        assert_eq!(
            prefilter(&c, &repo("o/my-EMBED-model")),
            Err(Reason::ExcludedName)
        );
        let mut draft = repo("o/m");
        draft.gguf = Some(GgufInfo {
            architecture: Some("dflash".into()),
            ..GgufInfo::default()
        });
        assert_eq!(prefilter(&c, &draft), Err(Reason::ExcludedArchitecture));
        assert_eq!(prefilter(&c, &repo("o/m")), Ok(()));
    }

    #[test]
    fn a_projectors_architecture_is_never_a_reason_to_leave_a_model_out() {
        let c = Curated::parse("exclude_architectures = [\"clip\", \"dflash\"]").unwrap();
        assert!(
            !architecture_excluded(&c, "clip"),
            "even when the file lists it"
        );
        assert!(!architecture_excluded(&c, "CLIP"));
        assert!(architecture_excluded(&c, "dflash"));
        let mut clip = repo("o/m");
        clip.gguf = Some(GgufInfo {
            architecture: Some("clip".into()),
            ..GgufInfo::default()
        });
        assert_eq!(prefilter(&c, &clip), Ok(()));
    }

    #[test]
    fn an_id_that_is_not_owner_slash_name_is_refused() {
        let c = Curated::default();
        for id in ["gpt2", "/x", "x/", "a/b/c"] {
            assert!(prefilter(&c, &repo(id)).is_err(), "{id}");
        }
        assert_eq!(split_id("a/b"), Some(("a", "b")));
        assert_eq!(split_id("a/b/c"), None);
    }

    #[test]
    fn a_misspelt_key_in_curated_toml_is_an_error_and_an_empty_file_is_the_defaults() {
        assert!(Curated::parse("max_model = 5").is_err());
        let c = Curated::parse("").unwrap();
        assert_eq!(
            (c.max_models, c.max_candidates, c.max_list_pages),
            (300, 900, 5)
        );
        assert!(Curated::parse("max_models = 'many'").is_err());
    }
}
