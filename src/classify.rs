// SPDX-License-Identifier: AGPL-3.0-only
//! What a model can do, for the app to filter and sort by: `vision`, `audio`, `video`, `tools`, `code`,
//! `agent`.
//!
//! Every verdict comes from something checkable in the repository, never from a guess about quality:
//!
//! - `vision`: the repository has a vision projector (`mmproj-…`), the file llama.cpp needs to read
//!   an image. A label saying "image-text-to-text" without one does not count.
//! - `audio`, `video`: the model takes sound or moving pictures in. llama.cpp reads both through the
//!   same projector file as an image, so a repository with no projector has neither, whatever its
//!   labels say. With one, Hugging Face's own labels say which: the pipeline `any-to-any` or
//!   `audio-text-to-text`, a tag `audio` or `speech`, for audio; `video-text-to-text` or a tag `video`,
//!   for video; and `omni` (a whole word of its name, or of the model it was made from, or a tag) for
//!   both, those being the models that take every kind of input.
//! - `tools`: the chat template takes a list of tools. When the metadata Hugging Face gave was a
//!   projector's and the template is unknown, an explicit tag (`function-calling`, `tool-use`) stands in.
//! - `code`: made for programming, by its name (a whole word: `coder`, `code`, `devstral`, not
//!   `encoder`), its tags, or the name of the model it was made from.
//! - `agent`: its tags, its name or its README put it forward for agents and tool use, and its
//!   template, when known, takes tools. Most models take tools, so this is the narrower claim: the
//!   author says it is meant for agents.
//!
//! "Agent and coding" is both flags at once; "chat" is every model, so it has no flag.

pub const VISION: &str = "vision";
pub const AUDIO: &str = "audio";
pub const VIDEO: &str = "video";
pub const TOOLS: &str = "tools";
pub const CODE: &str = "code";
pub const AGENT: &str = "agent";

const CODE_TAGS: &[&str] = &[
    "code",
    "coding",
    "coder",
    "code-generation",
    "code-assistant",
    "programming",
];
const CODE_WORDS: &[&str] = &[
    "coder",
    "coding",
    "code",
    "codestral",
    "devstral",
    "starcoder",
    "starcoder2",
    "codellama",
    "codegemma",
    "opencoder",
    "codeqwen",
    "codegeex",
];
const TOOL_TAGS: &[&str] = &[
    "function-calling",
    "function_calling",
    "tool-use",
    "tool_use",
    "tool-calling",
    "tool_calling",
];
const AGENT_TAGS: &[&str] = &["agent", "agents", "agentic"];
const AUDIO_TAGS: &[&str] = &["audio", "speech", "audio-text-to-text", "omni"];
const AUDIO_PIPELINES: &[&str] = &["audio-text-to-text", "any-to-any"];
const AUDIO_WORDS: &[&str] = &["audio", "omni", "voxtral", "ultravox"];
const VIDEO_TAGS: &[&str] = &["video", "video-text-to-text", "omni"];
const VIDEO_PIPELINES: &[&str] = &["video-text-to-text"];
const VIDEO_WORDS: &[&str] = &["video", "omni"];
const AGENT_WORDS: &[&str] = &["agent", "agents", "agentic"];
/// Whether a chat template (Jinja) takes a list of tools: `tools` as a whole word inside a `{{ … }}` or
/// `{% … %}`, where a template uses it. The word in a comment (`{# … #}`) or in the plain text between
/// (a template that says it does not support tools) is no use of it.
pub fn template_takes_tools(template: &str) -> bool {
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        let after = &rest[open + 1..];
        let (close, from) = match after.as_bytes().first() {
            Some(b'{') => ("}}", 1),
            Some(b'%') => ("%}", 1),
            Some(b'#') => {
                // A comment: skipped whole.
                match after.find("#}") {
                    Some(end) => {
                        rest = &after[end + 2..];
                        continue;
                    }
                    None => return false,
                }
            }
            _ => {
                rest = after;
                continue;
            }
        };
        let body = &after[from..];
        let end = body.find(close).unwrap_or(body.len());
        if names_word(&body[..end], TOOLS) {
            return true;
        }
        rest = &body[end..];
    }
    false
}

/// Whether `word` stands in `text` as a whole word (not inside `tools_list` or `mytools`).
fn names_word(text: &str, word: &str) -> bool {
    text.match_indices(word).any(|(at, _)| {
        let before = text[..at].chars().next_back();
        let after = text[at + word.len()..].chars().next();
        let part = |c: Option<char>| c.is_some_and(|c| c.is_alphanumeric() || c == '_');
        !part(before) && !part(after)
    })
}

/// How often a README must speak of agents and tools before it counts as putting the model forward.
const README_MENTIONS: usize = 3;
const README_PHRASES: &[&str] = &["agent", "tool call", "function call", "tool use"];

pub struct Signals<'a> {
    /// `owner/name`.
    pub id: &'a str,
    pub tags: &'a [String],
    /// Hugging Face's label for what the repository does (`text-generation`, `any-to-any`, …).
    pub pipeline_tag: Option<&'a str>,
    /// Whether the chat template takes tools; `None` when the template is not known.
    pub tools: Option<bool>,
    pub has_projector: bool,
    pub readme: Option<&'a str>,
}

/// Whole words of a name: `Qwen3-Coder-30B` gives `qwen3`, `coder`, `30b`; `encoder` stays one word.
fn words(name: &str) -> Vec<String> {
    name.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect()
}

/// The names that count for a repository: its own and those of the models it was made from.
fn names(signals: &Signals) -> Vec<String> {
    let own = signals.id.rsplit('/').next().unwrap_or(signals.id);
    let mut names = words(own);
    for tag in signals.tags {
        if let Some(base) = tag.strip_prefix("base_model:") {
            names.extend(words(base.rsplit('/').next().unwrap_or(base)));
        }
    }
    names
}

fn has_tag(tags: &[String], wanted: &[&str]) -> bool {
    tags.iter()
        .any(|t| wanted.contains(&t.to_lowercase().as_str()))
}

fn mentions(readme: &str) -> usize {
    let lower = readme.to_lowercase();
    README_PHRASES
        .iter()
        .map(|phrase| lower.matches(phrase).count())
        .sum()
}

/// Sorted, without repeats, so the file does not change when nothing else did.
pub fn capabilities(signals: &Signals) -> Vec<String> {
    let names = names(signals);
    let mut found = Vec::new();

    if signals.has_projector {
        found.push(VISION);
        let pipeline = signals.pipeline_tag.map(str::to_lowercase);
        let in_pipelines = |list: &[&str]| pipeline.as_deref().is_some_and(|p| list.contains(&p));
        let in_names = |list: &[&str]| names.iter().any(|w| list.contains(&w.as_str()));
        if has_tag(signals.tags, AUDIO_TAGS)
            || in_pipelines(AUDIO_PIPELINES)
            || in_names(AUDIO_WORDS)
        {
            found.push(AUDIO);
        }
        if has_tag(signals.tags, VIDEO_TAGS)
            || in_pipelines(VIDEO_PIPELINES)
            || in_names(VIDEO_WORDS)
        {
            found.push(VIDEO);
        }
    }

    let tool_tag = has_tag(signals.tags, TOOL_TAGS);
    let tools = match signals.tools {
        Some(known) => known,
        None => tool_tag,
    };
    if tools {
        found.push(TOOLS);
    }

    if has_tag(signals.tags, CODE_TAGS) || names.iter().any(|w| CODE_WORDS.contains(&w.as_str())) {
        found.push(CODE);
    }

    let put_forward = has_tag(signals.tags, AGENT_TAGS)
        || tool_tag
        || names.iter().any(|w| AGENT_WORDS.contains(&w.as_str()))
        || signals
            .readme
            .is_some_and(|r| mentions(r) >= README_MENTIONS);
    if put_forward && signals.tools != Some(false) {
        found.push(AGENT);
    }

    found.sort_unstable();
    found.dedup();
    found.into_iter().map(str::to_string).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tags(list: &[&str]) -> Vec<String> {
        list.iter().map(|t| t.to_string()).collect()
    }

    fn of(
        id: &str,
        tags: &[String],
        tools: Option<bool>,
        projector: bool,
        readme: Option<&str>,
    ) -> Vec<String> {
        in_pipeline(id, tags, None, tools, projector, readme)
    }

    fn in_pipeline(
        id: &str,
        tags: &[String],
        pipeline_tag: Option<&str>,
        tools: Option<bool>,
        projector: bool,
        readme: Option<&str>,
    ) -> Vec<String> {
        capabilities(&Signals {
            id,
            tags,
            pipeline_tag,
            tools,
            has_projector: projector,
            readme,
        })
    }

    #[test]
    fn a_plain_chat_model_has_no_flag_at_all() {
        assert!(of("o/Llama-3-8B-Instruct-GGUF", &[], Some(false), false, None).is_empty());
    }

    #[test]
    fn a_projector_is_what_makes_a_model_see_and_a_label_alone_is_not() {
        assert_eq!(of("o/m", &[], Some(false), true, None), [VISION]);
        let labelled = tags(&["image-text-to-text", "vision-language"]);
        assert!(of("o/m", &labelled, Some(false), false, None).is_empty());
    }

    #[test]
    fn a_template_that_takes_tools_gives_tools_and_nothing_more() {
        assert_eq!(of("o/m", &[], Some(true), false, None), [TOOLS]);
    }

    #[test]
    fn code_is_read_from_whole_words_of_the_name_and_never_from_part_of_one() {
        assert_eq!(
            of(
                "o/Qwen3-Coder-30B-A3B-Instruct-GGUF",
                &[],
                Some(false),
                false,
                None
            ),
            [CODE]
        );
        assert_eq!(
            of("o/Devstral-Small-GGUF", &[], Some(false), false, None),
            [CODE]
        );
        assert_eq!(
            of("o/CodeLlama-7b-GGUF", &[], Some(false), false, None),
            [CODE]
        );
        assert_eq!(
            of("o/Opus-Code-Distill-GGUF", &[], Some(false), false, None),
            [CODE]
        );
        for not_code in [
            "o/bert-encoder-GGUF",
            "o/decoder-only-7b-GGUF",
            "o/Barcode-reader-GGUF",
            "o/Recode-GGUF",
        ] {
            assert!(
                of(not_code, &[], Some(false), false, None).is_empty(),
                "{not_code}"
            );
        }
    }

    #[test]
    fn code_is_also_read_from_the_tags_and_from_the_model_it_was_made_from() {
        assert_eq!(
            of("o/m", &tags(&["coding"]), Some(false), false, None),
            [CODE]
        );
        assert_eq!(
            of("o/m", &tags(&["Code-Generation"]), Some(false), false, None),
            [CODE]
        );
        let from_coder = tags(&[
            "gguf",
            "base_model:quantized:Qwen/Qwen2.5-Coder-7B-Instruct",
        ]);
        assert_eq!(
            of("o/neutral-name-GGUF", &from_coder, Some(false), false, None),
            [CODE]
        );
    }

    #[test]
    fn an_agent_is_one_put_forward_for_agents_by_tag_name_or_readme_and_able_to_take_tools() {
        assert_eq!(
            of("o/m", &tags(&["agentic"]), Some(true), false, None),
            [AGENT, TOOLS]
        );
        assert_eq!(
            of("o/Some-Agent-8B", &[], Some(true), false, None),
            [AGENT, TOOLS]
        );
        let three = "An agent model. It does tool calling well. Built for agents.";
        assert_eq!(
            of("o/m", &[], Some(true), false, Some(three)),
            [AGENT, TOOLS]
        );
        let two = "A chat model with tool calling, good for one agent task.";
        assert_eq!(
            of("o/m", &[], Some(true), false, Some(two)),
            [TOOLS],
            "two mentions are not enough"
        );
    }

    #[test]
    fn a_model_whose_template_takes_no_tools_is_no_agent_whatever_its_tags_say() {
        assert!(
            of(
                "o/m",
                &tags(&["agentic", "function-calling"]),
                Some(false),
                false,
                None
            )
            .is_empty()
        );
    }

    #[test]
    fn when_the_template_is_unknown_an_explicit_tag_stands_in_for_it() {
        assert_eq!(
            of("o/m", &tags(&["function-calling"]), None, true, None),
            [AGENT, TOOLS, VISION]
        );
        assert_eq!(
            of("o/m", &[], None, true, None),
            [VISION],
            "no tag, no claim"
        );
    }

    #[test]
    fn an_agent_that_codes_is_both_flags_and_the_list_is_sorted_and_free_of_repeats() {
        let got = of(
            "o/Agent-Coder-GGUF",
            &tags(&["agentic", "coding"]),
            Some(true),
            true,
            None,
        );
        assert_eq!(got, [AGENT, CODE, TOOLS, VISION]);
        let mut sorted = got.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(got, sorted);
    }

    #[test]
    fn a_template_takes_tools_only_where_it_uses_the_list() {
        for takes in [
            "{%- if tools %}\n{{ tools | tojson }}{%- endif %}",
            "{{- bos_token }}{%- if tools is defined and tools is not none %}x{% endif %}",
            "{% for tool in tools %}{{ tool.name }}{% endfor %}",
            "{%-if tools-%}",
        ] {
            assert!(template_takes_tools(takes), "{takes}");
        }
        for takes_none in [
            "",
            "{{ messages }} This model does not support tools.",
            "{# tools are not supported #}{{ messages }}",
            "{% set tools_list = [] %}{{ mytools }}{{ tools_x }}",
            "{{ messages }}{",
            "{{ messages }} tools",
            "{# never closed tools",
        ] {
            assert!(!template_takes_tools(takes_none), "{takes_none}");
        }
        // Past a comment and a plain block, a later use is still found.
        assert!(template_takes_tools(
            "{# tools #}{% set a = 1 %}text {%- if tools %}x{% endif %}"
        ));
    }

    // The repositories behind these cases are real ones of the catalog: Gemma 4 E2B (any-to-any with a
    // projector), LFM3-Audio (tags audio and speech), Qwen2.5-Omni (the word only in the base model's tag),
    // and a text model that is tagged `audio` with no projector.
    #[test]
    fn a_model_takes_sound_and_moving_pictures_in_only_with_a_projector_and_a_label_that_says_so() {
        let any = Some("any-to-any");
        assert_eq!(
            in_pipeline(
                "ggml-org/gemma-4-E2B-it-GGUF",
                &[],
                any,
                Some(true),
                true,
                None
            ),
            [AUDIO, TOOLS, VISION]
        );
        assert_eq!(
            in_pipeline(
                "l/LFM3-Audio-2B-GGUF",
                &tags(&["audio", "speech"]),
                None,
                Some(true),
                true,
                None
            ),
            [AUDIO, TOOLS, VISION]
        );
        let base = tags(&["base_model:quantized:Qwen/Qwen2.5-Omni-7B"]);
        assert_eq!(
            in_pipeline(
                "ggml-org/Qwen2.5-Omni-7B-GGUF",
                &base,
                any,
                Some(true),
                true,
                None
            ),
            [AUDIO, TOOLS, VIDEO, VISION],
            "omni takes every kind of input"
        );
        assert_eq!(
            in_pipeline(
                "o/Clips-GGUF",
                &[],
                Some("video-text-to-text"),
                Some(true),
                true,
                None
            ),
            [TOOLS, VIDEO, VISION]
        );
        // A label with no projector is not a way to take the input in.
        assert_eq!(
            in_pipeline(
                "o/Tagged-GGUF",
                &tags(&["audio", "video"]),
                any,
                Some(true),
                false,
                None
            ),
            [TOOLS]
        );
        // A projector with no label says only that the model sees.
        assert_eq!(
            in_pipeline(
                "o/Vision-GGUF",
                &[],
                Some("image-text-to-text"),
                Some(true),
                true,
                None
            ),
            [TOOLS, VISION]
        );
        // `audio` has to be a whole word of the name: a model about audiobooks is not one.
        assert_eq!(
            in_pipeline(
                "o/Audiobook-Writer-GGUF",
                &[],
                None,
                Some(false),
                true,
                None
            ),
            [VISION]
        );
        assert_eq!(
            in_pipeline("o/Voxtral-Mini-GGUF", &[], None, Some(false), true, None),
            [AUDIO, VISION]
        );
    }
}
