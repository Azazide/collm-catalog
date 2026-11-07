// SPDX-License-Identifier: AGPL-3.0-only
//! One sentence or two about a model, taken from the start of its README.
//!
//! No model writes it: the first paragraph that reads as a statement about a model, and is not a
//! badge, a heading, a table, a list, a code block, a bar of links or an invitation to read
//! something else. A README with no such paragraph gives nothing; the caller may then fall back on
//! [`base_model_note`], and failing that the app shows the developer instead.

const MAX_CHARS: usize = 280;
const MIN_CHARS: usize = 30;
const MIN_WEIGHT: usize = 8;
/// A slogan is a string of short sentences; a statement about a model opens with a longer one.
const MIN_FIRST_SENTENCE: usize = 5;

/// Openings of paragraphs that point elsewhere instead of saying what the model is.
const POINTS_ELSEWHERE: &[&str] = &[
    "see ",
    "read ",
    "learn ",
    "join ",
    "follow ",
    "click ",
    "check out",
    "visit ",
    "download",
    "sign up",
    "star ",
    "support ",
    "note:",
    "note that",
    "disclaimer",
    "warning",
    "important",
    "all uploads",
    "for more",
    "to run ",
    "to use ",
    "how to ",
    "this is a gguf",
    "gguf quant",
    "don't know",
    "grab ",
    "which quant",
    "which file",
    "the first thing to figure out",
    "find ",
    "you can ",
    "for the full",
    "for full",
    "built by",
    "for large files",
    "disable thinking",
    "lm studio",
    "ollama",
];

/// A paragraph that says nothing of the kind is a slogan, a news item or an advertisement.
const ABOUT_A_MODEL: &[&str] = &[
    "model",
    "llm",
    "fine-tun",
    "finetun",
    "trained",
    "training",
    "parameter",
    "instruct",
    "mixture-of",
    "reasoning",
    "assistant",
    "multimodal",
    "chat",
    "modèle",
    "модел",
    "模型",
    "モデル",
    "모델",
];

pub fn description(readme: &str) -> String {
    let text = readme.replace("\r\n", "\n");
    let text = strip_front_matter(&text);
    let text = strip_html_comments(text);
    for block in blocks(&text) {
        if let Some(sentence) = prose(&block) {
            return shorten(&sentence, MAX_CHARS);
        }
    }
    String::new()
}

fn strip_front_matter(text: &str) -> &str {
    let Some(rest) = text.strip_prefix("---\n") else {
        return text;
    };
    match rest.find("\n---") {
        Some(end) => {
            let after = &rest[end + 4..];
            after.strip_prefix('\n').unwrap_or(after)
        }
        None => text,
    }
}

fn strip_html_comments(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("<!--") {
        out.push_str(&rest[..start]);
        match rest[start..].find("-->") {
            Some(end) => rest = &rest[start + end + 3..],
            None => return out,
        }
    }
    out.push_str(rest);
    out
}

/// Runs of non-blank lines, with fenced code left out.
fn blocks(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current: Vec<&str> = Vec::new();
    let mut fence: Option<&str> = None;
    let flush = |current: &mut Vec<&str>, out: &mut Vec<String>| {
        if !current.is_empty() {
            out.push(current.join(" "));
            current.clear();
        }
    };
    for line in text.lines() {
        let trimmed = line.trim_start();
        if let Some(marker) = fence {
            if trimmed.starts_with(marker) {
                fence = None;
            }
            continue;
        }
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            fence = Some(&trimmed[..3]);
            flush(&mut current, &mut out);
            continue;
        }
        if line.trim().is_empty() {
            flush(&mut current, &mut out);
        } else {
            // A list item, a table row, a quote or a heading ends the paragraph above it even with
            // no blank line between them.
            if starts_like_structure(line) {
                flush(&mut current, &mut out);
            }
            current.push(line.trim());
        }
    }
    flush(&mut current, &mut out);
    out
}

fn starts_like_structure(block: &str) -> bool {
    let b = block.trim_start();
    let first = b.chars().next().unwrap_or(' ');
    if matches!(first, '#' | '|' | '>') {
        return true;
    }
    if b.starts_with("---") || b.starts_with("***") || b.starts_with("===") {
        return true;
    }
    if matches!(first, '-' | '*' | '+') && b[1..].starts_with(' ') {
        return true;
    }
    let digits = b.bytes().take_while(u8::is_ascii_digit).count();
    if digits > 0 && b[digits..].starts_with(". ") {
        return true;
    }
    let lower = b.to_ascii_lowercase();
    [
        "<table", "<tr", "<td", "<details", "<summary", "<pre", "<code", "<style", "<script",
    ]
    .iter()
    .any(|tag| lower.starts_with(tag))
}

/// Mostly capital letters: a notice or a plea, not a statement about a model.
fn is_shouting(text: &str) -> bool {
    let letters = text.chars().filter(|c| c.is_alphabetic()).count();
    let upper = text.chars().filter(|c| c.is_uppercase()).count();
    if letters >= 20 && upper * 2 > letters {
        return true;
    }
    // A headline in capitals over ordinary sentences: four of the first six words are all capitals.
    let capital_words = text
        .split_whitespace()
        .take(6)
        .filter(|word| {
            let letters: Vec<char> = word.chars().filter(|c| c.is_alphabetic()).collect();
            letters.len() >= 2 && letters.iter().all(|c| c.is_uppercase())
        })
        .count();
    capital_words >= 4
}

/// Words, with two characters of Chinese, Japanese or Korean counted as one.
fn weight_of(text: &str) -> usize {
    text.split_whitespace().count() + text.chars().filter(|c| is_cjk(*c)).count() / 2
}

/// Up to the first full stop that ends a sentence (not the one inside `Qwen3.8`).
fn first_sentence(text: &str) -> &str {
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        let ends = match c {
            '。' | '！' | '？' => true,
            '.' | '!' | '?' => chars.peek().is_none_or(|(_, next)| next.is_whitespace()),
            _ => false,
        };
        if ends {
            return &text[..i + c.len_utf8()];
        }
    }
    text
}

fn is_cjk(c: char) -> bool {
    matches!(c as u32, 0x3040..=0x30FF | 0x3400..=0x9FFF | 0xAC00..=0xD7AF)
}

/// The paragraph as plain text, when it reads as a description.
fn prose(block: &str) -> Option<String> {
    if starts_like_structure(block) {
        return None;
    }
    let text = clean(block);
    let text = text
        .trim_start_matches(|c: char| !c.is_alphanumeric())
        .to_string();
    let chars = text.chars().count();
    let weight = weight_of(&text);
    if chars < MIN_CHARS || weight < MIN_WEIGHT || text.ends_with(':') {
        return None;
    }
    if text.matches(['|', '·', '•']).count() >= 2 || is_shouting(&text) {
        return None;
    }
    if weight_of(first_sentence(&text)) < MIN_FIRST_SENTENCE {
        return None;
    }
    let lower = text.to_lowercase();
    if names_a_file_or_a_command_or_points_around(&text, &lower) {
        return None;
    }
    if POINTS_ELSEWHERE.iter().any(|p| lower.starts_with(p)) {
        return None;
    }
    if !ABOUT_A_MODEL.iter().any(|word| lower.contains(word)) {
        return None;
    }
    Some(text)
}

/// The openings of the sentences [`base_model_note`] writes, so a note can be told from a README's words.
const NOTE_PREFIXES: [&str; 4] = [
    "Quantized build of ",
    "Fine-tune of ",
    "Merge that includes ",
    "Based on ",
];

/// Whether `text` is one of the sentences [`base_model_note`] writes.
pub fn is_note(text: &str) -> bool {
    NOTE_PREFIXES.iter().any(|prefix| text.starts_with(prefix))
}

/// A paragraph that opens with a file name (`Model-Q4.gguf — a conventional Q4`), carries a command
/// line option (`--chat-template-kwargs`), or points at something "below" or "above" says how to use
/// or find something, not what the model is.
fn names_a_file_or_a_command_or_points_around(text: &str, lower: &str) -> bool {
    if text.split_whitespace().next().is_some_and(|first| {
        first
            .to_lowercase()
            .trim_end_matches([',', ':', ';'])
            .ends_with(".gguf")
    }) {
        return true;
    }
    if text.match_indices(" --").any(|(at, _)| {
        text[at + 3..]
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic())
    }) {
        return true;
    }
    lower
        .split(|c: char| !c.is_alphanumeric())
        .any(|word| word == "below" || word == "above")
}

/// What the tags say a repository was made from: `base_model:quantized:owner/name`.
pub fn base_model_note(tags: &[String]) -> Option<String> {
    let of = |prefix: &str| {
        tags.iter()
            .find_map(|t| t.strip_prefix(prefix).filter(|rest| rest.contains('/')))
    };
    if let Some(base) = of("base_model:quantized:") {
        return Some(format!("{}{base}.", NOTE_PREFIXES[0]));
    }
    if let Some(base) = of("base_model:finetune:") {
        return Some(format!("{}{base}.", NOTE_PREFIXES[1]));
    }
    if let Some(base) = of("base_model:merge:") {
        return Some(format!("{}{base}.", NOTE_PREFIXES[2]));
    }
    // A plain `base_model:owner/name`, but not one of the typed forms with a second colon.
    tags.iter()
        .find_map(|t| {
            t.strip_prefix("base_model:")
                .filter(|rest| rest.contains('/') && !rest.contains(':'))
        })
        .map(|base| format!("{}{base}.", NOTE_PREFIXES[3]))
}

/// Links give their text, images and tags vanish, emphasis marks go, entities become characters.
fn clean(block: &str) -> String {
    let linked = strip_links(block);
    let untagged = strip_tags(&linked);
    let text = untagged
        .replace("**", "")
        .replace("__", "")
        .replace(['`', '*'], "");
    let text = decode_entities(&text);
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// `&amp;`, `&mdash;`, `&#8212;` and `&#x2014;` become the character they name; an entity that is not
/// known stays as written.
fn decode_entities(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find('&') {
        out.push_str(&rest[..start]);
        let tail = &rest[start..];
        let decoded = tail.find(';').filter(|end| *end <= 10).and_then(|end| {
            let name = &tail[1..end];
            let c = match name {
                "nbsp" => Some(' '),
                "amp" => Some('&'),
                "lt" => Some('<'),
                "gt" => Some('>'),
                "quot" | "ldquo" | "rdquo" => Some('"'),
                "apos" | "lsquo" | "rsquo" => Some('\''),
                "mdash" => Some('—'),
                "ndash" => Some('–'),
                "middot" => Some('·'),
                "bull" => Some('•'),
                "hellip" => Some('…'),
                "copy" => Some('©'),
                "reg" => Some('®'),
                _ => name.strip_prefix('#').and_then(|num| {
                    let code = match num.strip_prefix(['x', 'X']) {
                        Some(hex) => u32::from_str_radix(hex, 16).ok(),
                        None => num.parse().ok(),
                    };
                    code.and_then(char::from_u32)
                }),
            };
            c.map(|c| (c, end))
        });
        match decoded {
            Some((c, end)) => {
                out.push(c);
                rest = &tail[end + 1..];
            }
            None => {
                out.push('&');
                rest = &tail[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

fn strip_tags(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        let opens_tag = chars[i] == '<'
            && chars
                .get(i + 1)
                .is_some_and(|c| c.is_ascii_alphabetic() || *c == '/' || *c == '!');
        if opens_tag {
            match chars[i..].iter().position(|c| *c == '>') {
                Some(close) => {
                    out.push(' ');
                    i += close + 1;
                    continue;
                }
                None => break,
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// `[text](address)` becomes `text`; `![alt](address)` becomes nothing.
fn strip_links(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '!'
            && chars.get(i + 1) == Some(&'[')
            && let Some((_, end)) = link_at(&chars, i + 1)
        {
            i = end;
            continue;
        }
        if chars[i] == '['
            && let Some((label, end)) = link_at(&chars, i)
        {
            out.push_str(&strip_links(&label));
            i = end;
            continue;
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// The label and the end of a `[label](address)` that starts at `start`.
fn link_at(chars: &[char], start: usize) -> Option<(String, usize)> {
    let mut depth = 0usize;
    let mut close = None;
    for (offset, c) in chars[start..].iter().enumerate() {
        match c {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    close = Some(start + offset);
                    break;
                }
            }
            _ => {}
        }
    }
    let close = close?;
    if chars.get(close + 1) != Some(&'(') {
        return None;
    }
    let end = chars[close + 2..].iter().position(|c| *c == ')')? + close + 2;
    Some((chars[start + 1..close].iter().collect(), end + 1))
}

/// At most `max` characters, cut where a sentence ends or else at a word.
pub fn shorten(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let cut: String = text.chars().take(max).collect();
    let sentence_end = cut
        .char_indices()
        .filter(|(i, c)| matches!(c, '.' | '!' | '?') && cut[i + 1..].starts_with(' '))
        .map(|(i, _)| i + 1)
        .next_back();
    if let Some(end) = sentence_end
        && cut[..end].chars().count() >= 80
    {
        return cut[..end].to_string();
    }
    match cut.rfind(' ') {
        Some(space) if space > 0 => {
            format!("{}…", cut[..space].trim_end_matches([',', ';', ':', '-']))
        }
        _ => format!("{cut}…"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> String {
        let path = format!(
            "{}/tests/fixtures/{name}/readme.md",
            env!("CARGO_MANIFEST_DIR")
        );
        std::fs::read_to_string(path).unwrap()
    }

    #[test]
    fn the_first_paragraph_of_prose_is_the_description() {
        let readme = "---\nlicense: mit\n---\n# Title\n\n![badge](https://x/y.svg)\n\n\
            A compact instruction model that follows long prompts and writes clean code in many languages.\n\n\
            More text that is not wanted.";
        assert_eq!(
            description(readme),
            "A compact instruction model that follows long prompts and writes clean code in many languages."
        );
    }

    #[test]
    fn badges_links_html_and_emphasis_are_reduced_to_their_words() {
        let readme = "[![Chat](https://img/b.svg)](https://chat) <p align=\"center\"></p>\n\n\
            <p>This **model** is a [fine-tune](https://hf.co/x) of <code>Base-7B</code> &amp; keeps \
            its `tool calling` skills for agents.</p>";
        assert_eq!(
            description(readme),
            "This model is a fine-tune of Base-7B & keeps its tool calling skills for agents."
        );
    }

    #[test]
    fn tables_lists_quotes_code_and_comments_are_not_prose() {
        let readme = "<!-- A long hidden comment that says many different words about nothing at all today. -->\n\n\
            | Quant | Size |\n|---|---|\n| Q4 | 4 GB |\n\n\
            - first thing in a list that goes on for quite a few words in a row here\n\n\
            1. a numbered step that also goes on for quite a few words in a row here\n\n\
            > a quoted line that also goes on for quite a few words in a row here for sure\n\n\
            ```\nprint('this code is long enough to look like prose if it were not fenced off')\n```\n\n\
            ## Heading with many words in it that would pass for a sentence if it were allowed\n\n\
            A reasoning model tuned for mathematics and logic puzzles, released under a permissive licence.";
        assert_eq!(
            description(readme),
            "A reasoning model tuned for mathematics and logic puzzles, released under a permissive licence."
        );
    }

    #[test]
    fn paragraphs_that_point_elsewhere_or_introduce_something_are_passed_over() {
        let readme = "See the guide for how to run this model locally with all the right settings.\n\n\
            Here are the quantized files that are available in this repository today:\n\n\
            Read our blog post for the full story behind the release and its training recipe.\n\n\
            A multilingual chat model with a long context window and strong coding ability.";
        assert_eq!(
            description(readme),
            "A multilingual chat model with a long context window and strong coding ability."
        );
    }

    #[test]
    fn a_readme_without_prose_gives_nothing() {
        assert_eq!(description(""), "");
        assert_eq!(description("# Only\n\n![x](y)\n\n| a | b |\n"), "");
        assert_eq!(
            description("---\nlicense: mit\n"),
            "",
            "an unclosed header is not a description either"
        );
        assert_eq!(description("too short"), "");
    }

    #[test]
    fn a_paragraph_in_chinese_counts_by_its_characters() {
        let readme = "# 模型\n\n这是一个面向对话和代码生成的大型语言模型，支持很长的上下文窗口并且速度很快。";
        assert!(description(readme).starts_with("这是一个面向对话"));
    }

    #[test]
    fn a_long_paragraph_is_cut_at_a_sentence_and_never_inside_a_character() {
        let sentence = "This model was trained on a large corpus of text and code collected from many sources. ";
        let long = sentence.repeat(6);
        let out = shorten(long.trim(), 280);
        assert!(out.ends_with('.'), "{out}");
        assert!(out.chars().count() <= 280);

        let no_stops = "слово ".repeat(80);
        let cut = shorten(no_stops.trim(), 100);
        assert!(cut.ends_with('…') && cut.chars().count() <= 101, "{cut}");
        assert!(
            cut.trim_end_matches('…').ends_with("слово"),
            "cut at a word: {cut}"
        );

        let han = "字".repeat(400);
        let cut = shorten(&han, 280);
        assert_eq!(cut.chars().count(), 281, "280 characters and the ellipsis");
    }

    #[test]
    fn a_short_paragraph_is_kept_whole() {
        assert_eq!(shorten("Short and sweet.", 280), "Short and sweet.");
    }

    #[test]
    fn a_bar_of_links_a_slogan_and_an_advertisement_are_not_descriptions() {
        let readme = "Hugging Face | GitHub | Launch Blog | Documentation License : Apache 2.0 | Authors : Google DeepMind\n\n\
            \u{1F44B} Join our community and chat with the people who build this model every day of the week.\n\n\
            One Gateway. Every Provider. &mdash; Route Smarter &middot; Ship Faster &middot; Spend Less today.\n\n\
            Analysis of the best providers: our dynamic quants deliver a better accuracy at the same size as the others.\n\n\
            Gemma is a family of open models from a large lab, built from the same research as its flagship assistant.";
        assert_eq!(
            description(readme),
            "Gemma is a family of open models from a large lab, built from the same research as its flagship assistant."
        );
    }

    #[test]
    fn a_notice_in_capitals_and_a_bar_of_bullet_separated_links_are_not_descriptions() {
        let readme = "I HAVE REACHED HUGGING FACE'S FREE STORAGE LIMIT, SO THIS MODEL IS MOVING SOON.\n\n\
            Website &bull; Learn more &bull; BF16 model &bull; Discord for questions about the model.\n\n\
            A small multilingual model that translates between thirty languages with a short context.";
        assert_eq!(
            description(readme),
            "A small multilingual model that translates between thirty languages with a short context."
        );
        assert!(!is_shouting(
            "A model with a few CAPS in an otherwise ordinary sentence of words."
        ));
        assert!(!is_shouting("SHORT"), "too little to judge");
        let headline_over_prose = "I HAVE REACHED HUGGING FACE'S FREE STORAGE LIMIT \u{26A0}\u{1F6A8} I can no longer upload new \
            models unless I can cover the cost of additional storage. I host many free models as an independent contributor.";
        assert!(is_shouting(headline_over_prose));
        assert!(!is_shouting(
            "The NVIDIA Nemotron model is a small language model post-trained by the team."
        ));
        assert_eq!(
            description(headline_over_prose),
            "",
            "nothing else in that README to describe"
        );
    }

    #[test]
    fn advice_on_which_file_to_download_is_not_a_description_of_the_model() {
        let readme = "Don't know which to choose? Grab the middle one: usually a good mix of size and quality for this model.\n\n\
            The first thing to figure out is how big a model you can run on the memory that you have.\n\n\
            A distilled reasoning model for maths and code that runs well on a laptop with a small context.";
        assert_eq!(
            description(readme),
            "A distilled reasoning model for maths and code that runs well on a laptop with a small context."
        );
    }

    #[test]
    fn instructions_credits_and_pointers_seen_in_real_readmes_are_not_descriptions() {
        let cases = [
            "Find architecture and usage details in the original model card.",
            "Find more details in the original model card: https://huggingface.co/org/model for this model.",
            "You can either specify a new local-dir or download them using the command line interface.",
            "For the full capability, training, and evaluation writeup, see the base model card of this model.",
            "Built by the team behind Atomic Chat, the app that runs open-weight models locally and offline.",
            "For large files, we split them into multiple segments due to the limitation of file upload here.",
            "Disable thinking via the chat template option when you want a fast model answer without a long trace.",
            "LM Studio and Ollama load any text quant of this model; for images keep the projector beside it.",
            "Model-Q4_K_M.gguf is a conventional quantization of this model where most tensors are Q4_K.",
            "Run the model with --chat-template-kwargs set so that the chat model answers without thinking.",
            "The sizes below are the live model weight sizes of the files in this repository at the head.",
            "The table above lists every file of this model and the quality that each of them keeps.",
        ];
        for case in cases {
            assert_eq!(
                description(&format!(
                    "{case}\n\nA small model that is made for careful answers to maths questions."
                )),
                "A small model that is made for careful answers to maths questions.",
                "{case}"
            );
        }
    }

    #[test]
    fn a_real_description_that_merely_touches_those_words_is_kept() {
        for ok in [
            "A reasoning model that finds bugs in code and explains the cause of each in plain words.",
            "Fine-tuned from a base model by training further on a curated set of long documents for chat.",
            "An instruction model with a long context that keeps the answer of every question short and exact.",
        ] {
            assert_eq!(description(ok), ok);
        }
    }

    #[test]
    fn a_string_of_short_sentences_is_a_slogan_even_when_it_says_model() {
        let slogan = "One Gateway. Every Model. Route Smarter and ship safer while you spend a lot less.\n\n\
            A routing layer that sends each request to the model that suits it best and the price you set.";
        assert_eq!(
            description(slogan),
            "A routing layer that sends each request to the model that suits it best and the price you set."
        );
        assert_eq!(
            first_sentence("Qwen3.8 is new. It is good."),
            "Qwen3.8 is new."
        );
        assert_eq!(first_sentence("no stop at all"), "no stop at all");
        assert_eq!(first_sentence("模型很好。第二句"), "模型很好。");
    }

    #[test]
    fn a_paragraph_is_not_glued_to_the_list_that_follows_it_without_a_blank_line() {
        let readme = "A compact instruction model for chat and coding with a long context window.\n\
            - Type: Causal Language Models\n- Parameters: 4B\n\nMore text that is not wanted here at all.";
        assert_eq!(
            description(readme),
            "A compact instruction model for chat and coding with a long context window."
        );
        let introduces = "This repository holds the instruction-tuned model, which has these features:\n\
            - Type: Causal Language Models\n- Parameters: 4B\n\n\
            A second paragraph that says the model is good at long documents and code.";
        assert_eq!(
            description(introduces),
            "A second paragraph that says the model is good at long documents and code.",
            "a sentence that only introduces a list is not a description"
        );
    }

    #[test]
    fn a_note_is_told_from_a_readme_sentence() {
        for tags in [
            "base_model:quantized:o/b",
            "base_model:finetune:o/b",
            "base_model:merge:o/b",
            "base_model:o/b",
        ] {
            let note = base_model_note(&[tags.to_string()]).unwrap();
            assert!(is_note(&note), "{note}");
        }
        assert!(!is_note("A compact model."));
    }

    #[test]
    fn a_leading_emoji_is_dropped_so_the_opening_is_judged_by_its_words() {
        let readme = "\u{1F44B} Join our community to talk about the model and share what you have built with it.\n\n\
            \u{1F4A1} A small model that answers in the language it is asked in, with a long context and tool calling.";
        assert_eq!(
            description(readme),
            "A small model that answers in the language it is asked in, with a long context and tool calling."
        );
    }

    #[test]
    fn entities_become_their_characters_and_unknown_ones_stay() {
        assert_eq!(
            decode_entities("a &mdash; b &amp; c &#8212; d &#x2014; e &lt;f&gt;"),
            "a — b & c — d — e <f>"
        );
        assert_eq!(
            decode_entities("R&D and &unknown; &#99999999; &"),
            "R&D and &unknown; &#99999999; &"
        );
        assert_eq!(decode_entities("tail &amp"), "tail &amp");
    }

    #[test]
    fn the_base_model_is_named_from_the_tags_the_typed_ones_first() {
        let tags = |list: &[&str]| list.iter().map(|t| t.to_string()).collect::<Vec<_>>();
        assert_eq!(
            base_model_note(&tags(&[
                "gguf",
                "base_model:org/Base-7B",
                "base_model:quantized:org/Base-7B"
            ]))
            .as_deref(),
            Some("Quantized build of org/Base-7B.")
        );
        assert_eq!(
            base_model_note(&tags(&["base_model:finetune:org/Base-7B"])).as_deref(),
            Some("Fine-tune of org/Base-7B.")
        );
        assert_eq!(
            base_model_note(&tags(&["base_model:merge:org/Base-7B"])).as_deref(),
            Some("Merge that includes org/Base-7B.")
        );
        assert_eq!(
            base_model_note(&tags(&["base_model:org/Base-7B"])).as_deref(),
            Some("Based on org/Base-7B.")
        );
        assert_eq!(
            base_model_note(&tags(&["base_model:adapter:org/Base-7B"])),
            None,
            "an adapter is not a base"
        );
        assert_eq!(base_model_note(&tags(&["gguf", "conversational"])), None);
        assert_eq!(
            base_model_note(&tags(&["base_model:quantized:no-owner"])),
            None
        );
    }

    #[test]
    fn real_readmes_give_a_statement_about_a_model_or_nothing() {
        for name in ["gemma", "glm", "gated", "qwen"] {
            let text = description(&fixture(name));
            eprintln!("{name}: {text}");
            assert!(
                !text.contains("](")
                    && !text.contains('<')
                    && !text.contains('|')
                    && !text.contains("&mdash;"),
                "{name}: {text}"
            );
            assert!(text.chars().count() <= 281, "{name}");
            let lower = text.to_lowercase();
            assert!(
                text.is_empty() || ABOUT_A_MODEL.iter().any(|w| lower.contains(w)),
                "{name}: {text}"
            );
            assert!(
                !lower.starts_with("join")
                    && !lower.starts_with("hugging face")
                    && !lower.starts_with("analysis")
                    && !lower.starts_with("one gateway"),
                "{name}: {text}"
            );
        }
    }
}
