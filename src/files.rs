// SPDX-License-Identifier: AGPL-3.0-only
//! From a repository's file tree to the files the app can download.
//!
//! Only whole files are offered: a weights file cut into `-00001-of-00003` parts is skipped, as the
//! app downloads one file per model. Every address points at the commit the tree was read at.
//!
//! Files are told apart as the models, the vision projectors (`mmproj-…`) and the drafts for
//! speculative decoding, which pair with a model and are no model themselves.

use crate::hf::{TreeEntry, encode_path};
use crate::model::File;
use std::collections::HashMap;

pub struct Files {
    /// The weights files that run on their own, smallest first.
    pub quants: Vec<File>,
    /// The drafts for speculative decoding (`mtp`, `dspark`, `dflash`, `eagle3`), smallest first.
    pub drafts: Vec<File>,
    pub mmproj: Vec<File>,
}

/// The kinds of draft llama.cpp takes beside a model, as upstream names them.
const DRAFT_KINDS: [&str; 4] = ["mtp", "dspark", "dflash", "eagle3"];

/// A file this many times smaller than the repository's largest is a draft whatever else its name
/// says: a draft head is a sliver of a model (those measured were 0.001 to 0.04 of the largest file
/// beside the smallest real quant at 0.2).
const DRAFT_SIZE_RATIO: u64 = 10;

/// A file this small is no model: the smallest chat model of the catalog is 149 MB, and what is
/// smaller (a vocabulary, the data of a quantization, an adapter, a part of a speech model) is not
/// something to run.
const MIN_MODEL_BYTES: u64 = 50_000_000;

/// Words that name what a repository keeps beside its models: importance data, adapters, vocabularies,
/// tokenizers, the parts of a speech model. A file with one of them in its path is no model when it is
/// a sliver of the largest file (measured: 0.001 to 0.05 of it, against 0.1 and over for the models
/// that merely carry `imatrix` in their name).
const NOT_WEIGHTS_WORDS: [&str; 6] = ["imatrix", "lora", "vocab", "tokenizer", "tts", "token2wav"];

/// The words of a file name, lower case: `Qwen3-Coder-MTP-Q4_K_M` is qwen3, coder, mtp, q4, k, m.
fn words(name: &str) -> Vec<String> {
    name.to_lowercase()
        .split(['-', '_', '.'])
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect()
}

/// Whether a file of a repository is something that is not weights to run: see [`MIN_MODEL_BYTES`]
/// and [`NOT_WEIGHTS_WORDS`].
fn is_not_a_model(path: &str, size: u64, largest: u64) -> bool {
    if size < MIN_MODEL_BYTES {
        return true;
    }
    let sliver = size.saturating_mul(DRAFT_SIZE_RATIO) < largest;
    sliver
        && path
            .to_lowercase()
            .split(['-', '_', '.', '/'])
            .any(|word| NOT_WEIGHTS_WORDS.contains(&word))
}

/// A name by its words only, so `Model-Q8_0` and `model.q8-0` are one name.
fn normal(name: &str) -> String {
    words(name).join("-")
}

/// A draft named by its folder or its first word that weighs at least this share (one in this many) of
/// the file that is its name without the kind is a model with the head kept in it, not a draft: the
/// real ones measured 0.001 to 0.11 of their mate, the models of a folder named for the kind 1.03 to
/// 1.04.
const MODEL_OF_THE_KIND_SHARE: u64 = 2;

/// Whether a file is a draft, from its path without `.gguf`, its size, the largest file of the
/// repository and the sizes of its whole files by their normal names.
///
/// A draft is a file in a folder of the kind's name (`MTP/…`), or one that begins with it
/// (`mtp-…`, the way upstream names them), unless it weighs half of the file that is its name without
/// the kind or more: such a file (`mtp/Model-MTP-Q4_K_M` beside `Model-Q4_K_M`) is a whole model that
/// is only filed under the name. A name that holds the kind anywhere else, as `Model-MTP-Q4_K_M` or
/// `Model-Q8_0-MTP`, is a model made with the head kept in it: such a file weighs about what the same
/// model without the head weighs (measured, 1.0 to 1.07 times, in 132 files of 40 repositories), and
/// none of them was small. It is a draft only when it is a sliver, under a tenth of the file that is
/// its name without the kind, or of the largest file when no file is.
fn is_draft(path: &str, size: u64, largest: u64, sizes: &HashMap<String, u64>) -> bool {
    let (folder, base) = match path.rsplit_once('/') {
        Some((folder, base)) => (Some(folder), base),
        None => (None, path),
    };
    let parts = words(base);
    DRAFT_KINDS.iter().any(|kind| {
        let named_by_folder = folder.is_some_and(|f| {
            f.split('/')
                .next()
                .is_some_and(|d| d.eq_ignore_ascii_case(kind))
        });
        let at = parts.iter().position(|w| w == kind);
        if !named_by_folder && at.is_none() {
            return false;
        }
        let without: Vec<&str> = parts
            .iter()
            .enumerate()
            .filter(|(i, _)| Some(*i) != at)
            .map(|(_, w)| w.as_str())
            .collect();
        // Without the kind in its own name a file has no mate: its name without the kind is its name.
        let mate = at.and_then(|_| sizes.get(&without.join("-")).copied());
        if named_by_folder || at == Some(0) {
            // Named a draft by where it is filed or how it begins: a model only when it is as big as
            // the model it shares its name with.
            return mate.is_none_or(|mate| size.saturating_mul(MODEL_OF_THE_KIND_SHARE) < mate);
        }
        size.saturating_mul(DRAFT_SIZE_RATIO) < mate.unwrap_or(largest)
    })
}

/// `name-00001-of-00003`, with or without `.gguf`: one part of a weights file that was cut up.
pub fn is_split_part(name: &str) -> bool {
    let stem = name.strip_suffix(".gguf").unwrap_or(name);
    let digits = |s: &str| s.len() >= 3 && s.bytes().all(|b| b.is_ascii_digit());
    let Some((head, total)) = stem.rsplit_once("-of-") else {
        return false;
    };
    let Some((_, part)) = head.rsplit_once('-') else {
        return false;
    };
    digits(part) && digits(total)
}

fn is_sha256(text: &str) -> bool {
    text.len() == 64 && text.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Whether a path of a repository's tree is one that can be put into an address below the commit
/// the files are pinned to: no `.` or `..` segment, no empty one, no backslash or control character.
/// `rev/../../other/repo/resolve/main/x` would leave the commit, and the repository, once an
/// address is resolved.
fn is_plain_path(path: &str) -> bool {
    !path.is_empty()
        && !path.contains('\\')
        && !path.chars().any(char::is_control)
        && path
            .split('/')
            .all(|segment| !segment.is_empty() && segment != "." && segment != "..")
}

pub fn classify(repo: &str, rev: &str, tree: &[TreeEntry]) -> Files {
    let mut whole = Vec::new();
    let mut mmproj = Vec::new();
    // The largest weights file of the repository, parts included: a repository whose real weights are
    // cut into parts has nothing whole but an importance matrix, which is a sliver of a part and not
    // the repository's largest whole file.
    let mut overall = 0u64;
    for entry in tree {
        if entry.kind != "file"
            || !entry.path.to_ascii_lowercase().ends_with(".gguf")
            || !is_plain_path(&entry.path)
        {
            continue;
        }
        let name = entry.path.rsplit('/').next().unwrap_or(&entry.path);
        let size = entry
            .lfs
            .as_ref()
            .map(|l| l.size)
            .filter(|s| *s > 0)
            .or(entry.size)
            .unwrap_or(0);
        if size == 0 {
            continue;
        }
        let is_projector = name.to_ascii_lowercase().contains("mmproj");
        if !is_projector {
            overall = overall.max(size);
        }
        if is_split_part(name) {
            continue;
        }
        // A file with no hash from Hugging Face's LFS cannot be checked once downloaded: not offered.
        let Some(sha256) = entry
            .lfs
            .as_ref()
            .map(|l| l.oid.to_ascii_lowercase())
            .filter(|oid| is_sha256(oid))
        else {
            continue;
        };
        let file = File {
            model_id: entry.path[..entry.path.len() - ".gguf".len()].to_string(),
            path: format!(
                "https://huggingface.co/{repo}/resolve/{rev}/{}",
                encode_path(&entry.path)
            ),
            size,
            sha256: Some(sha256),
        };
        if is_projector {
            mmproj.push(file);
        } else {
            whole.push(file);
        }
    }

    whole.retain(|f| !is_not_a_model(&f.model_id, f.size, overall));
    let largest = whole.iter().map(|f| f.size).max().unwrap_or(0);
    let sizes: HashMap<String, u64> = whole
        .iter()
        .map(|f| {
            (
                normal(f.model_id.rsplit('/').next().unwrap_or(&f.model_id)),
                f.size,
            )
        })
        .collect();
    let (mut drafts, mut quants): (Vec<File>, Vec<File>) = whole
        .into_iter()
        .partition(|f| is_draft(&f.model_id, f.size, largest, &sizes));

    let by_size = |a: &File, b: &File| {
        a.size
            .cmp(&b.size)
            .then_with(|| a.model_id.cmp(&b.model_id))
    };
    quants.sort_by(by_size);
    drafts.sort_by(by_size);
    mmproj.sort_by(|a, b| a.model_id.cmp(&b.model_id));
    Files {
        quants,
        drafts,
        mmproj,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hf::Lfs;

    fn fixture(name: &str) -> Vec<TreeEntry> {
        let path = format!(
            "{}/tests/fixtures/{name}/tree.json",
            env!("CARGO_MANIFEST_DIR")
        );
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    fn file(path: &str, size: u64, oid: Option<&str>) -> TreeEntry {
        TreeEntry {
            kind: "file".into(),
            path: path.into(),
            size: Some(size),
            lfs: oid.map(|o| Lfs {
                oid: o.into(),
                size,
            }),
        }
    }

    const OID: &str = "0a270ec9fe0a270ec9fe0a270ec9fe0a270ec9fe0a270ec9fe0a270ec9fe1234";

    #[test]
    fn a_repository_with_vision_and_draft_files_gives_whole_weights_projectors_and_drafts() {
        let files = classify("unsloth/gemma-4-12b-it-GGUF", "fc034cf", &fixture("gemma"));
        assert_eq!(
            files.quants.len(),
            21,
            "28 gguf files: 3 projectors, 4 drafts"
        );
        assert_eq!(files.mmproj.len(), 3);
        let drafts: Vec<_> = files.drafts.iter().map(|d| d.model_id.as_str()).collect();
        assert_eq!(drafts.len(), 4);
        for draft in [
            "MTP/mtp-gemma-4-12b-it-BF16",
            "MTP/mtp-gemma-4-12b-it-F16",
            "MTP/mtp-gemma-4-12b-it-Q8_0",
            "mtp-gemma-4-12b-it",
        ] {
            assert!(drafts.contains(&draft), "{draft} is a draft");
        }
        assert!(
            files
                .quants
                .iter()
                .all(|q| !q.model_id.to_lowercase().contains("mtp"))
        );
        assert!(
            files
                .mmproj
                .iter()
                .all(|m| m.model_id.starts_with("mmproj-"))
        );
        assert!(
            files.quants.windows(2).all(|w| w[0].size <= w[1].size),
            "smallest first"
        );
        assert!(files.drafts.windows(2).all(|w| w[0].size <= w[1].size));
    }

    #[test]
    fn the_hash_is_the_lfs_sha256_and_never_the_git_object_id() {
        let tree = fixture("gemma");
        let entry = tree
            .iter()
            .find(|e| e.path == "gemma-4-12b-it-Q4_K_M.gguf")
            .unwrap();
        let files = classify("unsloth/gemma-4-12b-it-GGUF", "fc034cf", &tree);
        let q = files
            .quants
            .iter()
            .find(|q| q.model_id == "gemma-4-12b-it-Q4_K_M")
            .unwrap();
        let sha = q.sha256.as_deref().expect("an LFS file has a hash");
        assert_eq!(sha.len(), 64);
        assert!(sha.starts_with("0a270ec9fe"), "{sha}");
        assert_eq!(sha, entry.lfs.as_ref().unwrap().oid);
        assert_eq!(q.size, 7_121_861_440);
    }

    #[test]
    fn every_address_points_at_the_commit_not_at_main() {
        let files = classify("unsloth/gemma-4-12b-it-GGUF", "fc034cf", &fixture("gemma"));
        for f in files.quants.iter().chain(&files.mmproj) {
            assert!(
                f.path.starts_with(
                    "https://huggingface.co/unsloth/gemma-4-12b-it-GGUF/resolve/fc034cf/"
                ),
                "{}",
                f.path
            );
            assert!(f.path.ends_with(".gguf"));
        }
    }

    #[test]
    fn a_weights_file_cut_into_parts_is_skipped_and_its_projectors_stay() {
        let glm = classify("unsloth/GLM-5.3-Flash-GGUF", "a38483c", &fixture("glm"));
        assert!(
            glm.quants.is_empty(),
            "68 of 70 gguf files are parts, 2 are projectors"
        );
        assert_eq!(glm.mmproj.len(), 2);
        let qwen = classify("unsloth/Qwen3.8-27B-GGUF", "4ca7207", &fixture("qwen"));
        assert!(!qwen.quants.is_empty());
        assert!(
            qwen.quants.iter().all(|q| !q.model_id.contains("-of-")),
            "no part is offered"
        );
        // A gated repository's tree comes with every hash scrubbed to asterisks (no such repository
        // is read: the list leaves it out); here the hashes are put back so the files count.
        let mut gated = fixture("gated");
        for entry in &mut gated {
            if let Some(lfs) = entry.lfs.as_mut() {
                lfs.oid = OID.to_string();
            }
        }
        let mixed = classify(
            "orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF",
            "98eb335",
            &gated,
        );
        assert_eq!(
            mixed.quants.len(),
            1,
            "53 gguf: 51 parts, 1 projector, 1 whole file"
        );
    }

    #[test]
    fn what_counts_as_a_part() {
        assert!(is_split_part("model-Q4_K_M-00001-of-00003.gguf"));
        assert!(is_split_part("model-00002-of-00014"));
        assert!(!is_split_part("model-Q4_K_M.gguf"));
        assert!(!is_split_part("mmproj-of-things.gguf"));
        assert!(!is_split_part("a-of-b.gguf"));
        assert!(!is_split_part("-of-.gguf"));
    }

    #[test]
    fn folders_other_files_and_files_without_a_size_are_not_offered() {
        let tree = vec![
            TreeEntry {
                kind: "directory".into(),
                path: "Q4.gguf".into(),
                size: None,
                lfs: None,
            },
            file("README.md", 10, None),
            file("empty.gguf", 0, None),
            file("Weights.GGUF", 5 * GB, Some(OID)),
            file("plain-Q4.gguf", GB, Some(OID)),
            file("not-in-lfs.gguf", 2 * GB, None),
        ];
        let files = classify("a/b", "r", &tree);
        let ids: Vec<_> = files.quants.iter().map(|q| q.model_id.as_str()).collect();
        assert_eq!(
            ids,
            ["plain-Q4", "Weights"],
            "the extension is read in any case, smallest first"
        );
    }

    #[test]
    fn a_file_whose_hash_is_not_sha256_is_not_offered_and_one_in_capitals_is_lowered() {
        // What cannot be checked after the download is not offered: a file outside LFS, or with an oid
        // that is not a SHA-256.
        let tree = vec![
            file("a.gguf", GB, Some("not-a-hash")),
            file("b.gguf", 2 * GB, Some(&OID.to_ascii_uppercase())),
            file("c.gguf", 3 * GB, None),
        ];
        let files = classify("a/b", "r", &tree);
        assert_eq!(files.quants.len(), 1);
        assert_eq!(files.quants[0].model_id, "b");
        assert_eq!(files.quants[0].sha256.as_deref(), Some(OID));
    }

    #[test]
    fn an_awkward_file_name_is_encoded_in_the_address() {
        let files = classify("a/b", "r", &[file("dir one/m+x.gguf", GB, Some(OID))]);
        assert_eq!(
            files.quants[0].path,
            "https://huggingface.co/a/b/resolve/r/dir%20one/m%2Bx.gguf"
        );
        assert_eq!(files.quants[0].model_id, "dir one/m+x");
    }

    const GB: u64 = 1_000_000_000;

    fn tree_of(files: &[(&str, u64)]) -> Vec<TreeEntry> {
        files
            .iter()
            .map(|(path, size)| file(&format!("{path}.gguf"), *size, Some(OID)))
            .collect()
    }

    fn split(files: &[(&str, u64)]) -> (Vec<String>, Vec<String>) {
        let found = classify("o/r", "rev", &tree_of(files));
        let ids = |v: &[File]| v.iter().map(|f| f.model_id.clone()).collect::<Vec<_>>();
        let (mut quants, mut drafts) = (ids(&found.quants), ids(&found.drafts));
        quants.sort();
        drafts.sort();
        (quants, drafts)
    }

    #[test]
    fn the_words_of_a_name_are_cut_at_dashes_underscores_and_dots() {
        assert_eq!(
            words("Qwen3-Coder-MTP-Q4_K_M"),
            ["qwen3", "coder", "mtp", "q4", "k", "m"]
        );
        assert_eq!(normal("Model.Q8_0"), normal("model-q8-0"));
    }

    #[test]
    fn a_draft_is_in_a_folder_of_its_kind_or_begins_with_it() {
        let (quants, drafts) = split(&[
            ("Model-Q8_0", 8 * GB),
            ("MTP/Model-BF16", GB),
            ("mtp-Model", GB),
            ("DSpark/anything", GB),
            ("eagle3-Model-Q4_0", GB),
            ("dflash_Model", GB),
        ]);
        assert_eq!(quants, ["Model-Q8_0"]);
        assert_eq!(drafts.len(), 5);
    }

    #[test]
    fn a_name_that_holds_the_kind_in_the_middle_is_a_model_when_nothing_pairs_with_it() {
        // Models made with the head kept in them, named so in every file of their repository.
        let (quants, drafts) = split(&[
            ("Tiel-Coder-35B-MTP-UD-Q2_K_XL", 12 * GB),
            ("Tiel-Coder-35B-MTP-Q4_K_M", 20 * GB),
            ("Tiel-Coder-35B-MTP-BF16", 60 * GB),
        ]);
        assert_eq!(quants.len(), 3, "{quants:?}");
        assert!(drafts.is_empty());
        // One whole file only, named with the kind at its end.
        let (quants, drafts) = split(&[("Qwen-iMatrix-NVFP4-MTP", 15 * GB)]);
        assert_eq!((quants.len(), drafts.len()), (1, 0));
    }

    #[test]
    fn a_variant_with_the_head_kept_in_it_weighs_what_its_mate_weighs_and_is_a_model() {
        // Measured: every such file was 1.0 to 1.07 times the size of the file it is named after.
        let (quants, drafts) = split(&[
            ("RVN-IQ1_S", 7_100_000_000),
            ("RVN-IQ1_S-mtp", 7_600_000_000),
            ("model.q4_0", 4 * GB),
            ("Model-MTP-Q4_0", 4_300_000_000),
            // Far larger than the others: against the largest, a 4.3 GB file would look like a sliver.
            ("Big-BF16", 60 * GB),
            ("mtp-RVN", 1_800_000_000),
        ]);
        assert_eq!(quants.len(), 5, "{quants:?}");
        assert_eq!(drafts, ["mtp-RVN"]);
    }

    #[test]
    fn a_file_under_a_tenth_of_the_file_it_is_named_after_is_its_draft_across_separators() {
        let (quants, drafts) = split(&[
            ("Model-Q8_0", 8 * GB),
            ("Model-Q8_0-MTP", GB / 2),
            ("Big-BF16", 60 * GB),
            ("model.q4_0", 4 * GB),
            ("Model-MTP-Q4_0", GB / 4),
        ]);
        assert_eq!(quants, ["Big-BF16", "Model-Q8_0", "model.q4_0"]);
        assert_eq!(drafts, ["Model-MTP-Q4_0", "Model-Q8_0-MTP"]);
    }

    #[test]
    fn a_sliver_of_the_largest_file_with_the_kind_in_its_name_is_a_draft_though_nothing_pairs() {
        let (quants, drafts) = split(&[
            ("Big-Q4_K_M", 20 * GB),
            ("Big-Q8_0", 36 * GB),
            ("Name-DFlash-Q4_K", GB / 5),
        ]);
        assert_eq!(quants, ["Big-Q4_K_M", "Big-Q8_0"]);
        assert_eq!(drafts, ["Name-DFlash-Q4_K"]);
        // The same name at a model's size, and as the largest, is a model.
        let (quants, drafts) = split(&[("Name-DFlash-Q4_K", 20 * GB), ("Other-Q8_0", 36 * GB)]);
        assert_eq!((quants.len(), drafts.len()), (2, 0));
    }

    #[test]
    fn the_kind_must_be_a_whole_word_of_the_name() {
        let (quants, drafts) = split(&[
            ("Model-noMTP-Q4", 4 * GB),
            ("Model-mtpx-Q4", 4 * GB),
            ("Model-undflashed-Q4", 4 * GB),
            ("MTPlus/Model-Q4", 4 * GB),
        ]);
        assert_eq!(quants.len(), 4, "{quants:?}");
        assert!(drafts.is_empty());
    }

    #[test]
    fn a_repository_whose_only_whole_files_are_drafts_has_no_model_in_it() {
        let found = classify(
            "o/r",
            "rev",
            &tree_of(&[("mtp-q2_0", GB), ("dspark-Model-Q8_0", GB)]),
        );
        assert!(found.quants.is_empty());
        assert_eq!(found.drafts.len(), 2);
    }

    const MB: u64 = 1_000_000;

    #[test]
    fn a_file_under_fifty_megabytes_is_no_model_even_when_it_is_all_there_is() {
        let (quants, drafts) = split(&[("Vocab-only", 9 * MB)]);
        assert!(quants.is_empty() && drafts.is_empty());
        let (quants, _) = split(&[
            ("Tiny-Q4", 49_999_999),
            ("Edge-Q4", 50_000_000),
            ("Real-Q4", 4 * GB),
        ]);
        assert_eq!(
            quants,
            ["Edge-Q4", "Real-Q4"],
            "50 MB is the least a model weighs"
        );
    }

    #[test]
    fn importance_data_adapters_vocabularies_and_speech_parts_are_no_models_when_they_are_slivers()
    {
        let (quants, drafts) = split(&[
            ("Model-Q4_K_M", 20 * GB),
            ("Model-BF16", 40 * GB),
            ("Model-imatrix", 192 * MB),
            ("imatrix/Ling.imatrix", 488 * MB),
            ("lora/Model-LoRA", 106 * MB),
            ("tokenizer-Audio-F16", 143 * MB),
            ("token2wav-gguf/hifigan2", 83 * MB),
            ("tts/Model-projector-F16", 150 * MB),
        ]);
        assert_eq!(quants, ["Model-BF16", "Model-Q4_K_M"]);
        assert!(drafts.is_empty());
    }

    #[test]
    fn a_model_that_carries_imatrix_in_its_name_at_the_size_of_a_model_stays() {
        let (quants, _) = split(&[
            ("Qwen-iMatrix-NVFP4-MTP", 15 * GB),
            ("Qwen-Q4_K_M", 20 * GB),
            ("thinking/Qwen-IQ3_XS-thinking", 5 * GB),
        ]);
        assert_eq!(quants.len(), 3, "{quants:?}");
    }

    #[test]
    fn the_word_must_be_whole_for_a_file_to_be_left_out() {
        let (quants, _) = split(&[
            ("Model-imatrixed-Q4", 200 * MB),
            ("Model-lora2-Q4", 300 * MB),
            ("Model-tokenizers", 300 * MB),
            ("Big-BF16", 60 * GB),
        ]);
        assert_eq!(quants.len(), 4, "{quants:?}");
    }

    // Found on the real catalog: JetBrains/Mellum2.1-12B-A2.5B-Thinking-GGUF files whole models in a folder
    // `mtp/` as `…-MTP-<quant>`, each 1.03 to 1.04 times its mate; its one draft is `mtp-…-BF16`.
    #[test]
    fn a_whole_model_filed_in_a_folder_of_the_kind_is_a_model_and_the_real_draft_beside_it_stays_one()
     {
        let (quants, drafts) = split(&[
            ("Mellum-Q4_K_M", 7_900_000_000),
            ("Mellum-Q8_0", 13_000_000_000),
            ("Mellum-BF16", 24_000_000_000),
            ("mtp/Mellum-MTP-Q4_K_M", 8_300_000_000),
            ("mtp/Mellum-MTP-Q8_0", 13_400_000_000),
            ("mtp/Mellum-MTP-BF16", 25_000_000_000),
            ("mtp-Mellum-BF16", 1_770_000_000),
        ]);
        assert_eq!(drafts, ["mtp-Mellum-BF16"]);
        assert_eq!(quants.len(), 6, "{quants:?}");
        assert!(quants.contains(&"mtp/Mellum-MTP-Q4_K_M".to_string()));
    }

    #[test]
    fn a_draft_in_a_folder_of_its_kind_named_for_its_mate_is_a_draft_while_a_sliver_of_it() {
        let (quants, drafts) = split(&[
            ("Model-Q8_0", 8 * GB),
            ("MTP/mtp-Model-Q8_0", 900_000_000),
            ("MTP/Model-MTP-Q8_0", 3 * GB),
        ]);
        assert_eq!(quants, ["Model-Q8_0"]);
        assert_eq!(drafts, ["MTP/Model-MTP-Q8_0", "MTP/mtp-Model-Q8_0"]);
    }

    // Found on the real catalog: three repositories offered their importance matrix as their one model,
    // the real weights being parts and the matrix the largest whole file.
    #[test]
    fn an_importance_matrix_beside_weights_that_are_all_parts_is_not_offered_as_a_model() {
        let (quants, _) = split(&[
            ("imatrix", 580_000_000),
            ("Model-Q4_K_M-00001-of-00004", 12 * GB),
            ("Model-Q4_K_M-00002-of-00004", 12 * GB),
        ]);
        assert!(quants.is_empty(), "{quants:?}");
        // The same file beside a whole model is no model either, and a model that only carries the word is.
        let (quants, _) = split(&[
            ("imatrix", 580_000_000),
            ("Model-Q4_K_M", 12 * GB),
            ("Model-imatrix-Q8", 11 * GB),
        ]);
        assert_eq!(quants, ["Model-Q4_K_M", "Model-imatrix-Q8"]);
    }

    #[test]
    fn a_path_that_leaves_the_commit_is_not_offered() {
        let bad = [
            "../../../other/repo/resolve/main/evil",
            "a/../b",
            "./a",
            "a//b",
            "/abs",
            "a\\b",
            "a\u{7}b",
        ];
        let mut tree: Vec<TreeEntry> = bad
            .iter()
            .map(|p| file(&format!("{p}.gguf"), 2 * GB, Some(OID)))
            .collect();
        tree.push(file("fine/Model-Q4.gguf", 2 * GB, Some(OID)));
        let found = classify("o/r", "rev", &tree);
        let ids: Vec<_> = found.quants.iter().map(|f| f.model_id.as_str()).collect();
        assert_eq!(ids, ["fine/Model-Q4"]);
    }
}
