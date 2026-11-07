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
use std::collections::HashSet;

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

/// The words of a file name, lower case: `Qwen3-Coder-MTP-Q4_K_M` is qwen3, coder, mtp, q4, k, m.
fn words(name: &str) -> Vec<String> {
    name.to_lowercase()
        .split(['-', '_', '.'])
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect()
}

/// A name by its words only, so `Model-Q8_0` and `model.q8-0` are one name.
fn normal(name: &str) -> String {
    words(name).join("-")
}

/// Whether a file is a draft, from its path without `.gguf`, its size, the largest file of the
/// repository and the normal names of all its whole files.
///
/// A draft is a file in a folder of the kind's name (`MTP/…`), or one that begins with it
/// (`mtp-…`, the way upstream names them). A name that holds the kind anywhere else, as
/// `Model-MTP-Q4_K_M`, is a model made with it when its repository has no file of that name without
/// the kind and no sliver of a size: the real ones are named so in their repository. It is a draft
/// when another file is its name without the kind (`Model-Q8_0` beside `Model-Q8_0-MTP`) or it is a
/// sliver of the largest file.
fn is_draft(path: &str, size: u64, largest: u64, names: &HashSet<String>) -> bool {
    let (folder, base) = match path.rsplit_once('/') {
        Some((folder, base)) => (Some(folder), base),
        None => (None, path),
    };
    let parts = words(base);
    DRAFT_KINDS.iter().any(|kind| {
        if folder.is_some_and(|f| {
            f.split('/')
                .next()
                .is_some_and(|d| d.eq_ignore_ascii_case(kind))
        }) {
            return true;
        }
        let Some(at) = parts.iter().position(|w| w == kind) else {
            return false;
        };
        if at == 0 {
            return true;
        }
        let without: Vec<&str> = parts
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != at)
            .map(|(_, w)| w.as_str())
            .collect();
        size.saturating_mul(DRAFT_SIZE_RATIO) < largest || names.contains(&without.join("-"))
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

pub fn classify(repo: &str, rev: &str, tree: &[TreeEntry]) -> Files {
    let mut whole = Vec::new();
    let mut mmproj = Vec::new();
    for entry in tree {
        if entry.kind != "file" || !entry.path.to_ascii_lowercase().ends_with(".gguf") {
            continue;
        }
        let name = entry.path.rsplit('/').next().unwrap_or(&entry.path);
        if is_split_part(name) {
            continue;
        }
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
        let sha256 = entry
            .lfs
            .as_ref()
            .map(|l| l.oid.to_ascii_lowercase())
            .filter(|oid| is_sha256(oid));
        let file = File {
            model_id: entry.path[..entry.path.len() - ".gguf".len()].to_string(),
            path: format!(
                "https://huggingface.co/{repo}/resolve/{rev}/{}",
                encode_path(&entry.path)
            ),
            size,
            sha256,
        };
        if name.to_ascii_lowercase().contains("mmproj") {
            mmproj.push(file);
        } else {
            whole.push(file);
        }
    }

    let largest = whole.iter().map(|f| f.size).max().unwrap_or(0);
    let names: HashSet<String> = whole
        .iter()
        .map(|f| normal(f.model_id.rsplit('/').next().unwrap_or(&f.model_id)))
        .collect();
    let (mut drafts, mut quants): (Vec<File>, Vec<File>) = whole
        .into_iter()
        .partition(|f| is_draft(&f.model_id, f.size, largest, &names));

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
        let mixed = classify(
            "orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF",
            "98eb335",
            &fixture("gated"),
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
            file("Weights.GGUF", 5, None),
            file("plain-Q4.gguf", 100, Some(OID)),
        ];
        let files = classify("a/b", "r", &tree);
        let ids: Vec<_> = files.quants.iter().map(|q| q.model_id.as_str()).collect();
        assert_eq!(
            ids,
            ["Weights", "plain-Q4"],
            "the extension is read in any case"
        );
    }

    #[test]
    fn a_hash_that_is_not_sha256_is_left_out_and_one_in_capitals_is_lowered() {
        let tree = vec![
            file("a.gguf", 10, Some("not-a-hash")),
            file("b.gguf", 20, Some(&OID.to_ascii_uppercase())),
        ];
        let files = classify("a/b", "r", &tree);
        assert_eq!(files.quants[0].sha256, None);
        assert_eq!(files.quants[1].sha256.as_deref(), Some(OID));
    }

    #[test]
    fn an_awkward_file_name_is_encoded_in_the_address() {
        let files = classify("a/b", "r", &[file("dir one/m+x.gguf", 7, None)]);
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
    fn a_file_that_is_another_files_name_plus_the_kind_is_its_draft_across_separators() {
        let (quants, drafts) = split(&[
            ("Model-Q8_0", 8 * GB),
            ("Model-Q8_0-MTP", 3 * GB),
            ("model.q4_0", 4 * GB),
            ("Model-MTP-Q4_0", 2 * GB),
        ]);
        assert_eq!(quants, ["Model-Q8_0", "model.q4_0"]);
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
}
