// SPDX-License-Identifier: AGPL-3.0-only
//! From a repository's file tree to the files the app can download.
//!
//! Only whole files are offered: a weights file cut into `-00001-of-00003` parts is skipped, as the
//! app downloads one file per model. Every address points at the commit the tree was read at.

use crate::hf::{TreeEntry, encode_path};
use crate::model::File;

pub struct Files {
    pub quants: Vec<File>,
    pub mmproj: Vec<File>,
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
    let mut quants = Vec::new();
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
            quants.push(file);
        }
    }
    quants.sort_by(|a, b| {
        a.size
            .cmp(&b.size)
            .then_with(|| a.model_id.cmp(&b.model_id))
    });
    mmproj.sort_by(|a, b| a.model_id.cmp(&b.model_id));
    Files { quants, mmproj }
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
            25,
            "28 gguf files, 3 of them projectors"
        );
        assert_eq!(files.mmproj.len(), 3);
        assert!(
            files
                .quants
                .iter()
                .any(|q| q.model_id == "MTP/mtp-gemma-4-12b-it-Q8_0")
        );
        assert!(
            files
                .quants
                .iter()
                .any(|q| q.model_id == "mtp-gemma-4-12b-it")
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
}
