// SPDX-License-Identifier: AGPL-3.0-only
//! The whole run against answers recorded from Hugging Face (`tests/fixtures`), with the requests
//! it makes counted.

use collm_catalog::assemble::{Stats, build};
use collm_catalog::hf::{Hf, HfError, ListPage, Repo, TreeEntry};
use collm_catalog::model::{Catalog, FORMAT, Model, check_shrink, render};
use collm_catalog::select::{Curated, Reason};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

const GEMMA: &str = "unsloth/gemma-4-12b-it-GGUF";
const GLM: &str = "unsloth/GLM-5.3-Flash-GGUF";
const GATED: &str = "orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF";
const QWEN: &str = "unsloth/Qwen3.8-27B-GGUF";

fn fixtures() -> String {
    format!("{}/tests/fixtures", env!("CARGO_MANIFEST_DIR"))
}

fn read(path: &str) -> String {
    std::fs::read_to_string(format!("{}/{path}", fixtures())).unwrap()
}

#[derive(Default)]
struct Fake {
    pages: Vec<Vec<Repo>>,
    details: HashMap<String, Repo>,
    trees: HashMap<String, Vec<TreeEntry>>,
    readmes: HashMap<String, String>,
    calls: RefCell<Vec<String>>,
    list_fails_at: Option<usize>,
    tree_fails: HashSet<String>,
}

impl Fake {
    /// The two recorded list pages and the four recorded repositories behind them.
    fn recorded() -> Fake {
        let mut fake = Fake::default();
        for page in ["list_page_1.json", "list_page_2.json"] {
            fake.pages.push(serde_json::from_str(&read(page)).unwrap());
        }
        for (key, id) in [
            ("gemma", GEMMA),
            ("glm", GLM),
            ("gated", GATED),
            ("qwen", QWEN),
        ] {
            fake.details.insert(
                id.into(),
                serde_json::from_str(&read(&format!("{key}/detail.json"))).unwrap(),
            );
            fake.trees.insert(
                id.into(),
                serde_json::from_str(&read(&format!("{key}/tree.json"))).unwrap(),
            );
            fake.readmes
                .insert(id.into(), read(&format!("{key}/readme.md")));
        }
        fake
    }

    fn called(&self, prefix: &str) -> Vec<String> {
        self.calls
            .borrow()
            .iter()
            .filter(|c| c.starts_with(prefix))
            .cloned()
            .collect()
    }

    fn set_sha(&mut self, id: &str, sha: &str) {
        for page in &mut self.pages {
            for repo in page.iter_mut().filter(|r| r.id == id) {
                repo.sha = Some(sha.into());
            }
        }
        self.details.get_mut(id).unwrap().sha = Some(sha.into());
    }

    fn set_downloads(&mut self, id: &str, downloads: u64) {
        for page in &mut self.pages {
            for repo in page.iter_mut().filter(|r| r.id == id) {
                repo.downloads = downloads;
            }
        }
    }
}

impl Hf for Fake {
    fn list_page(&self, cursor: Option<&str>) -> Result<ListPage, HfError> {
        let index = cursor
            .map(|c| c.trim_start_matches("page:").parse::<usize>().unwrap())
            .unwrap_or(0);
        self.calls.borrow_mut().push(format!("list:{index}"));
        if self.list_fails_at == Some(index) {
            return Err(HfError("the list did not answer".into()));
        }
        let repos = self.pages.get(index).cloned().unwrap_or_default();
        let next = (index + 1 < self.pages.len()).then(|| format!("page:{}", index + 1));
        Ok(ListPage {
            repos,
            next,
            skipped: 0,
        })
    }

    fn repo(&self, id: &str) -> Result<Repo, HfError> {
        self.calls.borrow_mut().push(format!("repo:{id}"));
        self.details
            .get(id)
            .cloned()
            .ok_or_else(|| HfError(format!("{id}: HTTP 404")))
    }

    fn tree(&self, id: &str, rev: &str) -> Result<Vec<TreeEntry>, HfError> {
        self.calls.borrow_mut().push(format!("tree:{id}@{rev}"));
        if self.tree_fails.contains(id) {
            return Err(HfError(format!("{id}: HTTP 503")));
        }
        self.trees
            .get(id)
            .cloned()
            .ok_or_else(|| HfError(format!("{id}: HTTP 404")))
    }

    fn readme(&self, id: &str, _rev: &str) -> Result<Option<String>, HfError> {
        self.calls.borrow_mut().push(format!("readme:{id}"));
        Ok(self.readmes.get(id).cloned())
    }
}

fn run(fake: &Fake, curated: &Curated, previous: &Catalog) -> (Catalog, Stats, Vec<String>) {
    let mut log = Vec::new();
    let (catalog, stats) = build(fake, curated, previous, &mut |line| log.push(line)).unwrap();
    (catalog, stats, log)
}

fn names(catalog: &Catalog) -> Vec<String> {
    catalog.models.iter().map(|m| m.repo_id()).collect()
}

fn find<'a>(catalog: &'a Catalog, id: &str) -> &'a Model {
    catalog
        .models
        .iter()
        .find(|m| m.repo_id() == id)
        .unwrap_or_else(|| panic!("{id} is not in the catalog"))
}

#[test]
fn a_run_keeps_the_chat_models_with_a_whole_file_and_says_why_it_left_the_rest_out() {
    let fake = Fake::recorded();
    let (catalog, stats, _) = run(&fake, &Curated::default(), &Catalog::empty());

    assert_eq!(catalog.format, FORMAT);
    assert_eq!(
        names(&catalog),
        [GEMMA, QWEN],
        "sorted by owner and name, not by downloads"
    );
    assert_eq!(stats.fetched, 2);
    assert_eq!(stats.reused, 0);
    assert_eq!(
        stats.rejected.get(&Reason::Gated),
        Some(&1),
        "the one behind a form"
    );
    assert_eq!(
        stats.rejected.get(&Reason::NoWholeFile),
        Some(&1),
        "the one cut into parts"
    );
    assert!(
        stats
            .rejected
            .get(&Reason::NotAChatPipeline)
            .copied()
            .unwrap_or(0)
            >= 10,
        "speech, image, video"
    );
    assert!(
        fake.called("repo:orcarouter").is_empty(),
        "a gated repository costs no request"
    );
    assert!(fake.called("tree:orcarouter").is_empty());
    assert!(
        stats.failed > 0,
        "repositories the fake does not know are counted as failed"
    );
    assert_eq!(stats.failed, stats.examined - stats.fetched - 1);
}

#[test]
fn a_model_carries_what_the_app_shows_and_checks() {
    let fake = Fake::recorded();
    let (catalog, _, _) = run(&fake, &Curated::default(), &Catalog::empty());
    let gemma = find(&catalog, GEMMA);

    assert_eq!(gemma.developer, "unsloth");
    assert_eq!(gemma.model_name, "gemma-4-12b-it-GGUF");
    assert_eq!(gemma.sha, "fc034cfff751157913579611efad8462ac1be606");
    assert_eq!(gemma.architecture, "gemma4");
    assert_eq!(gemma.context_length, Some(262_144));
    assert_eq!(gemma.license.as_deref(), Some("apache-2.0"));
    assert_eq!(gemma.pipeline_tag.as_deref(), Some("image-text-to-text"));
    assert_eq!(
        gemma.capabilities,
        ["agent", "tools", "vision"],
        "its README speaks of agentic workflows three times; the template takes tools; projectors lie beside the weights"
    );
    assert_eq!(gemma.quants.len(), 25);
    assert_eq!(gemma.mmproj_models.len(), 3);
    assert!(
        gemma
            .description
            .starts_with("Gemma is a family of open models"),
        "{}",
        gemma.description
    );
    assert_eq!(
        gemma.readme,
        "https://huggingface.co/unsloth/gemma-4-12b-it-GGUF/resolve/fc034cfff751157913579611efad8462ac1be606/README.md"
    );
    assert!(gemma.quants.iter().chain(&gemma.mmproj_models).all(|f| {
        f.path
            .contains("/resolve/fc034cfff751157913579611efad8462ac1be606/")
            && f.sha256.as_ref().is_some_and(|h| h.len() == 64)
            && f.size > 0
    }));
    let qwen = find(&catalog, QWEN);
    assert!(
        qwen.quants.iter().all(|q| !q.model_id.contains("-of-")),
        "no part of a cut file"
    );
}

#[test]
fn a_repository_that_has_not_changed_costs_no_request_but_its_downloads_are_renewed() {
    let mut fake = Fake::recorded();
    let (first, _, _) = run(&fake, &Curated::default(), &Catalog::empty());
    fake.calls.borrow_mut().clear();
    fake.set_downloads(GEMMA, 1);

    let (second, stats, _) = run(&fake, &Curated::default(), &first);

    assert_eq!(stats.reused, 2);
    assert_eq!(stats.fetched, 0);
    for id in [GEMMA, QWEN] {
        assert!(
            !fake
                .called("tree:")
                .iter()
                .any(|c| c.starts_with(&format!("tree:{id}@"))),
            "{id} was read again"
        );
        assert!(
            !fake
                .called("readme:")
                .iter()
                .any(|c| c == &format!("readme:{id}"))
        );
        assert!(
            !fake
                .called("repo:")
                .iter()
                .any(|c| c == &format!("repo:{id}"))
        );
    }
    assert_eq!(
        find(&second, GEMMA).downloads,
        1,
        "the count is the list's, today's"
    );
    let mut same = find(&second, GEMMA).clone();
    same.downloads = find(&first, GEMMA).downloads;
    assert_eq!(&same, find(&first, GEMMA), "everything else is as it was");
}

#[test]
fn a_new_commit_is_read_again_and_every_address_follows_it() {
    let mut fake = Fake::recorded();
    let (first, _, _) = run(&fake, &Curated::default(), &Catalog::empty());
    fake.calls.borrow_mut().clear();
    let new_sha = "0123456789abcdef0123456789abcdef01234567";
    fake.set_sha(GEMMA, new_sha);

    let (second, stats, _) = run(&fake, &Curated::default(), &first);

    assert_eq!((stats.fetched, stats.reused), (1, 1));
    assert_eq!(
        fake.called(&format!("tree:{GEMMA}@")),
        [format!("tree:{GEMMA}@{new_sha}")]
    );
    let gemma = find(&second, GEMMA);
    assert_eq!(gemma.sha, new_sha);
    assert!(gemma.quants.iter().all(|q| q.path.contains(new_sha)));
    assert!(gemma.readme.contains(new_sha));
}

#[test]
fn a_repository_that_fails_keeps_its_old_entry_and_one_never_seen_is_left_out() {
    let mut fake = Fake::recorded();
    let (first, _, _) = run(&fake, &Curated::default(), &Catalog::empty());
    fake.set_sha(GEMMA, "0123456789abcdef0123456789abcdef01234567");
    fake.tree_fails.insert(GEMMA.into());

    let (kept, stats, log) = run(&fake, &Curated::default(), &first);
    assert_eq!(names(&kept), [GEMMA, QWEN], "yesterday's entry stands");
    assert_eq!(find(&kept, GEMMA).sha, find(&first, GEMMA).sha);
    assert_eq!(stats.kept_after_error, 1);
    assert!(
        log.iter().any(|l| l.contains(GEMMA) && l.contains("503")),
        "{log:?}"
    );

    let (without, stats, _) = run(&fake, &Curated::default(), &Catalog::empty());
    assert_eq!(names(&without), [QWEN]);
    assert_eq!(stats.kept_after_error, 0);
}

#[test]
fn a_list_that_does_not_answer_ends_the_run_instead_of_giving_half_a_catalog() {
    let mut fake = Fake::recorded();
    fake.list_fails_at = Some(1);
    let result = build(&fake, &Curated::default(), &Catalog::empty(), &mut |_| {});
    assert!(result.is_err());
    assert_eq!(fake.called("list:"), ["list:0", "list:1"]);
}

#[test]
fn the_catalog_stops_at_max_models_and_pinned_repositories_come_first() {
    let fake = Fake::recorded();
    let one = Curated {
        max_models: 1,
        ..Curated::default()
    };
    let (top, _, _) = run(&fake, &one, &Catalog::empty());
    assert_eq!(
        names(&top),
        [QWEN],
        "the most downloaded chat model of the list"
    );
    assert!(
        fake.called(&format!("tree:{GEMMA}@")).is_empty(),
        "the second one is never read"
    );

    let pinned = Curated {
        max_models: 1,
        pinned: vec![GEMMA.into()],
        ..Curated::default()
    };
    let (first, _, _) = run(&Fake::recorded(), &pinned, &Catalog::empty());
    assert_eq!(names(&first), [GEMMA]);
}

#[test]
fn a_pinned_repository_that_cannot_be_read_stays_if_it_was_listed_before() {
    let mut fake = Fake::recorded();
    let pinned_repo = bare("pinned/only-here", &[], "llama", Some("{{ messages }}"));
    fake.trees
        .insert(pinned_repo.id.clone(), fake.trees[GEMMA].clone());
    fake.details.insert(pinned_repo.id.clone(), pinned_repo);
    let pinned = Curated {
        pinned: vec!["pinned/only-here".into()],
        max_models: 5,
        ..Curated::default()
    };

    let (before, _, _) = run(&fake, &pinned, &Catalog::empty());
    assert_eq!(
        names(&before)[0],
        "pinned/only-here",
        "the list does not hold it; pinning does"
    );

    fake.details.remove("pinned/only-here");
    let (after, stats, log) = run(&fake, &pinned, &before);
    assert!(
        names(&after).contains(&"pinned/only-here".to_string()),
        "yesterday's entry stands"
    );
    assert!(stats.kept_after_error >= 1);
    assert!(
        log.iter().any(|l| l.contains("pinned/only-here")),
        "{log:?}"
    );

    let (fresh, stats, _) = run(&fake, &pinned, &Catalog::empty());
    assert!(!names(&fresh).contains(&"pinned/only-here".to_string()));
    assert!(stats.failed >= 1);
}

fn bare(id: &str, tags: &[&str], architecture: &str, template: Option<&str>) -> Repo {
    let template =
        template.map(|t| format!("\"chat_template\":{}", serde_json::to_string(t).unwrap()));
    serde_json::from_str(&format!(
        "{{\"id\":\"{id}\",\"sha\":\"abc\",\"downloads\":5,\"likes\":1,\"pipeline_tag\":\"text-generation\",\
          \"tags\":{},\"gguf\":{{\"architecture\":\"{architecture}\"{}}}}}",
        serde_json::to_string(tags).unwrap(),
        template.map(|t| format!(",{t}")).unwrap_or_default()
    ))
    .unwrap()
}

fn synthetic(repos: Vec<Repo>) -> Fake {
    let mut fake = Fake::default();
    let tree: Vec<TreeEntry> = serde_json::from_str(
        r#"[{"type":"file","path":"m-Q4.gguf","size":4,"lfs":{"oid":"00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff","size":4}}]"#,
    )
    .unwrap();
    for repo in &repos {
        fake.details.insert(repo.id.clone(), repo.clone());
        fake.trees.insert(repo.id.clone(), tree.clone());
    }
    fake.pages.push(repos);
    fake
}

#[test]
fn a_file_without_a_chat_template_is_no_chat_model_and_an_excluded_architecture_is_left_out() {
    let fake = synthetic(vec![
        bare("o/plain", &[], "llama", None),
        bare("o/clipper", &[], "dflash", Some("{{ messages }}")),
        bare(
            "o/chatty",
            &[],
            "llama",
            Some("{% for m in messages %}{{ m }}{% endfor %}"),
        ),
    ]);
    let curated = Curated {
        exclude_architectures: vec!["DFlash".into()],
        ..Curated::default()
    };
    let (catalog, stats, _) = run(&fake, &curated, &Catalog::empty());
    assert_eq!(names(&catalog), ["o/chatty"]);
    assert_eq!(stats.rejected.get(&Reason::NoChatTemplate), Some(&1));
    assert_eq!(stats.rejected.get(&Reason::ExcludedArchitecture), Some(&1));
    assert!(
        find(&catalog, "o/chatty").capabilities.is_empty(),
        "a plain chat model has no flag"
    );
}

#[test]
fn without_a_readme_the_description_names_the_base_model_and_there_is_no_readme_address() {
    let fake = synthetic(vec![bare(
        "o/quant",
        &[
            "gguf",
            "base_model:org/Base-7B",
            "base_model:quantized:org/Base-7B",
        ],
        "llama",
        Some("{% if tools %}x{% endif %}"),
    )]);
    let (catalog, _, _) = run(&fake, &Curated::default(), &Catalog::empty());
    let m = find(&catalog, "o/quant");
    assert_eq!(m.description, "Quantized build of org/Base-7B.");
    assert_eq!(m.readme, "");
    assert_eq!(m.capabilities, ["tools"]);
    assert!(
        !render(&catalog).contains("\"readme\""),
        "an empty address is not written"
    );
}

#[test]
fn the_file_is_one_model_per_line_reads_back_the_same_and_is_the_same_every_time() {
    let (catalog, _, _) = run(&Fake::recorded(), &Curated::default(), &Catalog::empty());
    let text = render(&catalog);
    assert!(text.starts_with("{\"format\":1,\"models\":[\n"));
    assert!(text.ends_with("]}\n"));
    assert_eq!(text.lines().count(), catalog.models.len() + 2);
    assert_eq!(Catalog::from_text(&text), catalog);
    assert_eq!(render(&Catalog::from_text(&text)), text);
    let (again, _, _) = run(&Fake::recorded(), &Curated::default(), &Catalog::empty());
    assert_eq!(render(&again), text);
}

#[test]
fn a_file_that_is_not_a_catalog_of_this_format_reads_as_empty() {
    assert!(Catalog::from_text("").models.is_empty());
    assert!(Catalog::from_text("[1,2,3]").models.is_empty());
    assert!(
        Catalog::from_text("{\"format\":99,\"models\":[]}")
            .models
            .is_empty()
    );
    assert!(
        Catalog::from_text("{\"format\":1,\"models\":[{\"model_name\":1}]}")
            .models
            .is_empty()
    );
}

#[test]
fn a_catalog_far_smaller_than_the_last_is_a_failed_run() {
    assert!(check_shrink(0, 0, 0.7).is_err(), "never publish nothing");
    assert!(check_shrink(300, 0, 0.7).is_err());
    assert!(check_shrink(300, 209, 0.7).is_err());
    assert!(check_shrink(300, 210, 0.7).is_ok());
    assert!(
        check_shrink(10, 3, 0.7).is_ok(),
        "a small catalog may change a lot"
    );
    assert!(check_shrink(0, 5, 0.7).is_ok());
}

fn readme_for(fake: &mut Fake, id: &str, text: &str) {
    fake.readmes.insert(id.into(), text.into());
}

fn with_projector_tree(fake: &mut Fake, id: &str) {
    let tree: Vec<TreeEntry> = serde_json::from_str(
        r#"[
        {"type":"file","path":"mmproj-F16.gguf","size":9,"lfs":{"oid":"ffeeddccbbaa99887766554433221100ffeeddccbbaa99887766554433221100","size":9}},
        {"type":"file","path":"weights-Q4.gguf","size":4,"lfs":{"oid":"00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff","size":4}}]"#,
    )
    .unwrap();
    fake.trees.insert(id.into(), tree);
}

#[test]
fn a_repository_whose_first_file_is_a_projector_is_still_listed_with_nothing_taken_from_the_wrong_metadata()
 {
    // Hugging Face summarizes the first gguf file alphabetically; here that is `mmproj-F16.gguf`.
    let mut fake = synthetic(vec![
        bare("o/seer", &["function-calling"], "clip", None),
        bare("o/unlabelled", &[], "clip", None),
    ]);
    fake.details.get_mut("o/unlabelled").unwrap().pipeline_tag = None;
    fake.pages[0][1].pipeline_tag = None;
    with_projector_tree(&mut fake, "o/seer");
    with_projector_tree(&mut fake, "o/unlabelled");

    let (catalog, stats, _) = run(&fake, &Curated::default(), &Catalog::empty());

    assert_eq!(
        names(&catalog),
        ["o/seer"],
        "a chat label is what vouches for it; no label, no entry"
    );
    let seer = find(&catalog, "o/seer");
    assert_eq!(
        seer.architecture, "",
        "the projector's `clip` is not the model's architecture"
    );
    assert_eq!(seer.context_length, None);
    assert_eq!(
        seer.capabilities,
        ["agent", "tools", "vision"],
        "tools from the explicit tag, vision from the projector"
    );
    assert_eq!(seer.quants.len(), 1);
    assert_eq!(seer.mmproj_models.len(), 1);
    assert_eq!(stats.rejected.get(&Reason::NoChatTemplate), Some(&1));
    assert!(
        fake.called("repo:").is_empty(),
        "the list's answer is the same projector summary, so asking again would not help"
    );
}

#[test]
fn a_paragraph_that_opens_three_readmes_is_a_blurb_and_the_model_is_described_by_its_base_instead()
{
    let blurb = "A highlights program for new and noteworthy models made by the community, curated every week.";
    let mut repos = vec![
        bare(
            "a/one",
            &["base_model:quantized:org/Base-1"],
            "llama",
            Some("{{ messages }}"),
        ),
        bare(
            "a/two",
            &["base_model:quantized:org/Base-2"],
            "llama",
            Some("{{ messages }}"),
        ),
        bare("a/three", &[], "llama", Some("{{ messages }}")),
        bare(
            "a/own",
            &["base_model:quantized:org/Base-4"],
            "llama",
            Some("{{ messages }}"),
        ),
    ];
    for repo in &mut repos {
        repo.downloads = 10;
    }
    let mut fake = synthetic(repos);
    for id in ["a/one", "a/two", "a/three"] {
        readme_for(&mut fake, id, blurb);
    }
    readme_for(
        &mut fake,
        "a/own",
        "A small model that was tuned for careful step by step answers to maths questions.",
    );

    let (catalog, _, _) = run(&fake, &Curated::default(), &Catalog::empty());

    assert_eq!(
        find(&catalog, "a/one").description,
        "Quantized build of org/Base-1."
    );
    assert_eq!(
        find(&catalog, "a/two").description,
        "Quantized build of org/Base-2."
    );
    assert_eq!(
        find(&catalog, "a/three").description,
        "",
        "no tag to name a base: nothing rather than a blurb"
    );
    assert_eq!(
        find(&catalog, "a/own").description,
        "A small model that was tuned for careful step by step answers to maths questions."
    );
}

#[test]
fn the_same_note_on_many_repositories_is_kept_because_many_quantizations_of_one_model_say_so() {
    let repos = (1..=4)
        .map(|n| {
            bare(
                &format!("q/quant{n}"),
                &["base_model:quantized:org/Base"],
                "llama",
                Some("{{ messages }}"),
            )
        })
        .collect();
    let fake = synthetic(repos);
    let (catalog, _, _) = run(&fake, &Curated::default(), &Catalog::empty());
    assert_eq!(catalog.models.len(), 4);
    assert!(
        catalog
            .models
            .iter()
            .all(|m| m.description == "Quantized build of org/Base.")
    );

    // Carried over from the last catalog there are no tags to rebuild a note from, so a note that
    // were taken for a blurb would be lost.
    let (again, stats, _) = run(&fake, &Curated::default(), &catalog);
    assert_eq!(stats.reused, 4);
    assert!(
        again
            .models
            .iter()
            .all(|m| m.description == "Quantized build of org/Base.")
    );
}

#[test]
fn a_model_kept_from_the_last_run_is_dropped_when_its_architecture_is_excluded_since() {
    let fake = Fake::recorded();
    let (before, _, _) = run(&fake, &Curated::default(), &Catalog::empty());
    fake.calls.borrow_mut().clear();
    let curated = Curated {
        exclude_architectures: vec!["gemma4".into()],
        ..Curated::default()
    };
    let (after, stats, _) = run(&fake, &curated, &before);
    assert!(!names(&after).contains(&GEMMA.to_string()));
    assert!(names(&after).contains(&QWEN.to_string()));
    assert!(
        stats
            .rejected
            .get(&Reason::ExcludedArchitecture)
            .copied()
            .unwrap_or(0)
            >= 1
    );
}
