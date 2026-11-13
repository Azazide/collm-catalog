// SPDX-License-Identifier: AGPL-3.0-only
//! What the generator asks Hugging Face, and the one place that asks it.
//!
//! [`Hf`] is the whole conversation, so a test answers it from files. [`HttpHf`] is the real one:
//! it paces itself by the `RateLimit` header Hugging Face sends (anonymous callers get 500 API
//! requests per 5 minutes), retries what is worth retrying and follows only links that stay on
//! `huggingface.co`.

use serde::Deserialize;
use std::fmt;
use std::io::Read;
use std::time::Duration;

#[derive(Debug, Clone, PartialEq)]
pub struct HfError(pub String);

impl fmt::Display for HfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for HfError {}

/// Whether anyone may download the repository's files.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum Gated {
    Flag(bool),
    /// `"auto"` or `"manual"`: a form has to be accepted first.
    Mode(String),
}

impl Default for Gated {
    fn default() -> Self {
        Gated::Flag(false)
    }
}

impl Gated {
    pub fn is_open(&self) -> bool {
        matches!(self, Gated::Flag(false))
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct GgufInfo {
    #[serde(default)]
    pub architecture: Option<String>,
    #[serde(default)]
    pub context_length: Option<u64>,
    /// Only the single-repository answer carries it; the list leaves it out.
    #[serde(default)]
    pub chat_template: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct CardData {
    #[serde(default)]
    pub license: Option<serde_json::Value>,
}

impl CardData {
    /// The card writes one licence as a string and several as a list.
    pub fn license(&self) -> Option<String> {
        match self.license.as_ref()? {
            serde_json::Value::String(s) if !s.is_empty() => Some(s.clone()),
            serde_json::Value::Array(items) => items
                .iter()
                .find_map(|v| v.as_str().filter(|s| !s.is_empty()).map(str::to_string)),
            _ => None,
        }
    }
}

/// What Hugging Face reads from a repository's `config.json` (asked for in the MLX list).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct RepoConfig {
    #[serde(default)]
    pub model_type: Option<String>,
}

/// A repository as the list and the single-repository answer both describe it.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Repo {
    pub id: String,
    #[serde(default)]
    pub downloads: u64,
    #[serde(default)]
    pub likes: u64,
    #[serde(default)]
    pub sha: Option<String>,
    #[serde(default, rename = "createdAt")]
    pub created_at: Option<String>,
    #[serde(default, rename = "lastModified")]
    pub last_modified: Option<String>,
    #[serde(default)]
    pub pipeline_tag: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub gated: Gated,
    #[serde(default)]
    pub disabled: bool,
    #[serde(default)]
    pub private: bool,
    #[serde(default)]
    pub gguf: Option<GgufInfo>,
    #[serde(default, rename = "cardData")]
    pub card_data: Option<CardData>,
    /// Only the MLX list asks for it.
    #[serde(default)]
    pub config: Option<RepoConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Lfs {
    pub oid: String,
    #[serde(default)]
    pub size: u64,
}

/// One entry of a repository's file tree.
#[derive(Debug, Clone, Deserialize)]
pub struct TreeEntry {
    #[serde(rename = "type")]
    pub kind: String,
    pub path: String,
    #[serde(default)]
    pub size: Option<u64>,
    /// Present for files kept in LFS, which every weights file is. Its `oid` is the SHA-256; the
    /// entry's own `oid` is the git object id and says nothing about the bytes.
    #[serde(default)]
    pub lfs: Option<Lfs>,
}

pub struct ListPage {
    pub repos: Vec<Repo>,
    /// Where the next page is, as the full address Hugging Face gave.
    pub next: Option<String>,
    /// Items the page held that could not be read; one odd entry does not lose the other 999.
    pub skipped: usize,
}

pub trait Hf {
    /// The first page of GGUF repositories by downloads, or the page `cursor` names.
    fn list_page(&self, cursor: Option<&str>) -> Result<ListPage, HfError>;
    /// One repository with its GGUF metadata (context length, chat template).
    fn repo(&self, id: &str) -> Result<Repo, HfError>;
    /// Every file of the repository at `rev`, folders included.
    fn tree(&self, id: &str, rev: &str) -> Result<Vec<TreeEntry>, HfError>;
    /// The start of the README at `rev`; `None` when there is none.
    fn readme(&self, id: &str, rev: &str) -> Result<Option<String>, HfError>;
    /// The first page of MLX chat repositories by downloads, or the page `cursor` names.
    fn mlx_list_page(&self, _cursor: Option<&str>) -> Result<ListPage, HfError> {
        Err(HfError("this source has no MLX list".into()))
    }
    /// One repository as the MLX list describes it.
    fn mlx_repo(&self, id: &str) -> Result<Repo, HfError> {
        Err(HfError(format!("this source has no MLX repository {id}")))
    }
    /// A small file of the repository at `rev` (a configuration, a tokenizer's text), whole: `None` when
    /// there is none, an error when it is larger than `limit` bytes.
    fn small_file(&self, id: &str, _rev: &str, path: &str, _limit: u64) -> Result<Option<Vec<u8>>, HfError> {
        Err(HfError(format!("this source has no file {path} of {id}")))
    }
}

const API: &str = "https://huggingface.co/api/models";
const EXPAND: &str = "expand[]=gguf&expand[]=cardData&expand[]=likes&expand[]=downloads\
&expand[]=createdAt&expand[]=lastModified&expand[]=sha&expand[]=pipeline_tag&expand[]=tags\
&expand[]=gated&expand[]=disabled&expand[]=private";
/// The MLX list also asks for what Hugging Face reads from `config.json` (the architecture).
const MLX_EXPAND: &str = "expand[]=config";
const PAGE_SIZE: usize = 1000;
/// A README is read for its first paragraph; more than this is never needed and never wanted.
const README_LIMIT: u64 = 200_000;
const BODY_LIMIT: u64 = 64 * 1024 * 1024;

/// What `ratelimit: "api";r=497;t=34` says: requests left in the window and seconds until it
/// resets.
pub fn parse_ratelimit(header: &str) -> Option<(u32, u64)> {
    let mut remaining = None;
    let mut reset = None;
    for part in header.split(';') {
        let part = part.trim();
        if let Some(v) = part.strip_prefix("r=") {
            remaining = v.parse().ok();
        } else if let Some(v) = part.strip_prefix("t=") {
            reset = v.parse().ok();
        }
    }
    Some((remaining?, reset?))
}

/// Segment by segment, the way a file path goes into an address.
pub fn encode_path(path: &str) -> String {
    let mut out = String::with_capacity(path.len());
    for byte in path.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' | b'/' => {
                out.push(byte as char)
            }
            other => out.push_str(&format!("%{other:02X}")),
        }
    }
    out
}

pub struct HttpHf {
    agent: ureq::Agent,
    token: Option<String>,
}

struct Reply {
    status: u16,
    link: Option<String>,
    body: Vec<u8>,
}

impl HttpHf {
    /// `token` is optional: a Hugging Face token only raises the rate limit.
    pub fn new(token: Option<String>) -> Self {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(120)))
            .http_status_as_error(false)
            .user_agent(concat!("collm-catalog/", env!("CARGO_PKG_VERSION")))
            .build()
            .into();
        HttpHf {
            agent,
            token: token.filter(|t| !t.trim().is_empty()),
        }
    }

    fn get(&self, url: &str, limit: u64) -> Result<Reply, HfError> {
        let mut last = String::new();
        for attempt in 0..6u32 {
            let mut request = self.agent.get(url);
            if let Some(token) = &self.token {
                request = request.header("Authorization", format!("Bearer {token}"));
            }
            match request.call() {
                Ok(mut response) => {
                    let status = response.status().as_u16();
                    let header = |name: &str| {
                        response
                            .headers()
                            .get(name)
                            .and_then(|v| v.to_str().ok())
                            .map(str::to_string)
                    };
                    let link = header("link");
                    let limits = header("ratelimit").and_then(|h| parse_ratelimit(&h));
                    let mut body = Vec::new();
                    response
                        .body_mut()
                        .as_reader()
                        .take(limit)
                        .read_to_end(&mut body)
                        .map_err(|e| HfError(format!("reading {url}: {e}")))?;
                    if status == 429 {
                        let wait = limits.map(|(_, t)| t).unwrap_or(60).clamp(1, 330) + 1;
                        eprintln!("  rate limited; waiting {wait}s");
                        std::thread::sleep(Duration::from_secs(wait));
                        last = "429 too many requests".to_string();
                        continue;
                    }
                    if status >= 500 {
                        last = format!("HTTP {status}");
                        std::thread::sleep(Duration::from_secs(2 << attempt.min(3)));
                        continue;
                    }
                    if let Some((remaining, reset)) = limits
                        && remaining <= 1
                    {
                        let wait = reset.clamp(1, 330) + 1;
                        eprintln!("  rate limit reached; waiting {wait}s");
                        std::thread::sleep(Duration::from_secs(wait));
                    }
                    return Ok(Reply { status, link, body });
                }
                Err(e) => {
                    last = e.to_string();
                    std::thread::sleep(Duration::from_secs(2 << attempt.min(3)));
                }
            }
        }
        Err(HfError(format!("{url}: {last}")))
    }

    fn json<T: for<'de> Deserialize<'de>>(&self, url: &str) -> Result<T, HfError> {
        let reply = self.get(url, BODY_LIMIT)?;
        if reply.status != 200 {
            return Err(HfError(format!("{url}: HTTP {}", reply.status)));
        }
        serde_json::from_slice(&reply.body).map_err(|e| HfError(format!("{url}: {e}")))
    }
}

/// The address after `rel="next"` in a `Link` header.
pub fn next_link(header: &str) -> Option<String> {
    header.split(',').find_map(|part| {
        let (url, rel) = part.split_once(';')?;
        if !rel.contains("rel=\"next\"") {
            return None;
        }
        Some(
            url.trim()
                .trim_start_matches('<')
                .trim_end_matches('>')
                .to_string(),
        )
    })
}

/// Only Hugging Face's own API is followed from a `Link`.
fn trusted_api_link(url: &str) -> bool {
    url.starts_with("https://huggingface.co/api/")
}

impl Hf for HttpHf {
    fn list_page(&self, cursor: Option<&str>) -> Result<ListPage, HfError> {
        self.page(
            cursor,
            &format!("{API}?filter=gguf&sort=downloads&direction=-1&limit={PAGE_SIZE}&{EXPAND}"),
        )
    }

    fn mlx_list_page(&self, cursor: Option<&str>) -> Result<ListPage, HfError> {
        self.page(
            cursor,
            &format!(
                "{API}?filter=mlx&pipeline_tag=text-generation&sort=downloads&direction=-1&limit={PAGE_SIZE}&{EXPAND}&{MLX_EXPAND}"
            ),
        )
    }

    fn mlx_repo(&self, id: &str) -> Result<Repo, HfError> {
        self.json(&format!("{API}/{}?{EXPAND}&{MLX_EXPAND}", encode_path(id)))
    }

    fn small_file(&self, id: &str, rev: &str, path: &str, limit: u64) -> Result<Option<Vec<u8>>, HfError> {
        let url = format!("https://huggingface.co/{}/resolve/{rev}/{}", encode_path(id), encode_path(path));
        // One byte more than allowed tells a file that is too large from one that is exactly the limit.
        let reply = self.get(&url, limit + 1)?;
        match reply.status {
            200 if reply.body.len() as u64 > limit => Err(HfError(format!("{url}: larger than {limit} bytes"))),
            200 => Ok(Some(reply.body)),
            404 => Ok(None),
            other => Err(HfError(format!("{url}: HTTP {other}"))),
        }
    }

    fn repo(&self, id: &str) -> Result<Repo, HfError> {
        self.json(&format!("{API}/{}?{EXPAND}", encode_path(id)))
    }

    fn tree(&self, id: &str, rev: &str) -> Result<Vec<TreeEntry>, HfError> {
        let mut entries = Vec::new();
        let mut url = format!("{API}/{}/tree/{rev}?recursive=true", encode_path(id));
        for _ in 0..50 {
            let reply = self.get(&url, BODY_LIMIT)?;
            if reply.status != 200 {
                return Err(HfError(format!("{url}: HTTP {}", reply.status)));
            }
            let mut page: Vec<TreeEntry> =
                serde_json::from_slice(&reply.body).map_err(|e| HfError(format!("{url}: {e}")))?;
            entries.append(&mut page);
            match reply.link.as_deref().and_then(next_link) {
                Some(next) if trusted_api_link(&next) => url = next,
                _ => return Ok(entries),
            }
        }
        Err(HfError(format!(
            "{id}: the file tree has more than 50 pages"
        )))
    }

    fn readme(&self, id: &str, rev: &str) -> Result<Option<String>, HfError> {
        let url = format!(
            "https://huggingface.co/{}/resolve/{rev}/README.md",
            encode_path(id)
        );
        let reply = self.get(&url, README_LIMIT)?;
        match reply.status {
            200 => Ok(Some(String::from_utf8_lossy(&reply.body).into_owned())),
            404 => Ok(None),
            other => Err(HfError(format!("{url}: HTTP {other}"))),
        }
    }
}

impl HttpHf {
    /// A page of a list: the first one at `first`, or the one `cursor` names (on Hugging Face only).
    fn page(&self, cursor: Option<&str>, first: &str) -> Result<ListPage, HfError> {
        let url = match cursor {
            Some(next) if trusted_api_link(next) => next.to_string(),
            Some(other) => return Err(HfError(format!("not following {other}"))),
            None => first.to_string(),
        };
        let reply = self.get(&url, BODY_LIMIT)?;
        if reply.status != 200 {
            return Err(HfError(format!("{url}: HTTP {}", reply.status)));
        }
        let items: Vec<serde_json::Value> =
            serde_json::from_slice(&reply.body).map_err(|e| HfError(format!("{url}: {e}")))?;
        let mut repos = Vec::with_capacity(items.len());
        let mut skipped = 0;
        for item in items {
            match serde_json::from_value::<Repo>(item) {
                Ok(repo) => repos.push(repo),
                Err(_) => skipped += 1,
            }
        }
        Ok(ListPage {
            repos,
            next: reply.link.as_deref().and_then(next_link),
            skipped,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_rate_limit_header_gives_what_is_left_and_when_it_resets() {
        assert_eq!(parse_ratelimit("\"api\";r=497;t=34"), Some((497, 34)));
        assert_eq!(parse_ratelimit("\"resolvers\";r=0;t=300"), Some((0, 300)));
        assert_eq!(parse_ratelimit("\"api\";r=5"), None);
        assert_eq!(parse_ratelimit("nonsense"), None);
    }

    #[test]
    fn a_file_path_is_encoded_by_segment_and_keeps_its_slashes() {
        assert_eq!(encode_path("MTP/mtp-x-Q8_0.gguf"), "MTP/mtp-x-Q8_0.gguf");
        assert_eq!(encode_path("a b/c+d.gguf"), "a%20b/c%2Bd.gguf");
        assert_eq!(
            encode_path("модель.gguf"),
            "%D0%BC%D0%BE%D0%B4%D0%B5%D0%BB%D1%8C.gguf"
        );
    }

    #[test]
    fn the_next_page_is_read_from_the_link_header_and_only_huggingface_is_followed() {
        let header = "<https://huggingface.co/api/models?limit=1000&cursor=abc%3D>; rel=\"next\"";
        let next = next_link(header).unwrap();
        assert_eq!(
            next,
            "https://huggingface.co/api/models?limit=1000&cursor=abc%3D"
        );
        assert!(trusted_api_link(&next));
        assert!(!trusted_api_link(
            "https://evil.example/api/models?cursor=1"
        ));
        assert!(!trusted_api_link("http://huggingface.co/api/models"));
        assert!(!trusted_api_link(
            "https://huggingface.co.evil.example/api/x"
        ));
        assert_eq!(
            next_link("<https://huggingface.co/api/models>; rel=\"prev\""),
            None
        );
    }

    #[test]
    fn a_gated_repository_is_not_open_whatever_way_the_answer_spells_it() {
        let open: Repo = serde_json::from_str(r#"{"id":"a/b","gated":false}"#).unwrap();
        let auto: Repo = serde_json::from_str(r#"{"id":"a/b","gated":"auto"}"#).unwrap();
        let manual: Repo = serde_json::from_str(r#"{"id":"a/b","gated":"manual"}"#).unwrap();
        let flagged: Repo = serde_json::from_str(r#"{"id":"a/b","gated":true}"#).unwrap();
        let unsaid: Repo = serde_json::from_str(r#"{"id":"a/b"}"#).unwrap();
        assert!(open.gated.is_open() && unsaid.gated.is_open());
        assert!(!auto.gated.is_open() && !manual.gated.is_open() && !flagged.gated.is_open());
    }

    #[test]
    fn the_licence_is_a_string_or_the_first_of_a_list() {
        let one: CardData = serde_json::from_str(r#"{"license":"mit"}"#).unwrap();
        let many: CardData = serde_json::from_str(r#"{"license":["apache-2.0","mit"]}"#).unwrap();
        let none: CardData = serde_json::from_str(r#"{"license":""}"#).unwrap();
        let odd: CardData = serde_json::from_str(r#"{"license":7}"#).unwrap();
        assert_eq!(one.license().as_deref(), Some("mit"));
        assert_eq!(many.license().as_deref(), Some("apache-2.0"));
        assert_eq!(none.license(), None);
        assert_eq!(odd.license(), None);
    }
}
