// SPDX-License-Identifier: AGPL-3.0-only
//! The start of a GGUF file: its metadata keys, read from the first bytes of the file (a Range request), never the weights.
//!
//! Hugging Face's summary of a repository describes one file of it. Files of one repository can differ, and the
//! summary cannot tell: a file made before a model's architecture was reworked has the same name scheme but a layout
//! the current llama.cpp does not load.

use std::collections::HashSet;

/// How much of a file's start is read to see its metadata keys. The architecture keys come before the tokenizer's long
/// lists, so these bytes hold them (measured on the files this is used for).
pub const HEAD_BYTES: u64 = 256 * 1024;

/// What the start of a GGUF file says.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Head {
    /// `general.architecture`, when it was within the bytes read.
    pub architecture: Option<String>,
    /// Every metadata key read, in full (`inkling2.num_loops`).
    pub keys: HashSet<String>,
}

struct Reader<'a> {
    buf: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.at.checked_add(n)?;
        let part = self.buf.get(self.at..end)?;
        self.at = end;
        Some(part)
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
    fn string(&mut self) -> Option<String> {
        let n = usize::try_from(self.u64()?).ok()?;
        Some(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    /// Reads a value of GGUF type `t`; its text when it is a string, `Some(None)` for any other.
    fn value(&mut self, t: u32) -> Option<Option<String>> {
        let fixed = match t {
            0 | 1 | 7 => 1,
            2 | 3 => 2,
            4..=6 => 4,
            10..=12 => 8,
            8 => return self.string().map(Some),
            9 => {
                let inner = self.u32()?;
                let count = usize::try_from(self.u64()?).ok()?;
                if let Some(size) = match inner {
                    0 | 1 | 7 => Some(1),
                    2 | 3 => Some(2),
                    4..=6 => Some(4),
                    10..=12 => Some(8),
                    _ => None,
                } {
                    self.take(count.checked_mul(size)?)?;
                } else {
                    for _ in 0..count {
                        self.value(inner)?;
                    }
                }
                return Some(None);
            }
            _ => return None,
        };
        self.take(fixed)?;
        Some(None)
    }
}

/// The keys at the start of a GGUF file. `None` when the bytes are not a GGUF file; the keys read before the bytes ran
/// out when they are cut short.
pub fn read_head(bytes: &[u8]) -> Option<Head> {
    let mut r = Reader { buf: bytes, at: 0 };
    if r.take(4)? != b"GGUF" {
        return None;
    }
    r.u32()?;
    r.u64()?;
    let count = r.u64()?;
    let mut head = Head::default();
    for _ in 0..count {
        let Some(key) = r.string() else { break };
        let Some(kind) = r.u32() else { break };
        let Some(value) = r.value(kind) else { break };
        if key == "general.architecture" {
            head.architecture = value;
        }
        head.keys.insert(key);
    }
    Some(head)
}

/// Whether a file is an Inkling 2 Mini made before the model was reworked as `inkling2_loop`: the header says
/// `inkling2` with `block_count` 48 and a `num_loops`, but the file holds the 24 blocks of the reworked model, so the
/// current llama.cpp fails to load it (`tensor 'blk.24.attn_q.weight' not found`). The reworked files say `inkling2_loop`.
pub fn is_old_inkling_layout(head: &Head) -> bool {
    head.architecture.as_deref() == Some("inkling2") && head.keys.contains("inkling2.num_loops")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn string(s: &str) -> Vec<u8> {
        let mut v = (s.len() as u64).to_le_bytes().to_vec();
        v.extend(s.as_bytes());
        v
    }

    fn file(arch: &str, extra: &[(&str, u32, Vec<u8>)]) -> Vec<u8> {
        let mut pairs: Vec<(String, u32, Vec<u8>)> = vec![("general.architecture".into(), 8, string(arch))];
        pairs.extend(extra.iter().map(|(k, t, v)| (k.to_string(), *t, v.clone())));
        let mut out = b"GGUF".to_vec();
        out.extend(3u32.to_le_bytes());
        out.extend(0u64.to_le_bytes());
        out.extend((pairs.len() as u64).to_le_bytes());
        for (k, t, v) in pairs {
            out.extend(string(&k));
            out.extend(t.to_le_bytes());
            out.extend(v);
        }
        out
    }

    #[test]
    fn the_old_layout_is_inkling2_with_loops() {
        let old = file("inkling2", &[("inkling2.block_count", 4, 48u32.to_le_bytes().to_vec()), ("inkling2.num_loops", 4, 3u32.to_le_bytes().to_vec())]);
        let head = read_head(&old).unwrap();
        assert_eq!(head.architecture.as_deref(), Some("inkling2"));
        assert!(is_old_inkling_layout(&head));
    }

    #[test]
    fn the_reworked_layout_and_other_models_are_not_old() {
        let new = file("inkling2_loop", &[("inkling2_loop.block_count", 4, 24u32.to_le_bytes().to_vec())]);
        assert!(!is_old_inkling_layout(&read_head(&new).unwrap()));
        let plain = file("inkling2", &[]);
        assert!(!is_old_inkling_layout(&read_head(&plain).unwrap()), "no loops key: not the old layout");
        let other = file("llama", &[("inkling2.num_loops", 4, 3u32.to_le_bytes().to_vec())]);
        assert!(!is_old_inkling_layout(&read_head(&other).unwrap()));
    }

    #[test]
    fn arrays_are_skipped_and_cut_bytes_keep_what_was_read() {
        let mut array = 8u32.to_le_bytes().to_vec(); // array of strings
        array.extend(2u64.to_le_bytes());
        array.extend(string("a"));
        array.extend(string("b"));
        let whole = file("inkling2", &[("tokenizer.ggml.tokens", 9, array), ("inkling2.num_loops", 4, 3u32.to_le_bytes().to_vec())]);
        assert!(is_old_inkling_layout(&read_head(&whole).unwrap()));
        // Cut inside the array: the architecture was read, the later key was not.
        let cut = &whole[..whole.len() - 20];
        let head = read_head(cut).unwrap();
        assert_eq!(head.architecture.as_deref(), Some("inkling2"));
        assert!(!head.keys.contains("inkling2.num_loops"));
    }

    #[test]
    fn something_that_is_not_gguf_is_none() {
        assert_eq!(read_head(b"<html>"), None);
        assert_eq!(read_head(b""), None);
    }
}
