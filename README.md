# collm-catalog

The list of models [Collm](https://github.com/Azazide/Collm) offers for download, as one file:
[`catalog.json`](catalog.json). A program in this repository builds it from the
[Hugging Face](https://huggingface.co) API and a workflow runs it every night. Collm reads the file
from

```
https://raw.githubusercontent.com/Azazide/collm-catalog/main/catalog.json
```

Nothing else is needed to use the catalog: it is public data, the program is only how it is made.

## What is in the catalog

GGUF chat models, at most 300, taken from Hugging Face by downloads over the last 30 days. A
repository is listed when

- anyone can download it (not gated, not private, not disabled);
- Hugging Face does not label it as speech, image, video or another non-chat kind;
- its GGUF metadata carries a chat template (a model with no template is not a chat model). Hugging
  Face reads that metadata from the repository's first file alphabetically, which is often a vision
  projector (`mmproj-…`, architecture `clip`); then the metadata says nothing of the model and only a
  chat label such as `text-generation` or `image-text-to-text` vouches for it;
- not a build for a big-endian machine: `BE` as a word of its name and a README that says its files are
  byte-swapped or for IBM Z (those do not load on a Mac);
- at least one weights file is whole and runs on its own. A file cut into `-00001-of-00003` parts is
  skipped, and a repository that has only those is left out; so is one whose only whole files are
  drafts (below).

`curated.toml` holds the choices a person makes: the size limit, repositories that are always listed
(on top of the limit, so a pinned model never takes the place of one that is not),
and owners, names and architectures that never are. An unknown key in it is an error.

## The file

```json
{"format":1,"rules":3,"models":[
{"model_name":"gemma-4-12b-it-GGUF","developer":"unsloth","sha":"fc034cf…","description":"…", …},
…
]}
```

One model per line, sorted by owner and name, so a day's change reads as a diff. `format` changes
when a field is renamed or changes meaning; a reader that does not know it should refuse the file.
`rules` is the version of the rules the entries were made under (what counts as a model, a draft, a
description); the program reads every repository again when it differs, and a reader may ignore it.

| field | |
|---|---|
| `model_name`, `developer` | the repository as `owner/name`, which two owners cannot share, and its owner |
| `sha` | the commit every address below points at; a later push to the repository cannot make a download fail its hash |
| `description` | the first paragraph of the README that reads as a statement about a model, not a heading, list, table, bar of links, slogan or a blurb that opens the README of three or more repositories; failing that, which model it was made from; left out if neither |
| `downloads`, `likes` | as Hugging Face counts them (downloads over 30 days) |
| `created_at`, `updated_at`, `license`, `pipeline_tag` | from the repository, left out when it has none |
| `architecture`, `context_length` | from the GGUF header; empty (`architecture`) and left out (`context_length`) when Hugging Face read a projector's instead |
| `capabilities` | what the model can do, sorted: `vision`, `audio`, `video`, `tools`, `code`, `agent` (see below) |
| `readme` | the README at `sha`, left out when there is none |
| `quants` | every whole weights file that runs on its own, smallest first: `{model_id, path, size, sha256}` |
| `drafts` | the drafts for speculative decoding (`mtp`, `dspark`, `dflash`, `eagle3`) that pair with a model, the same shape; left out when there are none (see below) |
| `mmproj_models` | vision projectors, the same shape |

## Capabilities

For the app to filter and sort by. Each comes from something checkable in the repository, never from a
guess about quality; every model is a chat model, so that has no entry.

| | the repository |
|---|---|
| `vision` | has a vision projector (`mmproj-…`), the file llama.cpp needs to read an image. A label saying "image-text-to-text" without one does not count |
| `audio` | takes sound in: has a vision projector (llama.cpp reads sound through the same file as an image) and Hugging Face labels it so: the pipeline `any-to-any` or `audio-text-to-text`, a tag `audio`, `speech` or `omni`, or `audio`, `omni`, `voxtral` or `ultravox` as a whole word of its name or of the model it was made from |
| `video` | takes moving pictures in: has a vision projector and is labelled `video-text-to-text`, tagged `video` or `omni`, or has `video` or `omni` as a whole word of its name or of its base model's |
| `tools` | its chat template uses a list of tools, `tools` as a whole word inside a `{{ … }}` or `{% … %}` (not in a comment, nor in text that says the model takes none); when the template is unknown, an explicit tag (`function-calling`, `tool-use`) stands in |
| `code` | is made for programming: a whole word of its name (`coder`, `code`, `devstral`, not `encoder`), its tags, or the name of the model it was made from say so |
| `agent` | is put forward for agents by its tags, its name, or three mentions of agents and tool calling in its README, and its template, when known, takes tools |

Most models take tools, so `agent` is the narrower claim: the author says it is meant for agents. "Agent
and coding" is `agent` and `code` together.

`size` is in bytes. `sha256` is the file's SHA-256 as Hugging Face's LFS states it; a file that is not in
LFS, or whose hash is not 64 hex digits, is not offered, since it could not be checked after the download.
`model_id` is the path in the repository without `.gguf`. Every `path` is
`https://huggingface.co/<owner>/<name>/resolve/<commit>/<file>`, with no `.` or `..` segment.

## Files that are not models

A repository keeps files beside its models, and none of them is offered as a model: a file under
50 MB (the smallest chat model of the catalog is 149 MB); and a file with `imatrix`, `lora`, `vocab`,
`tokenizer`, `tts` or `token2wav` as a whole word in its path that is under a tenth of the largest weights
file of the repository, the parts of a cut file counted (importance data of a quantization, an adapter,
a vocabulary, the parts of a speech model). Counting the parts matters: a repository whose real weights
are all parts has no whole file but its importance matrix, and that is not offered as a model. A model that only carries `imatrix` in its name, at the size of a model, is a model. A repository
left with no model file is left out.

## Drafts

A draft is a small file for speculative decoding that pairs with a model and runs with no model of its
own. It is told from a model by its name and its size: a file in a folder of the kind's name (`MTP/…`) or
beginning with it (`mtp-…`), with `mtp`, `dspark`, `dflash` or `eagle3` as a whole word, unless it
weighs half of the file it shares its name with, or more (`mtp/Model-MTP-Q4_K_M` beside `Model-Q4_K_M`:
a whole model that is only filed under the name); or a file with the
word elsewhere in its name that is under a tenth of the file it is named after (`Model-Q8_0-MTP` beside
`Model-Q8_0`), or of the largest file when no file is. Most files that carry the word elsewhere are models
made with the head kept in them: they weigh what the same model weighs, and they are models.

## Running it

```bash
cargo run --release -- --out catalog.json
```

| | |
|---|---|
| `--out PATH` | where the catalog is written (default `catalog.json`); also read as the last catalog |
| `--curated PATH` | the choices file (default `curated.toml`, or the defaults when there is none) |
| `--force` | write the catalog even if it is far smaller than the last |
| `--refresh` | read every repository again instead of carrying over the unchanged ones; needed after the program's rules change, since a carried-over entry is not redone |
| `HF_TOKEN` | optional; a Hugging Face token only raises the rate limit |

Exit code 0: written. 1: the run failed and the file is untouched. 2: the new catalog has fewer than 70%
of the last one's models, or fewer than 200, so it is taken for a failed run and the file is untouched.
(The first guard alone lets a catalog lose a third of itself a night, five nights running.)

Before the file is written, every model is checked as a download would meet it: a name that is
`owner/name`, a commit that is a 40-digit hash, a file with a size and a SHA-256 and an address under
this repository and commit, a README that is this repository's at this commit, no name listed twice.
A model that fails is left out and the run says why; it never stops the others.

Every run starts from the last `catalog.json`, unless that was made under other `rules`. A repository whose commit has not changed is carried
over without a request; only new and changed ones are read in full (the file tree, the README). A
failure on one repository keeps its last entry; a failure on the list ends the run, so a half-read list
never becomes a catalog. Hugging Face allows an anonymous caller 500 API requests per 5 minutes; the
program reads the `RateLimit` header and waits when the window is used up.

```bash
cargo test
```

The tests run on answers recorded from Hugging Face (`tests/fixtures`), none on the network.

## Licence

[GNU Affero General Public License v3.0](LICENSE). The models the catalog names belong to their authors
and are used under the licence each publishes; the catalog gives each one's `license` field as the
repository states it.

`tests/fixtures` holds answers recorded from the Hugging Face API and the start of four READMEs of
Apache-2.0 and MIT licensed repositories, kept as they are so the tests run on real data.
