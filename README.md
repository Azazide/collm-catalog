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
- at least one weights file is whole and runs on its own. A file cut into `-00001-of-00003` parts is
  skipped, and a repository that has only those is left out; so is one whose only whole files are
  drafts (below).

`curated.toml` holds the choices a person makes: the size limit, repositories that are always listed,
and owners, names and architectures that never are. An unknown key in it is an error.

## The file

```json
{"format":1,"models":[
{"model_name":"gemma-4-12b-it-GGUF","developer":"unsloth","sha":"fc034cf…","description":"…", …},
…
]}
```

One model per line, sorted by owner and name, so a day's change reads as a diff. `format` changes
when a field is renamed or changes meaning; a reader that does not know it should refuse the file.

| field | |
|---|---|
| `model_name`, `developer` | the repository as `owner/name`, which two owners cannot share, and its owner |
| `sha` | the commit every address below points at; a later push to the repository cannot make a download fail its hash |
| `description` | the first paragraph of the README that reads as a statement about a model, not a heading, list, table, bar of links, slogan or a blurb that opens the README of three or more repositories; failing that, which model it was made from; left out if neither |
| `downloads`, `likes` | as Hugging Face counts them (downloads over 30 days) |
| `created_at`, `updated_at`, `license`, `pipeline_tag` | from the repository, left out when it has none |
| `architecture`, `context_length` | from the GGUF header; left out when Hugging Face read a projector's instead |
| `capabilities` | what the model can do, sorted: `vision`, `tools`, `code`, `agent` (see below) |
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
| `tools` | its chat template takes a list of tools; when the template is unknown, an explicit tag (`function-calling`, `tool-use`) stands in |
| `code` | is made for programming: a whole word of its name (`coder`, `code`, `devstral`, not `encoder`), its tags, or the name of the model it was made from say so |
| `agent` | is put forward for agents by its tags, its name, or three mentions of agents and tool calling in its README, and its template, when known, takes tools |

Most models take tools, so `agent` is the narrower claim: the author says it is meant for agents. "Agent
and coding" is `agent` and `code` together.

`size` is in bytes. `sha256` is the file's SHA-256 as Hugging Face's LFS states it; it is left out for a
file that is not in LFS. `model_id` is the path in the repository without `.gguf`.

## Drafts

A draft is a small file for speculative decoding that pairs with a model and runs with no model of its
own. It is told from a model by its name and its size: a file in a folder of the kind's name (`MTP/…`) or
beginning with it (`mtp-…`), with `mtp`, `dspark`, `dflash` or `eagle3` as a whole word; or a file with the
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
of the last one's models, so it is taken for a failed run and the file is untouched.

Every run starts from the last `catalog.json`. A repository whose commit has not changed is carried
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
