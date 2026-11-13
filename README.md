# collm-catalog

Two JSON files that list local LLMs worth downloading, built from [Hugging Face](https://huggingface.co) and refreshed every night.

| File | For | Models |
|---|---|---|
| [`catalog.json`](catalog.json) | GGUF models, run by [llama.cpp](https://github.com/ggml-org/llama.cpp) | 390 |
| [`mlx.json`](mlx.json) | MLX models, run on Apple Silicon | 1125 |

Both are public data. Read them straight from GitHub:

```
https://raw.githubusercontent.com/Azazide/collm-catalog/main/catalog.json
https://raw.githubusercontent.com/Azazide/collm-catalog/main/mlx.json
```

A desktop app reads them, and each build of it reads only the file for its own engine. Nothing stops you from using them for anything else.

## What you get

Every file looks the same: a small header, then one model per line, sorted by owner and name, so a night's change reads as a clean diff.

```json
{"format":1,"rules":3,"models":[
{"model_name":"owner/name", …},
…
]}
```

- `format` changes when a field is renamed or changes meaning. A reader that does not know it should refuse the file.
- `rules` is the version of the selection rules the entries were made under. You can ignore it.
- `mlx.json` also has `"engine":"mlx"`, so a reader can tell the two files apart.
- Every download address is pinned to a commit (`…/resolve/<40-digit sha>/<file>`), so a later push to a repository cannot break a download.
- Every file has its size in bytes and its SHA-256, so a download can be checked. A file with no known SHA-256 is not offered.

### `catalog.json` (GGUF)

Chat models only, at most 300 by downloads over the last 30 days, plus a hand-picked list.

```json
{
  "model_name": "agentionai/Qwen3.8-27B-AP-GGUF",
  "developer": "agentionai",
  "sha": "2c387d95ea40…",
  "description": "Every 27B quant gives something up. …",
  "downloads": 924346,
  "likes": 103,
  "license": "apache-2.0",
  "pipeline_tag": "image-text-to-text",
  "architecture": "qwen35",
  "context_length": 262144,
  "capabilities": ["agent", "tools", "vision"],
  "readme": "…",
  "quants": [
    { "model_id": "Qwen3.8-27B-AP-IQ2_S",
      "path": "https://huggingface.co/agentionai/Qwen3.8-27B-AP-GGUF/resolve/2c387d95ea40…/Qwen3.8-27B-AP-IQ2_S.gguf",
      "size": 9607980896, "sha256": "daf741050fd1…" }
  ],
  "mmproj_models": [ { "model_id": "mmproj-BF16", … } ],
  "drafts": []
}
```

| Field | Meaning |
|---|---|
| `model_name`, `developer` | the repository as `owner/name`, and its owner |
| `sha` | the commit every address points at |
| `description` | the first paragraph of the README that reads as a real description, or a note on what the model was made from; left out if neither exists |
| `base_model` | the repository it was made from (`owner/name`), taken from its `base_model` tags; left out if there is none |
| `downloads`, `likes` | as Hugging Face counts them (downloads over 30 days) |
| `created_at`, `updated_at`, `license`, `pipeline_tag` | from the repository; left out if missing |
| `architecture`, `context_length` | from the GGUF header |
| `capabilities` | see below |
| `readme` | the README at `sha`; left out if there is none |
| `quants` | every whole weights file that runs on its own, smallest first: `model_id`, `path`, `size`, `sha256` |
| `mmproj_models` | vision projector files, same shape |
| `drafts` | small draft files for speculative decoding (`mtp`, `dspark`, `dflash`, `eagle3`), same shape; left out if none |

### `mlx.json` (MLX)

An MLX model is a folder, not one file, so an entry lists every file of it.

```json
{
  "model_name": "2001Y/Ternary-Bonsai-2-27B-Abliterated-v2-PQ2_0-MTP-MLX",
  "developer": "2001Y",
  "sha": "ea576e1e005b…",
  "architecture": "qwen3_5",
  "bits": 2,
  "capabilities": ["tools"],
  "size": 8998833332,
  "files": [
    { "name": "config.json",
      "path": "https://huggingface.co/2001Y/…/resolve/ea576e1e005b…/config.json",
      "size": 4062, "sha256": "95a1ef8781dc…" },
    …
  ]
}
```

It has the same fields as above (`description`, `downloads`, `likes`, `base_model`, `readme` and so on), except `quants`, `mmproj_models` and `drafts`, and adds:

| Field | Meaning |
|---|---|
| `bits` | bit width of quantized weights; left out for unquantized ones |
| `size` | the whole folder, in bytes |
| `files` | every file of the folder: `name`, `path`, `size`, `sha256` |

## Capabilities

`capabilities` is a sorted list. Each entry comes from something checkable in the repository, never from a guess about quality. Every model is a chat model, so that has no entry.

- `vision`: it has a vision projector (`mmproj-…`) for GGUF, or a vision tower and an image processor in the folder for MLX.
- `audio`: GGUF only. It has a projector and is labelled as taking sound (`any-to-any`, `audio-text-to-text`, an `audio`, `speech` or `omni` tag, or such a word in its name).
- `video`: GGUF only. It has a projector and is labelled as taking video (`video-text-to-text`, a `video` or `omni` tag, or such a word in its name).
- `tools`: its chat template uses a list of tools. When the template is unknown, a tag such as `function-calling` or `tool-use` stands in.
- `code`: made for programming, by its name, tags or the model it was made from (`coder`, `devstral`, …).
- `agent`: put forward for agents by its tags, its name or its README, and its template takes tools. This is narrower than `tools`.

The MLX engine takes text and pictures only, so `mlx.json` never has `audio` or `video`.

## What gets in

For GGUF, a repository is listed when:

- anyone can download it, and Hugging Face does not label it as speech, image, video or another non-chat kind;
- it has a chat template;
- it is not a big-endian build, and at least one whole weights file runs on its own (files cut into `-00001-of-0000N` parts do not count);
- it is not an embedding, reranker, LoRA, Whisper or VAE repository, or a non-language architecture.

Adapters, vocabularies, importance matrices and other small files kept beside the weights are not offered as models.

For MLX, a repository is listed when it is open to everyone, tagged `mlx`, has a chat template, a whole folder (config, tokenizer, every weights part), and an architecture and quantization the engine can load.

The choices a person makes live in [`curated.toml`](curated.toml) (GGUF) and [`mlx.toml`](mlx.toml) (MLX): size limits, always-listed repositories, and what to keep out. An unknown key in either is an error. The MLX list also adds each chosen model's other quantizations by the same owner (`-4bit`, `-8bit`, `-bf16`, …), so one model comes with all its variants.

## Run the generator

You need a recent Rust toolchain.

```bash
cargo run --release -- --out catalog.json                  # GGUF
cargo run --release -- --engine mlx --out mlx.json         # MLX
```

| Flag | Meaning |
|---|---|
| `--engine gguf\|mlx` | which catalog to build (default `gguf`) |
| `--out FILE` | where to write it (default `catalog.json`, or `mlx.json` for MLX); also read as the last catalog |
| `--curated FILE` | the choices file (default `curated.toml`, or `mlx.toml` for MLX) |
| `--refresh` | read every repository again instead of reusing unchanged ones |
| `--force` | write the file even if it shrank sharply |
| `HF_TOKEN` | optional environment variable; a Hugging Face token only raises the rate limit |

Each run starts from the last file and rereads only repositories whose commit changed. A change of `rules` rereads everything. Before writing, every entry is checked the way a download would meet it, and a bad one is dropped with a reason.

Exit codes: `0` written, `1` failed (file untouched), `2` the new catalog is under 70% of the last one, or too small (under 200 models for GGUF, 100 for MLX), so it is taken for a failed run (file untouched).

A GitHub workflow ([`update.yml`](.github/workflows/update.yml)) runs the tests, builds both files every night and commits the change. It rereads everything on Sundays and on request.

```bash
cargo test
```

The tests use answers recorded from Hugging Face in `tests/fixtures`; none touch the network.

## License

[AGPL-3.0](LICENSE). The models the files list belong to their authors and keep the license each one publishes; every entry carries it as `license`.
