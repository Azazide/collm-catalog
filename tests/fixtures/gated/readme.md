---
license: apache-2.0
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: gguf
language:
  - en
  - zh
tags:
  - abliterated
  - uncensored
  - qwen
  - qwen3.8
  - qwen4
  - flash-next
  - moe
  - gguf
  - llama.cpp
  - vision-language
  - mmproj
  - function-calling
  - reasoning
  - ai-red-team
  - red-teaming
---

<div align="center">

<a href="https://www.orcarouter.ai" target="_blank">
  <img src="https://www.orcarouter.ai/orca-logo-classic.png" alt="OrcaRouter" width="110">
</a>

<h1>Qwen3.8-Flash-Next-Uncensored-GGUF</h1>

<p><em>GGUF quants (2-bit &rarr; 8-bit) of the abliterated (refusal-removed) Qwen3.8-Flash-Next &mdash; for llama.cpp</em></p>


<!-- orcacyber-promo:start -->
<table>
<tr>
<td width="150" style="vertical-align: middle;">
<div align="center"><a href="https://www.orcarouter.ai/models/orca/orcacyber-zero-1.5"><img src="https://www.orcarouter.ai/orca-logo-classic.png" width="90" alt="OrcaCyber Zero 1.5"></a></div>
</td>
<td>
<p>🛡️ <strong>New from Orca &mdash; <a href="https://www.orcarouter.ai/models/orca/orcacyber-zero-1.5">OrcaCyber Zero 1.5</a></strong></p>
<p>Post-trained for cybersecurity: vulnerability analysis, vulnerability reproduction, red teaming &amp; offensive-security research. <strong>98.01% pass@1 on CyberGym Level&nbsp;1</strong> &middot; 1M context &middot; function calling &middot; OpenAI-compatible API.</p>
<p><a href="https://www.orcarouter.ai/models/orca/orcacyber-zero-1.5"><img src="https://img.shields.io/badge/Try%20it%20on%20OrcaRouter%20%E2%86%92-FF6A00?style=for-the-badge&amp;labelColor=0B1F3A" alt="Try it on OrcaRouter"></a></p>
</td>
</tr>
<tr>
<td colspan="2">
<p>🔌 <strong>New from Orca &mdash; <a href="https://www.orcarouter.ai/solutions/orca-connect">Orca Connect</a></strong></p>
<p>Running this GGUF in llama.cpp? <strong>Orca Connect</strong> puts the rig under your desk behind your usual OrcaRouter API key, next to 200+ hosted models, reachable from a laptop, a phone or a cloud agent. Outbound only: no port forwarding, no tunnel, no firewall rule. <strong>Requests your own hardware serves cost $0</strong>, and <code>orca-connect claude</code> points Claude Code at your rig for one session.</p>
<video src="https://huggingface.co/spaces/orcarouter/README/resolve/main/assets/orca-connect.mp4" controls autoplay muted loop playsinline width="100%"></video>
<p><a href="https://www.orcarouter.ai/solutions/orca-connect"><img src="https://img.shields.io/badge/Connect%20your%20rig%20%E2%86%92-FF6A00?style=for-the-badge&amp;labelColor=0B1F3A" alt="Connect your rig"></a></p>
</td>
</tr>
</table>
<!-- orcacyber-promo:end -->

<p>
<a href="https://www.orcarouter.ai"><img src="https://img.shields.io/badge/Website-orcarouter.ai-1E6FEB" alt="Website"></a>
<a href="https://www.orcarouter.ai/models"><img src="https://img.shields.io/badge/OrcaRouter-Model%20Catalog-2EA043" alt="Model Catalog"></a>
<a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-4C8BF5" alt="License"></a>
<img src="https://img.shields.io/badge/Format-GGUF-00A67E" alt="GGUF">
<img src="https://img.shields.io/badge/Quants-Q2__K%20%E2%86%92%20Q8__0-FF8800" alt="Quants">
<img src="https://img.shields.io/badge/Vision-mmproj-9B59B6" alt="Vision">
<img src="https://img.shields.io/badge/Arch-Qwen4%20MoE-E67E22" alt="Qwen4 MoE">
</p>

<p><strong>One Gateway. Every Model.</strong> &mdash; Route Smarter &middot; Ship Safer &middot; Spend Less.</p>

<p>
<a href="https://www.orcarouter.ai">Website</a> &middot;
<a href="https://www.orcarouter.ai/models">Model Catalog</a> &middot;
<a href="https://github.com/Continuum-AI-Corp">GitHub</a> &middot;
<a href="https://ollama.com/orcarouter">Ollama</a> &middot;
<a href="https://discord.gg/yAh6Tex6kx">Discord</a> &middot;
<a href="https://x.com/OrcaRouter">X</a>
</p>

</div>

---

> **GGUF conversions** of the **abliterated** (refusal-removed) build of Qwen's
> [`Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) &mdash; a large **Mixture-of-Experts**
> (512 experts, 10 routed + 1 shared active) preview of the **Qwen4 architecture** (`qwen4_exp`):
> **Gated DeltaNet** linear attention + **Qwen Sparse Attention (QSA)** at the micro-block level,
> **HyperConnections** in place of layer norms, **PLE n-gram** hash embeddings, native
> **vision-language**, reasoning, and tool-calling. These files run in **llama.cpp** (CPU / CUDA /
> Metal / ROCm), quantized from **2-bit to 8-bit**, with a separate **mmproj** file that
> restores **vision**. Browse all models in the [OrcaRouter Model Catalog](https://www.orcarouter.ai/models).

---

## Disclaimer &mdash; read before use

This model has had its **safety alignment substantially removed** via *abliteration* (orthogonalizing
the refusal direction out of the residual stream). It will **comply with harmful, unethical, or illegal
requests** the original `Qwen3.8-Flash-Next` would refuse. Released **strictly for legitimate research** &mdash;
interpretability, AI-safety / refusal-mechanism study, red-teaming, and robustness evaluation. **You
assume full responsibility** for how you use it and everything it generates; add your own safety and
moderation layers before any deployment. Use must comply with the
[Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0) inherited from the base model and all
applicable law. The authors accept **no liability** for misuse.

---

## Requirements &mdash; build a recent llama.cpp

The `qwen4_exp` architecture (Gated DeltaNet + QSA + HyperConnections + PLE n-gram) was **merged into
mainline llama.cpp on 2026-08-27** ([PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742),
`model: add Qwen3.8-Flash-Next (qwen4exp)`, plus follow-up #27880). Any llama.cpp built from `master`
at/after that date loads these files. **Older or bundled runtimes that predate it will not** &mdash; e.g.
LM Studio's current build stops at `qwen35moe` an