---
language:
- en
- zh
license: mit
pipeline_tag: text-generation
base_model:
- zai-org/GLM-5.3-Flash
tags:
- unsloth
- glm5_next
---
# Read our How to [Run GLM-5.3-Flash Guide!](https://unsloth.ai/docs/models/glm-5.3-flash)
<div>
  <p style="margin: 0 0 0px 0; margin-top: 0px;">
    <em><a href="https://unsloth.ai/docs/basics/dynamic-3.0-ggufs">Unsloth Dynamic 3.0</a> achieves superior accuracy & outperforms other leading quants.</em>
  </p>
  <div style="display: flex; gap: 5px; align-items: center; margin-bottom: 0px;">
    <a href="https://github.com/unslothai/unsloth/">
      <img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="133">
    </a>
    <a href="https://discord.gg/unsloth">
      <img src="https://github.com/unslothai/unsloth/raw/main/images/Discord%20button.png" width="173">
    </a>
    <a href="https://unsloth.ai/docs/models/glm-5.3-flash">
      <img src="https://raw.githubusercontent.com/unslothai/unsloth/refs/heads/main/images/documentation%20green%20button.png" width="143">
    </a>
  </div>
      <ul style="margin: 0;">
    <li>To run, please use our <a href="https://github.com/ggml-org/llama.cpp/pull/27754">llama.cpp PR</a> or use the <a href="https://unsloth.ai/docs/desktop">Unsloth Desktop</a> app.</li>
    <li>You can now run GLM-5.3-Flash in our Unsloth Desktop UI.</li>
 <li>See below for 1-bit GLM-5.3-Flash (Low) run inside of Unsloth Desktop:</li>
</div>

<img width="600" alt="glm unsloth desktop" src="https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FqPVJA6BHcCJIkQQrDvpD%252FScreenshot%25202026-08-27%2520at%25207.38.11%25E2%2580%25AFAM.png%3Falt%3Dmedia%26token%3D576a6810-3f32-4e4c-a0d8-8a84cb733a52&width=768&dpr=3&quality=100&sign=c29f4f58dbf49733ae3d4cb8d790c899&sv=3" />

---

# GLM-5.3-Flash

<div align="center">
<img src=https://raw.githubusercontent.com/zai-org/GLM-5/refs/heads/main/resources/logo.svg width="15%"/>
</div>
<p align="center">
    👋 Join our <a href="https://raw.githubusercontent.com/zai-org/GLM-5/refs/heads/main/resources/wechat.png" target="_blank">WeChat</a> or <a href="https://discord.gg/QR7SARHRxK" target="_blank">Discord</a> community.
    <br>
    📖 Check out the GLM-5.3-Flash <a href="https://z.ai/blog/glm-5.3-flash" target="_blank">blog</a> and GLM-5 <a href="https://arxiv.org/abs/2602.15763" target="_blank">Technical report</a>.
    <br>
    📍 Use GLM-5.3-Flash API services on <a href="https://docs.z.ai/guides/llm/glm-5.3-flash">Z.ai API Platform. </a>
</p>

## Introduction

We introduce GLM-5.3-Flash, the first natively multimodal model in the GLM-5 series. With 320B total parameters and just 18B active parameters, it outperforms GLM-5.2 across benchmarks and real-world workloads at one-tenth the price, while approaching Claude Opus 4.8 on coding and agentic benchmarks.

GLM-5.3-Flash starts from a newly trained base model, with its architecture and training recipe redesigned around capability and efficiency. For the first time in the GLM series, we introduce a hybrid architecture combining sparse and linear attention, sharply reducing long-context serving costs while preserving precise long-context capabilities. The model also adopts Manifold-Constrained Hyper-Connections (mHC) to further improve scaling efficiency. Together with our latest 30T-token multimodal pre-training corpus, these changes enable GLM-5.3-Flash to deliver more intelligence with less compute.

![bench_53](https://raw.githubusercontent.com/zai-org/GLM-5/refs/heads/main/resources/bench_53.png)


## Footnotes

* **HLE w/ tools (full set)**: We use sampling parameters of `temperature=1.0` and `top_p=0.95` for evaluation, with a maximum generation length of `163,840` tokens. The evaluation is conducted with a maximum context length of `300,000` tokens, using a context management strategy. We use GPT-5.6-luna (medium) as the judge model.
* **NL2Repo**: We evaluated NL2Repo with temperature=1.0, top_p=1.0, and max_new_tokens=64k under 1M context. To prevent hacking, we use rule-based and a LLM-based judgement to prevent malicious behaviors (e.g., unauthorized pip or curl operations).
* **DeepSWE**: We run DeepSWE using the mini-swe-agent harness with `temperature=0.95`, `top_p=1.0`, `timeout=6h` and 400K context.
* **Terminal-Bench 2.1**: We evaluate in Claude Code 2.1.207 with temperature=1.0, top_p=1, max_new_tokens=65536 with 6h timeout.
* **Agent’s Last Exam**: 
* **Toolathlon Verified**: We obtain all results via the official evaluation service and report pass@1 averaged over 3 independent runs.
* **AutomationBench**: We evaluate on AutomationBench **v1.0.6**, incorporating the fix for the `null`-type handling issue introduced in [PR #13](https://github.com/zapier/AutomationBench/pull/13).
* **GDPval-AA v2**: Models are evaluated by Artificial Analysis.
* **BabyVision**: We use temperature=1.0, top_p=0.95, and a maximum context length of 164K tokens. We resize the input images such that their shorter side is at least 1.5K pixels, consistent with other baselines.

## Citation

If you find GLM-5.3-Flash useful in your research, please cite our technical report:

```bibtex
@misc{glm5team2026glm5vibecodingagentic,
      title={GLM-5: from Vibe Coding to Agentic Engineering},
      author={GLM-5-Team and : and Aohan Zeng and Xin Lv and Zhenyu Hou and Zhengxiao Du and Qinkai Zheng and Bin Chen and Da Yin and Chendi Ge and Chenghua Huang and Chengxing Xie and Chenzheng Zhu and Congfeng Yin and Cunxiang Wang and Gengzheng Pan and Hao Zeng and Haoke Zhang and Haoran Wang and Huilong Chen and Jiajie Zhang and Jian Jiao and Jiaqi Guo and Jingsen Wang and Jingzhao Du and Jinzhu Wu and Kedong Wang and Lei Li and Lin Fan and Lucen Zhong and Mingdao Liu and Mingming Zhao and Pengfan Du and Qian Dong and Rui Lu and Shuang-Li and Shulin Cao and Song Liu and Ting Jiang and Xiaodong Chen and Xia