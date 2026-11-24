# `reflective_t2i_v1` 架构重构与算法升级设计文档

> **版本**：v1.0（基于 2026-11-22 `modelarts-qwen` 4-NPU 零训练诊断消融结果修订）
>
> **目标**：将当前强耦合于 SenseNova-U1.5 的 `u15_target_correction` 升级为 **Model-Agnostic Reflective T2I Framework（模型无关的自反思文生图框架）**，并在核心算法层根治 `add / count` 指令遵循不足与复杂场景 `Damage Rate` 偏高的问题。

---

## 1. 研究定位与实证驱动（来自 Step 1 零训练诊断的关键结论）

### 1.1 论文叙事升级
- **旧叙事**：*How to improve self-correction in a Unified Multimodal Model?*（局限于单一 UMM 底座，易被质疑收益来自 UMM 原生编辑先验，且受限于 GenEval 96.5% 天花板）。
- **新叙事 (`reflective_t2i_v1`)**：*Can a frozen text-to-image foundation model acquire visual self-reflection and self-correction through an external understanding model, without edit pretraining?*
  - **Generator 完全冻结**，不要求经过任何图像编辑预训练，完整保留原先生图先验与尾段高频动力学。
  - **独立 VLM Observer** 负责视觉观察、约束核查与结构化纠偏反馈（与 Generator 彻底解耦、跨模型共享）。
  - **轻量级模型专属 State Corrector** 将外部语义反馈转化为该 Generator 原生轨迹上的可执行流形跃迁算子。

### 1.2 Step 1 零训练推理消融实证（决定 Step 2 的核心设计边界）
我们在 `modelarts-qwen`（4× Ascend 910C NPU）上，基于冻结的 `GenEval2` 180 题分层触发子集（含全量 65 题 `trig_pass` 以测精确 Damage Rate，加 75 `add` / 20 `remove` / 15 `attr` / 5 `move` 失败样本；复用同一份 `Qwen30B` 反思指令与 `Qwen3-VL-8B-Instruct` 独立判分），对当前 U1.5 `step_006001` 完成了 5 臂正交消融（完整数据见 [output/ablate_zero_train_gv2_180_20261122/summary.md](../output/ablate_zero_train_gv2_180_20261122/summary.md)）：

| 评测臂 | Exact Acc ↑ | Atomic QA Acc ↑ | Fix / TrigFail ↑ | Damage / TrigPass ↓ | `add` Fix / Dmg / Net | Mean HF | Mean OOB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline`（原生直出） | 0.3611 | 0.8269 | 0/115 (0.0%) | 0/65 (0.0%) | 0/75 · 0/24 · +0 | 0.06602 | 0.03120 |
| `vanilla_iso`（原空条件锚点） | 0.2944 | 0.8185 | 8/115 (7.0%) | 20/65 (30.8%) | 5/75 · 7/24 (29.2%) · **-2** | 0.07541 | 0.03475 |
| **`arm1_cond_anchor`**（条件原生锚点） | **0.3167** | **0.8216** | 8/115 (7.0%) | **16/65 (24.6%)** | 5/75 · **3/24 (12.5%)** · **+2** | 0.07498 | 0.03346 |
| `arm2_soft_mask_eta`（软掩码+局部刷新） | 0.2778 | 0.8138 | 3/115 (2.6%) | 18/65 (27.7%) | 2/75 · 3/24 (12.5%) · -1 | **0.06658** | **0.02935** |
| `arm3_mask_tmid`（`arm2`+$t_{\text{out}}=0.48$） | 0.2722 | 0.8144 | 4/115 (3.5%) | 20/65 (30.8%) | 3/75 · 5/24 (20.8%) · -2 | 0.06642 | **0.02866** |
| **`arm4_edit_cfg`**（条件锚点+`edit_cfg=1.75`） | 0.2611 | 0.7986 | **14/115 (12.2%)** | 32/65 (49.2%) | **7/75 (9.3%)** · 7/24 · +0 | 0.07605 | 0.04559 |
| `arm5_combo` | 0.2556 | 0.8043 | 10/115 (8.7%) | 29/65 (44.6%) | 5/75 · 6/24 · -1 | 0.06656 | 0.03153 |

**四条直接锁定 `reflective_t2i_v1` 架构与训练配方的铁律**：
1. **默认切换为条件原生锚点（零代价帕累托改进）**：仅将锚点从 $H_0(z_{t_0}, t_0, \varnothing)$ 改为 $H_0(z_{t_0}, t_0, p_{\text{orig}})$，`Exact Acc +2.23%`、`Damage` 相对下降 20%（`30.8% → 24.6%`），**`add` 的 Damage 直接腰斩（`29.2% → 12.5%`）、Net Fixed 由 `-2` 翻正至 `+2`**，且可复用前缀最后一步计算（0 额外 NFE）。
2. **Edit-CFG 实证打破 MMSE 均值塌缩，但必须配套修复训练态无条件分支与 Observer 误触发门控**：`arm4_edit_cfg (1.75)` 使**全动作 Fix Rate 暴涨 +75%（`7.0% → 12.2%`：`add +40%`、`remove 3×`、`attr +50%`、`move 0%→20%`）**；但它同时把 `remove` 误触发的 Damage 放大到 `87.0% (20/23)`，且因旧训练将 `prompt_dropped` 仍监督向 $T$ 而非 $S$ 导致 `OOB` 升至 `0.0456`。这证明新框架必须同时做到：
   - 训练期将 `edit_prompt_dropout` 改为**源图 $S$ 恒等锚定**（使 $\hat{x}_{\text{cond}} - \hat{x}_{\text{id}}$ 成为纯净编辑增量）+ 推理期引入 `CorrFlow` 风格 Trust-Region 范数裁剪；
   - Observer 侧增加针对 **`remove` 与 `add` 的双向视觉聚焦复核与置信度门控**，把假触发率压低。
3. **后处理软掩码无法“无中生有”，`add / count` 的核心破局必须在训练侧完成**：`arm2` 虽然把高频梯度（`0.0666` vs 原生 `0.0660`）与 OOB（`0.0294`）完美拉回原生分布，但由于 1-epoch 无加权 MSE 编辑器输出的 `add` 幅度本身偏弱，高斯平滑后 `mask_max` 仅 `0.73`，`(1-m)z_{t0} + m z_{reproj}` 反而衰减了本就微弱的 `add` 信号（`Fix 7.0% → 2.6%`）。因此**不能指望推理侧软掩码修补弱编辑器**；必须在 `reflective_t2i_v1` 训练侧原生内置 **无 bbox 自发差分加权 + 余弦残差损失（`CorrFlow`）** 与 **无目标泄露 1-Step 自滚出桥训练（`HY-SOAR`）**！
4. **纯 T2I 文本编码器的指令理解瓶颈必须提前化解**：SD3.5（CLIP-L/G + T5-XXL）未经指令微调，纯祈使句 `feedback_only`（如 `"Remove 1 hotdog"`）会在 CLIP 词袋特征中反向激发 `"hotdog"`。因此框架必须在条件层原生支持 `feedback_only`、`native_prompt_plus_feedback` 与 `vlm_hidden_bridge` 三种模式。

---

## 2. 统一数学框架与 `NoiseGeometry` 抽象

### 2.1 归一化去噪进度 $p \in [0, 1]$
为彻底屏蔽不同 Generator 的时间轴方向（$t=0$ 噪声 vs $\sigma=1$ 噪声）和预测空间（Clean Head vs $v$-pred vs $\epsilon$-pred），框架内部所有算法一律使用**归一化线性进度 $p \in [0, 1]$**：
- $p = 0$：纯噪声流形（$z_0 = \sigma_{\text{scale}}\epsilon$）；
- $p = 1$：干净图像/潜变量流形（$z_1 = x_0$）；
- 统一前向线性路径：
  $$z_p = p \cdot x_0 + (1 - p) \cdot \sigma_{\text{scale}}\,\epsilon,\qquad v_{\text{to-clean}} \triangleq \frac{\mathrm{d} z_p}{\mathrm{d} p} = x_0 - \sigma_{\text{scale}}\,\epsilon = \frac{x_0 - z_p}{1 - p}.$$

### 2.2 三个目标 Generator 的精确几何映射表

| Generator | 空间与形状（1024 / 512） | 原生时间参数 | 原生网络输出 | $p \in [0,1]$ 与 $(\hat{x}_0, \hat{\epsilon}, v_{\text{to-clean}})$ 精确换算 |
|---|---|---|---|---|
| **SD3.5-Large** (8B, MMDiT) | 16-ch VAE Latent `(B, 16, H/8, W/8)` (`shift_factor`, `scaling_factor`) | `FlowMatchEulerDiscreteScheduler`: $\sigma \in [1, 0]$ (`timestep = 1000 * sigma`) | $v_{\text{sd3}} = \epsilon - x_0$ | $p = 1 - \sigma$；$v_{\text{to-clean}} = -v_{\text{sd3}}$；$\hat{x}_0 = z_p - (1-p)v_{\text{sd3}}$；$\hat{\epsilon} = z_p + p\,v_{\text{sd3}}$ |
| **Z-Image** (6B, S3-DiT) | 16-ch VAE Latent `(B, 16, H/8, W/8)`（复用 `zimage_core`） | `t_clean = 1 - sigma` $\in [0, 1]$（动态 resolution shift） | $v_{\text{zi}} = x_0 - \epsilon$（注意 scheduler 吃 `-v`） | $p = t_{\text{clean}} = 1 - \sigma$；$v_{\text{to-clean}} = v_{\text{zi}}$；$\hat{x}_0 = z_p + (1-p)v_{\text{zi}}$；$\hat{\epsilon} = z_p - p\,v_{\text{zi}}$ |
| **SenseNova-U1.5** (8B, UMM MoT) | RGB Pixel `(B, 3, 512, 512)`，`noise_scale()` $\sigma_s$ | $t \in [0, 1]$（`timestep_shift=3.0`） | `fm_head` 直接输出干净图 $\hat{x}_0$ | $p = t$；$v_{\text{to-clean}} = \frac{\hat{x}_0 - z_p}{\max(1-p, t_\epsilon)}$；$\hat{\epsilon} = \frac{z_p - p\hat{x}_0}{1-p}$（含 $\sigma_s$） |

> **关键对齐协议（Cross-Model `read_progress` 标定）**：严禁把 SenseNova 的 `0.36` 直接硬拷贝到 SD3.5/Z-Image。在 `timestep_shift=3.0` 下，SenseNova 的 `p=0.36` 对应 50 步中的第 32 步（**64% 步数进度**）。每个新接入的 Generator 必须先跑 10 分钟无训练探针（在 50 题难样本上扫 `step_fraction ∈ {0.45, 0.55, 0.65}` 的 4-step preview 清晰度与可编辑性），将标定后的 `read_progress` 与 `stop_progress` 写入该模型专属 config。

### 2.3 升级版条件锚点等时流形跳转算子（`IsochronicTransitionV2`）
在接入点 $p_0 = \text{read\_progress}$ 处，设当前原生轨迹状态为 $z_{p_0}$：
1. **条件原生锚点提取（零额外 NFE，复用前缀最后一步）**：
   $$\hat{x}_S = H_0(z_{p_0}, p_0, c_{\text{orig}}),\qquad \hat{\epsilon}_{\text{anchor}} = \frac{z_{p_0} - p_0 \hat{x}_S}{1 - p_0}.$$
2. **短程语义流推进 + Trust-Region 限幅的受控 Edit-CFG**：
   在 $[p_0, p_{\text{stop}}]$ 内沿子网格积分，每步速度为：
   $$\hat{x}_{\text{guided}} = \hat{x}_{\text{id}} + s_{\text{edit}}\cdot \operatorname{clip}_{\rho}(\hat{x}_{\text{cond}} - \hat{x}_{\text{id}}),\qquad v_\theta = \frac{\hat{x}_{\text{guided}} - q_\tau}{\max(1 - \tau, t_\epsilon)}$$
   其中 $\operatorname{clip}_{\rho}(\delta) = \delta \cdot \min\!\big(1,\; \frac{\rho\,\|\hat{x}_S\|_{\text{rms}}}{\|\delta\|_{\text{rms}} + 10^{-8}}\big)$（借鉴 `CorrFlow` Trust-Region 裁剪），防止 `edit_cfg_scale`（推荐 `1.3–1.5`）触发 OOB 爆炸。
3. **等时回投 + 可选方差保持噪声刷新（仅在新训练强编辑器验证有效时按动作开启）**：
   $$\hat{\epsilon}_{\text{eff}} = \sqrt{1 - \eta^2}\,\hat{\epsilon}_{\text{anchor}} + \eta\,\xi,\qquad \hat{z}_{p_0} = p_0\,\hat{x}_T + (1 - p_0)\,\hat{\epsilon}_{\text{eff}}.$$
4. **交还完全冻结原生 Generator**：停用 Corrector adapter，用原始 `original_prompt` 与原生 CFG schedule（`cfg_interval=[0.0, 0.80]`）从 $p_0$ 续采至 $1.0$。

---

## 3. 三层解耦仓库架构与精确接口契约

### 3.1 目标目录结构
不继续在 `qwen_latent_cot/u15_target_correction/` 内堆砌条件分支；保持旧目录只读稳定（兼容历史 checkpoint），新建三层解耦包 `qwen_latent_cot/reflective_t2i/`：

```text
qwen_latent_cot/
├── u15_target_correction/           # 保留历史 U1.5 入口与已训权重兼容
└── reflective_t2i/
    ├── __init__.py
    ├── core/
    │   ├── contracts.py             # LatentState / ConditionBundle / ReflectionDecision / 协议抽象基类
    │   ├── noise_geometry.py        # LinearFlowGeometry (clean/v/eps 统一换算 + CPS noise refresh + TrustRegion)
    │   ├── reflection_policy.py     # 单次视觉审图 + add/remove 双向聚焦复核 + 算术/置信度门控
    │   ├── correction.py            # IsochronicTransitionV2 (短程演进 + 条件锚点回投 + K-step 可选迭代)
    │   └── pipeline.py              # Model-agnostic Generate -> Preview -> Observe -> Correct -> Continue 闭环
    ├── generators/
    │   ├── base.py                  # GeneratorAdapter ABC
    │   ├── diffusers/
    │   │   ├── sd35.py              # SD3.5-Large (8B MMDiT, 双流隔离, Diffusers + Ascend NPU)
    │   │   ├── z_image.py           # Z-Image (6B S3-DiT, 单流 image-token mask 隔离, 复用 zimage_core)
    │   │   └── qwen_image.py        # Qwen-Image-2512 (20B 质量上界预留)
    │   └── u15.py               # SenseNova-U1.5 MoT 封装适配器
    ├── observers/
    │   ├── base.py                  # ObserverBackend ABC (统一 deterministic JSON 输出)
    │   ├── vllm_client.py           # OpenAI-compatible 独立 VLM 推理客户端 (压低延迟至 ~1.3x)
    │   ├── qwen35.py                # Qwen3.5-9B 主 Observer
    │   ├── qwen3vl.py               # Qwen3-VL-8B 对照 Observer
    │   └── internvl.py              # InternVL3.5-8B 独立 Judge / 交叉 Observer
    ├── editors/
    │   ├── base.py                  # StateCorrector ABC
    │   ├── latent_lora.py           # Switchable LoRA + TokenMaskedLoRA (单流 S3-DiT 必备)
    │   └── vlm_bridge.py            # [Phase 2 预留] VLM 语义隐状态 -> DiT 轻量 Cross-Attn Projector
    ├── training/
    │   ├── dataset.py               # 复用 prod_1m_glm53_20261118 40w 清单 + shared_field_of_view_v2 + 动态 VAE 编码
    │   ├── losses.py                # clean_mse / delta_weighted_clean_mse / cosine_residual_loss
    │   ├── rollout.py               # HY-SOAR 1-step ODE/CPS 自滚出辅助状态构造
    │   └── train.py                 # 跨 Generator 共享训练主循环 (DDP / ZeRO / 50k Pilot -> 400k)
    └── evaluation/
        ├── t2i_benchmark.py         # 统一 5-Arm 实验矩阵 (A-E) 与 6 榜单评测入口
        └── reflection_diagnostics.py# Fix / Damage / Net Fixed / Atomic QA / HF / OOB / Latency 统计
```

### 3.2 核心接口契约（`core/contracts.py` & `generators/base.py`）

```python
@dataclass
class LatentState:
    tensor: torch.Tensor              # Pixel [B,3,H,W] or VAE latent [B,C,h,w], float32
    progress: float                   # Normalized p in [0.0, 1.0] (0=noise, 1=clean)
    valid_mask: torch.Tensor | None   # [B,1,H_or_h,W_or_w] in [0,1], excludes letterbox padding
    meta: dict[str, Any] = field(default_factory=dict)

@dataclass
class ConditionBundle:
    raw_text: str | None
    payload: Any                      # Generator-specific (e.g., SD3.5 (embeds, pooled); Z-Image cap_feats; U1.5 Prefix)

@dataclass
class ReflectionDecision:
    outcome: Literal["satisfied", "edit", "gated_noop"]
    action: Literal["none", "add", "remove", "attr", "move", "other"]
    instruction: str                  # Canonical edit instruction
    target_rewrite: str               # Optional rewritten target prompt (for Arm B & prompt_plus_feedback)
    coarse_region_hint: str           # e.g., "open space on the right" (no pixel bbox!)
    confidence: float                 # [0, 1] observer self-assessed certainty
    gated_by: str | None
    raw: dict[str, Any]

class GeneratorAdapter(ABC):
    @property
    @abstractmethod
    def geometry(self) -> LinearFlowGeometry: ...
    @abstractmethod
    def encode_prompt(self, text: str | None) -> ConditionBundle: ...
    @abstractmethod
    def encode_clean(self, rgb_minus1_1: torch.Tensor) -> torch.Tensor: ...
    @abstractmethod
    def decode_preview(self, clean_tensor: torch.Tensor) -> Image.Image: ...
    @abstractmethod
    def sample_until(self, z_init: torch.Tensor, target_p: float, cond: ConditionBundle,
                     uncond: ConditionBundle | None, cfg_scale: float, *,
                     start_p: float = 0.0, return_clean_at_end: bool = True) -> tuple[LatentState, torch.Tensor | None]: ...
    @abstractmethod
    def lookahead_preview(self, state: LatentState, cond: ConditionBundle,
                          uncond: ConditionBundle | None, cfg_scale: float, steps: int = 4) -> Image.Image: ...
    @abstractmethod
    def predict_clean(self, z: torch.Tensor, p: float | torch.Tensor, cond: ConditionBundle,
                      *, use_corrector: bool = False) -> torch.Tensor: ...
    @abstractmethod
    def continue_native(self, state: LatentState, cond: ConditionBundle,
                        uncond: ConditionBundle | None, cfg_scale: float,
                        cfg_interval: tuple[float, float] = (0.0, 0.8)) -> torch.Tensor: ...
```

### 3.3 两个主战场 Generator 的专属 Adapter 与 Corrector 细化设计

#### A. `SD35Adapter`（工程首选：`stabilityai/stable-diffusion-3.5-large`，8B MMDiT）
- **为什么首选跑通**：Diffusers 原生支持成熟、华为云 ModelArts 已有 SD3.5 Ascend NPU 推理兼容记录、双流 MMDiT 天然将 `image stream` 与 `text stream` 物理隔离。
- **双流 LoRA 挂载范围（严禁污染文本流）**：
  - 仅挂载在 `SD3Transformer2DModel.transformer_blocks[i]` 的**图像通道投影**：`attn.to_q`, `attn.to_k`, `attn.to_v`, `attn.to_out.0`（消融臂可加 `ff.net.0.proj`, `ff.net.2`）。
  - **严格冻结**所有文本流模块：`attn.add_q_proj`, `attn.add_k_proj`, `attn.add_v_proj`, `attn.to_add_out`, `ff_context.*`，以及全部 `CLIP-L / CLIP-G / T5-XXL` 编码器与 VAE。
  - 层范围 Pilot 对照：`middle_third`（blocks `12..25`, `rank=16`）vs `all_image_attn`（blocks `0..37`, `rank=32`），检验 `add/count` 所需容量。
- **解决 CLIP/T5 祈使句盲区（SD3.5 必做消融）**：
  - `condition_mode = "native_prompt_plus_feedback"`：CLIP-L/G 喂入结构化目标描述 `target_rewrite`（避免 `"Remove 1 hotdog"` 反向激发 hotdog），T5-XXL 喂入 `"Original: {prompt} | Edit: {instruction}"`（利用 T5-XXL 的强指令语义理解），并在训练数据中通过双图/原题模板对齐。
  - 保留 `feedback_only` 作为严格对照臂，并在 `editors/vlm_bridge.py` 预留直接注入 `Qwen3.5-9B` 反馈隐状态的轻量 Cross-Attn 接口。

#### B. `ZImageAdapter`（研究首选：`Tongyi-MAI/Z-Image`，6B 单流 S3-DiT）
- **现成资产复用**：权重已下载至 `/data/zyd_workspace/models/Z-Image`，且 `/data/zyd_workspace/zimage-isochronic-correction/src/zcorr/zimage_core/` 已在 Ascend NPU 上验证全量 40w 训练与推理，直接迁入 `generators/diffusers/z_image.py` 零移植风险！
- **必须修复现有 Z-Image 仓库的 2 个结构性缺陷**：
  1. **单流 Token Mask 隔离（`ImageTokenMaskedLoRA`）**：现有 `zcorr/lora.py` 对 `layers.`、`noise_refiner.`、`context_refiner.` 的所有 token 无差别加 LoRA，修改了文本 token 表征。新 Adapter 必须：
     - 冻结 `context_refiner.*`；
     - 在单流 `layers.*` 中显式传入当前批次的 `image_token_mask`（标记前 `L_img` 个图像潜变量 token 与后 `L_txt` 个文本 token），使 LoRA 仅作用于图像 token：`y = base(x) + image_token_mask * (B(A(x)) * scale)`。
  2. **原生替换掉旧配方**：将现有 `zcorr/manifold.py` 的 `empty` 锚点、无加权 `masked_clean_mse` 与 `feedback_only` 全部升级为第 2 节与第 5 节的统一配方。

---

## 4. 独立 Observer 与高速高精 Reflection Policy（`observers/`）

### 4.1 解耦模型配置
- **主 Observer**：`Qwen/Qwen3.5-9B`（独立服务部署，`temperature=0.0`，结构化 JSON 输出）。
- **对照 Observer**：`Qwen/Qwen3-VL-8B-Instruct`（与历史基准对齐）。
- **独立 Judge**：`OpenGVLab/InternVL3_5-8B`（打破 Observer 与 Judge 同家族偏差），同时保留 `Qwen3-VL-8B-Instruct` 复核历史榜单。

### 4.2 针对 Step 1 暴露的「假触发高损伤」与「3.25× 延迟」的双重优化
Step 1 明确显示：一旦编辑器执行力提升（如 `arm4_edit_cfg`），VLM 对 `trig_pass` 的误触发（尤其误发 `remove`）会直接转化为高 Damage Rate；且历史上 `35s` 耗时里超过 24s 花在串行多跳长文本解码。新 `ReflectionPolicy` 改为两级漏斗：
1. **Stage 1 — 单次视觉结构化初审（Fast-Pass JSON，所有样本仅 1 次 VLM 调用）**：
   - 保留“先盲述后对照”的防确认偏置机制，但在**同一次带图前向**内输出固定 JSON（避免旧版 Judge 不看图只看文本造成的二次信息丢失！）：
     ```json
     {
       "observed_objects": [{"name": "dog", "count": 2, "color": "brown", "location": "left"}],
       "missing_or_wrong": [],
       "match": true,
       "action": "none",
       "instruction": "",
       "target_rewrite": "",
       "coarse_region_hint": "",
       "confidence": 0.92
     }
     ```
   - 判定 `match: true` 的样本立即放行（0 额外开销），配合独立 vLLM 服务将端到端延迟压回 **1.3×–1.5×**。
2. **Stage 2 — 双向聚焦复核 + 确定性门控（仅对 `match: false` 触发）**：
   - **双向聚焦复核**：不仅对 `add` 做无 prompt 计数泄露的小物体复扫（现有 Gate 1），**新增对 `remove` 的聚焦计数复核**（专门拦截把 5 个看成 6 个而错发 `remove` 导致的 `39%–87%` 损伤！）。
   - **置信度 + 算术一致性门控**：`confidence < 0.75` 或端点计数矛盾直接降级为 `gated_noop`。
   - **粗方位自然语言提示（无 bbox）**：允许 `add` 附带粗粒度空位描述（如 `"in the open area on the right"`），将空间后验熵 $H(T \mid S, e)$ 降低一个量级而不会触发构图重绘。
   - **可选 `max_corrections = 2`**：在多约束难榜（`GenEval2`）上消融 1 轮 vs 2 轮反思（第 2 轮仅对第 1 轮后仍未满足的样本触发）。

---

## 5. 训练目标与算法升级（零 bbox 根治 `add / count`）

基于 Step 1 结论（推理侧无法凭空创造被训练损失淹没的 `add` 信号），`training/losses.py` 与 `training/train.py` 默认整合三项来自 `CorrFlow`、`HY-SOAR`、`Correction Training` 的零 bbox 算法升级：

### 5.1 自发差分加权 + 余弦残差方向损失（借鉴 `CorrFlow`，零 bbox）
利用每个训练对 $(S, T)$ 在潜空间/像素空间自带的精确增量 $\Delta^* = T - S$：
1. **有效画布掩码** $M_{\text{valid}}$（同 `shared_field_of_view_v2`，屏蔽 padding 单元）。
2. **自发空间重要性权重**（自动把占画布 3%–8% 的小物体 `add/count` 区域权重抬高 5–10×，背景保持基础权重 `1`）：
   $$d(x,y) = \|T(x,y) - S(x,y)\|_2,\qquad w(x,y) = 1 + \lambda_\Delta \cdot \operatorname{clip}\!\left(\frac{d(x,y)}{\operatorname{mean}_{M_{\text{valid}}}(d) + \varepsilon},\; 0,\; w_{\max}\right)$$
   推荐初值（经 4 卡 Z-Image Pilot 实测校准，防止过激权重稀释背景导致 MMSE 模糊）：$\lambda_\Delta = 1.0,\; w_{\max} = 3.0$（上限 4× 聚焦变化区，保住 >60% 背景保持梯度）。
3. **编辑增量余弦方向损失**（带 `CorrFlow` 的 `has_direction` 零初始化防梯度爆炸保护）：
   $$\Delta\hat{x} = \hat{x}_T - S,\qquad \mathcal{L}_{\text{dir}} = \mathbb{1}\!\big(\|\Delta\hat{x}\| > 10^{-6} \wedge \|\Delta^*\| > 10^{-6}\big)\cdot \big(1 - \cos_{M_{\text{valid}}}(\Delta\hat{x},\; \Delta^*)\big)$$
   总损失：
   $$\mathcal{L}_{\text{total}} = \operatorname{mean}_{M_{\text{valid}}}\!\big(w \cdot \|\hat{x}_T - T\|^2\big) + \lambda_{\text{cos}}\,\mathcal{L}_{\text{dir}}\qquad (\lambda_{\text{cos}} = 0.02).$$

### 5.2 混合桥 + 预热式 `HY-SOAR` 1-Step 自滚出训练（消除 $\alpha T$ 目标泄露与 Exposure Bias）
- **85% 标准中段桥样本**（$p \in [p_0, p_{\max}]$，`bridge_start_prob = 0.25` 提高无泄露入口比例）。
- **15% 预热后 SOAR 自滚出辅助样本**（经 `soar_warmup_steps` 初始稳定后启用，避免冷启动噪声放大）：
  1. 严格从无 $T$ 泄露的源噪声态 $a_{p_0} = p_0 S + (1-p_0)\epsilon$ 出发；
  2. 用当前 Corrector 在 `torch.no_grad()` 下推进 1 步 ODE 至 $p_1 \sim \operatorname{Uniform}(p_0, p_{\max}]$，得到模型自己产生的真实 off-manifold 中间态 $q'_{p_1}$；
  3. 对 $q'_{p_1}$ 做前向并计算 $\mathcal{L}_{\text{total}}(H_\theta(q'_{p_1}, p_1, e),\; T)$。

### 5.3 修复 `edit_prompt_dropout` 的目标定义（全窗口覆盖，释放干净 Edit-CFG）
- 当以 `edit_prompt_dropout = 0.10` 丢弃反馈指令时：
  - **在全中段窗口 $\tau \in [p_0, p_{\max}]$ 上构造纯源图加噪态 $z_\tau^{(S)} = \tau S + (1-\tau)\epsilon$，并将监督目标严格设为原图 $S$（恒等/无编辑分支）**！
  - 由此在推理期全程 $\tau \in [p_0, p_{\max}]$ 执行 `edit_cfg_scale`（1.3–1.5）+ Trust-Region 裁剪时，$\hat{x}_{\text{cond}} - \hat{x}_{\text{id}}$ 严格对应“从保留原图 $S$ 到执行指定编辑 $T$ 的纯净方向”，既保留 `+75%` 的 `add` 执行增益，又根治 OOB 漂移。

---

## 6. 第一轮跨模型配置与因果实验矩阵

### 6.1 两套 Pilot 配置（共享协议，各自专属 Adapter 与几何）

#### Pilot 1 — SD3.5-Large（NPU 工程首选，双流 MMDiT）
```json
{
  "protocol": "reflective_t2i_v1",
  "generator": {
    "adapter": "diffusers_sd35",
    "model_id": "stabilityai/stable-diffusion-3.5-large",
    "dtype": "bfloat16",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 40,
    "guidance_scale": 4.5,
    "cfg_interval": [0.0, 0.80],
    "scheduler": "native_flow_match"
  },
  "observer": {
    "backend": "vllm_qwen35",
    "model_id": "Qwen/Qwen3.5-9B",
    "temperature": 0.0,
    "max_new_tokens": 384
  },
  "reflection": {
    "mode": "single_pass_blind_describe_then_compare_json",
    "lookahead_steps": 4,
    "read_progress": "calibrated_by_probe",
    "stop_progress": "read_progress + 0.24",
    "max_corrections": 1,
    "focused_recheck_actions": ["add", "remove"],
    "confidence_threshold": 0.75,
    "contradiction_gate": true
  },
  "corrector": {
    "type": "dual_stream_image_lora",
    "rank": 16,
    "rank_ablation": 32,
    "zero_init": true,
    "layer_scope": "middle_third",
    "target_modules": ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0"],
    "anchor_mode": "conditional_native",
    "edit_cfg_scale": 1.4,
    "trust_region_rho": 0.65,
    "noise_refresh": 0.0,
    "condition_mode": "feedback_only"
  },
  "training": {
    "lr": 3e-5,
    "lr_final": 5e-6,
    "global_batch": 64,
    "pilot_pairs": 50000,
    "epochs": 2,
    "bridge_start_prob": 0.25,
    "soar_rollout_prob": 0.40,
    "edit_prompt_dropout": 0.10,
    "dropout_target": "source_identity",
    "loss": "delta_weighted_clean_mse_plus_cosine",
    "delta_weight_lambda": 4.0,
    "delta_weight_max": 8.0,
    "cosine_lambda": 0.10
  }
}
```

#### Pilot 2 — Z-Image Base（研究首选，单流 S3-DiT，权重已就绪）
与 Pilot 1 完全共用 `observer`、`reflection` 与 `training` 协议，仅替换 `generator` 与 `corrector` 专属字段：
- `generator.adapter = "z_image_s3dit"`，`model_id = "/data/zyd_workspace/models/Z-Image"`，`num_inference_steps = 50`，`guidance_scale = 4.0`（复用 `zimage_core` 100% NPU 算子）。
- `corrector.type = "single_stream_token_masked_lora"`（`image_token_mask` 隔离文本 token，冻结 `context_refiner`），同时对照 `condition_mode = "feedback_only"` vs `"native_prompt_plus_feedback"`。

### 6.2 因果验证实验矩阵（每套 Generator 严格同 seed / 同初噪 / 同预算）

| 实验组 | 生成策略 | 验证目的 |
|---|---|---|
| **A** | Native T2I | 原生纯 T2I 基线（无编辑预训练） |
| **B** | VLM Observer + Prompt Rewrite + Resample | 外层提示词改写重采基线（控制额外算力与 VLM 信息增益） |
| **C** | **VLM Observer + State Corrector (Ours)** | 完整反思闭环主方法 |
| **D** | Oracle Feedback + State Corrector | Corrector 自身能力上界（剥离 VLM 误检与漏检） |
| **E** | Shuffled Feedback + State Corrector | 因果性反事实检验（证明提升确实来自反馈指令而非额外去噪步/LoRA 偏置） |

**核心评测榜单与指标**：
- **主力榜单**（告别 GenEval 96.5% 天花板）：`GenEval2 (800)`、`GenEval++ (280)`、`T2I-CompBench++ (2,400)`、`DPG-Bench (1,065)`、`T2I-CoReBench (1,080)`、`SpatialGenEval`。
- **必报指标**：`Exact Acc`、`Atomic QA Acc`、`Fix Rate (Fix / TrigFail)`、`Damage Rate (Damage / TrigPass)`、`Net Fixed`、分动作分解（`add / remove / attr / move`）、`HF Grad / OOB`、端到端平均延迟与人工画质偏好胜率。

---

## 7. 分阶段落地路线图（Phases A–E）与验证门禁

| 阶段 | 核心交付物 | 强制验证门禁（不通过不进下一阶段） |
|---|---|---|
| **Phase A：骨架抽离与 U1.5 封装** | 建立 `qwen_latent_cot/reflective_t2i/core/`、新 `losses.py`/`rollout.py` 与 `generators/u15.py` | CPU 单测覆盖 `LinearFlowGeometry` 正逆换算恒等、`DeltaWeighted+Cosine` 梯度非零、U1.5 `cond_anchor` 数值与 Step 1 精确一致 |
| **Phase B：Z-Image Adapter 升级 + SD3.5 Adapter 打通** | (1) 复用 `zimage_core` 接入 `TokenMaskedLoRA`（立即可在已有权重 `/data/zyd_workspace/models/Z-Image` 上开跑）；(2) 接入 Diffusers `SD35Adapter`（验证 Ascend NPU 前向与 LoRA 反传算子） | 零初始化 LoRA 严格等价性检验（`max_abs_diff < 1e-5`）；文本 token 梯度与输出零扰动检验；10 分钟 `read_progress` 甜点标定完成 |
| **Phase C：独立 VLM Observer 服务化** | 接入 `Qwen3.5-9B`（主 Observer）+ `InternVL3.5-8B`（独立 Judge），上线单次结构化 JSON + `add/remove` 双向复核 | 在 180 题 `GenEval2` 上实测 `False-Trigger Rate` 较当前 `29.4%` 显著下降，且单题反思延迟 < 3s |
| **Phase D：50k Pilot 训练与新损失消融** | 从 `prod_1m_glm53_20261118` 抽 50k 均衡子集，在 **Z-Image** 与 **SD3.5** 上分别验证：`原 Clean MSE` vs `Delta-Weighted + Cosine + SOAR 1-step + S-anchored Dropout`；以及 `rank=16 (mid)` vs `rank=32 (all)` | 50k Pilot 上 `add` Fix Rate 与 Net Fixed 显著超过原配方（`arm1`/`vanilla_iso`），且 OOB 不恶化 |
| **Phase E：400k 全量训练与跨架构 A–E 评测** | 在 Z-Image（单流 S3-DiT）与 SD3.5-Large（双流 MMDiT）上完成全量 400k 训练，跑齐实验矩阵 A–E 与 6 大基准 | 两个完全不同架构的纯 T2I 底座在同一 Observer 与协议下均获得统计显著的 Net Fixed 与整体精度提升 |
