# Reflective T2I Framework (`reflective_t2i_v1`) & Isochronic Manifold Correction

本项目提供 **模型无关的自反思文生图框架（`reflective_t2i_v1`）**，统一训练和推理入口支持两个冻结的流匹配 T2I 基础模型（`Z-Image` 单流 S3-DiT、`SD3.5-Large` 双流 MMDiT），外挂独立 VLM Observer（`Qwen3.5-9B` / `Qwen3-VL-8B` / `InternVL3.5-8B`），并通过轻量级图像流状态纠偏算子（`IsochronicTransitionV2`）在生成中途完成 **Generate → Lookahead Preview → Observe & Gate → Concise Imperative Compilation → Isochronic Correction → Frozen Native Continuation** 闭环（支持多轮迭代原子自反思）。

> 历史 SenseNova-U1.5 专用接口 `qwen_latent_cot/u15_target_correction/`（`isochronic_manifold_v1`）继续使用原入口。U1.5 的 Adapter 可直接从 Python 导入，统一命令行尚未接入。

---

## 核心跨架构实证结果（两类完全不同的零编辑预训练纯 T2I 底座）

在 64× Ascend 910C（`modelarts-64b`）上分别以**同一套配方**训练 **Z-Image Base 6B（单流 S3-DiT + Qwen3-4B）** 与 **SD3.5-Large 8B（双流 MMDiT + CLIP-L/G + T5-XXL）** 至完整 2 Epochs（`step_012000` / 768k pairs），并在独立 `Qwen3-VL-8B-Instruct` 裁判下完成端到端六臂闭环因果与两轮迭代评测（完整报告见 [docs/reflective_t2i_v1_pilot_report.md](docs/reflective_t2i_v1_pilot_report.md)）：

| 冻结纯 T2I 底座（架构 · 可训占比） | Benchmark (`n`) | `A_native` | `E_shuffled` (因果对照) | `C_r1_imp` (单轮) | **`C_r2_atomic` (两轮 Ours)** | `B_rewrite` (全图重采) | **`R2 - A` (95% CI)** | **`R2 - E` 纯因果 (95% CI)** |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Z-Image Base 6B** (`S3-DiT` 单流) | **`GenEval2` (800)** | 0.1812 | 0.1925 | 0.2162 | **0.2288** (`0% Dmg`, `+38`) | 0.2462 | **`+4.75%` `[+3.25, +6.25]`** | **`+3.62%` `[+2.13, +5.25]`** |
| **Z-Image Base 6B** (`S3-DiT` 单流) | **`GenEval++` (280)** | 0.6929 | 0.7000 | 0.7393 | **0.7500** (**反超 `B`**, `+16`) | 0.7464 | **`+5.71%` `[+1.79, +9.64]`** | **`+5.00%` `[+2.14, +8.21]`** |
| **SD3.5-Large 8B** (`MMDiT` 双流 · `0.29%`) | **`GenEval++` (280)** | 0.3964 | 0.4071 | 0.4357 | **0.4464** (**`0% Dmg`**, `+14`) | 0.4643 | **`+5.00%` `[+2.50, +7.86]`** | **`+3.93%` `[+1.07, +6.79]`** |

**三大核心跨架构科学结论**：
1. **跨架构一致的统计显著增益**：单流 S3-DiT 与双流 MMDiT 上 `C_r2_atomic` 相对冻结原生底座均稳定提升 **`+4.75% ~ +5.71%`**，且相对打乱反馈控制臂 `E_shuffled` 的纯因果增益全部严格 > 0（**`+3.62% ~ +5.00%`**，净修复量为 `E` 的 **4.3× ~ 8.0×**）。
2. **局部精修反超全图重采且零画质/构图塌缩**：两类底座在局部属性精修上均大幅领先全图重采 `B`（Z-Image `GenEval2 attr`: `19.1% vs 11.9%`；SD3.5 `GenEval++ attr`: **`36.4% vs 22.7%`**），SD3.5 两轮迭代全程 **Damage = `0/12 (0.0%)`**（`B` 误伤率高达 `33.3%`），Z-Image `GenEval++` 总分更是以 **`0.7500` 直接反超 `B (0.7464)`**。
3. **100 对未见 `val` Scaling 规律一致**：Z-Image `Strict Pass` **`29.0% → 47.0%`**（`add 50%`、`remove 70%`、`attr 55%`、`HF 0.986`）；零编辑能力的 SD3.5-Large 从 **`5.0% → 32.0% (6.4×)`**（`add 32.5% [6.5×, Quality 87.5%]`、`remove 60.0% [12×]`、`HF 1.016`）。

---

## 核心架构（三层解耦）

```text
qwen_latent_cot/
├── reflective_t2i/                    # [主框架] 模型无关自反思文生图框架 (reflective_t2i_v1)
│   ├── core/                          # 归一化流几何 LinearFlowGeometry、条件锚点跳转、双向门控、祈使句编译、5臂闭环管线
│   ├── generators/                    # ZImageAdapter (单流 S3-DiT) / SD35Adapter (双流 MMDiT) / U15Adapter
│   ├── observers/                     # 独立 VLM 后端 (OpenAIVLMObserver / LocalHFObserver)
│   ├── editors/                       # 零初始化 SwitchableLoRALinear + 单流 ImageTokenMaskedLoRA
│   ├── training/                      # 自发差分加权+余弦损失、预热式 HY-SOAR 自滚出、全窗源图恒等 dropout、DDP
│   └── evaluation/                    # read_progress 标定探针、未见验证集评测与 VLM 盲评工具
└── u15_target_correction/             # [向后兼容] SenseNova-U1.5 历史闭环与权重入口
```

- **架构与数学设计文档**：[docs/reflective_t2i_v1_architecture_design.md](docs/reflective_t2i_v1_architecture_design.md)
- **完整跨架构实验报告（诊断 + 双模型 Scaling + 闭环因果与两轮迭代）**：[docs/reflective_t2i_v1_pilot_report.md](docs/reflective_t2i_v1_pilot_report.md)
- **归档实验数据**：
  - U1.5 180 题推理诊断：[output/ablate_zero_train_gv2_180_20261122/](output/ablate_zero_train_gv2_180_20261122/)
  - Z-Image 6B 标定、Scaling 与 1,080 题闭环评测：[output/reflective_v1_zimage_pilot_20261122/](output/reflective_v1_zimage_pilot_20261122/) · [output/bench_reflective_v1_zimage64_step012000_20261123/](output/bench_reflective_v1_zimage64_step012000_20261123/)
  - SD3.5-Large 8B 标定、Scaling 与 GenEval++ 闭环评测：[output/reflective_v1_sd35_20261123/](output/reflective_v1_sd35_20261123/) · [output/bench_reflective_v1_sd35_gvpp280_step012000_20261124/](output/bench_reflective_v1_sd35_gvpp280_step012000_20261124/)
- **历史 U1.5 数学与评测文档**：[docs/isochronic_manifold_logic.md](docs/isochronic_manifold_logic.md) · [docs/benchmark.md](docs/benchmark.md) · [docs/data.md](docs/data.md)

---

## 准备与验证

在仓库根目录运行框架与 U1.5 纠偏测试。CPU 即可运行；实际 NPU 训练和推理另需 Ascend 环境：

```bash
pip install -e ".[dev]"
pytest -v tests/test_reflective_t2i.py tests/test_isochronic_manifold.py
```

---

## 快速入口（`reflective_t2i_v1`）

### 1. `modelarts-64b` 64 卡大规模训练（4 × 16 Ascend 910C）
```bash
# 从仓库根目录运行。集群默认布局：工作区/code/仓库；非此布局时请指定已有工作区。
export REFLECTIVE_WORKSPACE="${REFLECTIVE_WORKSPACE:-$(cd ../.. && pwd)}"
PY=${REFLECTIVE_WORKSPACE}/venvs/u15-NPU/bin/python

# Z-Image Base 6B（单流 S3-DiT，400k 数据 2 epochs，约 2 小时）
"$PY" scripts/launch_reflective64.py \
  --config configs/reflective_t2i/z_image_400k_64npu.json \
  --output /data/zyd_workspace/outputs/reflective_v1_zimage_400k_64npu_20261123

# SD3.5-Large 8B（双流 MMDiT，400k 数据 2 epochs，约 2.5 小时）
"$PY" scripts/launch_reflective64.py \
  --config configs/reflective_t2i/sd35_400k_64npu.json \
  --output /data/zyd_workspace/outputs/reflective_v1_sd35_400k_64npu_20261123
```

### 2. 64 卡全量闭环因果 + 两轮迭代评测（`GenEval2 800 + GenEval++ 280`）
```bash
# 四臂因果主评测（A / B / C / E）
"$PY" scripts/bench_reflective_t2i64.py run-all64 \
  --config configs/reflective_t2i/z_image_400k_64npu.json \
  --checkpoint /data/zyd_workspace/outputs/reflective_v1_zimage_400k_64npu_20261123/step_012000 \
  --out /data/zyd_workspace/outputs/bench_reflective_v1_zimage64_step012000

# 祈使句编译 + 两轮迭代原子自反思进阶评测（C_r1_imperative + C_r2_atomic）
"$PY" scripts/bench_reflective_t2i64.py run-iter64 \
  --config configs/reflective_t2i/z_image_400k_64npu.json \
  --checkpoint /data/zyd_workspace/outputs/reflective_v1_zimage_400k_64npu_20261123/step_012000 \
  --base-dir /data/zyd_workspace/outputs/bench_reflective_v1_zimage64_step012000 \
  --out /data/zyd_workspace/outputs/bench_reflective_v1_zimage64_step012000_iter2
```

### 3. 新底座 `read_progress` 甜点标定探针（5 分钟零训练）
```bash
"$PY" scripts/run_reflective_t2i.py probe-progress \
  --config configs/reflective_t2i/sd35_400k_64npu.json \
  --out /tmp/probe_progress --candidates 0.24,0.32,0.40 --device npu:0
```

### 4. 单图五臂因果闭环推理（`A_native` / `B_rewrite_resample` / `C_reflective` / `D_oracle_feedback` / `E_shuffled_feedback`）
```bash
"$PY" scripts/run_reflective_t2i.py infer \
  --config configs/reflective_t2i/z_image_400k_64npu.json \
  --checkpoint /path/to/step_012000 \
  --arm C_reflective \
  --observer-url http://<vllm-qwen35-host>:8000/v1 \
  --prompt "A photo of three red apples and two green pears on a wooden table" \
  --seed 42 --device npu:0 --out /tmp/reflective_out
```

---

## 历史 SenseNova-U1.5 入口（向后兼容）

```bash
"$PY" -m qwen_latent_cot.u15_target_correction.cli t2i \
  --model /data/zyd_workspace/models/SenseNova-U1.5-8B-MoT \
  --source "${REFLECTIVE_WORKSPACE}/code/sensenova-u15-c35c83b" \
  --config configs/u15_isochronic_state/isochronic_manifold.json \
  --prompt "a lion beside seven zebras" \
  --checkpoint-mode trained \
  --checkpoint /data/zyd_workspace/isochronic_data_20261107/isochronic_manifold_400k_20261119/step_006001 \
  --output /tmp/u15_out
```
