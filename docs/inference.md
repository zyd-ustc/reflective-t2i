# 严格等时流形跳转推理与突破口 A 自反思门控

项目唯一推理路径为 **突破口 A 多步显影审图门控 + 严格等时流形跳转算子（`isochronic_manifold_transition`）+ 冻结原生 `G1` 尾段（`cfg_interval = [0.0, 0.80]`）**。

## 1. 完整文生图自反馈闭环流程（`inference.t2i_feedback`）

1. **共享前缀生成（`0 -> 0.36`）**：使用冻结原生 `NativeT2I`（`original_prompt`，`cfg_scale=4.0, cfg_interval=[0.0, 0.80]`）推进至 $z_{0.36}$。
2. **突破口 A —— 多步子网格预览显影 + 双重零偏置防误触发门控**：
   - **4 步子网格预览（`field.preview_image(..., steps=4)`）**：在 `[0.36, 1.0]` 位移时间网格上均匀锚定 4 个子区间（首区间 Heun 预测-校正 + 后续 3 步 Euler），跨越 `[0.36, 0.75]` 物体涌现带，使小物体与次要物体充分显影，同时严保主轨迹 $z_{0.36}$ 不变。
   - **Gate 1（无提示词偏置的 `add` 聚焦复核 `recheck_add_on_preview`）**：当 UND 初审拟触发 `Add <subject>` 时，在不泄露目标 prompt 与目标数量的前提下对预览图做 `<subject>` 专项细粒度扫描并重判，拦截全局粗描述漏看小物体引发的假阳性 `add`。
   - **Gate 2（确定性端点与题目计数一致性门控 `gate_contradictory_edits`）**：确定性拦截算术矛盾指令、将题目明确要求存在的物体删至 `0` 的误删指令、以及当前计数已等于题目要求计数却仍下令增删的幻觉指令。
3. **严格等时流形跳转（$z_{0.36}^{(S)} \to \hat{z}_{0.36}^{(T)}$）**：
   - 冻结原生分支提取源干净锚点 $\hat{x}_S = H_0(z_{0.36}, 0.36, \varnothing)$ 与结构噪声锚点 $\hat{\epsilon}_{\text{anchor}} = \frac{z_{0.36} - 0.36\hat{x}_S}{0.64}$；
   - LoRA 分支在短程窗口 $\tau \in [0.36, 0.60]$ 内推进得到 $q_{0.60}$ 并读出干净目标估计 $\hat{x}_T$；
   - **严格等时回投至同一时刻 $t_0 = 0.36$ 的干净原生噪声流形**：
     $$\hat{z}_{0.36} = 0.36 \hat{x}_T + 0.64 \hat{\epsilon}_{\text{anchor}}$$
4. **冻结原生 `G1` 尾段精修（`0.36 -> 1.0`）**：将 $\hat{z}_{0.36}$ 交还冻结原生 `NativeT2I`（携带完整 `original_prompt`，`t > 0.80` 自动关闭 CFG），恢复全部原生高频纹理与次要约束。

## 2. 单题 CLI 与 64 卡全量闭环评测

```bash
PY=/data/zyd_workspace/bagel-LatentCoT/venvs/u15-NPU/bin/python
MODEL=/data/zyd_workspace/models/SenseNova-U1.5-8B-MoT
SOURCE=/data/zyd_workspace/bagel-LatentCoT/code/sensenova-u15-c35c83b
CONFIG=configs/u15_isochronic_state/isochronic_manifold.json

# 单题闭环文生图
"$PY" -m qwen_latent_cot.u15_target_correction.cli t2i \
  --model "$MODEL" --source "$SOURCE" --config "$CONFIG" \
  --prompt "a photo of a cell phone and a horse" \
  --checkpoint-mode trained --checkpoint /data/zyd_workspace/isochronic_data_20261107/isochronic_manifold_400k_20261119/step_006001 \
  --seed 17 --output /path/to/out

# 64 卡 GenEval(553) + GenEval++(280) 全量闭环生成与独立 Qwen3-VL 评测
"$PY" scripts/bench_t2i_closed_loop64.py launch-gen64 \
  --config "$CONFIG" --checkpoint /data/zyd_workspace/isochronic_data_20261107/isochronic_manifold_400k_20261119/step_006001 --out /path/to/bench_out
"$PY" scripts/bench_t2i_closed_loop64.py launch-score64 --out /path/to/bench_out
"$PY" scripts/bench_t2i_closed_loop64.py merge --out /path/to/bench_out

# 64 卡 GenEval2(800) + T2I-CompBench++(2400) 全量闭环评测
"$PY" scripts/bench_t2i_closed_loop64.py launch-gen64 \
  --skip-geneval --skip-geneval-pp \
  --geneval2 /data/zyd_workspace/bagel-LatentCoT/code/anchor-search-20261104/data/geneval2_800.jsonl \
  --t2i-compbench /data/zyd_workspace/isochronic_data_20261107/benchmark_assets/frozen/outputs/0_eval62326_qwen27b_sixbench_20261019/foundation_v1/inputs/score_manifests/t2i_compbench_pp.jsonl \
  --config "$CONFIG" --checkpoint /data/zyd_workspace/isochronic_data_20261107/isochronic_manifold_400k_20261119/step_006001 --out /path/to/bench_gv2_out
```
