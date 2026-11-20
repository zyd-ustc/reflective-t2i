# 严格等时流形跳转算子训练（`isochronic_manifold_v1`）

项目采用**严格等时流形跳转算子（Strict Isochronic Manifold Transition Operator）**作为统一训练与推理架构：将语义结构修正聚焦于中段低噪窗口 $\tau \in [0.36, 0.60]$，完整保留冻结原生 T2I 底座负责 $\tau \in [0.36, 1.0]$ 尾段高频细节与次要约束的能力。

## 1. 模型结构（`IsochronicManifoldEditor`）

- 冻结原生 `SenseNova-U1.5-8B-MoT` 的 UND 理解主干、`timestep_embedder` 与原生输出头 `fm_head`。
- 仅在每层 GEN 注意力（`q/k/v/o_proj_mot_gen`）与 GEN MLP（`gate/up/down_proj`）挂载秩为 `64` 的 `GENLoRA`（`b` 矩阵零初始化，零点与原生分支严格等价）。
- 唯一条件模板：`Edit request: <feedback>`（支持 `edit_prompt_dropout` 置空为无条件前缀 `None`，与推理侧编辑指令 CFG 完全对齐）。

## 2. 中段噪声-干净桥训练目标（`train.isochronic_manifold_update`）

令源图为 $S$、目标图为 $T$、读取时间为 $t_0 = 0.36$、训练上限时间为 $\tau_{\max} = t_{\text{stop}} = 0.60$：

1. 构造 $t_0$ 时刻源噪声态：$a_{t_0} = t_0 S + (1 - t_0)\sigma\epsilon$。
2. 仅在中段语义成型窗口 $\tau \in [t_0, \tau_{\max}]$ 内采样时间（以 `bridge_start_prob = 0.15` 精确抽中入口 $\tau = t_0$，其余均匀采自 $[t_0, \tau_{\max}]$），彻底避免在 $\tau > 0.60$ 的尾段高频区域浪费 LoRA 容量。
3. 沿噪声匹配桥构造中间态：
   $$\alpha = \frac{\tau - t_0}{1 - t_0}, \qquad q_\tau = (1 - \alpha)a_{t_0} + \alpha T$$
4. 按 `edit_prompt_dropout = 0.10` 随机丢弃编辑指令，在有效画布区域（非黑边区）最小化全频干净目标均方误差 `clean_target_mse(H_theta(q_tau, tau, feedback), T, valid)`。

## 3. 唯一项目配置与 64 卡启动

唯一配置文件：`configs/u15_isochronic_state/isochronic_manifold.json`（继承 `core.json`）。

```bash
cd /data/zyd_workspace/bagel-LatentCoT/code/bagel-loop-layer-NPU
PY=/data/zyd_workspace/bagel-LatentCoT/venvs/u15-NPU/bin/python

"$PY" scripts/launch_train64.py \
  --linear /path/to/train_manifest.jsonl \
  --config configs/u15_isochronic_state/isochronic_manifold.json \
  --output /path/to/new-training-output
```
