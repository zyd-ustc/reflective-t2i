# Isochronic Manifold T2I Self-Feedback (`isochronic_manifold_v1`)

本项目面向统一多模态生成模型（`SenseNova-U1.5-8B-MoT`）及纯文生图自反馈闭环场景，实现**严格等时流形跳转算子（Strict Isochronic Manifold Transition Operator）**与**多步显影审图门控**。

核心思路：在原生 T2I 生成轨迹的早期关键时刻 $t_0 = 0.36$ 暂停，通过 4 步子网格显影预览触发自反思审图与零偏置门控；当需要修改时，编辑器仅接收带噪状态和纯文本编辑指令 `Edit request: <feedback>`，在短程语义成型窗口 $\tau \in [0.36, 0.60]$ 内预测干净目标估计，并**严格等时回投至同一时刻 $t_0 = 0.36$ 的原生噪声流形** $\hat{z}_{0.36}$，再交还给冻结原生 T2I 底座精修 $\tau \in [0.36, 1.0]$ 的高频细节与次要约束。

## 准备与环境

在已安装 PyTorch 与 `torch_npu` 的 Ascend 环境中执行：

```bash
pip install -e .
```

`modelarts-64b` 默认路径（权重、源码与训练产物均独立存放在代码仓库之外）：

```bash
PY=/data/zyd_workspace/bagel-LatentCoT/venvs/u15-NPU/bin/python
MODEL=/data/zyd_workspace/models/SenseNova-U1.5-8B-MoT
SOURCE=/data/zyd_workspace/bagel-LatentCoT/code/sensenova-u15-c35c83b
CONFIG=configs/u15_isochronic_state/isochronic_manifold.json
CKPT=/data/zyd_workspace/isochronic_data_20261107/isochronic_manifold_400k_20261119/step_006001
```

## 快速入口

- **推理与文生图自反馈闭环**（见 [推理指南](docs/inference.md)）：
  ```bash
  "$PY" -m qwen_latent_cot.u15_target_correction.cli t2i \
    --model "$MODEL" --source "$SOURCE" --config "$CONFIG" \
    --prompt "a lion beside seven zebras" \
    --checkpoint-mode trained --checkpoint "$CKPT" --output /path/to/out
  ```
- **64 卡训练**（见 [训练指南](docs/train.md)）：
  ```bash
  "$PY" scripts/launch_train64.py \
    --linear /data/zyd_workspace/isochronic_data_20261107/prod_1m_glm53_20261118/final_manifests/manifest_*.jsonl \
    --config "$CONFIG" --output /path/to/new-training-output
  ```
- **数据构建与核验**（见 [数据指南](docs/data.md)）：支持 64 卡 `Qwen-Image-2.1` 反向合成计数/空间样本与多进程 `GLM-5.3-flash` 语义任务覆盖清洗。
- **评测体系**（见 [评测指南](docs/benchmark.md)）：支持 `GenEval`、`GenEval++`、`GenEval2`、`T2I-CompBench++` 64 卡全量闭环生成与独立 `Qwen3-VL-8B-Instruct` 判分，以及单题调试评测。
