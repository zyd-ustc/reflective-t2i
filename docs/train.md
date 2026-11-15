# Stage 1 中间状态转换训练

Stage 1 训练中间状态转换器：给定源图状态和编辑指令，在固定生图时间上推进状态，再由 SenseNova-U1.5 接续生成。数据字段见 [data.md](data.md)，推理见 [inference.md](inference.md)。

## 模型与条件

底座参数冻结。每层 GEN 注意力的 q、k、v、o 投影和 GEN MLP 的 gate、up、down 投影添加秩为 64 的 LoRA，另训练编辑进度嵌入 edit_s_embedder。LoRA 的 B 矩阵和进度嵌入末层从零初始化。

条件协议是 isochronic_condition_v2。编辑指令整理为 `Edit request: <instruction>`，分别构建无文本基底前缀与指令前缀。当前图像状态通过生成分支输入。

## 训练目标

令 x_a、x_b 为源图和目标图，t 为生图时间，s 为编辑进度，ε 为按原生模型噪声尺度采样的同一份噪声。训练构造：

```text
a_t = t × x_a + (1 - t) × ε
b_t = t × x_b + (1 - t) × ε
q_s = (1 - s) × a_t + s × b_t
```

s 从 0 到 1 随机采样。冻结分支计算源锚点预测 h0 = H0(a_t, t)，带 LoRA 的条件分支计算 Hθ(q_s, t, s, instruction)。编辑速度与损失是：

```text
uθ = t × [Hθ(q_s, t, s, instruction) - h0]
L  = mean_valid [((uθ - (b_t - a_t)) / t)²]
```

共享噪声使 b_t - a_t = t × (x_b - x_a)。因此损失让预测差分 Hθ - h0 拟合图像差分 x_b - x_a。mean_valid 仅在有效画布区域求均值。训练对一个随机编辑进度做一次梯度更新；推理沿编辑轴做多步 Euler 积分。

## 当前配方

`configs/u15_isochronic_state/stage1_full490k.json` 继承 `core.json`：

| 项目 | 值 |
|---|---:|
| 画布 | 512 × 512 |
| 生图步数 / 时间偏移 | 50 / 3.0 |
| 读取和训练时间 t | 0.36 |
| 推理编辑步数 | 8 |
| LoRA 秩 / 全局批次 | 64 / 64 |
| 数据与损失 | linear_pair、L_edit，权重均为 1 |
| 不变样本比例 | 0 |
| 最大学习率 / 末学习率 | 5e-5 / 1e-5 |
| 预热 | 500 步 |
| AdamW | betas (0.9, 0.95)，eps 1e-8，weight decay 0 |
| 梯度裁剪 | 范数 1.0 |
| 每对样本预算 / 最多步数 | 2 / 14,723 |
| 保存间隔 / 随机种子 | 1,000 步 / 17 |

步数预算按**遍历次数**计算：一遍为 `ceil(训练记录数 / 全局批次)`，`epochs` 遍即乘上它。旧的 `max_steps` 上限于本轮移除——它是按 49 万条数据定的，面对千万级数据会静默截断遍历。

采样是无放回打乱遍历：每个 epoch 打乱全部训练索引，各 rank 取不重叠的记录，所以 world_size 个 step 恰好覆盖 world_size 条不同记录，走完一遍再重新打乱。`task_weights` 已废弃，所有记录等权，配置里出现它会直接报错。梯度跨进程求平均；学习率在预热后按余弦曲线下降。训练循环按 save_every 和最后一步保存检查点，sampler 的 epoch/cursor 与打乱顺序一并写入检查点，恢复时从原位置继续而不是重开一遍。

以 1,510.9 万条、全局批次 64 计：一遍 236,091 步，两遍 472,182 步。

## 启动与恢复

以下在有四个 16 卡节点的 ModelArts 作业内执行。启动器读取 VC_WORKER_HOSTS 并检查节点状态：

```bash
cd /data/zyd_workspace/bagel-LatentCoT/code/bagel-loop-layer-NPU
PY=/data/zyd_workspace/bagel-LatentCoT/venvs/u15-NPU/bin/python
DATA=/data/zyd_workspace/isochronic_data_20261107/linear_nocaption/L_full_490k.json
CONFIG=configs/u15_isochronic_state/stage1_full490k.json

"$PY" scripts/launch_train64.py \
  --linear "$DATA" --config "$CONFIG" \
  --output /path/to/new-training-output
```

使用 `--dry-run` 查看启动命令。恢复时添加 `--resume /path/to/step_XXXXXX` 并指定新的输出目录。恢复校验阶段、进程数、配置、模型身份、数据版本和总步数，同时恢复优化器、采样器和各进程随机状态。

## 训练产物

项目使用的检查点是 `/data/zyd_workspace/isochronic_data_20261107/train_full490k/step_014723`。每个 step 目录包含 manifest.json 和各 rank 的状态文件；rank 0 文件保存可训练权重与优化器状态。train.jsonl 按步记录 loss、学习率、梯度范数、样本 ID、编辑进度、耗时和显存；complete.json 记录步数、数据版本、进程数和采样权重。推理时加载编辑器权重并校验底座身份、参数布局和条件协议。

## 画布预处理版本

`shared_field_of_view_v1` 在计算宽图的居中矩形时用了 `height = width * ratio`，把每一张横图都顶成整块方形画布，画面被横向挤压。v2 改为 `height = width / ratio`。在 490k 清单上抽查 1200 条，横图占 11.8%、方图 71.7%、竖图 16.6%，也就是约 5.8 万条训练样本受此影响，空间类编辑（T9 共 5.9 万条）首当其冲。`benchmarks/cases.py` 的 `restore_geometry` 一直是按 `source_size × scale` 做含入，所以只有修好 rect 之后，宽图输出才与评测的还原口径自洽。

版本记录方式：`GEOMETRY_PROTOCOL` 升为 `shared_field_of_view_v2`，并写入 `editor_identity(..., "geometry")`，随每个 checkpoint 的 `manifest.json` 落盘。加载时 `geometry_gate` 校验：记录为 v2 才放行；记录为 v1 或完全没有记录（旧 checkpoint 全部属于 v1）一律拒绝。确需查看旧权重时显式设 `U15_ALLOW_LEGACY_GEOMETRY=1`，训练不得使用。

现有 Stage 1 检查点 `step_014723` 的 `identity` 里没有 `geometry` 字段，即按 v1 预处理训练。因此它**不能再作为热启动来源**，否则新旧预处理会混在同一条训练曲线里。Stage 2 需要在修正后的预处理上重新训练，再在其基础上做 rollout 与尾段。

`launch_train64.py` 会在启动进程前读取 `--resume`、`--init-checkpoint`、`--tail-editor-checkpoint` 的 manifest 做同一项校验，所以配错时在单机就能失败，不会先把 64 卡作业拉起来再崩。

代码：`qwen_latent_cot/u15_target_correction/train.py`、`model.py`、`loss.py`、`checkpoint.py`；启动器：`scripts/launch_train64.py`。

## Stage 2：在线 rollout 与独立尾段生成器

Stage 2 在**修正预处理后重训的** Stage 1 检查点上继续训练，仍固定读取时间 t = 0.36（旧的 `step_014723` 属于 v1 几何，见下一节，不能直接用来热启动）。三份配方按用途分开：

| 配置 | 组件 | 参数化 | 关键项 |
|---|---|---|---|
| `editor_v2_rollout_lora.json` | editor | rank-64 LoRA | `rollout_prob` 0.6、`terminal_rollout_prob` 0.1、`edit_prompt_dropout` 0.1、`edit_cfg_scale` 1.5 |
| `editor_v2_rollout_gen_full.json` | editor | `gen_full` | 同上，另加 `zero_optimizer` 分片优化器状态 |
| `tail_v1_lora.json` | tail | rank-64 LoRA | `tail_t_min` 0.36、`tail_native_prob` 0.2、`tail_prompt_dropout` 0.1 |

编辑器每步按比例抽一种监督方式：`rollout_step` 做 k 步无梯度前缀 rollout 后在剩余终点上回归，`rollout_terminal` 对完整 K 步终端状态求导，`ifm` 保留 Stage 1 的线性插值样本。非 keep 样本按 `edit_prompt_dropout` 置空指令但仍监督到 b_t；keep 样本（x_b 取源图自身）不参与 rollout，直接把编辑进度上任何非零速度压到零，因此“保持不变”真正学成空操作。

尾段生成器独立于编辑器，覆盖 t ∈ [0.36, 1.0]，既接受采集到的 q_K → x_b，也混合原生 T2I 样本。

### 编辑 CFG 的 joint_prefix 不是无条件分支

`edit_state` 与 `rollout_prefix` 的 `joint_prefix` 参数含义是**替换有条件分支** `prefixes["joint"]`，无条件分支始终取 `prefixes["empty"]`。若把空前缀传进去，CFG 变成

```text
u_cfg = u_empty + w × (u_empty − u_empty) = u_empty
```

采集到的 q_K 完全不携带编辑指令，却仍与该指令的目标图配对，尾段就被迫补做编辑器根本没做的修改。修复前 `collect_tail_states.py`、`inference.py`、`heldout_edit_test.py`、`t2i_feedback_bench.py` 四处都有这个写法，现统一为不传 `joint_prefix`，由 `prefixes["joint"]` 作条件分支。`field._resolve_joint` 会在 joint 与 empty 相同时直接报错，避免同类错误再次静默发生。

该错误此前未造成数据污染：服务器上没有任何已采集的尾段状态目录。

### 尾段的条件来自 target caption

尾段模型需要“目标画面长什么样”的文本条件。`linear_nocaption/L_full_490k.json` 的每一行都没有 caption（`tail_condition` 出现 0 次，`tail_source` 全为 `none`），若不加约束，q_K → x_b 的监督就退化成无条件的自指复制。因此：

- 训练清单缺 target caption 时 `LinearManifest` 直接报错，不再静默放行；
- 用 `--target-captions` 传入 uid → caption 的 JSONL 覆盖（键名 `parent_condition` 亦可）；
- 现有 caption 只覆盖约 3 万条 uid（`isochronic_data_20261106/captions*/`，与 490k 清单交集 3 万），其余行需先补齐 caption 才能进入尾段训练；
- 清单声明 `derived_parent_plus_edit` 时受 `allow_derived_tail` 约束，当前 `core.json` 为 `false`。

### 启动

```bash
cd /data/zyd_workspace/bagel-LatentCoT/code/bagel-loop-layer-NPU
PY=/data/zyd_workspace/bagel-LatentCoT/venvs/u15-NPU/bin/python
DATA=/data/zyd_workspace/isochronic_data_20261107/linear_nocaption/L_full_490k.json
INIT=/path/to/stage1-retrained-on-shared_field_of_view_v2/step_XXXXXX

# 编辑器热启动（必须先按 v2 几何重训 Stage 1）
"$PY" scripts/launch_train64.py \
  --linear "$DATA" --config configs/u15_isochronic_state/editor_v2_rollout_lora.json \
  --init-checkpoint "$INIT" --output /path/to/editor-v2-output

# 尾段：先采集 q_K，再带 caption 训练
"$PY" scripts/collect_tail_states.py \
  --manifest "$DATA" --config configs/u15_isochronic_state/tail_v1_lora.json \
  --model /data/zyd_workspace/models/SenseNova-U1.5-8B-MoT \
  --source /data/zyd_workspace/bagel-LatentCoT/code/sensenova-u15-c35c83b \
  --checkpoint /path/to/editor-v2-output/step_XXXXXX --out /path/to/tail-states

"$PY" scripts/launch_train64.py \
  --linear /path/to/tail-states/manifest.json \
  --target-captions /path/to/target_captions.jsonl \
  --config configs/u15_isochronic_state/tail_v1_lora.json --output /path/to/tail-output
```

`launch_train64.py` 自身导入 `qwen_latent_cot`，因此需在仓库根目录执行（或设置 `PYTHONPATH`），否则会报 `ModuleNotFoundError`。`--dry-run` 只打印启动命令，但同样会校验配置并读取清单。

## 全量编辑数据池：转换与划分

UniMoE clean 池（`/data/zx_wang_shared_work/ModelScope-zx/clean`）的编辑集转换成一个清单，用 `scripts/build_unimoe_edit_manifest.py` 生成。

```bash
"$PY" scripts/build_unimoe_edit_manifest.py \
  --out /data/zyd_workspace/isochronic_data_20261107/unimoe_edit/L_edit.jsonl \
  --val-fraction 0.002 --test-fraction 0.002 --verify-geometry
```

字段映射：`image[0]` → `source_image`，`image_gen` → `target_image`，`conversations[human]`（去掉 `<image>` 与句首模板包装）→ `edit_instruction`，`input_caption` → `parent_condition`，`output_caption` → `tail_condition`，`uid` 取 `数据集名:原始id`。

**划分按图片家族，不按记录 id。** 同一张源图常对应多条编辑指令，按记录划分会让它们跨越训练与验证集，loader 的 `_remember` 会直接拒绝。`source_family` 由 `数据集名 + 规范化源图路径` 哈希得到，同族样本整体落在同一侧；seed 固定，重跑得到相同划分。

编辑器的理解侧仍是纯文本接口，caption 不进入它的图像条件。`output_caption` 补的是**尾段的文本条件**，以及与实际 T2I 接口的对应关系——缺它时尾段是无文本条件的“状态到图像”重建，而非自指复制，损失始终监督目标图 `x_b`。

完整链路是两段，不能只补字段就当尾段数据就绪：

```text
转换后的编辑对
  ├─ 直接训练编辑器
  └─ 经编辑器 rollout 得到 q_K（依赖上面修好的编辑 CFG）
        └─ q_K + output_caption + target_image 训练尾段
```

过滤只针对有效性：图片不可读、指令为空、源/目标宽高比不兼容、重复样本、同源跨 split 泄漏、尾段 caption 缺失。**不按任务类型排除样本**——风格、背景、属性、替换、全局编辑都保留。HQ-Edit 首轮排除的理由是几何协议不兼容（抽样源/目标宽高比一致率仅 8.8%），不是任务归属。每类过滤的计数写入同名 `.report.json`，便于追溯。通用 VLM 质量分不作为计数/空间标签已核验的依据。

## 全量数据池的启动实录

几何校验（`scripts/verify_manifest_geometry.py`，256 进程只读图片头）覆盖全部 15,103,630 条，拒绝 161,130 条（1.07%），全部是源/目标宽高比不一致：X2Edit 109,291、ImgEdit 37,245、OpenGPT-4o 14,361、AnyEdit 169、SEED 64。其中 ImgEdit 一段连续 id 的目标图是正方形（如源 1.5938 对目标 1.0000），Excel 之外的分块坏数据，抽样没有命中。过滤后 14,942,500 条，训练 14,882,904、验证 29,788、测试 29,808，0 个家族跨 split。

启动阶段踩到并修掉的三个问题：

1. **`prompt_family` 用错了键。** 原先取 `数据集名 + 指令文本`，导致同一句话配不同源图时跨 split，loader 的 `_remember` 直接拒绝。现与 `source_family` 一致，同源图上多指令的整体绑定。
2. **`max_steps` 已移除**，预算改为 `epochs × ceil(训练记录数 / 全局批次)`。14,882,904 条、64 卡、2 遍 = 465,092 步。
3. **清单读取必须预先分片。** 未分片时 64 个 rank 各自流式读完整清单，每个进程读 26 GB 后卡在 `rpc_wait_bit_killable`，35 分钟没有出第一步。`scripts/shard_manifest.py` 按行轮转切成 `shards64/shard_0000x.jsonl`，每片约 210 MB，训练器按 rank 只读自己那片。

实测每步 1.6–1.7 秒，显存峰值 36.2 GiB，465,092 步约 9 天。监控用 `scripts/monitor_training.py`，超过 30 分钟没有新行即判定 stalled 并以非零码退出。
