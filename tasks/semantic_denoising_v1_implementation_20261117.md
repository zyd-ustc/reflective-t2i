# 文本反馈驱动的语义纠错去噪：第一版实现说明

日期：2026-11-17。交付物：供用户实施的代码设计与配方；本文没有实施代码修改，也没有启动训练。

项目根目录：`/Users/zyd/Documents/LCoT-codex/bagel-loop-layer-NPU`。

## 1. 固定本版合同

从源场景噪声态 `a = 0.36*S + 0.64*epsilon` 出发，让保留原生 clean 输出头的 GEN LoRA 分支，依据文本反馈，多步去噪到目标图 T。

首版只实现这一条路线：

- 固定起始时间 `t0=0.36`，训练路径终点为 `T`，对应时间 1。
- 推理网络输入只有当前状态、当前去噪时间和反馈文本。没有图像前缀、源图拼接或目标图条件。
- 使用原生 clean 输出头及其预训练权重，不创建、加载或清零 displacement head，不增加编辑进度 s。
- 训练 GEN rank-64 LoRA；原生 GEN 主干、原生输入模块、原生输出头和 UND 都冻结。
- 一次训练样本只做一次 GEN 前向与反传。多步发生在推理，不在训练中展开 ODE。
- 从 t0 到 1 全程使用同一个适配分支。不在中途切回冻结底座，不在结果之后再接一次 tail。
- 无 bbox、区域损失、低频滤波、位移损失、H0 相减、教师缓存、蒸馏、时间课程或独立 tail 检查点。
- 旧 anchored、direct_displacement 和 tail 路径保留原行为。新增接口不能默默改变旧检查点含义。

### 首版的文本选择

选择 **feedback-only**：完整反馈指令进入同一个固定模板，训练与推理完全一致。

```
Edit request: {feedback}
```

这是有意的范围选择：当前配对数据可靠可用的条件是编辑指令，原始 T2I prompt 并非全量存在。原始 prompt 继续用于 T2I 初始生成和诊断，但第一版不作为纠错网络的第二文本输入。目标 caption、源图 caption、目标图生成的文字均不进入纠错分支。不补 caption 来满足旧 tail 接口。

首版纠错 CFG 为 1、反馈 dropout 为 0：每步只做有条件前向。不训练一个不知道反馈却仍要完成指定编辑的空文本分支。

## 2. 数学与张量合同

S、T：经现有 geometry v2 处理的 RGB 张量，`[B,3,H,W]`，范围 `[-1,1]`。当前训练仍是每卡一个样本。

epsilon：`randn_like(S) * rt.noise_scale()`。它已含原生噪声尺度；禁止再乘第二次 sigma。

```
t0 = 0.36
a = t0*S + (1-t0)*epsilon
alpha = (tau-t0)/(1-t0)
q_tau = (1-alpha)*a + alpha*T
```

其中 `tau` 是原生去噪时间，不是 alpha，也不是此前的编辑进度 s。

精确桥速度与原生 clean 预测的关系：

```
v_star = (T-a)/(1-t0)
v_theta = (H_theta(q_tau, tau, feedback)-q_tau)/(1-tau)
v_theta-v_star = (H_theta-T)/(1-tau)
```

因此监督 clean 输出为 T 与监督桥速度有同一个逐样本零误差目标。模型输入不包含 a、S、T 或 epsilon 的额外通道；它们只用于构造训练状态和损失。

### 时间采样

使用一个固定分布，不做课程：

- 10% 样本：`tau=t0`，直接覆盖部署入口，网络完全看不到插值进去的目标图。
- 90% 样本：`tau ~ Uniform[t0,1)`。
- 使用 PyTorch RNG，以便现有检查点的 RNG 恢复机制覆盖时间和噪声抽样。组合状态前转 float32，避免低精度时间被舍入到 1。

`sample['time']` / `state_times=[0.36]` 表示入口时间；训练另采 `tau`。不要把整个 `[0.36,1)` 写入 `state_times`，也不要调用时间课程。

## 3. 首版损失：全频 clean MSE，明确其 FM 权重

本版选择：

```
L = mean_valid((H_theta(q_tau,tau,e)-T)**2)
```

这是**全频、时间加权的 Flow Matching**。在未截断的连续公式下，它等价于：

```
L = (1-tau)**2 * mean_valid((v_theta-v_star)**2)
```

这是本说明的明确配方选择，不把它叫作“未加权速度 MSE”。选择原因：保持原生 clean 输出含义与全部像素频率，同时不人为把接近终点的误差放大为入口误差的上百倍。首版重点是学会从源状态产生正确内容，不是继续给几乎已混入完整 T 的训练点增加时间权重。

**不要直接调用当前 `tail_flow_matching`。** 它对 clean 误差除以 `max(1-tau,t_eps)`，实现的是另一种时间权重；不要不经说明地沿用。

新增 `semantic_clean_mse`，只复用已有有效画布归约：

```python
def semantic_clean_mse(predicted_clean, target_clean, valid):
    error = (predicted_clean.float() - target_clean.float()).square()
    weight = valid.float()
    return (error * weight).sum() / weight.sum().clamp_min(1.0) / error.shape[1]
```

valid 的形状为 `[B,1,H,W]`，用于排除 letterbox。它不是编辑区域。监督包含目标图的全部有效像素，不添加“贴近源图”的背景保持项。

训练中不 clamp H、q 或残差，不做 avg_pool、不做低通。只在保存展示图片时 clamp，并保留未截断输出的越界统计。

## 4. 模型：新增 SemanticDenoiser，不把旧 TailGenerator 改名混用

目标文件：`/Users/zyd/Documents/LCoT-codex/bagel-loop-layer-NPU/qwen_latent_cot/u15_target_correction/model.py`。

新增接口常量 `semantic_denoising_v1`，新增类 `SemanticDenoiser(_GENFlowBase)`，由 `build_editor` 显式分派。

接口示意：

```python
class SemanticDenoiser(_GENFlowBase):
    def __init__(self, rt, cfg):
        super().__init__(rt, cfg, parameterization="gen_lora",
                         lora_rank=int(cfg["lora_rank"]))
        self.clean_calls = 0

    @torch.no_grad()
    def feedback_prefix(self, feedback):
        # 调用唯一共享文本模板；空反馈在输入边界报错。
        prefix = self.rt.prefix(semantic_feedback_prompt(feedback))
        return (prefix.cache, prefix.indexes, None)

    def predict_clean(self, state, tau, prefix):
        self.clean_calls += 1
        return self._clean(state, float(tau), prefix,
                           s=None, use_adapted=True)

    def predict_velocity(self, state, tau, prefix):
        clean = self.predict_clean(state, tau, prefix)
        # 与当前原生推理使用的有效 t_eps 保持一致。
        t_eps = float(getattr(self.model.config, "t_eps", self.config["t_eps"]))
        return (clean - state.float()) / max(1.0-float(tau), t_eps)
```

注意：

- `_clean` 使用原生 `fm_modules['fm_head']`，不传 `head_module=self.displacement_head`。
- 不构造 `edit_s_embedder`、`displacement_head` 或第二个适配生成器。
- 不要给 head 做 `_zero_last_output_layer`。只有 LoRA 按原有方式 B=0 初始化。
- native/ adapted layer view 的现有隔离方式保留，UND 路径和原生 T2I 不载入纠错 LoRA。
- 首版只接 gen_lora；不为尚未使用的 gen_full 扩展功能。

在零 LoRA 初始化时，给完全相同的文本前缀、状态和时间，该分支应与冻结原生分支预测一致。它不是零位移，也不是恒等输出。

## 5. 训练更新：直接在线构造桥，不经过旧编辑器

目标文件：`/Users/zyd/Documents/LCoT-codex/bagel-loop-layer-NPU/qwen_latent_cot/u15_target_correction/train.py`。

新增 `_semantic_denoising_update`。在 `train` 中按新接口分派；仍走 `LinearManifest + TrainingSampler`，不走 `TailManifest`，不读取 teacher-cache 或 real-states。

核心代码示意：

```python
def _semantic_denoising_update(module, sample, cfg):
    source = sample["source"].float()
    target = sample["target"].float()
    valid = sample["valid"]
    t0 = float(cfg["read_time"])

    eps = torch.randn_like(source) * module.rt.noise_scale()
    anchor = t0 * source + (1.0-t0) * eps

    start_draw = torch.rand((), device=source.device).item()
    if start_draw < float(cfg["bridge_start_prob"]):
        tau = t0
    else:
        u = torch.rand((), device=source.device).item()
        tau = t0 + (1.0-t0) * u

    alpha = (tau-t0)/(1.0-t0)
    state = (1.0-alpha)*anchor + alpha*target
    prefix = module.feedback_prefix(sample["edit_request"])
    predicted_clean = module.predict_clean(state, tau, prefix)
    loss = semantic_clean_mse(predicted_clean, target, valid)
    # 返回现有训练循环所需 fields；见下文。
    return loss, fields
```

返回 fields 至少包含当前训练循环读取的所有键：

```
mode=semantic_denoising
role=sample['role']
time=tau
progress=alpha
start_time=t0
loss_branch=full_clean_mse
keep=0
prompt_dropped=0
rollout_step=None
rollout_cfg_used=0
preview_calls=0
baseline_calls=0
state_calls=1
clean_calls=1
```

这里 `progress=alpha` 只是兼容日志，不能送入模型。将 `start_time/clean_calls` 加入可选日志字段。`time` 一律表示实际网络输入时间 tau。

训练继续复用当前优化器、广播、梯度同步、裁剪、存档和完成记录。新模式启动时校验实际 world size 等于配置 global_batch，避免现有“每卡一条”实现与配置口径不一致。不要新增梯度累积机制。

**首版从原生基础模型初始化，不加载旧 anchored 或 displacement 检查点。** 旧 adapter 曾为不同输出语义优化，不用它来定义本次起点。

## 6. 数据：复用现有图像对，不引入 bbox 或 caption 前置条件

目标文件：`/Users/zyd/Documents/LCoT-codex/bagel-loop-layer-NPU/qwen_latent_cot/u15_target_correction/data.py`。

继续使用现有 128k train / 1k val 清单与 source_family 划分；保留 geometry v2 的 load_pair。

新模式训练真正需要的字段只有：

```
uid, split, source_family,
source_image, target_image, edit_instruction,
geometry（仅当图像对需要声明裁剪）
```

task / semantic_task / 来源字段继续用于日志。不需要 edit_region、target_caption、tail_condition、teacher seed 或教师张量。

当前 `LinearManifest` 与 `sample` 还有一些旧 tail/region 逻辑，新接口应明确绕过：

- 不要求 target caption，不根据 `tail_source` 或 `derived_parent_plus_edit` 拒绝本版不会读取的尾段条件。
- 不读取区域图或区域框，不因无区域标注而失败。
- 不改变 S/T 方向，不将 `T-S` 写作训练目标。
- 旧 manifest 中的 `condition_protocol=isochronic_condition_v2` 作为已有配对数据的来源声明允许读入；进入新训练样本时绑定新的文本协议。不要要求重写整批图片和清单，也不要把旧 checkpoint 条件协议放宽。
- 这些豁免仅属于新接口，旧接口保持原校验。
- 已明确发现的错指令不进入本轮训练；不要把 `semantic_approved` 的缺省 true 当成做过语义核验。

keep_fraction=0，native_mix=0：第一版不增加额外样本混合。原生 T2I 通过关闭 adapter 保持原行为，不靠本轮追加训练来保持。

## 7. 推理：新增纠错接续入口，返回时间 1 的状态

目标文件：

- `/Users/zyd/Documents/LCoT-codex/bagel-loop-layer-NPU/qwen_latent_cot/u15_target_correction/field.py`
- `/Users/zyd/Documents/LCoT-codex/bagel-loop-layer-NPU/qwen_latent_cot/u15_target_correction/inference.py`
- `/Users/zyd/Documents/LCoT-codex/bagel-loop-layer-NPU/qwen_latent_cot/u15_target_correction/cli.py`

新增明确的接续函数，而不是把新类送进旧 `_edit` 的 duck typing 分支：

```python
@torch.no_grad()
def semantic_correct(module, state, start_time, feedback, grid):
    # 首版由入口校验 start_time == cfg['read_time'] == 0.36。
    prefix = module.feedback_prefix(feedback)
    times = segment_times(grid, start_time, 1.0)
    calls_before = module.clean_calls
    final = euler(
        state.detach().clone(), times,
        lambda z, tau: module.predict_velocity(z, tau, prefix),
    )
    return final, {
        "input_time": float(start_time),
        "output_time": 1.0,
        "correction_nfe": module.clean_calls-calls_before,
    }
```

采样规则：

- 复用原生 50 步、timestep_shift=3 的时间网格，仅插入 t0=0.36；不额外创建一个“纠错八步”网格。
- 使用 `segment_times(grid,0.36,1)` 与现有 Euler，调用次数根据实际网格记录，不硬编码为 19。
- prefix 每次纠错只建立一次；每个去噪步一次条件 GEN 前向。
- 中间状态与 head 输出不 clamp。不在 tau=1 调用网络。
- 推理保留原生 `max(1-tau,t_eps)`。连续桥推导在分母未截断时精确；终点附近采用原生数值截断，不宣称有限步采样严格重现理想直线。不要在本版另加末步投影或改变旧 sampler。

### 图像对入口

用相同预处理与固定 seed 构造 a，然后直接 `semantic_correct(a,0.36,e)` 得到结果。不得再调用 `_decode` 或原生 tail 从 0.36 跑第二遍。

验证图像对使用 `load_pair` 的共同 rect；普通用户单图编辑仍使用 `load_canvas`。不能在验证时忽略配对的声明裁剪。

### T2I 反馈入口

```
原生模型 + 原始 prompt：0 → 0.36
复制状态做预览 → 诊断得到反馈 e
若无需编辑：原生模型继续到 1
若需要编辑：SemanticDenoiser + e 从原始 z_0.36 继续到 1
```

预览不得修改主状态。纠错分支不读取预览图；第一版只执行一次反馈。已在时间 1 的输出不能作为仍处于 0.36 的状态再次送入编辑入口。

### CFG 与参数隔离

- `cfg_scale=4`、`cfg_interval=[0,0.8]` 仅用于原生 T2I、预览和未编辑原生接续。
- `semantic_cfg_scale=1` 用于纠错分支，整个后半程都只有有条件预测。
- 新接口拒绝 `--tail-checkpoint/--tail-config`；拒绝非 1 的 `--edit-cfg-scale`，不要静默忽略。
- 所谓 cfg=1 不表示无条件：纠错反馈一直保留。

### 返回产物与时间语义

新模式 CLI 保存 `final_state.pt`，而不是沿用让人误认为等时输出的 `edited_state.pt`。result.json 写入 `editor_interface`、`input_time=0.36`、`output_time=1.0`、反馈文本、实际采样时间节点、NFE、纠错耗时与原始输出越界率。

旧模式继续使用原有文件名和时间含义。不要全局改 CLI 存档文件名。

## 8. 检查点与配置：新语义单独登记

目标文件：

- `/Users/zyd/Documents/LCoT-codex/bagel-loop-layer-NPU/qwen_latent_cot/u15_target_correction/__init__.py`
- `/Users/zyd/Documents/LCoT-codex/bagel-loop-layer-NPU/qwen_latent_cot/u15_target_correction/config.py`
- `/Users/zyd/Documents/LCoT-codex/bagel-loop-layer-NPU/qwen_latent_cot/u15_target_correction/checkpoint.py`

新增 `EDITOR_INTERFACE_SEMANTIC="semantic_denoising_v1"`；component 仍为 editor，stage=1。避免仅因复用 TailGenerator 的计算，就保存为旧 tail checkpoint。

新 identity 至少绑定：

```
editor_interface: semantic_denoising_v1
parameter_layout: semantic_gen_lora_all_layers
condition_protocol: semantic_feedback_only_v1
prediction_type: clean_image
bridge_type: source_noised_to_target_clean_v1
read_time: 0.36
endpoint_time: 1.0
geometry: 当前 geometry v2 协议
native: 当前原生模型 identity
```

冻结原生 head 不需要重复存储在可训练参数文件；其身份随 native 模型绑定。新 identity 只在新分支增加字段，旧 identity 的构造结果保持不变。禁止新接口与旧 anchored/displacement/tail checkpoint 相互载入；不做自动参数迁移。

新配置校验应独立处理本模式条件，而不是放宽所有模式：

- 只接受 stage=1、component=editor、gen_lora。
- read_time=0.36、state_times=[0.36]、endpoint=1。
- 禁止 time_curriculum、rollout、terminal rollout、distill、teacher cache、real states、tail editor、keep 抽样和反馈 dropout。
- 不要求新模式具有 `edit_steps`，兼容继承字段若保留也不能用它控制采样或输出。
- 新条件协议只对新模式有效；旧协议仍严格匹配。

## 9. 建议配置

拟新增文件：`/Users/zyd/Documents/LCoT-codex/bagel-loop-layer-NPU/configs/u15_isochronic_state/semantic_denoising_v1.json`。

下面是实现完成后使用的配方，不是当前代码已经支持的配置：

```json
{
  "extends": "core.json",
  "stage": 1,
  "component": "editor",
  "editor_interface": "semantic_denoising_v1",
  "condition_protocol": "semantic_feedback_only_v1",
  "parameterization": "gen_lora",
  "lora_rank": 64,
  "read_time": 0.36,
  "state_times": [0.36],
  "state_time_probs": [1.0],
  "bridge_endpoint_time": 1.0,
  "bridge_start_prob": 0.10,
  "semantic_loss_type": "full_clean_mse",
  "semantic_cfg_scale": 1.0,
  "cfg_interval": [0.0, 0.8],
  "require_tail_condition": false,
  "keep_fraction": 0.0,
  "edit_prompt_dropout": 0.0,
  "edit_cfg_scale": 1.0,
  "rollout_prob": 0.0,
  "terminal_rollout_prob": 0.0,
  "lambda_distill": 0.0,
  "role_weights": {"L_edit": 1.0},
  "source_weights": {"linear_pair": 1.0},
  "lr": 0.00003,
  "lr_final": 0.000005,
  "warmup_steps": 200,
  "epochs": 1,
  "betas": [0.9, 0.95],
  "eps": 1e-8,
  "weight_decay": 0.0,
  "gradient_clip": 1.0,
  "global_batch": 64,
  "save_every": 500,
  "keep_checkpoints": 4,
  "seed": 31
}
```

core 继承的 512×512、原生 num_steps=50、timestep_shift=3、t_eps=0.05 和 activation_checkpointing 保留。epochs=1 是首轮预算，不预先承诺第二轮；128k train / 64 卡对应 2,000 个 optimizer steps。

## 10. 实现顺序与必要检查

按以下顺序改，避免先接上旧推理管线导致“纠错后又去噪一次”。

1. 新增接口、条件模板、模型类、identity 和配置分支。
2. 新增 full_clean_mse、桥采样和训练分派；新模式跳过无关区域/尾段字段。
3. 新增 semantic_correct，以及图像对/T2I 的明确推理分支和时间 1 存档。
4. 添加小范围测试，保留现有旧接口测试。
5. 单卡真实模型完成短训练和一次完整接续，再启动 64 卡首轮。

测试只覆盖本次改变的关键合同：

- 桥的两端：tau=t0 得 a；代数终点 tau=1 得 T；内部点满足 `q+(1-tau)*v_star=T`。
- 在未截断的内部时间，clean MSE 等于 `(1-tau)^2*velocity_MSE`。
- 高频错误有损失：用均值为零的棋盘差分，确认 full_clean_mse 非零；letterbox 不贡献损失。
- 零 LoRA 与原生同条件输出匹配；head 未被清零，head/UND 不更新；第一步 LoRA 梯度链路有效。LoRA 的零 B 初始化意味着不能要求 A/B 所有张量第一步都非零。
- 推理入口 state 不被原地改写；只有一个条件 prefix；NFE 等于所选 Euler 区间数；没有 H0/位移头/第二次 tail 调用。
- 新旧检查点交叉加载失败；新模式严格 resume 可恢复；旧接口原用法继续成立。
- 无 bbox、无 caption 的合格图像对可训练；新模式不读取这些无关字段。
- 新 cfg_scale 与原生 cfg_scale 分开，设置旧 edit CFG>1 在新模式明确报错。

不增加大规模探针矩阵。检查点图集固定使用已有人看的四类验证样本，至少保存 source、target、同状态同反馈的冻结原生接续、适配接续。冻结原生反馈基线使用相同模板、CFG=1、相同网格，避免把“给了反馈文本”本身算作训练收益。

主要判断是新增实体和位置变化是否实际出现。不得以低 loss、余弦、对象淡化或小位移通过作为晋级标准。先看 500/1000/2000 步的实际结果，不自动把失败结果送进 Stage 2。

## 11. 用户实施后的启动命令

前提：上述代码与配置已经实现并同步；现有 64 卡启动器所需的四 worker 环境可用，卡处于可启动状态。以下命令尚未执行。

在 modelarts-qwen 训练主节点执行：

```bash
cd /data/zyd_workspace/bagel-LatentCoT/code/bagel-loop-layer-NPU
PY=/data/zyd_workspace/bagel-LatentCoT/venvs/u15-NPU/bin/python
CFG=/data/zyd_workspace/bagel-LatentCoT/code/bagel-loop-layer-NPU/configs/u15_isochronic_state/semantic_denoising_v1.json
DATA=/data/zyd_workspace/isochronic_data_20261107/redesign_20261116/pairs_129k.jsonl
OUT=/data/zyd_workspace/isochronic_data_20261107/semantic_denoising_20261117/stage1_t036_128k

"$PY" scripts/launch_train64.py \
  --linear "$DATA" \
  --config "$CFG" \
  --output "$OUT" \
  --dry-run

# 上一条检查通过、配置与数据确认后启动。不要传旧 init-checkpoint。
"$PY" scripts/launch_train64.py \
  --linear "$DATA" \
  --config "$CFG" \
  --output "$OUT"
```

若在单独清单中剔除了已知错误样本，DATA 指向新清单，并使用该清单对应的 shards64，不沿用旧目录下自动发现的不同数据分片。实际步数按有效 train 数与 world size 计算，不强制保留 2,000。

新分支实现后的单图用法：

```bash
"$PY" -m qwen_latent_cot.u15_target_correction.cli edit \
  --model /data/zyd_workspace/models/SenseNova-U1.5-8B-MoT \
  --source /data/zyd_workspace/bagel-LatentCoT/code/sensenova-u15-c35c83b \
  --config "$CFG" \
  --device npu:0 \
  --checkpoint-mode trained \
  --checkpoint "$OUT/step_002000" \
  --image /absolute/path/to/source.png \
  --instruction "Add a blue umbrella on the right side of the scene." \
  --seed 17 \
  --output /absolute/path/to/new-semantic-result
```

替换最后一条中的输入/输出路径和实际检查点步数。新结果应包含 `final_state.pt` 和 output_time=1 的 result.json，不是一个等待再次续生成的等时编辑态。

## 12. 本版完成条件

工程完成：新模型按原生 clean 输出语义训练；一条适配分支从 t0 生成到 1；数据、检查点、推理返回时间均不与旧接口混淆。

能力目标：在固定验证图集中，看到新增对象形成、移动目标位置正确、删除后背景重新生成，而不是轮廓淡化。是否达到能力目标由实际图像决定；本文只交付实现方案。

本轮不交付多轮反馈、原生尾段切换或等时输出。真实 T2I 状态的首次反馈入口可以接通，但训练仍是合成图像对桥；两者分布差异保留为后续实际使用结果，而不宣称已被本轮训练消除。
