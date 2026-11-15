# 中间状态编辑与文生图反馈

推理使用 SenseNova-U1.5 原生生成模型和 Stage 1 状态转换器。配置在生图时间 t=0.36 读取状态，使用 8 步编辑，再接续生成终图。训练目标见 [train.md](train.md)，评测入口见 [benchmark.md](benchmark.md)。

## 环境与权重

以下命令在 modelarts-64 的项目目录执行：

```bash
cd /data/zyd_workspace/bagel-LatentCoT/code/bagel-loop-layer-NPU
PY=/data/zyd_workspace/bagel-LatentCoT/venvs/u15-NPU/bin/python
MODEL=/data/zyd_workspace/models/SenseNova-U1.5-8B-MoT
SOURCE=/data/zyd_workspace/bagel-LatentCoT/code/sensenova-u15-c35c83b
CKPT=/data/zyd_workspace/isochronic_data_20261107/train_full490k/step_014723
CONFIG=configs/u15_isochronic_state/stage1_full490k.json
```

`--checkpoint-mode trained --checkpoint "$CKPT"` 加载训练后的转换器；`--checkpoint-mode native` 使用零初始化 LoRA 和编辑进度嵌入。两种模式都执行状态编辑。默认设备是 npu:0，默认种子为 17；每次运行指定一个新输出目录。

## 源图编辑

```bash
"$PY" -m qwen_latent_cot.u15_target_correction.cli edit \
  --model "$MODEL" --source "$SOURCE" --config "$CONFIG" \
  --image /path/to/source.png --instruction 'Change the colour of the bag to blue.' \
  --checkpoint-mode trained --checkpoint "$CKPT" \
  --seed 17 --output /path/to/new-edit-output
```

1. 将源图加载到 512 × 512 画布，转换到 [-1, 1]。
2. 用种子采样噪声 ε，构造源状态 `a_t = t × source + (1 - t) × ε`。
3. 构建编辑指令前缀，在源状态计算一次冻结基底预测。
4. 固定生图时间 t，沿编辑进度 s 从 0 到 1 运行 8 步 Euler，得到编辑状态。
5. 用无条件原生文生图分支接续到终图；同一初始状态还生成不经过转换器的接续基线。

输出包括 `source.png`、`baseline.png`、`edited_preview.png`、`result.png`、`edited_state.pt` 和 `result.json`。baseline 是源图加噪后的无转换接续结果；result 是经过转换器的结果。result.json 记录指令、种子、读取时间、状态初始化、接续方式和检查点信息。

## UND 自反馈文生图

```bash
"$PY" -m qwen_latent_cot.u15_target_correction.cli t2i \
  --model "$MODEL" --source "$SOURCE" --config "$CONFIG" \
  --prompt 'a lion beside seven zebras' \
  --checkpoint-mode trained --checkpoint "$CKPT" \
  --seed 17 --output /path/to/new-t2i-output
```

1. 用原始 prompt 和噪声沿原生生图时间网格生成到 t=0.36。
2. 复制中间状态生成预览：先做一个 Heun 区间，再做一个 Euler 区间；主状态保持不变。
3. UND 看预览图，描述可见对象、数量、颜色、材质和位置。视觉输入经归一化、分块和特征提取后传给生成接口。
4. UND 文本判断读取原始 prompt 和图像描述，输出 MATCH 与 INSTRUCTION。提示词要求指令包含操作、对象、位置和数量变化。
5. MATCH 为 yes 时直接续生成图；为 no 且有指令时，编辑原始中间状态，再用原 prompt 续生成图。

`result.json` 保存描述、判断原文、解析结果和编辑指令。输出包含 `preview.png` 和 `result.png`；发生编辑时还包含 `edited_preview.png` 和 `edited_state.pt`。判断标签无法解析或需要编辑但没有指令时，流程报错。

## 原生生成与状态编辑

原生生成使用 50 步时间网格、`timestep_shift=3.0` 和 Euler 积分；分类器自由引导（CFG）系数为 4.0。网格中显式包含读取时间 0.36。

状态编辑的基底预测来自进入编辑时的锚点。每一步根据当前状态和编辑进度计算编辑速度，步长为 `1 / edit_steps`。源图编辑的续生成使用无文本前缀；文生图闭环的续生成使用原始 prompt。

## 评测生成的配对结果

`u15-benchmark generate --kind edit --checkpoint-mode both` 对同一源图、指令和种子输出：

- `baseline.png`：加噪状态直接接续；
- `native_result.png`：使用未加载检查点的转换器；
- `result.png`：使用训练后的转换器。

评测入口先将源图等比放入黑色画布，生成后裁去填充并恢复源图尺寸。输出还包含画布版本、编辑状态及绑定原题和图片的 `case.json`。文生图评测同样保存无编辑基线和闭环结果。数据、裁判和单题命令见 [benchmark.md](benchmark.md)。

代码：`qwen_latent_cot/u15_target_correction/inference.py`、`diagnosis.py`、`field.py` 和 `qwen_latent_cot/benchmarks/cli.py`。
