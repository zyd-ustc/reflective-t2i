# Stage 1 数据与评测数据

Stage 1 使用成对的源图、目标图和编辑指令。训练清单与图片位于代码仓库之外；读取、检查和采样由 `qwen_latent_cot/u15_target_correction/data.py` 实现。

## 训练清单

项目的全量清单路径为：

```text
/data/zyd_workspace/isochronic_data_20261107/linear_nocaption/L_full_490k.json
```

文件是包含 `records` 数组的 JSON。每条记录对应一组编辑前后图像，基本结构如下：

```json
{
  "uid": "example_0001",
  "split": "train",
  "source_image": "/path/to/source.png",
  "target_image": "/path/to/target.png",
  "edit_instruction": "Change the colour of the bag to blue, and keep everything else unchanged.",
  "source_family": "source_0001",
  "prompt_family": "prompt_0001",
  "semantic_approved": true,
  "condition_protocol": "isochronic_condition_v2"
}
```

`uid`、`split`、源图、目标图和编辑指令是训练记录的核心字段。`split` 取 `train`、`val` 或 `test`，训练采样只使用 `train`。指令按 `edit_instruction`、`edit_condition`、`instruction` 的顺序读取，连续空白会合并。图片路径既可为绝对路径，也可相对清单所在目录。

加载器检查图片存在、指令非空、样本的语义审批状态和条件协议。源图族及提示词族各自保持在同一数据划分。未显式提供家族时，源图族根据源图路径生成，提示词族使用 uid。

可选字段包括 `task`、`role`、`geometry`、`parent_condition`、`tail_condition` 和 `preserve`。Stage 1 的编辑前缀使用编辑指令；源图与目标图用于构造状态和训练目标。当前配方只采样 `L_edit`，不采样 keep 样本。

## 图像与几何

训练将源图和目标图转换为 RGB，并映射到同一 512 × 512 画布，数值归一化到 [-1, 1]。两张图共享画布矩形和有效区域 mask，损失在有效区域计算。

整图输入时，源图和目标图的宽高比差异阈值为 0.01。记录也可用 `geometry.source` 和 `geometry.target` 指定各自的裁剪框；两个裁剪区域需具有相近宽高比。图像按同一矩形缩放，其他画布区域填黑。

矩形由 `images._rect_for_ratio` 确定：窄图的矩形宽度为画布高度乘以宽高比；宽图的矩形高度按画布宽度乘以宽高比并限制在画布高度内。评测入口另行使用等比缩放和黑边填充，生成后恢复源图尺寸。

## 采样与版本

加载器建立 `train` 记录池，训练按 rank 初始化独立随机数生成器，每步随机选择记录与配置中的时间。当前配置的时间只有 0.36。一次样本将源图和目标图与同一份噪声混合，形成源状态与目标状态；模型学习两者之间的状态变化。

训练以清单路径、文件路径和内容生成 `data_version`，并把它写入检查点与 `complete.json`。恢复训练时校验该版本。训练和推理的条件与损失详见 [train.md](train.md)。

## 评测数据

评测按文生图和编辑两类组织，每次按 `record_id` 读取一条冻结记录：

- 文生图读取原始 prompt 和图像种子；清单提供评分 prompt 时，程序检查它与原始 prompt 一致。
- 编辑读取原始编辑指令、源图和种子。传入复制后的源图时，程序使用记录中的 `source_sha256` 校验。
- 生成结果的 `case.json` 保存题目、种子、生成配置和图片绑定；评分前再次校验题目与图片。

冻结清单、具体榜单和单题评分命令见 [benchmark.md](benchmark.md)。评分结果单独保存，记录榜单、题目、图片、模型、原始回答和有效状态。

代码：`qwen_latent_cot/u15_target_correction/data.py`、`images.py`、`conditions.py`；评测读取：`qwen_latent_cot/benchmarks/cases.py`。
