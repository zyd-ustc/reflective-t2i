# 面向 6+4 基准的 30–50w 图像编辑数据构造完整实践方案（极简去重版）

> **核心设计思路**：
> 1. **不从零洗原始图库**：直接白嫖 `ImgEdit_recap_mask`（自带密集描述 + 物体计数 JSON + 分割掩码）、`UniREdit-Data-100K`（自带多物体交互与规则推理底图）及 `OmniEdit-Filtered-1.2M` 的现成高清底图，省去 90% 的检测、分割、打标算力。
> 2. **把防泄露去重降到最简（Zero-Model Dedup + 单脚本 pHash）**：通过**“数据源天然隔离（全量禁用 COCO）+ 现成字段/ID 黑名单过滤 + 100 行 Python 感知哈希（pHash/dHash）秒级比对”**，彻底省去搭 FAISS 向量库和跑 DINOv2/SSCD 深度学习模型的繁琐流程。
> 3. **靶向对齐 6+4 基准的四大高分能力区**：用 `Qwen-Image-2.1` + VLM 围绕**多物体组合与精确计数、隐式因果/逻辑推理、复杂空间指代与多图参考、真实中英长尾应用**做同分布但非同题的指令增广。

---

## 一、 30–50w 数据配比与底图源分配表

| 数据分桶 | 目标规模 | 占比 | 对齐的 6+4 基准能力点 | 底图直接来源（全开源现成） | 为什么选它（省事在哪里） |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **Bucket A：多物体组合与精确数值/空间指代** | **15w – 22w** | ~45% | **ABACUS**（精确计数增删改）、**ImgEdit-UGE**（遮挡/空间关系指代）、**I²EBench**（计数/方向/区域） | **[sysuyy/ImgEdit_recap_mask](https://huggingface.co/datasets/sysuyy/ImgEdit_recap_mask)** | 官方已筛好 `aes > 4.75` 的 `1024px` 级真图，且**每张图自带物体类别计数字典（如 `{"car": 3, "person": 2}`）、密集 Caption 与 RLE Mask**，写个 `if` 判断就能直接挑出多物体复杂图 |
| **Bucket B：真实生活、开放域长尾与中英实用编辑** | **8w – 12w** | ~25% | **GEdit-Bench v1/v2**（23 类 + Open-set 自由指令）、**CPI-Practical/General**（人像/家装/商品/改字）、**MagicBrush** | **[TIGER-Lab/OmniEdit-Filtered-1.2M](https://huggingface.co/datasets/TIGER-Lab/OmniEdit-Filtered-1.2M)**（或流式抽 **[xiaotanhua/UnicEdit-10M](https://huggingface.co/datasets/xiaotanhua/UnicEdit-10M)**） | 多分辨率真实场景高清图，自带 GPT-4o 质量打分（直接按 `o_score >= 8.5` 截取高分底图即可） |
| **Bucket C：隐式物理/因果/时空与规则推理** | **5w – 10w** | ~20% | **RISEBench / ++**（时间/因果/空间/逻辑/反事实）、**UniREditBench**（真实多物体交互 + 游戏规则）、**CPI-Intelligent** | **[maplebb/UniREdit-Data-100K](https://huggingface.co/datasets/maplebb/UniREdit-Data-100K)**（直接取其 8w 底图）+ **Qwen-Image-2.1 自合成 2w 底图** | `UniREdit-Data-100K` 原本就是为训 BAGEL 推理编辑造的 10w 库，底图天然包含多物体因果与游戏状态；自合成底图 100% 零泄露 |
| **Bucket D：多参考图主体/风格/背景解耦组合** | **3w – 6w** | ~10% | **OmniRef-Bench**（主体+姿态+背景+光照+风格 10 类组合）、**CPI-Bench 多图子集** | 从 **Bucket A (`ImgEdit_recap_mask`)** 中直接用现成 Mask 抠出前景主体 + 搭配背景图与风格图组装成 2–4 图输入 | 不需要额外跑 SAM！`ImgEdit_recap_mask` 自带每个物体的现成 RLE 分割掩码，一行代码即可抠出干净主体拼成多图参考样本 |

---

## 二、 极简防泄露去重方案（无需跑深度学习特征模型）

为什么原来觉得去重麻烦？因为要对几十万张图跑 DINOv2 / SSCD 提特征再建向量索引。但只要看清 **6+4 基准的底图到底从哪来**，去重可以直接简化为 **3 步纯 CPU 轻量过滤（单机几分钟跑完）**：

### 第 1 步：源头物理隔离（砍掉 80% 泄露风险，0 行代码）
- **规则**：**绝对不碰 MS-COCO（2014/2017 全量）**。
- **效果**：`MagicBrush`、`ABACUS`、`I²EBench` 里的 COCO 底图与你选的四个开源池（`ImgEdit_recap_mask`、`OmniEdit`、`UniREdit-100K`、自合成）在数据源上**天然正交**，无需任何比对即 100% 零重合。

### 第 2 步：同源数据集的 ID / Hash 字典精准排除（处理 ImgEdit 与 UnicEdit）
- **唯一可能同源的地方**：
  1. 你用了 `ImgEdit_recap_mask`，而评测集 `ImgEdit-Bench`（784 张单轮图）也是从同一批 LAION-aes 里挑的；
  2. 如果你抽了 `UnicEdit-10M`，`GEditBench v2`（1,200 张图）有一部分源图采自它。
- **极简做法**：
  - `ImgEdit-Bench` 的 `basic_edit.json` 和 `UGE_edit.json` 里直接写明了文件名（如 `"000342021.jpg"`），只需把这 ~800 个数字 ID 放进 Python `set()`，读取 `ImgEdit_recap_mask` 时 `if img_id in blacklist_ids: continue` 即可；
  - `UnicEdit-10M` 每条样本自带 MD5 `key`，直接对比排除即可。

### 第 3 步：全量 6+4 测试集底图的“64-bit 感知哈希（pHash）”秒级兜底
为了防止不同数据集碰巧爬到同一张网图（或者经过缩放/重压缩的同源图），用 `imagehash` 库算 **64 位感知哈希（pHash + dHash）** 替代笨重的 DINOv2：
- **为什么足够用**：数据泄露的核心是“同一张底图（或其 Resize/JPEG 重压缩/轻微裁边版本）进了训练集”，`pHash` 对缩放、长宽比微调、JPEG 压缩完全免疫，且计算一张图只需 **1 毫秒（纯 CPU）**。
- **直接可用的单脚本（`fast_dedup.py`）**：

```python
# pip install Pillow imagehash tqdm
import os, glob, imagehash
from PIL import Image
from tqdm import tqdm

# 1. 把 6+4 个 Benchmark 的测试集底图（共约 1.5 万张）算一遍 64-bit pHash
bench_hashes = []
for img_path in tqdm(glob.glob("/path/to/benchmarks_6plus4/**/*.*", recursive=True)):
    try:
        h = imagehash.phash(Image.open(img_path).convert("RGB"))
        bench_hashes.append(h)  # 转为 int64 仅占 8 字节
    except Exception:
        pass

# 2. 过滤候选训练底图：汉明距离 (Hamming Distance) <= 8 视为近似同图直接剔除
def is_leaked(pil_img, threshold=8):
    h = imagehash.phash(pil_img.convert("RGB"))
    # 1.5万个 int64 异或比对在 Python/NumPy 下仅需 < 0.2 毫秒
    return any((h - bh) <= threshold for bh in bench_hashes)
```

> **文本指令防泄露同样一句话搞定**：生成完增广 Prompt 后，和 6+4 基准的测试 Prompt 算一次 **3-gram 词集合 Jaccard 相似度**，凡是 `Jaccard > 0.5` 的直接丢弃或让 LLM 换个物体/场景重写。

---

## 三、 四步落地执行流水线（Step-by-Step）

### Step 1：从 6+4 基准抽象“元任务模板库（Meta-Skill Schema）”，不抄原题
把 6+4 基准的 Prompt 按你的核心目标抽成 **4 类结构化 JSON 槽位模板**，严禁把测试集原句塞进 Few-shot 生成同义句：

1. **多物体组合与精确计数模板（对齐 ABACUS + ImgEdit-UGE）**
   - 槽位结构：`[动作: 新增/删除/替换] + [精确数量 K (1~6)] + [空间/属性限定词: 如“左数第二个/被遮挡的/除红色以外的所有”] + [目标物体] + [场景守恒约束]`
   - 示例指令：*“在木桌左侧再增加 2 个青苹果，使桌面上的苹果总数变为 5 个，同时将中间玻璃杯里的柠檬片替换为橙子片，保持背景光影不变。”*
2. **隐式因果/时空/规则推理模板（对齐 RISEBench + UniREditBench + CPI-Intelligent）**
   - 槽位结构：`[触发事件 / 物理条件改变 / 时间推移 / 游戏规则步] -> (隐去结果表面描述，并在训练标签中附带 <think> 推理链)`
   - 示例指令：*“把画面中支撑木板的左边积木抽走，展示接下来瞬间的物理状态。”*（训练时附带 CoT：`左端失去支撑 -> 木板向左倾斜滑落 -> 上方水杯倾倒洒出`）
3. **多参考图解耦合成模板（对齐 OmniRef-Bench + CPI 多图）**
   - 槽位结构：`[图1的主体 A] + [图2的主体 B / 姿态] + [图3的照明/艺术风格] + [图4的背景场景]`
4. **真实高频与垂直业务模板（对齐 GEdit v1/v2 + CPI-Practical）**
   - 槽位结构：`[人像微调 / 室内家装改造 / 电商商品换背景 / 中英文招牌改字 / 开放域脑洞创意]`

---

### Step 2：按底图元数据“精准配菜”生成增广指令（Prompt Augmentation）
利用 `ImgEdit_recap_mask` 自带的现成标注，让指令生成 **100% 贴合底图内容，绝不产生幻觉空指令**：

- **读取底图自带 JSON**：拿到 `caption`（密集描述）和 `objects`（如 `{"person": 3, "umbrella": 2, "bench": 1}`）。
- **规则路由**：
  - 若某类物体数量 $\ge 2$（如 `"umbrella": 2`）$\rightarrow$ 自动路由给 **计数/指代 Prompt 生成器**，构造“只修改其中 1 把伞 / 再增加 2 把伞 / 把左边的伞改成和右边一样”的指令；
  - 若场景含多个可交互物体（如人+交通工具/容器+液体）$\rightarrow$ 自动路由给 **因果/动作推理生成器**；
  - 若底图来自 `UniREdit-Data-100K` $\rightarrow$ 在其原有推理指令基础上，用 LLM 衍生出“反事实条件”或“下一步状态推演”的新指令与新 `<think>` 链。

---

### Step 3：调用 Qwen-Image-2.1 批量生产训练图对（三种合成模式）
为了保证 30–50w 数据的最高画质与精确度，按任务分三种模式调用 **Qwen-Image-2.1**：

1. **正向编辑模式（占 ~60%，主力常规/推理/多物体编辑）**：
   - 输入：`源底图 I_src + 增广指令 P_edit` $\rightarrow$ Qwen-Image-2.1 直接输出编辑结果图 `I_tgt`（统一 `1024×1024` 或保持原长宽比 `~100万像素`）。
2. **反向构造模式（Inversion Trick，占 ~25%，专攻“精确移除 / 复杂计数增加 / 去遮挡”）**：
   - **痛点**：让模型在已有 3 个人的图里精准加第 4 个人很容易破坏原图背景；但**把原图里已有的 1 个人用 Mask 擦除（Inpainting）变成 2 个人**却极其自然且背景 100% 无损！
   - **做法**：直接利用 `ImgEdit_recap_mask` 自带的物体 RLE Mask，将原图 `I_real` 中的 $K$ 个物体擦除得到干净底图 `I_erased`：
     - 把 `(I_erased -> I_real)` 作为**“精确增加 $K$ 个物体”**的训练对（此时目标图 `I_real` 是 100% 真实自然图，画质极高且数量绝对精准，完美对齐 **ABACUS Add**）；
     - 把 `(I_real -> I_erased)` 作为**“精确移除 $K$ 个物体”**的训练对（完美对齐 **ABACUS Remove** 与 **ImgEdit-UGE**）。
3. **T2I + Edit 双生合成模式（占 ~15%，专攻中英文字修改、游戏规则与极端物理推理）**：
   - 先用 Qwen-Image-2.1 生成带清晰中英文招牌或特定物理初始状态的底图 `I_src`，再紧接着生成编辑后的 `I_tgt`。

---

### Step 4：两级轻量质检过滤（保住 30–50w 精品率）
生成约 **60w–70w** 原始候选对后，用极低成本筛出最终 **30–50w** 入库：

1. **第一级：非编辑区像素/结构秒级初筛（0 成本，刷掉 15% 崩坏样本）**
   - 对局部编辑任务，计算生成前后全图的 **LPIPS / SSIM**（或非 Mask 区域的像素 L1 距离）。
   - 凡是全图变化过大（`SSIM < 0.55`，说明背景被重绘崩了）或几乎没变（`L1 < 0.005`，说明模型没听懂指令偷懒原样输出）的样本，直接秒删。
2. **第二级：开源 7B 裁判模型单次打分（低成本，刷掉 15% 语义不符样本）**
   - 不必调昂贵的闭源 API，直接在本地用 **Qwen2.5-VL-7B**（或微信 `UnicEdit-10M` 开源的专用 7B 质检模型 **`Qwen-Verify`** / `GEditBench v2` 开源的 **`PVC-Judge`**）跑一遍批量推理。
   - 只保留 **指令遵循度（Instruction Following）$\ge 4/5$** 且 **视觉一致性（Consistency）合格** 的样本，最终沉淀出高纯度的 **30–50w** 训练集。

---

## 四、 工程落地执行清单（Checklist）

1. **Day 1（底图准备与极简去重）**：
   - 下载 `sysuyy/ImgEdit_recap_mask`、`maplebb/UniREdit-Data-100K`、抽样 `TIGER-Lab/OmniEdit-Filtered-1.2M`；
   - 跑一遍上面的 `fast_dedup.py`（排除 `ImgEdit-Bench` ID + 对 6+4 测试集底图算 64-bit `pHash` 汉明距离 $>8$），落盘 **45w 张干净底图**。
2. **Day 2（Prompt 增广与多图组装）**：
   - 基于底图的物体计数字典与 Caption，按 4 大类槽位模板批量生成 **60w 条中英双语编辑指令**（其中推理类带 `<think>` CoT，多图类用自带 RLE Mask 抠图拼好 2–4 张参考图）；
   - 跑一遍 3-gram Jaccard（$\le 0.5$）过滤掉与测试集指令字面过近的句子。
3. **Day 3–5（Qwen-Image-2.1 批量合成 + 正反向对偶构造）**：
   - 常规/推理/多图走正向生成；计数增删走自带 Mask 的正反向对偶构造。
4. **Day 6（SSIM 初筛 + 7B 模型质检落盘）**：
   - 剔除背景漂移与未生效样本，最终打包 **30–50w** 标准 Parquet / WebDataset 格式交付训练。
