# 训练数据与评测数据（`isochronic_manifold_v1`）

训练仅消费成对的源图 `source_image`、目标图 `target_image` 与简短编辑指令 `edit_instruction`（无需 `tail_condition`、无需区域框、无需教师位移缓存）。读取、跨 split 家族隔离与无放回遍历由 `qwen_latent_cot/u15_target_correction/data.py` 实现。

## 现有训练数据资产总览

| 清单 | 路径 | 总记录数 | Train / Val / Test | 用途与特点 |
|---|---|---:|---|---|
| **400k 语义均衡主力集（6 类任务 + 新图合成）** | `isochronic_data_20261107/prod_1m_glm53_20261118/final_manifests/manifest_*.jsonl` | `400,098` | `384,004 / 8,022 / 8,072` | 流 A 64 卡 `Qwen-Image-2.1` 计数/空间合成 (`7,721`) + 流 B `GLM-5.3-flash` 6 类任务覆盖清洗 (`392,377`) |
| **全量底层编辑池（UniMoE 9 库过滤版）** | `isochronic_data_20261107/unimoe_edit/L_edit_final.jsonl` | `14,942,500` | `14,882,904 / 29,788 / 29,808` | 9 个公开编辑集经字段校验与宽高比几何校验后全量池，`shared_field_of_view_v2` |
| **四类均衡精选集（129k）** | `isochronic_data_20261107/semantic_denoising_20261117/pairs_129k_qwen.jsonl` | `129,000` | `128,000 / 1,000 / 0` | `add / remove / move / attr` 各 `32,000` train + `250` val，严格 `0` 跨 split 源图家族重叠 |
| **反向核验计数/空间试产集** | `isochronic_data_20261107/count_spatial_20261115/` | 单卡试产批次 | `96 / 2 / 2`（按家族） | 由 `qwen_latent_cot/count_spatial` 从正确 T2I 图反向破损并经 `GLM-5.3-flash` 双图核验 |

## 家族隔离与画布几何协议（`shared_field_of_view_v2`）

1. **按源图家族划分，杜绝跨 split 泄漏**：`LinearManifest` 强制校验 `source_family` 与 `prompt_family` 不得跨越 `train / val / test`。
2. **共享视野矩形与有效区域掩码**：源图和目标图等比缩放到同一 `512 × 512` 画布矩形（宽图 `height = width / ratio`，窄图 `width = height * ratio`），非有效区域填黑且 `valid = 0`，`clean_target_mse` 严格只在 `valid == 1` 像素上求均值。
3. **无放回打乱遍历与预分片**：大规模清单通过 `scripts/shard_manifest.py` 切分为 `shards64/shard_0000x.jsonl`，每个 epoch 打乱遍历全部训练记录一次。

## 数据准备与清洗工具链

- `scripts/build_unimoe_edit_manifest.py`：原始编辑池字段规范化与家族划分
- `scripts/verify_manifest_geometry.py` + `scripts/filter_manifest_by_rejections.py`：并行校验并剔除源/目标宽高比不一致样本
- `scripts/shard_manifest.py`：按 rank 轮转预切分片并生成 `shards.json`
- `scripts/build_1m_dataset_glm.py` + `scripts/launch_editor_fleet64.py`：4 节点 64 卡 `Qwen-Image-2.1` 并发合成 + 多进程 `GLM-5.3-flash` 语义任务覆盖清洗联合构建器
- `scripts/curate_edit_pool_glm.py`：存量编辑池 6 类任务均衡抽样与 `GLM-5.3-flash` 核验工具
- `scripts/count_spatial_edits.py`（`qwen_latent_cot/count_spatial/`）：从已验证 T2I 图反向合成并核验计数与空间编辑对
