# Real T2I Self-Feedback Closed-Loop 64-NPU Benchmark Summary

- Integrity: `gen=833/833`, `score=833/833` (`complete=True`)
- Independent Judge: `Qwen3-VL-8B-Instruct` (per-clause GenEval QA + GenEval++ checklist rubric)

## GenEval Full (553 Prompts, 6 Categories)
- Trigger rate: `0.141` | Parse fail: `0.0054` | Error-catch recall: `0.8` | False-trigger rate: `0.1163`

| Arm | Macro Acc ↑ | Micro Acc ↑ | Fix / TrigFail ↑ | Damage / TrigPass ↓ | Net Fixed | Triggered HF | Mean NFE | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | **0.9637** | 0.9638 | 0/16 (0.0) | 0/62 (0.0) | +0 | 0.05658 | 99.0 | 10.841 |
| `semantic_full` | **0.9652** | 0.9656 | 5/16 (0.3125) | 4/62 (0.0645) | +1 | 0.0421 | 102.74 | 34.87 |
| `reproj_mid060_g1` | **0.9621** | 0.962 | 2/16 (0.125) | 3/62 (0.0484) | -1 | 0.05689 | 104.15 | 34.992 |
| `reproj_iso036_g1` | **0.9688** | 0.9693 | 4/16 (0.25) | 1/62 (0.0161) | +3 | 0.0645 | 106.69 | 35.269 |

**Category Breakdown:**
| Arm | `single_object` | `two_object` | `counting` | `colors` | `position` | `color_attr` |
|---|---:|---:|---:|---:|---:|---:|
| `baseline` | 1.0 | 0.9798 | 0.925 | 0.9574 | 0.97 | 0.95 |
| `semantic_full` | 1.0 | 0.9798 | 0.9125 | 0.9787 | 0.97 | 0.95 |
| `reproj_mid060_g1` | 1.0 | 0.9798 | 0.925 | 0.9681 | 0.95 | 0.95 |
| `reproj_iso036_g1` | 1.0 | 0.9899 | 0.925 | 0.9681 | 0.98 | 0.95 |

## GenEval++ Full (280 Prompts)
- Trigger rate: `0.3714` | Parse fail: `0.0107` | Error-catch recall: `0.7636` | False-trigger rate: `0.2756`

| Arm | Macro Acc ↑ | Micro Acc ↑ | Fix / TrigFail ↑ | Damage / TrigPass ↓ | Net Fixed | Triggered HF | Mean NFE | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | **0.8036** | 0.8036 | 0/42 (0.0) | 0/62 (0.0) | +0 | 0.06236 | 99.0 | 11.336 |
| `semantic_full` | **0.8036** | 0.8036 | 11/42 (0.2619) | 11/62 (0.1774) | +0 | 0.0456 | 99.06 | 36.249 |
| `reproj_mid060_g1` | **0.825** | 0.825 | 13/42 (0.3095) | 7/62 (0.1129) | +6 | 0.06283 | 102.77 | 36.546 |
| `reproj_iso036_g1` | **0.8321** | 0.8321 | 14/42 (0.3333) | 6/62 (0.0968) | +8 | 0.07084 | 109.46 | 37.311 |

**Category Breakdown:**
| Arm | `color_attr` | `color_count_attr` | `color_spatial_attr` | `counting` | `multi_object_count_attr` | `size_spatial_attr` | `spatial_count_attr` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | 0.95 | 0.7 | 0.875 | 0.7 | 0.8 | 0.825 | 0.775 |
| `semantic_full` | 0.925 | 0.725 | 0.875 | 0.75 | 0.8 | 0.825 | 0.725 |
| `reproj_mid060_g1` | 0.925 | 0.725 | 0.875 | 0.8 | 0.825 | 0.825 | 0.8 |
| `reproj_iso036_g1` | 0.925 | 0.8 | 0.875 | 0.775 | 0.825 | 0.825 | 0.8 |

