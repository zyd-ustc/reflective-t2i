# Z-Image Base 6B Matched-Budget 4-NPU Pilot Comparison (Pre-64-NPU Gate)

## Overall (n=100 unseen val pairs: 40 add, 20 remove, 20 attr, 20 move)
| Arm | Edit Followed ↑ | BG Preserved ↑ | Quality OK ↑ | Strict Pass ↑ | Edit L1 ↓ | BG L1 ↓ | HF Ratio (~1.0) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `native_frozen` | 32.0% | 73.0% | 93.0% | **29.0%** | 0.4489 | 0.1361 | 1.049 |
| `legacy_500` | 28.0% | 68.0% | 93.0% | **25.0%** | 0.4268 | 0.1310 | 1.010 |
| `v1_uncal_500 (33x/soar0.4)` | 27.0% | 49.0% | 71.0% | **21.0%** | 0.3943 | 0.1359 | 0.917 |
| `v1_cal_500 (4x/soar0.15+warmup)` | 31.0% | 64.0% | 87.0% | **27.0%** | 0.4115 | 0.1307 | 0.965 |

## Action: `add`
| Arm | n | Edit Followed ↑ | BG Preserved ↑ | Quality OK ↑ | Strict Pass ↑ | Edit L1 ↓ | HF Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| `native_frozen` | 40 | 30.0% | 77.5% | 92.5% | **25.0%** | 0.4212 | 1.063 |
| `legacy_500` | 40 | 22.5% | 65.0% | 92.5% | **17.5%** | 0.4020 | 1.024 |
| `v1_uncal_500 (33x/soar0.4)` | 40 | 17.5% | 52.5% | 77.5% | **17.5%** | 0.3754 | 0.930 |
| `v1_cal_500 (4x/soar0.15+warmup)` | 40 | 27.5% | 62.5% | 92.5% | **22.5%** | 0.3893 | 0.981 |

## Action: `remove`
| Arm | n | Edit Followed ↑ | BG Preserved ↑ | Quality OK ↑ | Strict Pass ↑ | Edit L1 ↓ | HF Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| `native_frozen` | 20 | 60.0% | 70.0% | 85.0% | **55.0%** | 0.5104 | 1.042 |
| `legacy_500` | 20 | 55.0% | 65.0% | 80.0% | **50.0%** | 0.4801 | 0.993 |
| `v1_uncal_500 (33x/soar0.4)` | 20 | 70.0% | 45.0% | 55.0% | **40.0%** | 0.4309 | 0.890 |
| `v1_cal_500 (4x/soar0.15+warmup)` | 20 | 55.0% | 60.0% | 65.0% | **45.0%** | 0.4585 | 0.937 |

## Action: `attr`
| Arm | n | Edit Followed ↑ | BG Preserved ↑ | Quality OK ↑ | Strict Pass ↑ | Edit L1 ↓ | HF Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| `native_frozen` | 20 | 15.0% | 65.0% | 100.0% | **15.0%** | 0.4098 | 1.042 |
| `legacy_500` | 20 | 15.0% | 70.0% | 100.0% | **15.0%** | 0.3950 | 1.006 |
| `v1_uncal_500 (33x/soar0.4)` | 20 | 15.0% | 50.0% | 70.0% | **15.0%** | 0.3759 | 0.923 |
| `v1_cal_500 (4x/soar0.15+warmup)` | 20 | 20.0% | 65.0% | 90.0% | **20.0%** | 0.3847 | 0.964 |

## Action: `move`
| Arm | n | Edit Followed ↑ | BG Preserved ↑ | Quality OK ↑ | Strict Pass ↑ | Edit L1 ↓ | HF Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| `native_frozen` | 20 | 25.0% | 75.0% | 95.0% | **25.0%** | 0.4816 | 1.034 |
| `legacy_500` | 20 | 25.0% | 75.0% | 100.0% | **25.0%** | 0.4547 | 1.006 |
| `v1_uncal_500 (33x/soar0.4)` | 20 | 15.0% | 45.0% | 75.0% | **15.0%** | 0.4141 | 0.912 |
| `v1_cal_500 (4x/soar0.15+warmup)` | 20 | 25.0% | 70.0% | 95.0% | **25.0%** | 0.4355 | 0.964 |
