# `reflective_t2i_v1` 64-NPU Closed-Loop Causal Benchmark Summary (`1080/1080`)

## geneval2 (`n=800`, triggered=`467` [58.4%], gated=`{'focused_add_recheck': 44, 'target_count_contradicts_prompt': 11, 'focused_remove_recheck': 1, 'invalid_add_endpoints': 13, 'invalid_remove_endpoints': 2, 'removes_required_subject_to_zero': 2, 'current_count_already_matches_prompt': 1}`)
| Arm | Exact Acc ↑ | Atomic QA ↑ | Fix / TrigFail ↑ | Damage / TrigPass ↓ | Net Fixed | Mean HF | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| `A_native` | **0.1812** | 0.7593 | 0/467 (0.0%) | 0/0 (0.0%) | +0 | 0.04937 | 9.08s |
| `B_rewrite_resample` | **0.2462** | 0.7877 | 52/467 (11.1%) | 0/0 (0.0%) | +52 | 0.04851 | 24.44s |
| `C_reflective` | **0.2200** | 0.7724 | 31/467 (6.6%) | 0/0 (0.0%) | +31 | 0.05164 | 21.93s |
| `E_shuffled_feedback` | **0.1925** | 0.7530 | 9/467 (1.9%) | 0/0 (0.0%) | +9 | 0.05247 | 21.92s |
- Paired Δ Exact Acc (95% CI): `C - A = +0.0387 [0.025, 0.0525]` | `C - B = -0.0262 [-0.0462, -0.0063]` | `C - E (causal) = +0.0275 [0.0138, 0.0413]`

| Action (triggered) | n | A_native Acc | B_rewrite Acc | C_reflective Acc | E_shuffled Acc | C Fix / TF | C Dmg / TP | C Net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `add` | 367 | 0.0000 | 0.1035 | **0.0599** | 0.0163 | 22/367 | 0/0 | +22 |
| `remove` | 43 | 0.0000 | 0.1395 | **0.0000** | 0.0233 | 0/43 | 0/0 | +0 |
| `attr` | 42 | 0.0000 | 0.1190 | **0.1905** | 0.0476 | 8/42 | 0/0 | +8 |
| `move` | 15 | 0.0000 | 0.2000 | **0.0667** | 0.0000 | 1/15 | 0/0 | +1 |

## geneval_pp (`n=280`, triggered=`77` [27.5%], gated=`{'invalid_add_endpoints': 3, 'focused_remove_recheck': 3, 'focused_add_recheck': 2}`)
| Arm | Exact Acc ↑ | Atomic QA ↑ | Fix / TrigFail ↑ | Damage / TrigPass ↓ | Net Fixed | Mean HF | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| `A_native` | **0.6929** | — | 0/62 (0.0%) | 0/15 (0.0%) | +0 | 0.03946 | 9.06s |
| `B_rewrite_resample` | **0.7464** | — | 19/62 (30.6%) | 4/15 (26.7%) | +15 | 0.03883 | 20.05s |
| `C_reflective` | **0.7321** | — | 15/62 (24.2%) | 4/15 (26.7%) | +11 | 0.04030 | 18.87s |
| `E_shuffled_feedback` | **0.7000** | — | 10/62 (16.1%) | 8/15 (53.3%) | +2 | 0.04099 | 18.87s |
- Paired Δ Exact Acc (95% CI): `C - A = +0.0393 [0.0107, 0.0714]` | `C - B = -0.0143 [-0.05, 0.0214]` | `C - E (causal) = +0.0321 [0.0107, 0.0571]`

| Action (triggered) | n | A_native Acc | B_rewrite Acc | C_reflective Acc | E_shuffled Acc | C Fix / TF | C Dmg / TP | C Net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `add` | 55 | 0.1455 | 0.3818 | **0.3455** | 0.2000 | 13/47 | 2/8 | +11 |
| `remove` | 10 | 0.4000 | 0.4000 | **0.4000** | 0.3000 | 1/6 | 1/4 | +0 |
| `attr` | 11 | 0.2727 | 0.4545 | **0.2727** | 0.2727 | 1/8 | 1/3 | +0 |
| `move` | 1 | 0.0000 | 0.0000 | **0.0000** | 0.0000 | 0/1 | 0/0 | +0 |
