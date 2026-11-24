# `reflective_t2i_v1` 64-NPU Closed-Loop Causal Benchmark Summary (`280/280`)

## geneval_pp (`n=280`, triggered=`120` [42.9%], gated=`{'focused_add_recheck': 10, 'focused_remove_recheck': 1, 'removes_required_subject_to_zero': 2, 'current_count_already_matches_prompt': 2, 'invalid_remove_endpoints': 2, 'failed_json_parse': 1}`)
| Arm | Exact Acc ↑ | Atomic QA ↑ | Fix / TrigFail ↑ | Damage / TrigPass ↓ | Net Fixed | Mean HF | Mean Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| `A_native` | **0.3964** | — | 0/108 (0.0%) | 0/12 (0.0%) | +0 | 0.04613 | 8.51s |
| `B_rewrite_resample` | **0.4643** | — | 23/108 (21.3%) | 4/12 (33.3%) | +19 | 0.04619 | 20.70s |
| `C_reflective` | **0.4321** | — | 10/108 (9.3%) | 0/12 (0.0%) | +10 | 0.04664 | 19.56s |
| `E_shuffled_feedback` | **0.4071** | — | 5/108 (4.6%) | 2/12 (16.7%) | +3 | 0.04782 | 19.56s |
- Paired Δ Exact Acc (95% CI): `C - A = +0.0357 [0.0179, 0.0607]` | `C - B = -0.0321 [-0.0714, 0.0071]` | `C - E (causal) = +0.0250 [0.0, 0.0536]`

| Action (triggered) | n | A_native Acc | B_rewrite Acc | C_reflective Acc | E_shuffled Acc | C Fix / TF | C Dmg / TP | C Net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `add` | 68 | 0.0441 | 0.2059 | **0.1176** | 0.0588 | 5/65 | 0/3 | +5 |
| `remove` | 29 | 0.1724 | 0.4138 | **0.2414** | 0.2414 | 2/24 | 0/5 | +2 |
| `attr` | 22 | 0.1818 | 0.2273 | **0.3182** | 0.1818 | 3/18 | 0/4 | +3 |
| `move` | 1 | 0.0000 | 0.0000 | **0.0000** | 0.0000 | 0/1 | 0/0 | +0 |
