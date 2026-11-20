# BAGEL MoT：RecA 稠密条件 + 纠错 Δh 方案

日期：2026-10-23（Asia/Shanghai）。状态：实验草案（待实现）。路径：`/workspace/bagel-LatentCoT-RecA-deltaH-correction-plan.md`（可直接落入仓库 `docs/`）。

**一句话**：在 BAGEL MoT 上，用 UND hidden 作 RecA 式稠密条件对齐理解–生成；再在纠错回合对 GEN residual 加减对比激活差 \(\Delta h\)（CAA/steering 风格），把「看图 → 反馈 → 再生成」的显式多轮压进**单次采样**。数据**仅**高质量 GT 图文对 \((P,x_0)\)，失败/负例一律在线构造，不依赖外部 bad 标注集。

与姊妹草案 `bagel-LatentCoT-SOAR-trajectory-correction-plan.md` 的分工：SOAR 草案主打 **off-traj 速度回归** \(v_{\mathrm{corr}}\)；本草案主打 **RecA 稠密条件 + 表征层 \(\Delta h\) 纠错**。二者可串接（先 RecA 再 \(\Delta h\)，或 \(\Delta h\) 与 SOAR 损失并行），但叙事与验收以本文件为准。

---

## 1. 叙事与问题

### 1.1 显式上界（对照，不进部署）

```text
prompt P + seed
  → GEN 去噪出中间/最终图
  → 解码预览 + ViT/UND 文字反思（检出失败）
  → 拼约束句回退 t_early 再生成
```

这是**文字反思闭环**的天花板；贡献不是再做一个更好的外挂编辑器。

### 1.2 本方案目标

同一套 BAGEL MoT 权重内完成：

1. **观察（RecA）**：用 UND 对好图 \(x_0\)（或当前 clean proposal）抽出的 hidden / 语义特征，当作相对稀疏 caption \(P\) 的**稠密条件**，训练 GEN 重建 \(x_0\)，缩小理解–生成间隙。
2. **纠错（\(\Delta h\)）**：在采样中途选定 `t_read`，用对比激活差 \(\Delta h\) 对 GEN residual（或条件通道）做加减，把「偏了 → 往 good 方向推」压进同一次 ODE，无需外挂 VLM 写字。
3. **产品形态**：推理输入仍是原始 \(P\) + seed；**无外部图像、无 ViT、无失败标注库**。

记号与仓库一致：`UND`、`GEN`、`mot_edit`、`plan`、`t_read`、`t_early`。BAGEL 约定 \(t=1\) 噪声、\(t=0\) 干净；速度目标 \(v^*=\epsilon-x_0\)。

---

## 2. 为何 MoT / UMM 不可替代

若「更强 T2I + 外挂 VLM 多轮」能同等叙事，工作退化为工程集成。必须守住：

| 条件 | 说明 |
|---|---|
| 同模型自条件 | RecA 的稠密 prompt 来自**本模型 UND**，不是外挂 CLIP 当永久依赖；\(\Delta h\) 在**同一** residual 空间加减 |
| 中间态可读可写 | UND 读 GEN-hidden / clean proposal；\(\Delta h\) 写回 GEN 层激活——压缩的是计算图，不是磁盘 PNG |
| 无外部图像约束 | 纯 T2I 产品形态；显式文字反思只作上界对照 |
| 稠密条件 ≠ 稀疏 caption | RecA 证明理解编码器 embedding 可作 dense prompt（[2509.07295](https://arxiv.org/abs/2509.07295)）；MoT 才有原生 UND→GEN 跨专家通道 |
| \(\Delta h\) ≠ 外挂 DiT corrector | 对比激活差走 MoT 内部 residual / 条件 KV，不是另训速度头 |

写作禁止把方法概括成「给 BAGEL 加 RecA」或「加 CAA」；必须同时出现 **UND 稠密条件**、**在线构造正负**、**GEN 上加减 \(\Delta h\)**、**单次采样内闭环**。

---

## 3. 数据约束：只有 GT 高质量图文对

**唯一监督锚**：高质量 \((P, x_0)\)。  
**禁止假设**：外部失败图库、人工 bad caption、偏好对、奖励模型标签。

因此「正 / 负 / 失败」必须从 \((P,x_0)\) **在线**构造（§5）。锚点质量直接决定 RecA 重建目标与 \(\Delta h\) 方向是否语义合法——脏 caption 会把错误方向写进 steering。

建议沿用现网高质量 pairs（如 `pairs-20261022.jsonl`）；训练前用 CLIP/美学或规则过滤明显错配；原子可验证子集只作评测，不作唯一训练源。

---

## 4. 阶段 A：RecA —— 好图重建 / UND 特征作稠密条件

### 4.1 形式化（对齐 RecA，适配 BAGEL MoT）

对 GT 图 \(x_0\)：

1. **冻结或 stop-grad** 的 UND 路径（及/或视觉理解编码器）提取稠密特征 \(h_v=\mathrm{UND}(x_0)\)（或 GEN-hidden 观察前向在 \(t=0\) 上的图像 KV，与仓库 `mot_edit` 观察入口一致）。
2. 与模板文本 \(t_{\mathrm{tpl}}\)（如「Describe the image in detail.」及多样化模板）拼接，作为 GEN 的**稠密条件**（替代或增强稀疏 \(P\)）。
3. 在噪声态 \(x_t=(1-t)x_0+t\epsilon\) 上做原生速度 / flow 损失，重建 \(x_0\)：

\[
\mathcal L_{\mathrm{RecA}}=\mathbb E\bigl[w(t)\,\|\,v_\theta(x_t,t,\mathrm{concat}(t_{\mathrm{tpl}},h_v))-v^*\|\bigr]^2.
\]

可选保留弱 T2I：\(\mathcal L_{\mathrm{t2i}}\) 用原 prompt \(P\)（\(\lambda_{\mathrm{t2i}}\) 较小或阶段性为 0），以及共享参数时的 \(\mathcal L_{\mathrm{i2t}}\) 保理解（RecA 论文对共享 UMM 的设定）。

**信息瓶颈**：输入图缩放到理解编码器最小分辨率 / 对 \(h_v\) 做适度压缩，避免退化为像素 autoencoder（见 RecA reproduce guide）。

### 4.2 与 MoT 观察接口的关系

- 训练 RecA 时 \(h_v\) 来自**真实 \(x_0\)**（好图）。
- 推理时 T2I **不**再喂外部图；RecA 后训练的权重把「稠密语义 → 像素」能力迁移到纯文本条件（RecA 原论文结论）。
- 纠错回合（§5–6）中，对**当前 proposal** 再抽 \(h_{\mathrm{obs}}\)，与从 \(x_0\) 构造的正负激活对照，得到 \(\Delta h\)——观察入口与 RecA / `mot_edit` 一致：**读 GEN-hidden，推理无 ViT**。

### 4.3 RecA 伪代码

```text
# Phase A: RecA on GT pairs only
for each batch (P, x0):
  hv = stopgrad(UND_encode(x0))          # dense visual prompt
  tpl = sample_template()                 # 多样化重建模板
  ε ~ N(0,I); t ~ schedule
  xt = (1-t)*x0 + t*ε
  v_star = ε - x0
  v_hat = GEN_forward(xt, t, cond=concat(tpl, hv))
  L_reca = w(t) * MSE(v_hat, v_star)
  # optional: L_t2i on P; L_i2t if shared UND/GEN
  L = λ_reca*L_reca + λ_t2i*L_t2i + λ_i2t*L_i2t
  backward; step
```

---

## 5. 阶段 B：\(\Delta h\) —— 仅从 GT 对在线构造正负 / 失败

核心：\(\Delta h\) 是 **good 激活 − bad 激活**（或均值差）在选定层 / 条件位上的方向，推理时乘系数 \(\alpha\) 加到 GEN residual（CAA / ActAdd 风格：[2312.06681](https://arxiv.org/abs/2312.06681)、[2308.10248](https://arxiv.org/abs/2308.10248)）。

下列四种构造**全部只依赖 \((P,x_0)\) 与当前模型**，可组合使用。

### 5.1 自 rollout 坏态（主推荐，对齐 SOAR 几何）

```text
# 从好锚 x0 造离轨坏态，再取激活差
ε ~ N(0,I); sample t0
x_on = (1-t0)*x0 + t0*ε
v_cfg = stopgrad(CFG(GEN(x_on,t0,P)))
t1 = max(t0 - Δt, 0)
x_hat = x_on + (t1 - t0)*v_cfg          # 模型自身一步误差
# 同 ε 再噪到 t'
t' ~ Unif[t1, 1]; α = (t'-t1)/(1-t1)
x_bad = (1-α)*x_hat + α*ε

h_good = stopgrad(activations(GEN or observe, state=clean_proposal(x0), P))
h_bad  = stopgrad(activations(GEN or observe, state=clean_proposal(x_bad), P))
Δh = mean(h_good - h_bad)               # 层集合 L 上平均；可 EMA 累积
```

语义：坏态是**当前策略**推出来的，on-policy；目标仍锚定 GT \(x_0\)，无需外部失败图。

### 5.2 同 prompt 多样本筛（无标注的相对好坏）

对同一 \(P\)，用当前模型采样 \(K\) 张 \(\{\hat x^{(k)}\}\)，**不引入人工分数**，用相对 GT 的代理筛：

- 与 \(x_0\) 的 latent / 特征距离（VAE 空间 L2、UND 特征余弦）；
- 或简单启发式：CLIP(\(P,\hat x\)) 相对 CLIP(\(P,x_0\)) 的落差。

取距离最近的 1–2 个为 **pseudo-good**，最远的为 **pseudo-bad**（或阈值最低的一半）：

\[
\Delta h=\mathbb E[h(\hat x^{+})]-\mathbb E[h(\hat x^{-})].
\]

注意：这是**相对排序**，不是绝对质量标签；\(x_0\) 始终可作为最强 good 锚，避免全坏批次塌缩。

### 5.3 对 good 可控破坏（合成负例）

在 \(x_0\) 或其特征上施加**可逆、语义相关**的破坏，构造 \(x^{-}\) / \(h^{-}\)：

| 破坏 | 示例 | 用途 |
|---|---|---|
| 像素/latent | 强噪、局部遮挡、颜色抖动、空间错位 crop | 逼出「属性/布局」方向 |
| 条件 | 打乱 \(P\) 中属性词、删实体、换错误颜色词 | 逼出「遵从 prompt」方向 |
| 时间 | 高噪 proposal 当 bad，低噪 / \(x_0\) 当 good | 逼出「清晰语义」方向 |

\[
\Delta h = \mathrm{sg}\bigl[h(x_0,P)-h(x^{-},P^{-})\bigr].
\]

破坏强度需调度：过强 → 方向噪声大；过弱 → \(\Delta h\approx 0\)。

### 5.4 stop-grad 激活差（训练时即时 CAA）

训练步内不落盘：

```text
with no_grad:
  h+ = forward_acts(good_state)   # x0 或 on-traj 正确条件
  h- = forward_acts(bad_state)    # §5.1–5.3 之一
Δh = EMA(Δh, h+ - h-)             # 或当步直接用
# 可训前向：在 GEN 选定层注入 +α Δh，再算重建/速度损失
v = GEN_forward_with_steer(xt, t, P, Δh, α)
L_steer = MSE(v, v*) 或 MSE(v, v_corr)   # 与 SOAR 目标可选叠加
```

**梯度规则**：构造 \(\Delta h\) 的两条前向 **stop-grad**；只让「注入 \(\Delta h\) 之后」的 GEN/适配器吃梯度，避免恒等捷径把 \(\Delta h\) 学成 0。

### 5.5 正负定义小结（仅 GT）

| 角色 | 来源 | 禁止 |
|---|---|---|
| **正 / good** | GT \(x_0\)；或同 \(P\) 下最接近 \(x_0\) 的样本；或低破坏态 | 外部「好图集」以外的偏好库 |
| **负 / bad / 失败** | 自 rollout 离轨、多样本最差、可控破坏、错误条件 | 预标注失败集、人工 bad caption 库 |
| **锚** | 始终可用 \(x_0\) 定义 \(v^*\) / 重建目标 | 把模型失败图 + 原 caption 当 CE 正例 |

---

## 6. 训练损失

总损失建议两阶段或联合：

\[
\mathcal L=
\lambda_{\mathrm{RecA}}\mathcal L_{\mathrm{RecA}}
+\lambda_{\mathrm{t2i}}\mathcal L_{\mathrm{t2i}}
+\lambda_{\Delta}\mathcal L_{\Delta}
+\lambda_{\mathrm{corr}}\mathcal L_{\mathrm{corr}}^{\mathrm{(opt)}}.
\]

| 项 | 含义 | 备注 |
|---|---|---|
| \(\mathcal L_{\mathrm{RecA}}\) | §4，稠密 \(h_v\) 条件下重建 \(x_0\) | 阶段 A 主损失；\(\lambda_{\mathrm{t2i}}\) 可先 0 |
| \(\mathcal L_{\mathrm{t2i}}\) | 稀疏 \(P\) 上标准速度 MSE | 保纯文本推理接口 |
| \(\mathcal L_{\Delta}\) | 注入 \(\pm\alpha\Delta h\) 后，对 \(v^*\)（或 guidance-safe 参考速度）的 MSE；可选对称：\(+\alpha\) 拉向 good，\(-\alpha\) 应远离 / 加大误差作对比 | 阶段 B 主损失 |
| \(\mathcal L_{\mathrm{corr}}^{\mathrm{(opt)}}\) | SOAR 式 \(v_{\mathrm{corr}}=(x'-x_0)/t'\) 在 off-traj 上回归 | 可选；与 \(\Delta h\) 正交——一个改速度目标，一个改激活方向 |

**Guidance-safe**（继承现网）：修正只许与 \(g=v_g-v_c\) 同向/正交，禁止反向抵消：

\[
\mathcal H_g(d)=d-\min\!\Bigl(\frac{\langle d,g\rangle}{\max(\langle g,g\rangle,10^{-12})},0\Bigr)g.
\]

可训范围建议：UND 观察/条件 LoRA + GEN 注入点附近 LoRA；冻结 VAE、`llm2vae`、基础骨干。\(\Delta h\) 本身可 EMA 缓冲（无参）或由小网络预测尺度 \(\alpha\)（慎用，首版固定 \(\alpha\) 网格）。

### 6.1 联合训练步伪代码（可执行骨架）

```text
for each batch (P, x0):  # 仅 GT 高质量对
  # ----- RecA branch -----
  hv = stopgrad(UND_encode(x0))
  tpl = sample_template()
  t, ε = sample_noise()
  xt = (1-t)*x0 + t*ε
  L_reca = MSE(GEN(xt, t, concat(tpl, hv)), ε - x0)

  # ----- online bad / Δh (pick one or mix) -----
  x_bad = make_bad_online(P, x0, mode∈{rollout, multi_sample, corrupt})
  # mode=rollout: §5.1 stopgrad CFG step + same-ε renoising
  # mode=multi_sample: K samples, rank by dist to x0 / CLIP(P,·)
  # mode=corrupt: controllable destroy of x0 or P

  with no_grad:
    h_plus  = acts(observe_or_GEN, good=x0 or on_traj, P)
    h_minus = acts(observe_or_GEN, bad=x_bad, P or P_corrupt)
  Δh = ema_update(Δh, h_plus - h_minus)

  # ----- steered GEN -----
  vc, vu, vg = native_cfg(xt, t, P)
  v_steer = GEN_forward_inject(xt, t, P, delta_h=α*Δh, layers=L)
  Δ = H_g(v_steer - vc)
  L_delta = MSE(vg + Δ,  target_velocity)   # target = v* or H_g-aligned

  # optional SOAR corr on x_bad renoised state
  L_corr = 0
  if use_soar:
    x_off, t', v_corr = soar_renoised_target(x_bad, x0, ε)
    L_corr = MSE(GEN_inject(x_off, t', P, α*Δh), v_corr)

  L = (λ_reca*L_reca + λ_t2i*L_t2i + λ_delta*L_delta + λ_corr*L_corr)
  backward; optimizer_step()
```

要点：`make_bad_online` **禁止**读外部 bad 数据集；rollout **必须 stop-grad**；再噪 **必须同 \(\epsilon\)**（若走 SOAR 几何）；观察接口 on/off 一致。

---

## 7. 推理协议

```text
输入: P, seed
x ~ N(0,I)
for t in sample_grid:
  vc, vu, vg = native_CFG(x, t, P)
  if t in T_read:                          # 少量读点，如中高噪语义段
    X = clean_proposal(x, vg)              # x - t*vg
    h_obs = UND_observe(GEN_hidden(X))     # 无 ViT、无外图
    # Δh: (i) 用训练期 EMA 向量；或 (ii) 当步用「模板稠密 vs 当前 obs」差
    Δh_inf = Δh_ema  or  (h_reca_prior - h_obs)
    v = GEN_inject(x, t, P, +α * Δh_inf)
    v = vg + H_g(v - vc)
  else:
    v = vg
  x ← ODE_step(x, v, t)
decode(x)
```

| 设定 | 建议 |
|---|---|
| 默认 | 单次采样；\(R\in\{1,2\}\) 个读点注入 \(\Delta h\)；无外挂 ViT |
| \(\alpha\) | 验证集网格 \(\{0,0.5,1,2,\ldots\}\)；过大易毁画风 |
| 对照臂 | `native`（无注入）、`reca-only`（仅阶段 A 权重）、`reca+Δh`、等算力 `resample`、显式文字 `loop`（上界） |
| 跨时间 | 可选继承 `t_early` 存态 + plan 重放；与 \(\Delta h\) 注入点对齐训练 |

**与文字反思上界的关系**：显式 loop = 解码 + UND/VLM **文字**约束回退；本方案 = 同职责的 **连续激活差** 内化。上界只作天花板，不计入「隐式成功」。若 `reca+Δh` 在 native 失败子集上无法显著优于等算力 resample，则未证明内化纠错，只是换噪声运气。

---

## 8. 验收与 go / no-go

**必须同 seed、同网格、同分母** 对比：

1. native  
2. reca-only  
3. reca+Δh（本方法）  
4. 等算力 resample  
5. （可选）显式文字 loop 上界  

**指标**

- 原子问答 / GenEval、GenEval++ 子集（计数、属性、位置、存在）；检出与纠错分列。
- 配对 McNemar：在「native 失败」子集上，reca+Δh vs resample；**必须显著优于 resample**。
- RecA 诊断：重建保真（同 \(h_v\) 条件）相对基线提升；理解榜（POPE/MME 等）不崩。
- \(\Delta h\) 诊断：固定交换 good/bad 来源，注入方向应翻转；`α=0` 应回退到 native 字节级行为。
- 画质：人工 triptych + 可选 PickScore/美学；**禁止**仅以 MSE 判胜负。

| 判定 | 条件 |
|---|---|
| **go** | 原子子集上 reca+Δh > native，且 > resample（McNemar \(p<0.05\) 或效应稳定）；推理无 ViT；画风无系统性崩坏；\(\alpha=0\) 对照成立 |
| **no-go** | 仅 RecA 重建 MSE 降、T2I 不变；或改图但 ≤ resample；或依赖解码/ViT；或 guidance 反向抵消重现 |
| **改接口再训** | 观察交换失败 / \(\Delta h\) 不随图源变 → 先修 GEN-hidden 读口与 stop-grad 构造，再加大 \(\lambda_\Delta\) |

Pilot：先 32 条高难 prompt × 固定 seed；通过后再扩 GenEval / DPG 子集。

---

## 9. 风险与不做清单

**风险**

- **退化为 autoencoder**：RecA 无信息瓶颈 → 模式崩塌；必须分辨率瓶颈 / 模板多样 / 保留弱 T2I。
- **\(\Delta h\) 噪声**：多样本筛在全坏批次上不稳定 → 始终混入 \(x_0\) 作 good 锚；EMA 平滑。
- **画风漂移**：\(\alpha\) 过大或注入层过浅/过深 → 网格搜索层与 \(\alpha\)；保留 native \(K=0\) 路径。
- **捷径学习**：未 stop-grad 构造支路 → \(\Delta h\to 0\)；必须 sg + 只训注入后路径。
- **与 SOAR 重复叙事**：若只报速度 MSE、不报 UND 稠密条件与 \(\Delta h\) 翻转诊断，会被写成「又一个 diffusion corrector」。
- **多样性收窄**：接受「语义合法多样性」，用 LPIPS/Vendi 事后报告，不作首版主指标。

**不做**

- 不假设、不下载、不依赖外部 failure / preference 数据集作为训练必要条件。  
- 推理默认不加 ViT、不解码中间图喂回模型。  
- 不把外挂 VLM 多轮文字编辑写成贡献主体。  
- 不把 Flow-GRPO / 奖励模型当作本阶段必选项（可作达标后的第二阶段）。  
- 不删除可复现的 native / 旧 checkpoint loader。

---

## 10. 与相关工作的定位

| 工作 | 关系 |
|---|---|
| **RecA** [2509.07295](https://arxiv.org/abs/2509.07295) | 阶段 A 直接继承：UND embedding = dense prompt + 重建；本方案接到 BAGEL MoT 观察口，并接到阶段 B |
| **SOAR** [2604.12617](https://arxiv.org/abs/2604.12617) | 提供「自 rollout 坏态 + 同噪再锚回 \(x_0\)」几何；可选 \(\mathcal L_{\mathrm{corr}}\)；本方案主故事是 \(\Delta h\) 而非仅 \(v_{\mathrm{corr}}\) |
| **CAA / ActAdd** [2312.06681](https://arxiv.org/abs/2312.06681), [2308.10248](https://arxiv.org/abs/2308.10248) | \(\Delta h\) 的加减与对比均值差来自表征工程；迁移到 GEN residual / 条件位，正负对**在线**从 GT 构造 |
| 显式文字反思 / ReflectionFlow 等 | 上界与数据灵感；部署必须内化到 MoT，不改成外部反思 agent |
| 纯配对 MSE / 外挂 DiT corrector | 已否定为主故事：无失败态或无同模型理解通道 |

---

## 11. 落地检查清单

- [ ] 数据管道：**仅** GT \((P,x_0)\)；无 bad 集依赖  
- [ ] RecA：`UND(x0)` 稠密条件 + 重建损失 + 信息瓶颈  
- [ ] \(\Delta h\)：至少实现 §5.1 rollout；另备 multi-sample / corrupt  
- [ ] 构造支路 stop-grad；注入后路径可训  
- [ ] \(\mathcal H_g\) 继承；`α=0` / native 字节对照  
- [ ] 推理：单次采样内读点注入；无 ViT  
- [ ] 评测含 native / reca-only / reca+Δh / resample /（可选）显式 loop  
- [ ] 文档叙事含「显式多轮 → 单次采样」+「MoT 不可替代」+「仅 GT 在线造负例」

---

## 相关引用

- RecA: Reconstruction Alignment Improves Unified Multimodal Models. arXiv:2509.07295  
- SOAR: Self-Correction for Optimal Alignment and Refinement. arXiv:2604.12617  
- CAA: Steering Llama 2 via Contrastive Activation Addition. arXiv:2312.06681  
- ActAdd: Activation Addition. arXiv:2308.10248  
- BAGEL: Emerging Properties in Unified Multimodal Pretraining. arXiv:2505.14683  

仓库内对照：`bagel-LatentCoT-SOAR-trajectory-correction-plan.md`、`docs/bagel_mot_internal_edit_loop_plan_20261023.md`、显式上界 closed-loop 文档、guidance-safe / mot_edit 历史记录。
