# 严格等时流形跳转算子（isochronic_manifold_v1）系统逻辑与数学推导

本文档对 SenseNova-U1.5-8B-MoT 架构下的**纯文本中间态纠偏机制（严格等时流形跳转算子，`isochronic_manifold_v1`）**进行完整的数学与系统逻辑梳理。文档严格剥离所有比喻，从第一性原理、代码实现与微积分恒等式出发，明确训练与推理的全流程。

---

## 1. 基础物理框架与时间约定

### 1.1 时间轴与加噪约定
在标准 Flow Matching（如 Rectified Flow）与 SenseNova 的实现中，时间变量 $t \in [0, 1]$ 的方向定义如下：
* **$t = 0$**：完全随机的高斯噪声流形 $z_0 = \epsilon \sim \mathcal{N}(0, \sigma^2 I)$。
* **$t = 1$**：完全清晰的真实图像流形 $z_1 = x \in [-1, 1]^{3 \times H \times W}$。

给定任意真实图像 $x$ 与标准高斯噪声 $\epsilon$，前向线性加噪过程定义为两个边界点之间的直线插值：
$$z_t = t \cdot x + (1 - t) \cdot \epsilon$$

### 1.2 动力学速度与原生 Head 输出的数学本质
沿上述直线插值轨迹，理想的速度真值（Ground-Truth Velocity）为状态对时间的一阶导数：
$$v^* = \frac{\mathrm{d} z_t}{\mathrm{d} t} = x - \epsilon$$

在生成过程中，初始噪声 $\epsilon$ 是未知的。通过前向加噪方程变形，可将未知量 $\epsilon$ 用当前状态 $z_t$ 和终点 $x$ 严格表达：
$$(1 - t)\epsilon = z_t - t x \implies \epsilon = \frac{z_t - t x}{1 - t}$$

将该式代入常数速度公式 $v^* = x - \epsilon$，可得：
$$v^* = x - \frac{z_t - t x}{1 - t} = \frac{(1 - t)x - z_t + t x}{1 - t} = \frac{x - z_t}{1 - t}$$

#### SenseNova 源码中的 Head 定义
在 SenseNova 底座（`modeling_neo_chat.py` 第 628–638 行）中：
1. 神经网络最后一层输出头 `fm_head` 的实际输出张量即为**干净图像估计** $\hat{x}$（源码变量名 `x_pred`）：
   $$\hat{x} = \text{fm\_head}(h_{\text{last}})$$
2. 随后通过显式代数变换将其映射为速度：
   $$v_{\text{pred}} = \frac{\hat{x} - z_t}{(1 - t)}$$

因此，SenseNova 的物理网络头本质上是一个**以任意中间态 $z_t$ 为输入、直接回归清晰图像 $x$ 的全频 Clean Head**。

---

## 2. 训练阶段全逻辑（`train.isochronic_manifold_update`）

### 2.1 结构配置与参数冻结矩阵
* **冻结模块**：
  * 多模态理解主干（UND Branch，`visual_model_mot_und` 等）
  * 原生时间嵌入层（`timestep_embedder`）
  * 原生输出头（`fm_head`，直接复用其预训练权重）
  * 原生生成变换器主干权重
* **可训练模块**：
  * 仅在每层生成分支的自注意力矩阵（`q/k/v/o_proj_mot_gen`）与 MLP 矩阵（`gate/up/down_proj`）挂载秩为 64 的 `GENLoRA`。
  * LoRA 矩阵满足 $W = W_0 + \frac{\gamma}{r} B A$，其中 $B$ 矩阵全零初始化，初始状态在数学上严格恒等于原生底座。
* **输入文本条件**：
  * 仅输入纯文本指令：`Edit request: <feedback>`。
  * 引入 `edit_prompt_dropout = 0.10`：以 10% 概率将输入文本置为空（`None`），训练无条件流形分支以对齐推理期的 CFG 评估。

### 2.2 状态构造：中段噪声-干净桥（Noisy-to-Clean Bridge）
给定配对训练样本：源图 $S$、目标图 $T$、固定的介入时刻 $t_0 = 0.36$ 与训练上限 $\tau_{\max} = 0.60$。

1. **构造 $t_0$ 时刻源噪声状态**：
   $$a_{t_0} = t_0 \cdot S + (1 - t_0) \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)$$
2. **中段语义窗口时间采样**：
   采样时间 $\tau$ 严格约束在 $[t_0, \tau_{\max}] = [0.36, 0.60]$ 内：
   $$\tau = \begin{cases} t_0, & \text{with probability } p_{\text{start}} = 0.15 \\ t_0 + (\tau_{\max} - t_0) \cdot u, \quad u \sim \text{Uniform}(0, 1), & \text{with probability } 0.85 \end{cases}$$
3. **计算桥状态（Bridge State）**：
   定义相对进度 $\alpha \in [0, \frac{\tau_{\max} - t_0}{1 - t_0}]$：
   $$\alpha = \frac{\tau - t_0}{1 - t_0}$$
   在当前时间 $\tau$ 构造线性插值状态 $q_\tau$：
   $$q_\tau = (1 - \alpha) a_{t_0} + \alpha \cdot T$$
   * 当 $\tau = t_0$ 时，$\alpha = 0$，$q_\tau = a_{t_0}$，模型仅能观测到源图在 $0.36$ 时刻的带噪输入；
   * 当 $\tau \in (t_0, \tau_{\max}]$ 时，$q_\tau$ 沿直线向目标清晰图 $T$ 演化。

### 2.3 损失函数与时间权重设计
模型接收状态 $q_\tau$、时间标量 $\tau$ 与编辑指令 $e$，输出对目标图 $T$ 的干净预测：
$$\hat{T} = H_\theta(q_\tau, \tau, e)$$

损失函数定义为有效画布区域（排除非有效 padding）上的均方误差：
$$\mathcal{L}_{\text{manifold}} = \frac{1}{\sum \text{valid}} \sum_{i \in \text{valid}} \left( H_\theta(q_\tau, \tau, e)_i - T_i \right)^2$$

#### 为什么使用 Clean MSE 而非 Velocity MSE？
在原生 SenseNova 训练中，速度损失形式为：
$$\mathcal{L}_v = \| v_\theta - v^* \|^2 = \left\| \frac{H_\theta - q_\tau}{1 - \tau} - \frac{T - q_\tau}{1 - \tau} \right\|^2 = \frac{1}{(1 - \tau)^2} \| H_\theta - T \|^2$$
当 $\tau \to 1.0$ 时，系数 $\frac{1}{(1 - \tau)^2} \to \infty$，强制放大了尾段的微小高频误差。本项目限定 $\tau \le 0.60$，并直接采用 Clean MSE，相当于施加了显式的时间加权衰减因子 $(1 - \tau)^2$，确保梯度平稳集中于物体增删、位置重排等结构性拓扑修改，避免微调参数过拟合高频像素。

---

## 3. 推理阶段全闭环逻辑（`inference.t2i_feedback`）

推理过程为单轨自闭环，分为五个确定性执行阶段：

```
时间轴 t:  0 ──(原生CFG推进)──▶ z_{0.36} ───────────────────────────────────────▶ 最终图像 (t=1.0)
                                 │                                             ▲
                                 ├─▶ 4步子网格显影 ──▶ 预览图 P                   │ 冻结原生底座 G1
                                 │                         │                   │ (携带原始 prompt,
                                 │                    VLM 审图 + 门控          │  从 0.36 积分至 1.0)
                                 │                         │                   │
                                 ▼                         ▼ (触发 edit)        │
                             执行流形跳转:                                      │
                             1) 冻结原生提取结构噪声锚点: eps_anchor            │
                             2) GENLoRA 在 [0.36, 0.60] 短程演化读出 x_hat_T    │
                             3) 严格等时回投: z_hat_{0.36} = 0.36*x_hat_T + 0.64*eps_anchor ┘
```

### 步骤 1：原生前缀生成（$t \in [0, 0.36]$）
* 采用完全冻结的原生 T2I 模型（输入为原始生成提示词 `original_prompt`，设置 `cfg_scale = 4.0`）。
* 从纯高斯噪声 $z_0$ 开始，沿时间网格通过标准 Euler 积分演化至 $t_0 = 0.36$，输出潜变量 $z_{0.36}$。

### 步骤 2：多步显影预览与自反思门控
为了在不改变主轨迹 $z_{0.36}$ 的前提下使未充分显影的次要物体与拓扑结构可见：
1. **4 步子网格显影（`preview_image`）**：
   * 在区间 $[0.36, 1.0]$ 内提取 4 个离散子网格节点（覆盖 $[0.36, 0.75]$ 物体涌现窗口）。
   * 在分离的张量拷贝上，首个区间执行二阶 Heun 预估-校正积分，后续 3 个区间执行一阶 Euler 积分，得到预览图像 $P$。
   * **断言检查**：主状态 $z_{0.36}$ 必须严格保持恒定不变。
2. **多模态理解与双重零偏置门控（Dual Zero-Bias Gating）**：
   * 多模态理解模块（VLM/UND）输入 $P$，生成诊断文本。
   * **Gate 1（无提示词偏置的 `add` 聚焦复核）**：若拟触发增加物体指令，在隐藏原 prompt 目标计数的条件下，对小物体区域进行二次高精度扫描，拦截假阳性指令。
   * **Gate 2（端点一致性门控）**：利用确定性规则硬拦截算术矛盾指令（如：当前目标数量已满足提示词要求却仍发出增删命令；或将明确要求存在的实体计数修改为 0）。
   * 若门控未通过或判定无需修改，流程直接跳转至**步骤 5**。

### 步骤 3：结构噪声锚点提取与短程目标预测
若触发修改指令 `feedback`，在 $z_{0.36}$ 处执行两项解耦操作：

#### 3a. 提取原生结构噪声锚点（$\hat{\epsilon}_{\text{anchor}}$）
调用**完全冻结、未挂载 LoRA** 的原生生成分支 $H_0$，在无条件文本输入 $\varnothing$ 下评估当前状态：
$$\hat{x}_S = H_0(z_{0.36}, t_0, \varnothing)$$
利用加噪方程严格反解出蕴含整张图像宏观构图、光照与背景高频分布的结构噪声分量：
$$\hat{\epsilon}_{\text{anchor}} = \frac{z_{0.36} - t_0 \cdot \hat{x}_S}{1 - t_0} = \frac{z_{0.36} - 0.36 \cdot \hat{x}_S}{0.64}$$

#### 3b. 启用 GENLoRA 短程演化读出新目标（$\hat{x}_T$）
挂载训练好的 `GENLoRA` 权重，以 $z_{0.36}$ 为起点，条件设为 `Edit request: <feedback>`。
在短程时间窗口 $\tau \in [0.36, 0.60]$ 内执行 Euler 积分推进：
$$q_{\tau + \Delta \tau} = q_\tau + \Delta \tau \cdot v_\theta(q_\tau, \tau, \text{feedback})$$
在终止节点 $t_{\text{stop}} = 0.60$ 处，通过 Clean Head 读取纠正后的干净目标图预测：
$$\hat{x}_T = H_\theta(q_{0.60}, 0.60, \text{feedback})$$
*(注：若配置编辑 CFG，则在此处计算条件预测与无条件预测的线性外推：$\hat{x}_T = \hat{x}_{\text{uncond}} + s \cdot (\hat{x}_{\text{cond}} - \hat{x}_{\text{uncond}})$)*。

### 步骤 4：严格等时回投（Strict Isochronic Reprojection）
这是整个算子的核心几何操作：**放弃从 $0.60$ 状态继续向前采样的路径，将其投影回 $t_0 = 0.36$ 的原生流形**。

将修正后的干净目标预测 $\hat{x}_T$ 与原生结构噪声锚点 $\hat{\epsilon}_{\text{anchor}}$ 重新代入 $t_0 = 0.36$ 的加噪公式：
$$\hat{z}_{0.36} = t_0 \cdot \hat{x}_T + (1 - t_0) \cdot \hat{\epsilon}_{\text{anchor}} = 0.36 \cdot \hat{x}_T + 0.64 \cdot \hat{\epsilon}_{\text{anchor}}$$

* **等时性（Isochronic）**：输出潜变量的时间戳与输入介入时间严格一致（均为 $t_0 = 0.36$）。
* **流形保持性**：$\hat{z}_{0.36}$ 的噪声项直接沿用原生提取的 $\hat{\epsilon}_{\text{anchor}}$，语义项更新为 $\hat{x}_T$，在代数上完全符合标准高斯分布加噪形式，未产生流形偏离。

### 步骤 5：交还原生底座接续生成（$t \in [0.36, 1.0]$）
* 卸载/停用 `GENLoRA`，恢复完全冻结的原生 T2I 模型。
* 将回投后的状态 $\hat{z}_{0.36}$ 作为输入，条件恢复为原始提示词 `original_prompt`。
* 沿剩余离散时间网格 $t \in [0.36, 1.0]$ 执行标准 Euler 积分：
  $$z_{t + \Delta t} = z_t + \Delta t \cdot \frac{H_0(z_t, t, \text{prompt}) - z_t}{1 - t}$$
* 在 $t > 0.80$ 时自动关闭 CFG（即 `cfg_interval = [0.0, 0.80]`），由原生预训练底座完成全部毛发、微观纹理与高频渲染。

---

## 4. 与早期方案的本质技术差异

| 比较维度 | 早期“等时编辑”（StateEditor with $s$ 维度） | 传统“噪声映射”（Noise / Inversion Mapping） | 本方案：中段桥去噪 + 严格等时回投 (`isochronic_manifold_v1`) |
| :--- | :--- | :--- | :--- |
| **状态推进空间** | 在固定时间点 $t=0.36$ 内部人为定义第二进度维度 $s \in [0, 1]$，在切片内走 8 步 ODE。 | 沿反向 ODE 将带噪图或成图反演至高斯初态，在潜噪声空间执行特征变换。 | **回归单一真实物理扩散时间轴 $\tau$**。仅在 $[0.36, 0.60]$ 物理区间演化，无虚拟维度。 |
| **网络输出目标** | 必须新增专门初始化的“位移头”（Displacement Head）拟合像素差 $\Delta = T - S$。 | 拟合噪声残差差分 $\Delta \epsilon$ 或速度场残差 $v_{\text{tgt}} - v_{\text{src}}$。 | **零新增网络头**。直接复用原生 `fm_head`，直接输出全频干净图像 $\hat{x}_T$。 |
| **流形分布保真度** | $s$ 步推进后的状态容易脱离标准高斯加噪流形（流形漂移），导致背景异常或结构溃散。 | 反向反演（Inversion）存在累积离散化误差，极易破坏非编辑区域的一致性。 | **严格代数流形对齐**。通过闭式解 $\hat{z} = t_0 \hat{x}_T + (1 - t_0)\hat{\epsilon}$，确保状态严格落在原生流形上。 |
| **后续生成承接** | 原生底座无法直接承接漂移状态，被迫引入额外的 `TailGenerator` 模块拟合尾段。 | 严重依赖源图交叉注意力图与自注意力特征注入机制。 | **零额外参数承接**。完全交还给冻结的原生大模型以完整 prompt 自行渲染至 $1.0$。 |
| **条件输入依赖** | 往往需要构造目标图像的全局 caption，缺少目标 caption 时无法有效推断。 | 依赖源图像或参考图像的直接视觉条件输入。 | **纯文本指令驱动**（`Edit request: ...`），生成与编辑端完全无需额外视觉条件。 |

---

## 5. 核心代数恒等式速查与代码映射表

| 概念名称 | 数学表达式 | 代码实现位置（`bagel-loop-layer-NPU`） |
| :--- | :--- | :--- |
| **加噪状态方程** | $z_t = t \cdot x + (1 - t) \cdot \epsilon$ | `train.py:97` (`anchor = t0 * source + (1.0 - t0) * eps`) |
| **流匹配速度转换** | $v_t = \frac{\hat{x} - z_t}{1 - t}$ | `model.py:203` (`(clean - state.float()) / max(1.0 - tau, t_eps)`) |
| **中段桥状态插值** | $q_\tau = (1 - \alpha) a_{t_0} + \alpha T$ | `train.py:106` (`state = (1.0 - alpha) * anchor + alpha * target`) |
| **干净目标训练损失** | $\mathcal{L} = \| H_\theta(q_\tau, \tau, e) - T \|^2$ | `loss.py:6` (`clean_target_mse(predicted_clean, target, valid)`) |
| **原生噪声锚点提取** | $\hat{\epsilon}_{\text{anchor}} = \frac{z_{t_0} - t_0 \hat{x}_S}{1 - t_0}$ | `field.py:210` (`(state.float() - t0 * x_source_est.float()) / (1.0 - t0)`) |
| **短程目标干净预测** | $\hat{x}_T = H_\theta(q_{t_{\text{stop}}}, t_{\text{stop}}, e)$ | `field.py:222` (`editor.predict_guided_clean(q_stop, t_stop, ...)`) |
| **严格等时回投重组** | $\hat{z}_{t_0} = t_0 \cdot \hat{x}_T + (1 - t_0) \cdot \hat{\epsilon}_{\text{anchor}}$ | `field.py:167` (`t_out * clean_pred.float() + (1.0 - t_out) * eps_eff`) |
