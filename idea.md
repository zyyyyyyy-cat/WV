# ViSA→UAD：模态条件双 Token + ETGM/DLFM 迁移实验

**方案版本：V1.1 — 代码修改指导与实现验收规范**  
**整理日期：2026-10-22**  
**适用任务：WHU-MARS RGB/NIR/TIR × Ground/Aerial，基于 UAD ViT-B/16 的统一 Re-ID 检索**  
**文档用途：供开发者/Codex 依据现有 UAD 项目实施**

> **方案状态与边界。** 本文是经 ViSA、MDReID、Uni-Prompt 与 UAD 论文/公开代码比较后的**首轮实验设计**。结构和梯度定义可核验，但检索性能、信息互补性及超参数最优性尚未经实验确认。凡“保留/修正/不采用”均指本实验的工程选择，**不意味着相关论文或源码一定存在方法错误**。实现时以用户实际复现且运行稳定的 UAD 代码、权重、配置为准；本文中官方仓库路径用于定位参考实现。

---

## 0. 阅读说明与来源标记

为区分论文依据和我们主动设计，下文统一使用：

- **[UAD-原有]**：来自 UAD 既有 ViT、训练接口、CE/WRT、ProCA/GPD、数据采样或检索协议，不主动改动其定义。
- **[ViSA-原理]**：来自 ViSA 的 VAB / ETGM / DLFM 方法思想。
- **[ViSA-源码]**：来自 ViSA 当前官方代码的真实计算方式；若和论文描述不同，将单独说明。
- **[MDReID-借鉴]**：来自 MDReID 显式共享/特有信息、互补身份学习的设计动机，不等于复制其 ROL/KDL。
- **[Uni-Prompt-借鉴]**：来自身份与域条件分工的思想；原方法额外提示主要在 CLIP 文本侧，不等于在视觉 ViT 直接加相同数量的 Token。
- **[本方案-自适应]**：面向 WHU/UAD 的新增结构、参数、约束或有意偏离论文/代码的选择。

### 0.1 官方代码及论文定位

| 来源 | 参考代码/论文位置 | 本方案实际借鉴内容 |
|---|---|---|
| UAD / WHU-MARS | [官方仓库](https://github.com/msm8976/WHU-MARS)；`CVPR26_UAD/model/backbones/vit_pytorch.py`、`model/make_model.py`、`processor/processor.py`、`loss/wrt_loss.py` | 12 层 ViT-B/16、CLS/BNNeck、三模态 PKM、原始 CE/WRT/ProCA/GPD、评测接口 |
| ViSA (CVPR 2026) | [官方仓库](https://github.com/Cat-Zero/ViSA)；`fastreid/modeling/backbones/vision_transformer_multiview_ViSA.py` | View/Specific Token 选择思路、ETGM 专家 Router、局部 Top-K、GCN、`out_token` 汇聚 |
| ViSA 方法论文 | [CVPR 论文页](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_View-Aware_Semantic_Alignment_for_Aerial-Ground_Person_Re-Identification_CVPR_2026_paper.html)；VAB / ETGM / DLFM 部分 | 共享/特有信息及专家局部关系的总体框架；**不沿用逐层相减** |
| MDReID (NeurIPS 2025) | [官方仓库](https://github.com/stone96123/MDReID)；`modeling/clip/model.py`、`engine/processor.py`、`layers/triplet_loss.py` | 多特殊 Token 能建立不同身份表示；身份互补目标的重要性；不搬运原始 ROL/KDL |
| Multi-modal Multi-platform Re-ID / Uni-Prompt /MP(ICCV 2025) | [官方仓库](https://github.com/MP-ReID/mp-reid)；`model/make_model_uniprompt.py` | 将身份与模态/平台因素分开考虑的原则；**首轮只做模态，不做平台 Prompt** |

> 注意：本文没有对 ViSA 的全部参数和全部损失作逐值复刻，也没有将 Uni-Prompt 文本 Prompt 假装成 ViT 的图像 Token。相关“来源”说明的是思想/模块出处与自适应修改边界。

---

## 1. 实验目标、非目标与设计准则

### 1.1 问题定义

输入单张行人图像 $x$、训练身份 $y$（测试时不可用）和已知光谱模态 $m\in\{R,N,T\}$（RGB、NIR、TIR）。检索时仍使用 UAD 的同一 Query/Gallery 评测规则、L2 特征归一化、欧氏距离与同相机过滤；**不引入重排序、TTA、按模态分开的相似度打分函数**。

现有 UAD 已学习一个跨三模态可比较的 CLS 表示。本实验的核心假设是：**新增模态条件 Token 可以捕获 CLS 未充分使用的模态内身份线索，经 ViSA 的专家/局部聚合转换为可比较的 Local 特征，最终改善统一检索。**

### 1.2 模块分工（一句话明确）

| 表示 | 来自哪里 | 主要职责 | 直接身份监督 | 最终检索？ |
|---|---|---|---|---|
| $f_s$（Shared/CLS，768D） | ViT 第 12 层 CLS | UAD 统一身份锚点 | UAD CE/WRT/ProCA/GPD | **是** |
| $f_p$（Specific，768D） | ViT 第 12 层模态条件 Token | 当前光谱下的辅助身份线索 | **低权重模态内 WRT** | **否**，只用于专家输入 |
| $Q_s$（16×768） | Shared ETGM | 用共享身份信息生成条件语义查询 | 由 Local/Fusion 间接监督（对 CLS 输入 detach） | 通过 Local 间接参与 |
| $Q_p$（16×768） | RGB/IR ETGM | 用特有身份信息生成条件语义查询 | Local/Fusion 间接监督；保留到 $f_p$ 的梯度 | 通过 Local 间接参与 |
| $f_l$（Local，768D） | DLFM 双图 + 独立 `out_token` | 将共享和特有局部线索转换为统一身份补充 | 轻量 Local CE + Fusion WRT | **是** |
| $z$（Fusion，1536D） | 归一化 CLS 与 Local 加权拼接 | 唯一正式检索表示 | **直接 Fusion WRT** | **是，唯一主结果** |

**概念性边界：**“Shared”是受三模态身份监督和原型约束的统一表示，不意味着严格模态独立；“Specific”是模态条件身份表示，不等于纯模态噪声；Local 能否互补仍待实验验证。

### 1.3 版本边界

本轮**不做**：平台 Token、平台 CE、ViSA 逐层相减、最终 CLS−Specific、额外文本 Prompt、第三套 TIR 专家池、独立 Fusion CE、强正交、ROL、KDL、新型动态加权/额外门控/可微 Top-K。先保证当前闭环成立。

---

## 2. 一张表锁定整体参数与张量接口

以下张量的第一维 $B$ 指**展平后的每 GPU 图像数**；UAD 的原始三模态训练会按 RGB、NIR、TIR 顺序拼接图像，不能把原始单模态子批次大小误认为总 $B$。

| 位置 | 输入 → 输出 | 固定设置 |
|---|---|---|
| Patch Embedding | `[B,3,256,128]` → `[B,128,768]` | Patch 16，Stride 16，网格 16×8 |
| 输入特殊 Token | CLS `[B,1,768]` + Specific `[B,1,768]` | **每张图仅 2 个特殊 Token**，共 130 Token |
| ViT Blocks 1–11 | `[B,130,768]` → `[B,130,768]` | 不执行 CLS−Specific；缓存 $P_{11}=[B,128,768]$ |
| Block 12 + LN | `[B,130,768]` → `[B,130,768]` | $f_s=[B,768]$；$f_p=[B,768]$ |
| Shared ETGM | $f_s$ → $Q_s=[B,16,768]$ | Shared 8 专家、Top-2 |
| Specific ETGM | $f_p$ → $Q_p=[B,16,768]$ | RGB 8、IR 8；按模态选池、池内 Top-2 |
| Top-K Patch | $(f_s,P_{11})$ 与 $(f_p,P_{11})$ | 各选 $K=40$ 个 Patch，输出 `[B,40,768]` |
| 每路 16 张局部图 | Query `[B,16,768]` + Patch `[B,40,768]` | 节点 `[B,16,41,768]`，邻接 `[B,16,41,41]` |
| 两路独立 GCN | `[B,16,41,768]` → `[B,16,768]` | 每张局部图只返回**首个 Query 节点** |
| 局部聚合 | `[out_token, 16 shared,16 specific]` | Self-Attn + FFN，取首 Token，得 $f_l=[B,768]$ |
| 最终融合 | $f_s,f_l$ → $z=[B,1536]$ | 每支先 L2，系数 $\beta=0.5$，整体单位化 |

---

## 3. Backbone：三模态选择式 Specific Token

### 3.1 Token 参数定义 [ViSA-原理 / 本方案-自适应]

保留 UAD 原有 `cls_token`、12 层 Transformer、Patch Embedding、`norm`。新增**三组独立训练参数**：

$$T_R,T_N,T_T\in\mathbb R^{1\times1\times768}.$$

每张图根据真实模态选择其中一组：

$$T_m=\begin{cases}T_R&m=\mathrm{RGB},\\T_N&m=\mathrm{NIR},\\T_T&m=\mathrm{TIR}.\end{cases}$$

$$X_0=[T_{\mathrm{CLS}},T_m,P_1,\ldots,P_{128}].$$

- 不采用 RGB/IR 共用输入 Token：**NIR、TIR 是两种真实光谱，Backbone 应能分别学习条件参数**；ETGM 仍暂时可以共享 IR 专家池。
- 不采用“$T_0+\Delta_m+\Delta_{platform}$”：首次实验只解决模态 Gap，不引入平台因素与额外耦合。
- 三组 Token 用**同一个随机初始化张量的数值拷贝**初始化，但注册为三个独立的 `Parameter`，避免起点带有非必要模态偏差；训练后允许分化。可使用标准差约 0.02 的截断正态，**这只是初始化规范，不是身份对齐损失**。
- 模态标签仅作为选择 Token/专家池的已知输入条件；不是测试身份标签。测试时每张图也必须提供模态标签。

### 3.2 位置编码：禁止直接错误插值 [UAD-原有 / 本方案-自适应]

原 UAD 位置信息含 1 CLS + 128 Patch，形状 `[1,129,768]`。新序列是 1 CLS + 1 Specific + 128 Patch，需要 `[1,130,768]`：

$$E_{new}=[E_{CLS}^{old},E_{SP}^{new},E_{P_1}^{old},\ldots,E_{P_{128}}^{old}].$$

- 原 CLS/128 个 Patch 的位置编码**一一保留，不错位**；新增 Specific 位置编码初始可置零。
- 不允许把原 129 个位置直接重解释为两个特殊 Token 的格式；**不能不加区分调用 UAD 默认 `resize_pos_embed` 将 129 → 130**，因为它通常默认只有一个特殊 Token、剩余为规则网格，会导致插值网格错误。
- 加载基线权重的推荐顺序：先按原 129 位置正确载入主干，再显式扩展/重建 130 位的位置编码；或者在加载函数中对 `pos_embed` 专门映射。
- 若用户实际模型使用 SIE 相机/视角嵌入，保留现有配置及广播方式；本轮不新增平台 SIE，也不改变其参数或标签意义。

### 3.3 12 层执行与特征提取 [UAD-原有 / ViSA-借鉴]

$$X_{11}=\mathrm{Block}_{1:11}(X_0),\qquad P_{11}=X_{11}[:,2:,:].$$

$$X_{12}=\mathrm{LN}(\mathrm{Block}_{12}(X_{11})).$$

$$f_s=X_{12}[:,0,:],\qquad f_p=X_{12}[:,1,:].$$

- **无逐层减法、无末端 CLS−Specific。** 不能只关掉 `INNER_SUB` 却继续在 ETGM/身份分类前执行 `global_feats - view_feats`。
- 与 ViSA 源码的一个主动区别：ViSA 先取前 11 层输出，再用一份深拷贝的第 12 层 `b1` 生成全局 Token；本方案直接使用 UAD **唯一一份原始第 12 层与 LN**，保留第 11 层 Patch 用于 DLFM，避免重复权重、遗漏优化及额外 DDP 未使用参数。
- $f_s,f_p$ 都是最后 LayerNorm 后、任何身份 BNNeck 前的表示；Local CE 的 BNNeck 是另外建立的。不要混淆 Transformer `LayerNorm` 与身份分类的 `BNNeck`。
- 输入级 Token 确实参与 12 层 Self-Attention，因而会改变 CLS/Patch 的形成；**不能声称它是完全无侵入或保证不影响 UAD 原空间**。原主监督、低权重辅助及 Token-only 对照用于管理而非消除这一风险。

---

## 4. ETGM：全局条件查询专家（纠正单 Key 注意力描述）

### 4.1 结构与参数 [ViSA-ETGM / 本方案-自适应]

沿用 ViSA 的三类专家池，每池 8 个独立专家，Router Top-2，每个专家维护 16×768 可学习 Query：

| 池 | 输入 | 个数 | 激活规则 |
|---|---|---:|---|
| Shared | `sg(f_s)` | 8 | 每张图都经过 Shared Router，再 Top-2 |
| RGB | $f_p$ | 8 | RGB 图像选择 RGB 池，池内 Top-2 |
| IR | $f_p$ | 8 | NIR/TIR 选择 IR 池，池内 Top-2 |

三种模态 Token 与两类模态特有专家池**不矛盾**：Token 负责 RGB/NIR/TIR 真实成像条件，专家的首轮分组沿用 ViSA 的 RGB/IR 两组设计。NIR/TIR 输入内容和 Router 选择仍可不同。

### 4.2 为什么不直接照搬 Cross-Attention [ViSA-源码核查]

ViSA `MoEBlock` 的 `PromptRecapBlock` 对**单个全局向量**执行 `CrossAttention(queries, keys)`，其中 $Q$ 是 16 个 Query、$K/V$ 仅 1 个全局 Token。单 Key Softmax 恒为 1，**它不能在空间 Patch 之间做选择**；但仍会把经过 Value/Output 投影的全局向量广播给所有 Query，并由后续 Self-Attention/FFN 形成不同查询。

本方案**不虚构其空间选择能力**，而使用等价功能导向的简单条件注入：

$$H_0=\mathrm{LN}(T_e+W_c f),$$
$$H_1=\mathrm{LN}(H_0+\mathrm{SelfAttn}(H_0)),$$
$$E_e(f)=\mathrm{LN}(H_1+\mathrm{FFN}(H_1)),$$

其中 $T_e\in\mathbb R^{16\times768}$ 是第 $e$ 个专家的查询参数，$f\in\mathbb R^{768}$ 广播到 16 个 Query。可沿用原来的 Value/Output 投影组合来实现 $W_c$，**不必保留单 Key 下不起选择作用的 Query/Key 投影**。Self-Attn、FFN 与残差次序应固定，不能将其全部误删成普通线性层。

> **实现定位**：这不是 ViSA 论文声称的多空间位置 Cross-Attention，也不是把完整 Patch 序列送进 ETGM；空间 Patch 检索由下一阶段 DLFM 负责，避免两个模块重复建模与额外梯度通路。

### 4.3 Router 与加权求和 [ViSA-源码]

每个池有单独的 Router（768→8 logits）：

$$a=W_rf+b,\quad \mathcal K=\mathrm{Top2}(a),\quad w_k=\mathrm{Softmax}(a_{\mathcal K})_k.$$

$$Q=\sum_{k\in\mathcal K}w_kE_k(f),\quad Q\in\mathbb R^{B\times16\times768}.$$

- **Top-2 索引不可微**；被选专家、Top-2 的 Softmax 权重及所选 Router logits 可接收身份损失的梯度；未激活专家本轮可能没有身份梯度，属于正常稀疏计算。
- 不要误实现为从 24 个专家中统一 Top-2：先按 Shared/RGB/IR 选择池，**再在每个池内部 Top-2**。
- 3 个池各自独立的 Router；Shared 每张图参与，特有池视模态参与。两条路径输出 $Q_s,Q_p$，各 `[B,16,768]`。
- **避免破坏梯度的分派写法**：条件执行后将选中结果按原 Batch 索引汇总，需要保证加权专家输出到 loss 的计算图仍在；不要对结果使用 `.detach()`、`.data` 或在 `no_grad()` 下写入；可以采用可微的 `index_add`/按索引累积形式。

### 4.4 Shared Router 的精确 Stop-Gradient 规则 [本方案-自适应]

**必须在进入整个 Shared ETGM 前截断其输入**：

$$\boxed{Q_s=\mathrm{ETGM}_{shared}(\operatorname{sg}(f_s))}$$
$$\boxed{Q_p=\mathrm{ETGM}_{RGB/IR}(f_p)}.$$

含义：

- Shared Router 的 `gate` 权重、被选中 Shared 专家参数、16 个 Query 及后续 GCN **继续更新**。
- **只有 Shared ETGM 输入 $f_s$ 对 Local/Fusion/Balance 的直接梯度被截断**；不能把 Shared Experts 的权重也冻结。
- 特有路径 $f_p$ **不 detach**，Local/Fusion 对 Specific Token 和共享 Transformer 仍有梯度。
- 这不是彻底冻结 CLS：UAD 原始损失始终更新 $f_s$；新增辅助损失通过 $f_p$ 和共享 ViT 权重，甚至 Token 间注意力，还可能间接改变后续 CLS；必须如实描述。

### 4.5 负载均衡 [ViSA-源码 / 本方案保持]

使用原 ViSA Router 的全 8 专家 softmax 均衡思想：对池内概率 $p_e$ 在有效输入样本上求平均重要度 $\bar p_e$，计算 $L_b=E\sum_e\bar p_e^2$（$E=8$）。它均衡的是**概率分布倾向**，不保证实际 Top-2 激活次数完全相同。

- Shared/RGB/IR 的损失分别在具有有效输入的池上计算；无当前样本的池不造假值。
- 按有效样本数量进行池间加权，避免某个 rank 中缺少某模态导致不合理放大。
- 目标权重初始 $\lambda_b=10^{-4}$；不要同时引入另一套 MoE aux loss。

---

## 5. DLFM：Patch 筛选、16 张 Query 子图、双 GCN 与 Local 聚合

### 5.1 相似度与候选 Patch [ViSA-DLFM / 本方案适配 K]

以第 11 层 Patch $P_{11}=[B,128,768]$ 为共同视觉候选。

$$s_i^s=\langle\operatorname{Norm}(f_s),\operatorname{Norm}(P_{11,i})\rangle,$$
$$s_i^p=\langle\operatorname{Norm}(f_p),\operatorname{Norm}(P_{11,i})\rangle.$$

在 128 个位置内每路独立选 Top-$K$：

$$I_s=\mathrm{TopK}(s^s,K),\quad I_p=\mathrm{TopK}(s^p,K),\quad K=40.$$

$$P_s=\mathrm{Gather}(P_{11},I_s),\quad P_p=\mathrm{Gather}(P_{11},I_p).$$

说明：

1. **各分支所有 16 个 Expert Query 共享该分支 Top-40 Patch 索引**；不是每 Query 单独 Top-40。这与 ViSA 代码保持一致。
2. `TopK` 索引是离散选择，无索引梯度；选中的 Patch 特征仍参与图前向计算。
3. Top-40 对 128 Patch 是约 31.25%；来自将 ViSA Stride=12 的 Top-64 在 Patch 数量上作约比例适配。**这是首轮固定工程参数，不等于证实 K=40 优于其他 K**。
4. 只在相似度计算时按特征维度 L2 归一化用于排序；**GCN 节点的数值仍使用选中 Patch 的原特征**，不要把排序归一化结果误传为图节点特征。
5. 即使共享分支 $f_s$ 在计算相似度时没有显式 `.detach()`，Top-K 索引本身也不会将 Local 的梯度连续传回相似度；为语义清晰可在纯索引计算里用 `no_grad`，但**不能顺带 detach Expert Query**。

### 5.2 关键形状：不是 56 节点的大图 [ViSA-源码]

设 $M=16$，$K=40$，$D=768$。对每条分支的每个 Query $j$，分别建立一张 **41 节点的局部图**：

$$N_s^{(j)}=[Q_s^{(j)},\operatorname{sg}(P_s)]\in\mathbb R^{41\times768}.$$

将 16 张图堆叠：

$$\boxed{N_s\in\mathbb R^{B\times16\times41\times768}},$$
$$\boxed{N_p\in\mathbb R^{B\times16\times41\times768}}.$$

邻接对应：$A_s,A_p\in\mathbb R^{B\times16\times41\times41}$。

**GCN 每张图只读取第 0 个 Query 节点的更新结果**：

$$\mathrm{GCN}(N,A)=\phi(\hat A\,NW)[:,:,0,:]\in\mathbb R^{B\times16\times768}.$$

此处 $[:, :, 0, :]$ 是 41 节点图中的首节点 Query，不是从 56 节点整图中强行取前 16 个。与 ViSA 当前代码 `output[:, :, 0, :]` 完全对应。**绝不能让 GCN 返回 `[B,16,41,768]` 后直接与 `[B,16,768]` 残差相加。**

### 5.3 节点值/邻接的梯度策略 [本方案-自适应]

固定以下定义：

$$N_s^{(j)}=[Q_s^{(j)},\operatorname{sg}(P_s)],\quad N_p^{(j)}=[Q_p^{(j)},\operatorname{sg}(P_p)].$$
$$A_s=\operatorname{sg}(\operatorname{Adj}(N_s)),\quad A_p=\operatorname{sg}(\operatorname{Adj}(N_p)).$$

| 部分 | 是否保留 Local/Fusion 梯度 | 原因 |
|---|---|---|
| Expert Query 节点 $Q_s,Q_p$ | **是** | 局部监督需要更新选中 Expert 与 Router |
| 选中 Patch 的**节点值** $P_s,P_p$ | **否**（此条局部路径） | 首轮减少局部图损失直接扰动 UAD Patch 主空间 |
| `TopK` 索引 | 否 | 离散索引，天然不可微 |
| 图邻接 $A_s,A_p$ | 否 | 固定图权重，不沿图构造本身反传 |
| GCN 权重、LN、独立 `out_token` | **是** | 构成实际可学习 Local 路径 |

**不要照搬** ViSA 原实现把构造 `feat_sp/feat_vp` 的整个过程置于 `with torch.no_grad()`，因为那样连 Expert Query 的 GCN 节点值也被停止梯度。本方案只 detach 选中 Patch 节点值和图邻接，**不 detach Query**。与此同时，Patch/ViT 仍通过 UAD 主损失、Specific 路径与 token 交互获得更新，不是冻结 Backbone。

### 5.4 邻接构造、归一化、GCN 运算 [ViSA-源码 / 工程修正]

每张 41 节点图：

1. 以节点特征按最后维度 L2 归一化，构造余弦 Gram 矩阵 $S=\bar N\bar N^\top$；在当前方案中邻接对特征停止梯度。
2. 非负化：$A^+=\max(S,0)$。
3. 加单位自环：$\widetilde A=A^++I$。因余弦自身通常已为 1，这一步是**额外增加**一份自环，按 ViSA 代码保留，不擅自更换归一化约定。
4. 节点度 $d_i=\sum_j\widetilde A_{ij}$，数值下限 $\epsilon$；$\hat A_{ij}=\widetilde A_{ij}/\sqrt{d_i d_j}$。
5. 做线性变换、加权邻接聚合、`tanh` 激活，**只返回 Query 节点**（`[:,:,0,:]`）。

$$\widehat Q_s=\mathrm{GCN}_s(N_s,A_s),\qquad\widehat Q_p=\mathrm{GCN}_p(N_p,A_p).$$

$$\boxed{\bar Q_s=\mathrm{LN}_s(\widehat Q_s)+Q_s},\quad\boxed{\bar Q_p=\mathrm{LN}_p(\widehat Q_p)+Q_p}.$$

- 两路使用**独立参数** `shared_gcn` 与 `specific_gcn`。**ViSA 源码定义两套 GCN 却在两分支都调用 `share_gcn`；本方案主动改为分别调用，贴近论文双路径动机**。这不是已证实性能更好的结论，只是明确的实验设计选择。
- 原代码在 GCN 中 `torch.eye(..., device='cuda')`；应基于**节点所在的实际 `device` 与 dtype**创建，避免 DDP 多 GPU 的错误设备。
- 图相似度、度与归一化建议在 FP32 计算，使用批量张量算术代替逐 Batch/Query 的 Python 对角矩阵循环；防止 AMP 下不稳定。
- 不能把 $W\in\mathbb R^{768\times768}$ 的通道变换误解成节点数变换；GCN 保持 768 通道，首节点抽取将节点维收为 16 Query。
- 同一组 40 Patch 用 `unsqueeze/expand` 组织为 Query 子图即可；避免为每 Query 提前持久复制所有 Patch，注意广播后某些操作仍可能真实分配临时内存。
- 检查邻接、度、GCN 输出无 NaN/Inf；若某些相似边为 0，自环仍确保非零度，不因此额外添加新邻接学习器。

### 5.5 Local 聚合与论文/源码不一致 [ViSA-论文 vs 源码 / 本方案选择]

ViSA 论文的局部融合公式使用 CLS 参与最终查询聚合；当前公开代码实际拼接独立 `out_token`：

$$H_0=[T_{out},\bar Q_s,\bar Q_p]\in\mathbb R^{B\times33\times768}.$$
$$H=\mathrm{OutAttenBlock}(H_0),\qquad \boxed{f_l=H[:,0,:]\in\mathbb R^{B\times768}}.$$

- 选择**独立、三模态共享的 $T_{out}$**，随机初始化并可训练。它不属于 ViT 输入 Token，不增加 ViT 序列长度。
- `OutAttenBlock` 当前代码执行的是输入 33 Token 间的**Self-Attention + FFN**，不是对原图执行新的 Cross-Attention；因此描述必须与实际操作一致。
- 这样 CLS 最终仍单独进入检索，并已经用于 Shared Experts，局部汇聚不必再次直接以 CLS 为首位 Token；**降低冗余倾向，但不保证特征完全互补**。
- Local 的 768 维 $f_l$ 必须来自 `out_token` 输出，不能错误地取 32 个 Query 的平均或将所有 Query 拼接为最终 ReID 特征。

---

## 6. 损失：对象、归一化、强度、有效梯度与原型时序

### 6.1 主干共享损失 $L_{UAD}$ [UAD-原有]

**不替换 UAD 已有的 Loss 接口、采样器、正负样本定义、GPD 原型库或更新时序**。仅规定这些损失继续接受 $f_s$：

$$\boxed{L_{UAD}=L_{ID}^{s}+L_{WRT}^{s}+\lambda_{PCA}^{base}L_{ProCA}^{s}+\lambda_{GPD}^{base}L_{GPD}^{s}}.$$

其中 $L_{ID}^{s},L_{WRT}^{s}$ 原本在 UAD 中通过损失函数内部配置权重；上式是职责说明，**不要重复乘或漏乘 `ID_LOSS_WEIGHT/TRIPLET_LOSS_WEIGHT`**。

| 损失 | 输入 | 特别约束 |
|---|---|---|
| CLS CE | `classifier(BNNeck(f_s))` | 原共享身份分类器；所有 RGB/NIR/TIR 同一训练身份空间 |
| CLS WRT | **原始 pre-BN $f_s$** | 沿用现有 `TripletLoss_WRT(normalize_feature=False)` 默认输入处理 |
| ProCA | 按模态拼接顺序的 $f_s$，形状 `[3B_m,768]` | 与原 `target`、`NUM_INSTANCE`、组内身份顺序一致；中心 detach 与内部 L2 沿用原代码 |
| GPD | 同一 768D $f_s$ 和原型库 | 保留原温度、目标复制方式、原型 bank 的 DDP 更新与优化器 step 后更新 |

公开 UAD 配置含 `PCA_LOSS_WEIGHT=0.01`，但**实际运行的 Baseline 配置才是本项目的权威值**；不要将本文写成硬编码 ProCA=0.01 并覆盖已有工作配置。若使用 GPD 的 bank，则需要保存/正确恢复或明确重建其状态。

### 6.2 Specific 模态内 WRT $L_p$ [MDReID/IDKL 启发 / 本方案-自适应]

$$\boxed{L_p=\frac13\sum_{m\in\{R,N,T\}}\mathrm{WRT}(f_p^{m},y^{m})}.$$

- 按真实模态分三组；**不按平台再切分**。组内仍可以含 Ground/Aerial 同身份正样本。
- 直接使用最后 Transformer LN 后、BNNeck 前的 **$f_p$ 原始尺度**。**不先对 $f_p$ 做额外 L2**，因为 UAD 默认 WRT 的 `normalize_feature=False`，原始实现用 FP32 欧氏距离及 `SoftMarginLoss`，改变单位球尺度会改变难例权重分布。
- 不增加三个单独身份分类器或 Modality CE；条件 Token 已按已知模态索引，模态分类容易形成标签捷径。
- 每模态至少需要两个不同 PID，并确保每 PID 有至少两个同模态实例。UAD 的 WRT 源码把自身也记为正样本，因此需**确保不以自身作为唯一正样本**；首版不应不经对照就修改 WRT 本体的正例掩码。
- 只直接要求模态内部的身份距离合理，**不让原始 Specific 跨模态直接对齐**；它并不自动证明 Specific 与 CLS 不重复。

### 6.3 Local 身份分类 $L_l$ [ViSA-Local 监督 / 本方案减量]

$$\boxed{L_l=\mathrm{CE}(h_l(\mathrm{BN}_l(f_l)),y)}.$$

- 新建独立 Local BNNeck 和**全模态共享**的训练身份分类头；不使用 RGB/NIR/TIR 三个 Local 分类器。
- 该 CE 对 $f_l$ 形成统一身份语义；**不额外叠加 Local WRT**，因为最终融合已有直接 WRT。
- 推理取 **BNNeck 前 $f_l$**；分类头与 BN 后特征仅用于训练时的 Local CE。

### 6.4 统一检索向量与 Fusion WRT $L_f$ [本方案-自适应]

对全局和局部两支独立执行每图 L2：

$$u_s=\frac{f_s}{\|f_s\|_2+\epsilon},\quad u_l=\frac{f_l}{\|f_l\|_2+\epsilon}.$$

固定 $\beta=0.5$，组成：

$$\boxed{z=\frac{[u_s,\,\beta u_l]}{\sqrt{1+\beta^2}}\in\mathbb R^{1536}}.$$

训练 Fusion 分支使用**完全相同的前向数值**，仅截断共享分支直接梯度：

$$\boxed{z_f=\frac{[\operatorname{sg}(u_s),\,\beta u_l]}{\sqrt{1+\beta^2}}},$$
$$\boxed{L_f=\mathrm{WRT}(z_f,y)}.$$

- $z_f$ 与 $z$ **前向数值一致**；不要训练时对拼接另做一次独立 BN、Linear 或模态条件归一化，而推理却不做。
- Fusion WRT 使用 UAD 同一个 WRT 函数，输入已经是统一单位化特征；不调用 `normalize_feature=True` 做其他带参数的归一化。距离仍是欧氏距离。
- 不加 Fusion CE：CLS 与 Local 已有各自的身份 CE，融合优先用实际检索空间中的距离约束，避免第三套大分类头。
- `detach` 必须只用于计算 Fusion Loss 的 $u_s$ 视图，**不能使 UAD 主损失也读取 detach 后的 CLS**。
- 若计算中使用 `F.normalize`，请确保训练和测试代码共享**同一个融合函数**、同样的 `eps` 与顺序。若为数值精度再做一次最终整体 L2，应两端都做；禁止训练／推理各自实现不同版本。

**平方欧氏距离的可解释权重**（$u_s,u_l$ 真正单位化时）：

$$\|z_q-z_g\|_2^2=\frac{\|u_{s,q}-u_{s,g}\|_2^2+\beta^2\|u_{l,q}-u_{l,g}\|_2^2}{1+\beta^2}.$$

$\beta=0.5$ 对应共享/局部的**平方距离系数**约 80%/20%，**不等于 WRT 对局部分支的梯度固定为 20%，更不等于融合必定增益**。最终 Rank/AP 可用欧氏距离或其平方，排序相同；WRT 实际优化的是开平方的欧氏距离，不应把两个概念混淆。

### 6.5 总损失与渐进启用 [本方案-自适应]

$$\boxed{L_{total}=L_{UAD}+r(e)\,[\,0.05L_p+0.1L_l+0.5L_f+10^{-4}L_b\,]}.$$

$$r(e)=\min(1,e/5),\qquad e\in\{1,2,\ldots\}.$$

- 这里的 $r(e)$ 仅衰放**新增**损失；不能衰放原始 UAD 主损失。
- 5 epoch 为首轮确定的轻量暖启，不再增加多阶段冻结、重新学习文本或大规模权重搜索。
- 0.05/0.1/0.5/1e-4 是**固定首次实验的初始设置**，不是来自原论文的最优 WHU 系数。
- 由于 CLS WRT 与 Specific WRT 作用于未额外 L2 的特征，而 Fusion WRT 作用于单位化 $z$，**不要凭 loss 数值大小或 0.05/0.5 的表面大小判断实际梯度谁强**；一次性检查梯度尺度即可。
- UAD 源码有 `clip_grad_norm_(...,max_norm=1.0)`；保持现有基线梯度裁剪逻辑，避免新损失使步长异常。

---

## 7. 全部梯度路径：必须清楚写入代码注释与单测

### 7.1 前向链路（训练和测试共用）

```text
RGB / NIR / TIR image + known modality id
   ↓
Patch Embedding (128×768), CLS + one selected Specific token
   ↓
ViT blocks 1–11 ───────────────────→ P11 (128×768)
   ↓                                      ↓
Block 12 + final LN                          shared/specific Top-40 patch selection
   ↓                                      ↓
  f_s = CLS (768)                       selected patch values for GCN = stop_gradient
  f_p = Specific (768)                    ↓
   ├── f_s ───→ UAD CE/WRT/ProCA/GPD    16 query-centered subgraphs each branch
   │
   ├── stop_gradient(f_s) → Shared Router / selected shared experts → Q_s (16×768)
   │
   └── f_p → RGB/IR Router / selected specific experts → Q_p (16×768)
                       ↓
       Q_s, Q_p + selected P11 → independent GCNs → updated queries
                       ↓
             [learnable out_token, shared queries, specific queries]
                       ↓
                 OutAttenBlock (self-attention)
                       ↓
                   f_l = Local (768) ─────→ Local BNNeck/CE (train only)
                       ↓
         u_s = norm(f_s), u_l = norm(f_l)
                       ↓
        z = concat(u_s, 0.5*u_l)/sqrt(1.25) (1536)
                       ├── train: Fusion WRT, direct u_s path detached
                       └── test: same z, Euclidean retrieval
```

### 7.2 损失梯度影响矩阵

| 参数/输出 | UAD 主损失 | Specific WRT | Local CE | Fusion WRT | Balance |
|---|---|---|---|---|---|
| CLS 输出 $f_s$ 直接损失路径 | **有** | 非直接；Token 间耦合 | 经 Shared ETGM **截断**；Top-K 离散 | 两处直接路径都 **截断** | Shared Router 入口截断 |
| Specific 输出 $f_p$ | 可经 CLS 的 Self-Attention 间接影响 | **有** | **有**（Specific ETGM） | **有**（Local） | Specific Router **有** |
| ViT 共享 Block 参数 | **有** | **有** | **有**（Specific 路径） | **有**（Specific 路径） | Specific 路径可能有 |
| Shared Router/Shared Expert 权重 | 无直接主监督 | 无 | **有** | **有** | **有**（Router） |
| Specific Router/Specific Expert 权重 | 无直接主监督 | 无直接分支损失 | **有** | **有** | **有**（Router） |
| GCN_s、GCN_p、Local out_token | 无直接主监督 | 无 | **有** | **有** | 无 |
| Patch $P_{11}$ 经 GCN 节点值直接路径 | 原主干正常 | 原主干正常 | **截断** | **截断** | 不通过 GCN |
| BNNeck_s/Classifier_s | **有** | 无 | 无 | 无 | 无 |
| BNNeck_l/Classifier_l | 无 | 无 | **有** | 无 | 无 |

**特别重要的非对称性：** Shared ETGM 读 `sg(f_s)` 仍会学习 Shared Expert/Router；Local/Fusion 不会由这条路径直接驱动 CLS。Specific ETGM 读正常 $f_p$，所以新增局部身份信号会训练 Specific、输入 Token 与共享 ViT。**不能声称 Stop-Gradient 完全保护了 CLS 或冻结了共享 Transformer**，因为 ViT 自注意力参数、CLS Token 本身可能仍经 $f_p$ 路径受到梯度，而下一次 forward 的 $f_s$ 会随参数变化。

### 7.3 GCN 内图构造的具体停止梯度范围

实现时确保：

```python
# 仅示意关键梯度语义；非可直接替换原工程的完整代码
q_s = shared_etgm(f_s.detach())  # shared router/expert 可训练
q_p = specific_etgm(f_p)         # specific 路径保留梯度
p_s = gather(patch_11, idx_s).detach()
p_p = gather(patch_11, idx_p).detach()
node_s = concat_per_query(q_s, p_s)  # node_s 不可整体 detach
node_p = concat_per_query(q_p, p_p)
adj_s = adjacency(node_s).detach()
adj_p = adjacency(node_p).detach()
q_s_upd = ln_s(gcn_s(node_s, adj_s)) + q_s
q_p_upd = ln_p(gcn_p(node_p, adj_p)) + q_p
```

训练中可在 one-batch 上用 `autograd.grad` 或单独 backward 验证：$L_{Local/Fusion}$ 对选中 Expert、Router、两个 GCN、`out_token` 非零；对 `f_s` 的**上述截断路径**确实为 0，但不要在共享计算图中把 UAD 主损失的梯度也误断开。

---

## 8. 训练时序、初始化、数据接口与 DDP

### 8.1 初始化路线：只选一种，不能混用比较

**推荐作为公平主实验的默认路线：**使用与当前已复现 UAD 基线**完全相同的预训练 ViT 权重、训练批次、优化器、学习率调度、训练时长和原损失配置**，在网络构建时加入新 Token/Experts/GCN 并共同训练。这样不涉及“已训练 UAD checkpoint 中未保存的 GPD bank”如何接续，更便于与原 UAD 基线单变量比较。

如因计算预算选择**从已训练 UAD Checkpoint 热启动**，必须满足额外要求：

1. 原 CLS、原 12 层 ViT、Patch Embedding、位置编码、BNNeck、身份分类器（身份集合一致时）按匹配键完整恢复，不能调用会无条件跳过已有身份分类器的加载器。
2. 新 Specific Token、位置编码新位、三池 Router/专家、两套 GCN、Local Head 和 `out_token` 初始化；新增参数须进入优化器。
3. UAD GPD prototype bank 若没有存入模型 Checkpoint，**并不会因为恢复模型权重就自动恢复**；必须明确恢复保存的 bank 或按照一致规则重建，不能假称完全无缝续训。
4. 必须设置同样热启动权重、相同额外训练预算的 UAD-only 续训对照；不能把多训练的 epoch 带来的效果全算在 ViSA 上。
5. 优化器/scheduler 若重置，在对照实验中也要相同；禁止不同实验混合不同 warm start 策略。

不建议第一版先引入 ViSA 原始优化器、Uni-Prompt 两阶段训练或新文本侧训练阶段。**具体 SGD/Adam、LR、batch size、epoch 以本地有效运行配置为准，不由参考论文覆盖。**

### 8.2 数据与模式索引 [UAD-原有 / 本方案-自适应]

UAD 官方训练 `model(imgs,target,camids)` 接收 RGB/NIR/TIR 三组图像，内部按次序拼接，身份 `target.repeat(3)`。本方案需同步生成：

```text
m_input = 0 for RGB batch
m_input = 1 for NIR batch
m_input = 2 for TIR batch
m_expert = RGB if m_input==0 else IR
```

- 不可将 UAD 测试函数使用的 `mode=0`（训练路径标识）直接当成新模态 0；`mode=1/2/3` 若用于检索分组，也必须经显式映射转换为 0/1/2。
- ProCA/GPD 仍使用**严格 RGB、NIR、TIR 的块顺序**及 `target.repeat(3)`，Specific WRT 按同样的块切分；不能按三个 Token 的内部参数存储顺序推断标签。
- 若 Query/Gallery 模态元数据缺失，必须从既有评测数据记录准确获得，而不能在评测时靠 RGB 像素外观猜模态。
- 相机过滤规则、跨平台试验和九模态组合评测全部沿用原 UAD；**平台只作为评测维度，不参与本轮 Token 选择**。

### 8.3 训练步骤的正确顺序

```text
for each batch:
    1. 准备三模态 imgs、target、camids；显式生成每图模态 id
    2. 一次前向得到 f_s, f_p, f_l, z、训练 CE logits 和 balance
    3. 用 f_s + 原 target 排列计算原 UAD CE/WRT/ProCA/GPD
    4. 分别对 RGB/NIR/TIR 的 f_p 子批次计算 Specific WRT、求平均
    5. 对 f_l 的独立 BNNeck/分类头计算 Local CE
    6. 根据 stop_gradient(u_s) 与正常 u_l 构成 z_f，计算 Fusion WRT
    7. 组合总损失、AMP backward、沿用原梯度裁剪、optimizer.step()
    8. 按原 UAD 时序在 no_grad 下更新 GPD prototype bank
```

- 对外训练接口建议返回**结构化的特征、Logits、损失所需值**，但原始 ProCA/GPD 读取的 `global_feat` 必须仍明确是 768D $f_s$；不得把 1536D $z$ 误传入它们。
- 为避免 DDP 输出未被损失使用导致错误的 unused-parameter 判定，训练返回字段尽量只包含真实被 loss 使用的图张量；诊断特征输出应通过明确的可选接口而非默认大量返回。
- AMP 的图相似度与 WRT 距离建议至少在关键步骤转 FP32；保持原 UAD AMP、`GradScaler` 和全局梯度裁剪。

### 8.4 DDP 与稀疏专家

- 沿用 UAD 的 DDP 封装，动态 Expert 分支有正常的未选中参数；通常保留 `find_unused_parameters=True`，同时确保**所有实际激活参数的计算图能到 loss**。
- 每个 rank 的 RGB/IR 子批次不一定均匀分配，代码应处理某池没有样本的情形；没有样本的池不执行不必要的 Router/Expert，返回合适的零形状输出。
- 不使用 `.item()`、`.detach()` 参与 Expert 值加权，更不能为了“解决 DDP unused 参数”每轮执行所有专家再只取 Top-2；那会破坏预期的条件计算特性。
- 如果动态布尔索引赋值导致梯度异常或 DDP 重入问题，应改为等价的可微散射/分组计算；**算法仍保持池内 Top-2**。
- 检查真正选中的专家收到梯度、未选中专家允许零梯度；避免误将稀疏特性当作训练失败。

---

## 9. 测试与检索：明确单一前向接口

### 9.1 推理流程

每张 Query/Gallery 图像**与训练相同地**选择对应模态 Token、进入 Shared 与相应 RGB/IR 专家池、执行 DLFM，获得：

$$f_s(x)\in\mathbb R^{768},\qquad f_l(x)\in\mathbb R^{768}.$$

仅输出：

$$\boxed{z(x)=\frac{[\operatorname{Norm}(f_s(x)),\,0.5\operatorname{Norm}(f_l(x))]}{\sqrt{1.25}}\in\mathbb R^{1536}}.$$

- 查询和图库统一用 $d(q,g)=\|z_q-z_g\|_2$，不根据是否同模态改变距离、权重或特征维数。
- 不直接将原始 $f_p$ 拼接进统一检索，因为其直接辅助目标只限定模态内身份距离。
- `NECK_FEAT=before` 的全局分支语义保留：$f_s,f_l$ 均取 BNNeck 前；**推理代码需要显式返回 $z$ 而非旧 `model(..., mode!=0)` 返回的 768D CLS**。
- 原评测器如果再次对完整 $z$ 做 L2，一般不改变方向；为可控起见，在代码里统一调用训练/推理共享的融合函数，最终单位化与 `eps` 处理一致。
- 分类器、BNNeck 输出只用于训练 CE；`out_token` 参与推理但原始 Specific Token 只是条件特征来源，并非检索向量的一段。
- 训练特征输出、测试特征输出、调试导出输出三个接口要明确命名，杜绝“训练走 Fusion，测试退回 CLS”的隐性错误。

### 9.2 只需导出三个诊断特征

为了减少无休止诊断，训练完成后同一 Checkpoint 导出：

1. $u_s$：CLS 单独检索，衡量主干是否保持 UAD 身份能力；
2. $u_l$：Local 单独检索，衡量新增表征是否有身份信息；
3. $z$：**正式主结果**，衡量 Local 是否对同一协议的统一距离提供有效补充。

报告 WHU 既有总体 mAP/Rank-1，以及九模态组合、平台组合，尤其 TIR→TIR 与跨 TIR；**实验结果出来前不对收益作先验判断**。

---

## 10. 代码改动位置建议（抽象指导而非逐行补丁）

以下为官方 UAD 仓库内的参考路径；本地项目存在差异时以实际文件结构为准。

| 参考文件/组件 | 建议修改职责 | 不应该改变的部分 |
|---|---|---|
| `CVPR26_UAD/model/backbones/vit_pytorch.py` | 新增三个模态 Specific 参数与单独位置编码；扩展输入 Token；暴露 11 层 Patch、12 层 CLS/Specific；显式加载旧位置编码 | 原 ViT 权重、Patch Embed、12 层结构、现有 SIE 约定 |
| 新增 `model/visa_etgm.py` 或等价模块 | Shared/RGB/IR 三池、8 experts/池、Top-2、16 Query、全局条件注入、可微专家结果聚合 | 不要假装单 Key attention 在读 Patch |
| 新增 `model/visa_dlfm.py` 或等价模块 | 双 Top-40、16 张 41 节点子图、邻接计算、独立 GCN、`out_token` 汇总 | 不改 UAD Backbone 训练目标；不输出 41 节点到 16 Query 残差 |
| `CVPR26_UAD/model/make_model.py` | 组合 Backbone、ETGM、DLFM；维护原共享分类器，增加 Local 分类器；定义训练/评测返回接口；统一 $z$ 构造 | CLS 的原 BNNeck/分类语义、ProCA/GPD 读取 768D global_feat |
| `CVPR26_UAD/processor/processor.py` | 在原损失之后加入 Specific WRT、Local CE、Fusion WRT、balance 和 5-epoch warm-up；保留 bank 更新时序 | 原 PKM、原损失配置、AMP、grad clip、DDP、原型更新 |
| `CVPR26_UAD/loss/wrt_loss.py` | **优先不修改**，直接复用 `TripletLoss_WRT`；保证 Specific 不提前 L2、Fusion 输入已单位化 | 原距离函数/softmargin/正负掩码，除非已发现可复现错误 |
| `CVPR26_UAD/utils/metrics.py` / 评测入口 | 接受模型返回的 $z$；同一 Eval 计算 1536D 欧氏距离；可选导出 $u_s,u_l$ | Query/Gallery、相机过滤、九模态/平台协议、无 Re-ranking |
| 实验配置 | 增加 `USE_VISA_ADAPT`、`NUM_SPECIFIC=3`、`EXPERTS=8`、`TOP_K=2`、`PROMPT_LEN=16`、`GRAPH_K=40`、新增 loss 系数、融合 beta、ramp | Baseline 原学习率与有效 batch 配置、原 CE/WRT/ProCA/GPD 系数 |

**模块接口建议**：

```text
Backbone.forward(images, modality_ids, original_cam_ids?)
  -> {f_s:[B,768], f_p:[B,768], patch11:[B,128,768]}

ETGM.forward(f_s.detach(), f_p, modality_ids)
  -> {q_shared:[B,16,768], q_specific:[B,16,768], balance:scalar}

DLFM.forward(f_s, f_p, patch11, q_shared, q_specific)
  -> {f_local:[B,768]}

Model.forward_train(...)
  -> {cls_logits_s, f_s, f_p, cls_logits_local, f_local, z, balance}

Model.forward_test(images, modality_ids, ...)
  -> z:[B,1536]
```

**重要：** `forward_train` 中的 `z` 是未 detach 的相同前向值，真正计算 Fusion Loss 的 `z_f` 应在损失处对全局段视图执行 stop-gradient 再调用同一融合函数。不能在模型 forward 中全局地 detach $f_s$，否则会破坏原 UAD 主损失。若为降低多处实现差异，建议写一个唯一的 `make_retrieval_feature(f_s,f_l,detach_global_for_loss=False)`；该布尔选项只控制梯度，不改变结果数值。

---

## 11. 论文/源码差异与本方案最终裁决表

| 争议位置 | 论文叙述 / 公开代码事实 | 本方案裁决 | 裁决理由 |
|---|---|---|---|
| ViSA 层级减法 | ViSA VAB 使用逐层 CLS−View；其上层代码又进行最终 CLS−View | **都不采用** | 降低对 UAD CLS 的强干预，不预设模态特有内容都是噪声 |
| MDReID 双 Token 职责 | 官方原 CLS 与新增 Token 的具体 shared/specific 分工不同 | **保留 UAD CLS 为 shared，新增 Specific 为条件身份** | UAD 已有成熟 CLS CE/WRT/ProCA/GPD，保留其语义接口 |
| MDReID ROL/KDL | 论文与部分源码实现细节不完全一致；依赖三模态组合监督 | **不直接搬运** | WHU 非同步观测、ProCA/GPD 已有对齐约束，避免复杂竞争 |
| Uni-Prompt 附加提示 | 模态/平台/身份文本上下文，分阶段图文对比 | **只借鉴条件职责分离，不加平台或文本 Prompt** | 本轮仅研究模态 Gap，不在视觉端复制多文本 Token |
| ViSA ETGM Cross-Attn | 代码单 Key 全局输入，没有空间注意力选择 | **显式全局条件注入** | 保留查询生成实质、消除无用 Q/K 投影，不额外注入 Patch |
| ViSA GCN 输出 | 源码 `output[:,:,0,:]`，每 Query 子图输出首节点 | **严格保持 16 张 41 节点子图** | 解决 56/16 节点数误解与残差形状 |
| ViSA 两 GCN 调用 | 源码创建 `view_gcn` 却在两支调用 `share_gcn` | **独立 `gcn_s/gcn_p`** | 明确两类局部消息传播参数职责 |
| ViSA GCN `no_grad` | 源码连 Query 图节点值都停止梯度，仍有残差支路 | **只停止 Patch 节点和邻接，保留 Query 节点梯度** | 让局部图损失经 GCN 更新 Expert，同时限制直接 Patch 梯度 |
| ViSA Local 聚合 | 论文公式给 CLS；源码使用可学习 `out_token` | **采用独立 `out_token`** | CLS 已入最终检索，Local 应独立聚合身份补充 |
| ViSA 最终检索 | 源码拼接相减全局分支与 Local，Head eval 可能返回 BN 后 | **统一使用 pre-BN CLS 与 Local 的固定归一化融合** | 和 UAD `NECK_FEAT=before` 及 Fusion WRT 保持一致 |
| UAD WRT 归一化 | `normalize_feature=False` 默认，欧氏距离 | Specific **不额外 L2**；Fusion **本来就是单位向量** | 区分辅助监督尺度与最终实际检索尺度 |

---

## 12. 实验对照、轻量验收与禁止的假结论

### 12.1 最少实验结构

- **B0：原 UAD 基线**；使用已有已复现设置与指标。
- **T0：UAD + 输入级三模态 Specific Token，不加 ETGM/DLFM**；仅保留 UAD 主损失。这一对照隔离“多一个参与 Self-Attention 的 Token”对 CLS 的影响。若想隔离 Specific 辅助损失，可另做 T0+Specific WRT，但**不是首轮必需**。
- **V1.1：本方案完整结构**；与 T0 使用相同主干初始化、训练时长、采样及原损失系数。新增 Specific WRT、Local CE、Fusion WRT、Balance。

注意：V1.1 相比 T0 同时增加专家、局部模块和辅助损失，因此**只能检验整条新信息通路是否成立**，不能声称隔离证明了某一个模块单独贡献；需要发表严格消融时再据结果挑选最有价值的一项补充。

### 12.2 开训前一次性硬性验收

1. **尺寸**：输入 `[B,130,768]`、Patch11 `[B,128,768]`、$f_s/f_p$ `[B,768]`、两类 Query `[B,16,768]`、每路图 `[B,16,41,768]`、GCN `[B,16,768]`、Local `[B,768]`、Fusion `[B,1536]`。
2. **预训练加载**：原 129 位的位置编码严格拷贝为 CLS+128 Patch，新增位单独创建；原 12 层权重被完整正确恢复。
3. **标签**：训练 concat 顺序 RGB/NIR/TIR 与 `target.repeat(3)`、Specific Token 选择、Expert RGB/IR 映射完全一致；推理每张图有正确模态 id。
4. **GCN 功能**：独立两套 GCN 都真正被调用且获得梯度；图第 0 节点返回形状与 Query 残差一致；邻接有限、无 NaN/Inf；设备不硬编码 `cuda:0`。
5. **ETGM 条件注入**：选中 Top-2 Expert 与 Router 的相应参数真实得到梯度；16 Query 不被意外压成 1 Query；无样本专家允许无梯度。
6. **Stop-Gradient**：Shared ETGM 入口 `sg(f_s)` 和 Fusion Loss 的直接 $u_s$ 支路生效；UAD CE/WRT/ProCA/GPD 对 $f_s$ 的梯度必须仍存在。
7. **Specific WRT**：每模态有有效正负样本，直接使用原始尺度 $f_p$；不在新分支悄悄设置 `normalize_feature=True`。
8. **训练/推理相等**：在关闭 Dropout 等随机性、同一模型 eval 状态下，用同一个 `(f_s,f_l)` 调用测试融合和训练损失融合（仅 detach 不同），应数值相等；$\|z\|_2\approx1$。
9. **原始主损失**：ProCA 768D、原型库 768D、原 target 排列、GPD 更新时序、Baseline 真实权重完全不变。
10. **梯度强度**：抽样 1–2 个 Batch，检查新增损失对 ViT 后层的梯度不存在数量级异常；若异常，先减小 $\lambda_f$ 或核查实现错误，不自动追加 PCGrad/更多模块。

### 12.3 判读边界

- 有两 Token **不等于**已证明统计解耦；只有通过实际训练约束赋予了功能分工。
- Token 分组不同 **不等于** Specific 一定有互补身份信息；Shared/Specific 的 Top-K 区域可能高度重叠。
- Local CE 下降 **不等于**混合 Gallery mAP 一定提高；看 $f_s$、$f_l$、$z$ 的检索表现。
- 固定 $\beta=0.5$ **不等于**局部一定提供正收益；它只是确定的距离权重与推理协议。
- Stop-Gradient **不等于**共享 ViT 被完全保护；Specific 梯度和多 Token 自注意力仍可影响未来的 CLS。

---

## 13. 最终冻结清单（V1.1）

| 项目 | 当前决定 |
|---|---|
| Backbone | UAD ViT-B/16，256×128，Patch/Stride=16，12 层，128 Patch |
| 输入特殊 Token | CLS + RGB/NIR/TIR 三选一 Specific；无平台条件；共 130 Token |
| Token 初始化 | 三组相同随机数值初始化、独立参数；Specific 位置编码置零；旧位置编码精准保留 |
| 中间 CLS−Specific | 关闭，包含最终外层相减 |
| 全局特征 | 第 12 层 LN 后、BNNeck 前 CLS，768D，UAD 原 CE/WRT/ProCA/GPD |
| 特有特征 | 第 12 层 LN 后 Specific，768D；模态内原尺度 WRT ×0.05 |
| Shared ETGM 输入 | **`sg(f_s)`**；Shared Router/Expert 权重仍可训练 |
| Specific ETGM 输入 | 原始 `f_p`，不断梯度；RGB/IR 池按已知模态选择 |
| ETGM 专家 | 3 池×8 Experts，池内 Top-2，每专家 16 查询，**全局条件注入 + SelfAttn + FFN** |
| 局部候选 | Block11 128 Patch；共享与特有各 Top-40 |
| 图形状 | 每路 16 张 41 节点图：`[B,16,41,768]` |
| GCN | 共享/特有独立权重；每图仅 Query 节点输出 `[B,16,768]`；非负邻接＋自环＋对称度归一化 |
| 图梯度 | Expert Query 可微、Patch 节点值 detach、邻接 detach、Top-K 索引不反传 |
| Local | 共享 `out_token`＋32 专家查询，OutAttenBlock 首位输出，768D，独立 Local CE×0.1 |
| Fusion | $z=[\operatorname{Norm}(f_s),0.5\operatorname{Norm}(f_l)]/\sqrt{1.25}$，1536D |
| Fusion 训练 | WRT×0.5；**仅直接 CLS 拼接段 detach**，Local 路径正常训练 |
| Balance | ViSA 风格 Router 重要度均衡 ×1e-4 |
| 新损失暖启 | 5 epoch 线性加权，原 UAD 主损失恒定 |
| 检索 | 唯一正式输出 1536D $z$，各模态统一欧氏距离，无 modality-pair 特殊打分 |
| 初始化/对照 | 优先同 UAD baseline 初始化和完整训练预算；Token-only 对照；热启动须恢复/处理 GPD bank 并匹配续训基线 |

**最终交付目标**：实现人员应能够从同一个前向过程中取出 `f_s, f_p, f_l, z`，明确追踪每种损失的监督对象和梯度路径；在不改变 UAD 原 ProCA/GPD 和评测协议的前提下训练完整模型，并在推理时对所有模态只返回相同定义的 1536D 检索向量。

## 14. 补充说明

- ETGM / OutAttenBlock： 除了已经确定要替换的单 Key Cross-Attention，其余注意力层、FFN、LayerNorm 和输出聚合结构尽量沿用 ViSA 的实际实现，明确冻结所有超参数。
- 学习率： 保留 UAD 原有 Backbone 的优化策略，对新增参数设置统一、固定的 2 倍基础学习率，T0 和完整 V1.1 采用相同的 Specific Token 学习率设置。

## 一、ETGM 和 OutAttenBlock 应如何确定？

### 1. ETGM：每个专家独立完整的查询生成模块

条件投影 \\(W_c\\) 建议每个专家独立一套，不在专家池内部共享。

原因来自 ViSA 原始设计。官方的每个 `MoEBlock` 都独立实例化一个 `PromptRecapBlock`，其 Cross-Attention、Self-Attention、FFN 和可学习 Query 原本就属于该专家。

我们只是将其中退化的单 Key Cross-Attention 替换为等效功能的线性条件注入，没有必要额外改变专家之间的参数共享关系。

对于第 \\(e\\) 个专家：

\\[ \begin{aligned} H_0^e &=\mathrm{LN}\_1^e(T_e+W_c^e f+b_c^e)\\\ H_1^e &=\mathrm{LN}\_2^e(H_0^e+\mathrm{SA}\_e(H_0^e))\\\ Q_e &=\mathrm{LN}\_3^e(H_1^e+\mathrm{FFN}\_e(H_1^e)) \end{aligned} \\]

各个专家的 \\(T_e,W_c^e,\mathrm{SA}\_e,\mathrm{FFN}\_e\\) 均独立。

注意：这里的 \\(T_e\\) 是专家内部的 16 个 Query，不是 ViT 输入端的 RGB/NIR/TIR Specific Token。

### 2. ETGM 固定参数

| 参数                      | 最终建议                   | 来源                                  |
| ----------------------- | ---------------------- | ----------------------------------- |
| 专家池数量                   | 3：Shared / RGB / IR    | 本方案将 ViSA 视角池改为模态池                  |
| 每池专家数量                  | 8                      | ViSA 原配置                            |
| 池内路由                    | Top-2                  | ViSA 原配置                            |
| 每个专家的 Query             | 16 × 768               | ViSA 原配置                            |
| 条件投影 \\(W_c\\)          | Linear(768,768)，带 bias | 本方案替代单 Key Cross-Attention          |
| 条件投影共享方式                | 每个专家独立                 | 保留 ViSA 专家参数独立原则                    |
| Self-Attention          | 1 层、12 Heads、每头 64D    | ViSA 原代码                            |
| Attention QKV Bias      | False                  | ViSA `Attention` 默认值                |
| FFN                     | 768 → 1536 → 768       | ViSA `PromptRecapBlock`，MLP Ratio=2 |
| FFN 激活                  | GELU                   | ViSA 实际 `Mlp` 实现                    |
| Attention / FFN Dropout | 0                      | ViSA 原代码默认值                         |
| LayerNorm               | 3 个，Post-Norm，eps=1e-5 | ViSA 实际调用结构                         |
| 专家输出                    | 16 × 768               | 保持不变                                |

这里有两个特别值得在指南中明确的源码细节：

第一，ViSA 配置里的 `PROMPT_DEPTH=2` 不意味着每个专家有两层 `PromptRecapBlock`。 当前官方代码中，每个 `MoEBlock` 只实例化一个 `PromptRecapBlock`。本方案也固定为一个，不要由开发者根据配置名称自行堆叠两层。

第二，`PromptRecapBlock` 的构造函数虽然有 `activation=nn.ReLU` 参数，但实际创建 `Mlp` 时没有传入该参数，使用的是 `Mlp` 默认 GELU。 因此不能误实现成 ReLU。

### 3. OutAttenBlock：完整沿用实际代码的单层结构

我建议不再改造其注意力结构，仅将输入由 ViSA 原来的视角专家查询替换为我们的共享/模态专家查询。

输入：

\\[ H\_{\mathrm{in}} = [T\_{\mathrm{out}},\bar Q_s,\bar Q_p] \in\mathbb R^{B\times33\times768} \\]

计算：

\\[ \begin{aligned} H_1&=\mathrm{LN}\_1(H\_{\mathrm{in}}+\mathrm{SA}(H\_{\mathrm{in}}))\\\ H_2&=\mathrm{LN}\_2(H_1+\mathrm{FFN}(H_1))\\\ f_l&=H_2[:,0,:] \end{aligned} \\]

固定参数如下：

| 参数                      | 最终建议                        |
| ----------------------- | --------------------------- |
| Block 数量                | 1 层                         |
| 输入 Token 数              | 33                          |
| 隐藏维度                    | 768                         |
| Self-Attention Heads    | 12                          |
| QKV Bias                | False                       |
| FFN                     | 768 → 3072 → 768            |
| FFN 扩展倍数                | 4                           |
| 激活函数                    | GELU                        |
| Attention / FFN Dropout | 0                           |
| LayerNorm               | 2 个，Post-Norm，eps=1e-5      |
| 输出                      | 第 0 位 `out_token` 更新结果，768D |

不使用额外 Cross-Attention，不堆叠多个 OutAttenBlock，也不在末端增加新的投影层。

参考源码：[ViSA Vision Transformer / ETGM / OutAttenBlock](https://github.com/Cat-Zero/ViSA/blob/main/fastreid/modeling/backbones/vision_transformer_multiview_ViSA.py?utm_source=chatgpt.com)。

## 二、新增参数的学习率如何设置？

### 1. 先核实 UAD 原始优化器

UAD 官方 `solver/make_optimizer.py` 并没有自动区分预训练 Backbone 和新增模块。

它主要根据参数名称设置：

- 普通参数：`BASE_LR`
- Bias 参数：`BASE_LR × BIAS_LR_FACTOR`
- 可选分类器倍率：取决于 `LARGE_FC_LR`

官方配置使用 SGD、`BASE_LR=0.008`、`BIAS_LR_FACTOR=2`，但这不等于你当前复现实验一定使用这些数值。本方案必须以你实际运行的 UAD 配置为准。

参考：[UAD make_optimizer.py](https://github.com/msm8976/WHU-MARS/blob/main/CVPR26_UAD/solver/make_optimizer.py?utm_source=chatgpt.com)。

### 2. 推荐固定两类学习率

我建议统一采用：

\\[ \boxed{ LR\_{\mathrm{old}}=LR\_{\mathrm{UAD}} } \\]

\\[ \boxed{ LR\_{\mathrm{new}}=2LR\_{\mathrm{UAD}} } \\]

原 UAD 参数的学习率规则保持不变；新增模块在对应原规则的基础上统一乘以 2，包括原有的 Bias LR 规则。

| 参数类型                                         | 学习率策略        |
| -------------------------------------------- | ------------ |
| 原 CLS、Patch Embedding、12 层 ViT、原 LayerNorm   | 原 UAD 学习率    |
| 原 BNNeck、身份分类器                               | 完全保留原 UAD 规则 |
| RGB/NIR/TIR Specific Token                   | 新参数，2×       |
| 新增 Specific 位置编码                             | 新参数，2×       |
| Shared/RGB/IR Router                         | 新参数，2×       |
| 24 个专家，包括 Query、\\(W_c\\)、Self-Attention、FFN | 新参数，2×       |
| 两套 GCN、图后 LayerNorm                          | 新参数，2×       |
| 独立 `out_token`、OutAttenBlock                 | 新参数，2×       |
| Local BNNeck 和身份分类器                          | 新参数，2×       |

这里的 2 倍属于保守的初始工程选择，不是已经验证的最优倍率。

例如，若实际 UAD 主参数学习率为 \\(5\times10^{-6}\\)，新增参数就使用 \\(10^{-5}\\)；若使用官方 SGD 配置的 0.008，则新增普通参数为 0.016。这两种不同优化器与配置不能混在一次实验里使用。

### 3. 发现一个需要同步修正的细节：新增位置编码最好单独定义

这点与学习率分组直接相关，也是原指南值得补充的地方。

如果把原来的：

\\[ E\_{\mathrm{old}}\in\mathbb R^{1\times129\times768} \\]

直接扩展成一个完整的：

\\[ E\_{\mathrm{new}}\in\mathbb R^{1\times130\times768} \\]

那么它成为一个整体 `nn.Parameter` 后，不能仅通过普通优化器参数组给第 2 个位置使用 2 倍学习率，而让其他 129 个位置保持原学习率。

因此，我建议改为两个独立参数：

\\[ E\_{\mathrm{old}}\in\mathbb R^{1\times129\times768} \\]

\\[ E\_{\mathrm{SP}}\in\mathbb R^{1\times1\times768} \\]

前向时构造：

\\[ E\_{\mathrm{input}} = [E\_{\mathrm{CLS}},E\_{\mathrm{SP}},E\_{P_1},\ldots,E\_{P\_{128}}] \\]

示意：

```
pos_embed_input = torch.cat(    [        self.pos_embed[:, :1],        self.specific_pos_embed,        self.pos_embed[:, 1:]    ],    dim=1)
```

这样有三个好处：原来的 129 个位置编码能够原样加载、新位置编码可以独立初始化与设置 2 倍学习率，而且不需要为原位置编码设计额外的梯度缩放逻辑。

这项修正值得直接纳入 V1.1 指南。

### 4. T0 和完整 V1.1 必须采用相同规则

假设两个实验为：

- T0：UAD + 三模态 Specific Token。
- V1.1：T0 + ETGM + DLFM + Local/Fusion。

那么 T0 与 V1.1 共有的新增 Token、Specific 位置编码，都应使用相同的参数初始化、学习率倍率、优化器和训练预算。

不能出现 T0 的 Token 用 1 倍学习率，而 V1.1 的 Token 用 2 倍学习率，否则无法清楚区分新增专家系统的作用。

此外，应当在创建所有新模块后再构建优化器，避免漏掉新参数。学习率调度也要保留各参数组的固定倍率，不能在每个 epoch 把所有组重新覆盖为同一个值。

### 5. 不需要额外增加复杂训练策略

第一版继续保留：

- UAD 原优化器、调度器与原有学习率 Warmup。
- 原 UAD 主损失及其权重不变。
- Specific、Local、Fusion 等新增损失前 5 epoch 渐进启用。
- 新增参数学习率固定为相应 UAD 规则的 2 倍。

损失权重 Warmup 和学习率 Warmup 是两个不同机制，不需要因为损失已经渐进启用，就再单独设计新模块的第二套学习率调度。


## 四、最终敲定

这两项问题可以通过固定设计规范解决，无须引入新的监督损失或训练阶段。

| 项目                | 敲定选择                         |
| ----------------- | ---------------------------- |
| ETGM 条件投影         | 每个专家独立 `Linear(768,768)`     |
| 专家 Self-Attention | 单层，12 Heads                  |
| 专家 FFN            | 768→1536→768，GELU            |
| 专家归一化             | 3 个 Post-LayerNorm           |
| OutAttenBlock     | 单层，12 Heads，FFN 768→3072→768 |
| Dropout           | 均为 0                         |
| 新参数学习率            | 统一为原 UAD 对应规则的 2 倍           |
| 旧参数学习率            | 完全保留当前 UAD Baseline          |
| 新位置编码             | 单独定义为可学习参数                   |
| T0 / V1.1         | 共有参数采用相同 LR 和初始化             |
| 学习率调度             | 沿用 UAD，不额外增加新模块调度            |

我的建议是将以上内容作为 V1.1 实施指南的强制补充。 还需要实验判断的是这些参数配置的效果，而不是把结构与学习率的选择留给开发者自行决定。