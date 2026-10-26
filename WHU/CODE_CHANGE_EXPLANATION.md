# UAD + 模态条件 ETGM/DLFM：代码改动与方案核对说明

本文对应 `../idea.md` 的 V1.1 首轮实验。以下内容描述**实际代码**，不把未经数据训练验证的性能推断写成结论。原始 `configs/UAD.yml` 为 B0；`configs/UAD_T0.yml` 为 Token-only 对照；`configs/UAD_ViSA.yml` 为完整模型。三份配置沿用同一 UAD 优化器、采样、训练轮数及原损失系数，分别写入不同日志目录。

## 1. 改动文件和职责

| 文件 | 新增或修改的实际功能 | 目的 |
| --- | --- | --- |
| `model/backbones/vit_pytorch.py` | 可选三模态 Specific Token、独立 Specific 位置编码、`forward_specific`；返回第 11 层 Patch 与第 12 层 LN 后的 CLS/Specific | 在原 ViT 上建立模态条件输入，并保持旧位置编码和旧权重键 |
| `model/visa_adapt.py` | `QueryExpert`、`ExpertPool`、`ETGM`、`QueryGCN`、`DLFM`、唯一融合函数 | 实现三专家池、双局部图和固定检索向量 |
| `model/make_model.py` | `off/token_only/full` 三模式；完整模式的 Local BNNeck/分类器；训练/测试分支和诊断输出；完整适配权重严格加载 | 对接 UAD 原主损失，同时保证测试返回融合向量 |
| `processor/processor.py` | Specific WRT、Local CE、Fusion WRT、均衡损失、五轮暖启；测试传入真实模态 ID；多 GPU 推理的模态 ID 改为可拆分张量 | 维持 ProCA/GPD 时序，避免评测时选错 Token/专家池 |
| `config/defaults.py` | 全部新超参数，默认 `MODE=off` | 保持旧 B0 配置可直接运行 |
| `solver/make_optimizer.py` | 给新参数组打 `visa_new` 标记并设置对应原规则的 2 倍初始 LR | 区分旧参数和新参数 |
| `solver/scheduler_factory.py`、`solver/cosine_lr.py` | 暖启和余弦最低 LR 支持逐组值；旧组与 B0 完全同值，新组为对应旧组 2 倍 | 防止调度器覆盖新参数倍率，同时不改变旧 Bias/分类器暖启 |
| `configs/UAD_T0.yml`、`configs/UAD_ViSA.yml` | B0 配置的匹配副本，分别启用 `token_only` 和 `full`、分开输出目录；V1.1 验证 batch 调为 32 | 提供首轮对照与主实验入口，限制完整模型验证显存 |
| `tests/test_visa_adapt.py` | 无数据 CPU 契约检查 | 核对加载、形状、梯度、融合、优化器和调度器 |
| `VISA_ADAPT_IMPLEMENTATION.md` | 简要运行和结构说明 | 快速查看入口 |
| `CODE_CHANGE_EXPLANATION.md` | 本详细核对说明 | 审阅全部实现决定和验证边界 |

没有删除源文件；没有改 `loss/wrt_loss.py`、数据集/采样器、`utils/metrics.py`、`train.py`、`test.py` 和原 `configs/UAD.yml`。

## 2. 三种实验模式与输入输出

| 模式 | ViT 输入 | 模型训练输出 | 测试正式特征 | 损失 |
| --- | --- | --- | --- | --- |
| `off` / B0 | 原 CLS + 128 Patch | 原 `(cls_score, global_feat, feat)` | UAD 原 768D 特征 | 原 CE/WRT/ProCA/GPD |
| `token_only` / T0 | CLS + 当前模态 Specific + 128 Patch | 与 B0 相同接口，`global_feat=f_s` | 768D `f_s` | 仅原损失 |
| `full` / V1.1 | 同 T0 | `(CLS logits, f_s, f_p, Local logits, f_l, balance)` | 1536D `z` | 原损失 + 四项新损失 |

训练 DataLoader 的三组图像依次是 `RGB / IR(NIR) / Thermal(TIR)`，模型拼接后生成 0/1/2 模态 ID；对应身份为 `target.repeat(3)`。评估样本元数据存的是 1/2/3，模型显式减一；评估的 `mode=1/2/3` 仅表示遍历哪个 Query/Gallery 组，不用来猜测图像模态。若 `SIE_CAMERA` 打开，适配分支使用真实相机 ID；若 `SIE_VIEW` 打开，使用数据记录第四字段所对应的模态索引。默认两项均关闭，未增加平台条件。

完整模式的同一次前向取得 `f_s, f_p, f_l`。训练处理器从 `f_s,f_l` 调用和评估完全相同的 `make_retrieval_feature` 生成 Fusion WRT 输入；测试模型直接调用该函数返回 `z`。评估调用 `return_components=True` 可得到 `shared/local/fusion` 三个单位特征；正式评估只用 `fusion`。这三个分支尚未作为独立文件自动保存。

## 3. Backbone：旧参数与两个特殊 Token

当前 UAD 配置为 ViT-B/16，输入 256×128，Patch/Stride 16，生成 16×8=128 个 Patch，隐藏维 768，12 个 Block。输入序列为 `[CLS, T_m, P_1...P_128]`，总长 130；一个样本只选一个 `T_m`。三份 `T_RGB/T_NIR/T_TIR` 是独立 `Parameter`，但从同一个截断正态随机张量克隆，因此初值相同且训练后可分化。

原 `pos_embed` 仍是 `[1,129,768]`，由原 CLS 位置和 128 Patch 位置构成；新增 `specific_pos_embed` 是独立 `[1,1,768]`、零初始化。前向拼接为 `[原 CLS 位置, 新 Specific 位置, 原 Patch 位置]`。ImageNet 预训练加载器因此仍对原 129 位执行原有尺寸插值；没有把 129 位误当成 130 位重采样。第 1–11 个 Block 后缓存 `P11=x[:,2:,:]`；原第 12 个 Block 和原 LN 只执行一次，取 `f_s=x[:,0]`、`f_p=x[:,1]`。没有 ViSA 的逐层或末端 CLS−Specific。

T0 和 V1.1 的 Backbone 构造顺序相同；使用相同随机种子及预训练文件时，共有的 Token 和旧 ViT 起点相同。新增 Token 会参与自注意力，因此可能改变 CLS，T0 的用途正是测量这一影响。

## 4. ETGM 的计算与参数

`f_s.detach()` 进入 Shared 池；原始 `f_p` 按 0→RGB、1/2→IR 进入相应池。三池各 8 个专家、一个独立 768→8 Router，每池 Top-2；绝不跨 24 个专家统一 Top-2。Router 对有效样本算 logits，Top-2 logits 先 softmax，再加权所选专家输出。专家按实际选中样本稀疏执行，输出通过可微 `index_add` 回填原批次。未选专家当轮可无梯度，DDP 沿用原 `find_unused_parameters=True`。

每个专家有独立的 `prompt[1,16,768]`、带 bias 的条件线性层 768→768、三个 eps=1e-5 的 post LayerNorm、一次 12-head Self-Attention（QKV 无 bias，输出投影有 bias）、一次 768→1536→768 GELU FFN；两处 Dropout 均为 0。先算 `LN(prompt + W_c f)`，再各做 Self-Attention 残差与 FFN 残差。没有保留 ViSA 对单个全局 Key 的退化 Cross-Attention，也没有堆叠两层专家块。

Router 均衡取全池 8 个 softmax 概率的平均重要度 `p_bar`，每个有效池为 `8*sum(p_bar²)`；按该池有效样本数加权，再除以 Shared+Specific 两路总有效数。此项鼓励概率分布均匀，不保证实际 Top-2 选择次数均匀。

## 5. DLFM 的计算、形状与梯度

两路分别用 `f_s`、`f_p` 与同一份 `P11[B,128,768]` 的余弦相似度选 Top-40，索引形状 `[B,40]`。排序时各向量 L2 归一化，但图节点使用原尺度 Patch。每一路的 16 个 Query 共用该路的 40 个 Patch 位置；每个 Query 形成 `[Query,40 Patch]` 的 41 节点图。因此节点张量是 `[B,16,41,768]`，邻接是 `[B,16,41,41]`，不是一张 56 节点图。

邻接在 FP32 且无梯度的上下文中构造：节点先 L2，余弦 Gram 负值截零，加单位自环，用 `D^-1/2 A D^-1/2` 归一化。自环所在设备取节点设备，没有 `cuda:0` 硬编码。GCN 在 FP32 用独立的 768×768 权重先变换通道、再邻接聚合、`tanh`，仅取每图第 0 个 Query 节点，得 `[B,16,768]`。两路分别执行 `Q + LN(GCN(Q, stopgrad(P), stopgrad(A)))`。

这里的截断范围很重要：Top-K 索引不可微，选中 Patch 的**节点值**停止梯度，邻接停止梯度；Expert Query 节点**不断梯度**。因此 Local/Fusion 能更新 Router、被选专家、两套 GCN 和局部聚合模块，但不能通过图节点值直接推动 Patch11。Shared ETGM 入口还截断 `f_s` 的这条直接路径；UAD 原损失仍正常训练 `f_s`。Specific 路径仍可通过共享 ViT 参数间接影响未来的 CLS。

局部汇聚输入 `[out_token,16 Shared Query,16 Specific Query]` 共 33 Token。`out_token[1,1,768]` 独立可学习且三模态共享；经过一次 12-head Self-Attention、768→3072→768 GELU FFN 和两个 post LayerNorm，取首位为 `f_l[B,768]`。两路 GCN 真正调用各自权重。

## 6. 损失、系数与原型时序

| 项目 | 输入与实现 | 系数/时序 |
| --- | --- | --- |
| 原 UAD CE | `classifier(BN_s(f_s))`，身份 `target.repeat(3)` | `ID_LOSS_WEIGHT`，默认 1；原接口保持 |
| 原 UAD WRT | 768D、pre-BN、未额外 L2 的 `f_s` | `TRIPLET_LOSS_WEIGHT`，默认 1；原 WRT 实现不改 |
| ProCA | 按 RGB/NIR/TIR 顺序的 `f_s`，每身份每模态 `NUM_INSTANCE=4` | 当前配置 `PCA_LOSS_WEIGHT=0.01` |
| GPD | 768D `f_s` 与身份原型 bank；温度 0.03，bank 动量 0.2 | 当前 `base+pca+gpd` 配置中系数 1；仍在 optimizer step 后更新 bank |
| Specific WRT | `f_p` 按模态分 3 组，各自以原始尺度调用同一 `TripletLoss_WRT`，结果求均值 | `0.05 × r(e)` |
| Local CE | `local_classifier(BN_l(f_l))`，一个全模态共享分类器 | `0.1 × r(e)` |
| Fusion WRT | `z_f=[stopgrad(norm(f_s)),0.5 norm(f_l)]/sqrt(1.25)` | `0.5 × r(e)`；仅 CLS 直接拼接段停梯度 |
| Router balance | Shared/RGB/IR 有效池的概率重要度均衡 | `1e-4 × r(e)` |

`r(e)=min(1,e/5)`，epoch 从 1 开始；第 1–5 轮为 0.2/0.4/0.6/0.8/1，之后为 1。原 UAD 损失不乘这个系数。原 AMP、GradScaler、全局梯度裁剪上限 1.0、ProCA/GPD 计算和 GPD bank 的 DDP 汇总更新顺序未变。Specific WRT 要求每 rank 至少两个 PID，且每 PID 每模态至少两个实例；代码对前两个条件有显式检查，PKM 的 `NUM_INSTANCE=4` 提供后一个条件。

## 7. 检索、学习率与实验配置

新增配置的冻结值如下；T0 只实际使用 Token、位置编码与 LR 倍率，`full` 使用全部项。

| 配置键 `MODEL.VISA_ADAPT.*` | 当前值 | 用途 |
| --- | ---: | --- |
| `MODE` | `off` / `token_only` / `full` | 默认关闭，实验配置分别选择 T0/V1.1 |
| `EXPERTS` | 8 | 每池专家数，共 24 个 |
| `TOP_K` | 2 | 每池 Router 选中数 |
| `PROMPT_LEN` | 16 | 每专家 Query 数，也即每路局部图数 |
| `HEADS` | 12 | 专家和 Local 聚合注意力头数，每头 64D |
| `GRAPH_K` | 40 | 每路从 128 Patch 选取数 |
| `FUSION_BETA` | 0.5 | Local 检索分支系数 |
| `SPECIFIC_WRT_WEIGHT` | 0.05 | Specific 模态内 WRT |
| `LOCAL_CE_WEIGHT` | 0.1 | Local 身份 CE |
| `FUSION_WRT_WEIGHT` | 0.5 | Fusion WRT |
| `BALANCE_WEIGHT` | 0.0001 | Router 概率重要度均衡 |
| `LOSS_RAMP_EPOCHS` | 5 | 仅新增损失线性启用轮数 |
| `NEW_LR_FACTOR` | 2.0 | 新参数相对对应旧参数 LR 倍率 |

训练和测试共用 `make_retrieval_feature`：

```
u_s = normalize(float32(f_s)); u_l = normalize(float32(f_l))
z   = concat(u_s, 0.5*u_l) / sqrt(1+0.5^2)  # [B,1536]
```

正常非零向量时 `||z||≈1`；平方欧氏距离中 CLS/Local 系数为 80%/20%。Fusion WRT 使用同数值 `z_f`，只在 CLS 段应用 `detach()`。正式测试统一返回 `z`，沿用原 Query/Gallery 合并、相机过滤、欧氏距离、九模态/平台分析；`RE_RANKING=False`，不按模态对改变权重。

旧参数继续执行本地 UAD 的规则：SGD 的 `BASE_LR` 由启动命令确定，旧 Bias 乘 `BIAS_LR_FACTOR=2`；`LARGE_FC_LR` 若启用仍按旧代码处理（当前未启用）。新参数包括三份 Specific Token、Specific 位置编码、三池 Router/24 个专家、双 GCN/LN、`out_token`/汇聚块、Local BNNeck/分类器；它们在各自对应原规则上统一乘 `NEW_LR_FACTOR=2`。旧 CLS、Patch、12 层 Block/LN、旧 BNNeck/分类器原 LR 不变。权重衰减仍按旧普通/Bias 规则。`UAD.yml` 写入的是单卡训练默认值；README 的四卡命令通过 CLI 覆盖全局 batch 和基础学习率。两套运行值必须分别遵守：

| 运行方式 | 全局 `IMS_PER_BATCH` | `BASE_LR` | 旧普通/旧 Bias LR | 新普通/新 Bias LR | 原 README 验证 batch / V1.1 建议值 |
| --- | ---: | ---: | --- | --- | ---: |
| 单卡（README 训练配方） | 64 | 0.008 | 0.008 / 0.016 | 0.016 / 0.032 | 1024 / 32 |
| 四卡 DDP（README 训练配方） | 256 | 0.032 | 0.032 / 0.064 | 0.064 / 0.128 | 512 / 32 |

`IMS_PER_BATCH` 是 PKM sampler 的全局 batch；四卡时每 rank 为 64 个 sampler 项。每项取 RGB、IR、Thermal 三模态，所以实际前向图像数每 rank 为 192。单卡 64 时也为 192 张图像。T0、V1.1 和 B0 必须在同一运行方式内使用相同 batch、基础 LR、seed 与训练轮数比较；四卡不可省略 CLI 中 256/0.032 的覆盖。

以下命令均从 `WHU/` 目录运行，`ROOT_DIR` 与 ImageNet 预训练权重路径沿用 `configs/UAD.yml`，需在实际服务器上指向有效位置。V1.1 在 WHU-MARS-1000 上的单卡与四卡命令分别为：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config_file configs/UAD_ViSA.yml \
  MODEL.DIST_TRAIN False DATASETS.DATASET_DIR WHU-MARS \
  SOLVER.IMS_PER_BATCH 64 SOLVER.BASE_LR 0.008 SOLVER.SEED 1234 \
  TEST.IMS_PER_BATCH 32 OUTPUT_DIR logs/UAD_ViSA_1000_single

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  train.py --config_file configs/UAD_ViSA.yml \
  MODEL.DIST_TRAIN True DATASETS.DATASET_DIR WHU-MARS \
  SOLVER.IMS_PER_BATCH 256 SOLVER.BASE_LR 0.032 SOLVER.SEED 1234 \
  TEST.IMS_PER_BATCH 32 OUTPUT_DIR logs/UAD_ViSA_1000_DDP
```

T0 对照把配置文件换成 `configs/UAD_T0.yml`，并换独立 `OUTPUT_DIR`；B0 用原 `configs/UAD.yml`。T0/B0 可继续使用 README 的验证 batch，因其没有图分支。WHU-MARS-2337 将 `DATASETS.DATASET_DIR` 改为 `WHU-MARS-2337` 并使用另一输出目录。V1.1 单次验证 1024 张时，仅单条图分支的节点张量就约 1.9 GiB，尚未计入其他中间结果，因此 V1.1 配置与命令明确使用验证 batch 32。验证时特征逐图计算、不使用训练批次统计，缩小验证 batch 影响吞吐和显存，不改变检索定义；如显存允许，可再增大。若训练 batch 不足以运行，应为 B0/T0/V1.1 统一调整 batch 与基础 LR 后重新做可比实验，不能只对 V1.1 单独改训练配方。

原余弦调度器原先给**所有旧组**同一绝对暖启起点 `0.01*BASE_LR` 和最低值 `0.002*BASE_LR`，即使 Bias 的峰值 LR 为 2 倍。适配运行现在对旧组精确保留此行为；新组的暖启起点和最低值乘 2，峰值使用其新组初始 LR。这样“新组=对应旧组×2”在各 epoch 成立，同时旧组与 B0 调度相同。T0 与 V1.1 的共有新 Token 均使用同样初始化、2 倍 LR、120 epoch 和原 UAD 损失设置。

## 8. 与 `idea.md` 逐项核对及实现形式差异

| 方案要求 | 当前核对结果 | 备注 |
| --- | --- | --- |
| 三模态选择式双 Token、旧位置编码不移位 | 已实现 | 旧权重 129 位保持原键；新增位置单独参数 |
| 只执行原 12 层一次，取 P11、CLS、Specific | 已实现 | 不存在深拷贝第 12 层或 CLS−Specific |
| 三池×8、Top-2、16 Query、单层条件注入专家 | 已实现 | 三池写在一个 `visa_adapt.py` 文件中，不影响模块语义 |
| Shared 入口截断，Specific 路径不断梯度 | 已实现 | 单测检查输出梯度边界 |
| 双 Top-40、16 张 41 节点图、独立 GCN | 已实现 | 精确 768D 图路径已做 CPU 形状/有限值检查 |
| Patch 节点值/邻接截断，Query 不截断 | 已实现 | 单测检查 Patch 无直接梯度、两 GCN 有梯度 |
| 独立 out_token、33 Token 单层聚合 | 已实现 | FFN 768→3072→768 |
| 原 UAD 损失、原型库和评测协议保持 | 代码路径已核对 | 无数据/权重，真实数值和指标待 GPU 验收 |
| 四项新损失及 5 轮暖启 | 已实现 | 系数与公式一致 |
| 唯一 1536D 正式检索特征 | 已实现 | 训练与测试调用同一融合函数 |
| 新参数 2 倍 LR，旧参数原策略 | 已修正并测试 | 旧 Bias 暖启与 B0 完全一致 |
| 单卡和 DDP 保持 | 原路径保留，未完成 GPU 实测 | 稀疏专家依赖已有 `find_unused_parameters=True` |

**与方案“建议接口”不同的实现形式：**训练模型返回六元组，没有默认把未停梯度 `z` 一并返回；处理器在同一次前向取得 `f_s,f_l` 后调用同一融合函数生成 `z_f`。这样避免 DDP 前向输出包含不用于 loss 的额外图张量。方案建议拆成 `visa_etgm.py` 和 `visa_dlfm.py` 两文件；实际放在一个 `visa_adapt.py`，类和梯度边界仍分开。方案表中的 `USE_VISA_ADAPT` 布尔开关改为 `MODE=off/token_only/full`，便于 T0 对照；`NUM_SPECIFIC=3` 固定于 WHU 三模态顺序，没有独立配置以防与数据模态数冲突。这些是接口与文件组织选择，不改变冻结结构或损失公式。

**与 ViSA 原论文/源码不同、但与 `idea.md` 一致的选择：**无 CLS−View 相减；以线性条件注入替代单 Key Cross-Attention；两路分别调用独立 GCN；不把 Query 节点整体放进 `no_grad()`；Local 取独立 `out_token`，最终取统一的 pre-BN CLS+Local。MDReID 的 ROL/KDL、MP-ReID 的文本与平台 Prompt 均未移植。以上是首轮设计选择，不能据此声称性能更好。

**仍待后续实验或环境验证：**诊断特征可通过 `return_components=True` 获取，当前代码没有自动写入独立特征文件；需在真实数据评估时决定保存格式。模型 checkpoint 沿用 UAD 原格式，只保存模型参数而不保存 GPD bank，故完整 checkpoint 可作评估，但不能把仅加载模型权重视作 GPD 训练的无缝续训。没有实施可选的“已训练 UAD checkpoint 热启动”路线；主实验是同 ImageNet 预训练起点的完整训练。尚无数据、预训练权重和 CUDA 环境，真实单卡/NCCL-DDP、显存、吞吐及 mAP/Rank-1 不能由 CPU 测试推出。

## 9. 已执行的本地自检

1. `python -m unittest discover -s tests -p 'test_visa_adapt.py' -v`：9 项通过。覆盖配置与 UAD 训练参数一致性、四卡 CLI 覆盖及调度器、不同验证 batch 的特征一致性、Token/位置形状、旧 Backbone checkpoint 拷贝及第 12 层权重、768D 图路径、Expert/Router/双 GCN 梯度与共享/图 Patch 截断、融合数值/单位范数、优化器分组和新旧 LR 在暖启/余弦阶段的比较。
2. `python -m compileall -q model processor solver config tests`：语法检查通过。
3. YACS 分别加载 B0/T0/V1.1 配置：模式为 `off/token_only/full`，原 `BASE_LR=0.008`、`PCA_LOSS_WEIGHT=0.01` 一致。
4. 训练模块导入成功；检查评估模态元数据 1/2/3 与训练拼接 0/1/2 的映射及 DataParallel 张量拆分。

真实环境中首次开训仍需核对：每 rank 的 PID/模态分布、所有被选专家梯度、两 GCN 梯度、无 NaN/Inf、ProCA/GPD bank 更新、单卡与 NCCL-DDP 的一批前后向、同 checkpoint 的 `u_s/u_l/z` 检索以及峰值显存。本地检查只证明上述静态和 CPU 契约，不代表训练复现或检索收益。
