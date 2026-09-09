# SAM 3 模型结构与模式说明

> 本文汇总自 `SAM3_算法能力与模式说明.md` 的架构总览与能力矩阵，梳理 **SAM / SAM 2 / SAM 3 / SAM 3.1** 四代模型的演进与区别，并给出 **SAM3 原生模式** 与 **SAM2-task 模式** 的完整模型结构图（所有尺寸/层数均取自 `sam3/model_builder.py` 真实构建参数，连接关系经代码核实）。
> 代码位置均以本仓库实际文件为准，可执行示例见 `examples/` 目录。

---

## 1. SAM / SAM 2 / SAM 3 / SAM 3.1：演进与区别

### 1.1 演进脉络

- **SAM（2023.4）**：单图、几何提示交互分割——“点哪里/框哪里，分哪里”
- **SAM 2（2024.7）**：+ 视频流式记忆跟踪——“第一帧点一下，全程跟住”
- **SAM 3（2025.11）**：+ 概念提示与开放词汇检测——“说一句话，检出并跟踪所有同类”
- **SAM 3.1（2026.3）**：+ 多对象 Multiplex 效率优化——“对象再多也快”

### 1.2 核心区别对比表

| 维度 | SAM | SAM 2 | SAM 3 | SAM 3.1 |
|---|---|---|---|---|
| 任务定位 | 可提示分割（PS） | 可提示视觉分割（PVS，图像+视频） | **概念分割**（开放词汇检测+跟踪） | 同 SAM 3，多对象效率版 |
| 提示类型 | 点/框/mask（纯几何） | 同左 | **文本短语**/视觉示例/点/框 | 同左 |
| 视觉编码器 | ViT-H（约632M，重） | **Hiera**（分层，高效） | **共享 ViT**（1008/ViTDet 窗口式，trunk 一套双 Neck 出两路） | 同左 |
| 核心架构 | Encoder + PromptEncoder + MaskDecoder 三大件 | 左 + **MemoryAttention + MemoryEncoder**（流式记忆） | **检测器（DETR式）+ 跟踪器（SAM2血统）解耦**，共享 ViT；presence token / DAC 双查询 / DotProductScoring | SAM3 + **MultiplexMaskDecoder**（多对象分桶联合解码，共享 token） |
| 文本/开放词汇 | ✗ | ✗ | ✓（CLIP-L 风格 24 层文本编码器，27 万+概念） | ✓ |
| 视频能力 | ✗（单图） | ✓ 流式逐帧传播、双向 | ✓ dense tracking：**传播中可自动发现新目标**（det-trk 关联缝合） | 同左 |
| 多对象 | 单目标迭代 | 每 obj 一套独立记忆状态 | 同左 + 对象上限/keep-alive 竞争 | **分桶共享推理**（桶容量 16），128 对象约 7× 加速 |
| 记忆机制 | 无 | 7 帧 spatial 记忆 + obj_ptr（SAM3 tracker 原样继承） | 同左 | 同左 + 共享记忆注意力 |
| 训练数据 | SA-1B（11M 图 / 1B mask） | SA-V（5 万视频 / 60 万 masklet） | SA-Co（概念标注 gold/silver 集） | 同 SAM 3（multiplex 微调） |
| 多掩码输出 | 3 候选 + IoU 选优 | 同左 + dynamic multimask | 同左（tracker 侧）；检测侧 200 query 一次出全图 | 同左 |
| 推理效率 | 慢（重编码器） | 快（Hiera + 流式） | 848M 参数；多卡 SPMD 支持 | + torch.compile 约 2×；多对象场景大幅提速 |
| 本仓库对应 | `sam3/sam/` 三大件（被复用）；能力矩阵 B | tracker 的记忆机制；能力矩阵 F | 仓库主体；能力矩阵 A-D、G | `sam3_multiplex_*`；能力矩阵 E；`RELEASE_SAM3p1.md` |

### 1.3 关键增量解读（每个模型只讲“新加了什么”）

1. **SAM 的奠基**：确立了“重编码器 + 轻解码器”的分工（编码一次，解码可实时），以及点/框/mask 三类几何提示 + 3 候选 mask 的交互范式。它的三大件（[prompt_encoder.py](sam3/sam/prompt_encoder.py)、[mask_decoder.py](sam3/sam/mask_decoder.py)、[transformer.py](sam3/sam/transformer.py)）被 SAM2/SAM3 **逐字复用至今**。

2. **SAM 2 的记忆**：解决“视频”靠的不是更大的模型，而是**记忆机制**——MemoryAttention（当前帧特征 cross-attend 历史记忆）+ MemoryEncoder（预测 mask 写回记忆）+ obj_ptr（目标语义指针）。这套骨架原封不动成为 SAM3 的 tracker（`Sam3TrackerBase`）。

3. **SAM 3 的解耦**：加检测器容易，难在“检测找新目标（逐帧独立、无状态）”和“跟踪跟住旧目标（有记忆、有状态）”训练时互相干扰。SAM3 的答案是**共享 ViT trunk + 双路独立 Neck + 检测/跟踪两分支独立解码 + 模型外缝合（det-trk IoU 匹配）**；文本能力通过 presence token（判概念是否存在）和 DotProductScoring（query·文本点积，替代固定类别头）落地开放词汇。

4. **SAM 3.1 的提速**：不改算法语义，改**多对象的执行方式**——多路 mask token 打包进桶共享一次 transformer 前向（MultiplexMaskDecoder + 超网络 MLP），配 torch.compile；对象数多时瓶颈从“每对象一次解码”变成“每桶一次解码”。

### 1.4 传承链（本仓库的“活化石”证据）

```
SAM 三大件(sam3/sam/) ──复用──► SAM2 记忆骨架(Sam3TrackerBase)
                                    │
                                    ▼ 继承为 tracker
SAM3 = 共享ViT + 双路Neck + 检测器(新) + tracker(SAM2血统) + 缝合平面
                                    │
                                    ▼ decoder 换 multiplex 桶
SAM3.1 = SAM3 + MultiplexMaskDecoder + torch.compile
```

---

## 2. 模型架构总览

SAM 3 是"检测器 + 跟踪器"解耦架构的统一分割基础模型，共 848M 参数：

```
                 ┌────────────────────────────┐
   图像/视频帧 → │  共享 ViT 视觉编码器        │
                 │  (1008 分辨率, patch 14)    │
                 └───────┬────────────┬───────┘
                         │            │
              ┌──────────▼───┐  ┌─────▼──────────────┐
   文本提示 → │  Detector     │  │  Tracker           │
   几何提示 → │  (DETR 风格,  │  │  (SAM2 式 encoder- │
   图像示例 → │  presence     │  │   decoder + 记忆)  │
              │  token)       │  │                    │
              └──────────────┘  └────────────────────┘
               开放词汇检测/分割   视频跟踪 + 交互精修
```

- **Detector**：以文本（开放词汇短语）、几何（点/框）、图像示例为条件，一次性检测并分割某概念的**所有实例**。presence token 用于区分相近概念（如"白衣球员" vs "红衣球员"）。
- **Tracker**：继承 SAM 2 的记忆式跟踪架构，负责逐帧传播掩码并支持交互式点/框精修。
- 两者共享同一 ViT 视觉主干；`sam3.pt` checkpoint 同时包含 detector 与 tracker 权重。

SAM 3.1（2026-03 发布）在 tracker 侧引入 **Object Multiplex**：把多个对象分组进固定容量桶联合处理（共享记忆注意力），128 对象场景在 H100 上约 7 倍加速，精度基本持平。

**组件规格速查**（来自 `model_builder.py` 构建函数）：

| 组件 | 规格 |
|---|---|
| ViT trunk | 输入 1008 / patch 14（→72×72=5184 token）/ embed 1024 / **32 层** / 16 头 / MLP ratio 4.625；窗口注意力 window=24 + 第 (7,15,23,31) 块全局注意力 + RoPE（336 预训练插值到 1008） |
| Neck（双路） | SimpleFPN，scale_factors `[4.0, 2.0, 1.0, 0.5]` → 4 尺度 × 256；SAM2 路 = deepcopy（结构相同、权重独立） |
| 文本编码器 | TextTransformer **24 层 × 1024 宽 × 16 头**（CLIP-L 量级），BPE 分词 ≤77 token，输出对齐 256 |
| 检测 Encoder/Decoder | 各 **6 层**，d_model 256 / FFN 2048 / 8 头 / pre-norm |
| 检测 Decoder 查询 | **200 queries**（DAC 对半 = 100 o2o + 100 o2m）、box_refine、presence_token、boxRPB="log"、RoPE 交叉注意力 |
| MemoryAttention | **4 层**，RoPE self-attn（256 维 1 头）+ cross-attn（KV 64 维压缩）+ FFN 2048 |
| 记忆库 | `num_maskmem=7`（1 条件帧 + 6 近邻帧）、`max_cond_frames_in_attn=4`、`max_obj_ptrs_in_encoder=16`、记忆特征 64 通道 |
| SAM 头 | PromptEncoder（mask_in_chans=16）+ MaskDecoder（TwoWayTransformer depth 2 / 8 头 / 3 候选 mask + IoU 头 + obj_score 头）+ obj_ptr_proj（MLP×3 → 256） |
| 记忆编码器 | SimpleMaskEncoder：SimpleMaskDownSampler（conv stride2→1152）+ CXBlock×2 fuser → 64-d |

---

## 3. 能力矩阵

| # | 模式 | 输入 | 提示类型 | 核心能力 | 构建入口 |
|---|------|------|----------|----------|----------|
| A | 图像开放词汇分割 | 单图 | 文本、框(正/负) | 检测并分割某概念的所有实例 | `build_sam3_image_model()` + `Sam3Processor` |
| B | 图像交互式分割（SAM1-task） | 单图/批量图 | 点(正/负)、框、掩码 | 单目标交互式抠图，SAM1 风格 | `build_sam3_image_model(enable_inst_interactivity=True)` |
| C | 图像批量推理 | 图集 | 文本 | 批量开放词汇分割 | 同 A，`set_image_batch` |
| D | SAM3 原生视频跟踪 | 视频/帧目录 | 文本、点、框 | 文本检测所有实例 + 全程跟踪 + 点精修；多 GPU | `build_sam3_predictor(version="sam3")` |
| E | SAM 3.1 Multiplex 视频 | 视频/帧目录 | 文本、点 | 同 D，多对象共享记忆，大幅提速 | `build_sam3_predictor(version="sam3.1")` |
| F | SAM2-task 视频跟踪 | 视频/帧目录 | 点、框 | SAM2 风格交互式 VOS（无文本） | `build_sam3_video_model().tracker` |
| G | Agent 模式 | 单图 | 自然语言指令 | MLLM 多轮推理 + SAM3 工具调用，处理复杂指代表达 | `sam3/agent/` |

---

## 4. SAM3 原生模式完整模型结构

### 4.0 架构位置（承接章节 1）

SAM3 原生模式即能力矩阵 D：完整启用检测分支 + 跟踪分支 + 缝合平面，是四代演进中"解耦架构"的完整形态（见 1.3 第 3 条增量）。本章自底向上拆解其全部组件。

### 4.1 共享骨干：ViT trunk 与双路 Neck

```
输入图像 1008×1008×3
        │
┌─ Patch Embed ─────────────────────────────────────────────┐
│  Conv 14×14 / stride 14（bias_patch_embed=False）          │
│  → 72×72 = 5184 tokens × 1024-d   （无 CLS token）          │
│  ln_pre=True → LayerNorm                                   │
│  + 绝对位置编码（预训练 336 尺寸 tile 平铺到 1008）           │
└────────────────────────────┬───────────────────────────────┘
                             ▼
┌─ Transformer Blocks × 32（embed 1024 / 16 头 / MLP 4.625≈4736）────┐
│   块 0~6, 8~14, 16~22, 24~30：窗口注意力                          │
│     window_size=24 → 72×72 网格切成 3×3 = 9 个窗口                │
│     每窗口 24×24=576 tokens 内部 self-attn（省算力主体）           │
│   块 7, 15, 23, 31：全局注意力（每 8 块一次全图信息交换）           │
│   RoPE（use_rope + use_interp_rope：336→1008 插值）              │
│   残差 + LayerNorm + drop_path 0.1                               │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
              单尺度特征图 72×72×1024（stride 14）
                             │
        ┌────────────────────┴────────────────────┐
        ▼ 共享同一 trunk 输出                       ▼
┌─ SAM3 Neck（检测路）──────┐          ┌─ SAM2 Neck（跟踪路）──────┐
│  SimpleFPN，4 个尺度分支： │          │  sam2_convs = deepcopy   │
│ scale 4.0: ConvT×2        │          │  (convs)——结构相同、      │
│   → 288×288×256           │          │  权重完全独立             │
│ scale 2.0: ConvT          │          │                          │
│   → 144×144×512           │          │  输出与左侧对称的          │
│ scale 1.0: 直通 72×72     │          │  多尺度特征，专供          │
│ scale 0.5: MaxPool→36×36  │          │  tracker / 交互分支       │
│ 每尺度: 1×1 conv→256      │          │                          │
│  + 3×3 conv→256 + sine PE │          │                          │
└──────────┬───────────────┘          └──────────────────────────┘
           ▼
  4 尺度 × 256-d
```

要点：ViTDet 式"**窗口为主、全局点缀**"——32 块里 28 块只做窗口内注意力，计算量近似线性于图像面积，只靠 4 个全局块交换全图信息。trunk 只有一个输出，两个 Neck 各自卷积出多尺度——**检测/跟踪的特征解耦从这里开始**（各自 neck 权重塑造各自特征风格）。

### 4.2 检测分支（Detector = `Sam3Image.forward_grounding`）

```
      text_memory (≤77 token × 256)        几何 prompt 嵌入 (n_prompt × 256)
      ▲                                    ▲ geometry_encoders: 3层transformer
      │                                    │ box→拆2个point编码 + CLS token
      │            KV                      │            KV
      │        ┌───────────────────────────┴──────────┐
      │        │        prompt token 序列 (拼接)        │
      │        └───────────────────┬──────────────────┘
      │                            │ cross-attn KV
图像主网格 token                    │
72×72 × 256 (num_feature_levels=1)│
      │                            │
      ▼                            │
┌─ TransformerEncoderFusion × 6 层 ─┴───────────────────────────┐
│  每层 (pre-norm, 8头, FFN 2048, dropout 0.1):                  │
│    ① self-attn:  图像 token ↔ 图像 token（空间上下文）           │
│    ② cross-attn: 图像 token 作 Q → 读 prompt tokens（文本+几何） │
│    ③ FFN                                                       │
│  输出: 融合了"文本语义"的图像 memory (5184 × 256)                │
└──────────────────────────────┬─────────────────────────────────┘
                               │ memory (KV)
┌──────────────────────────────┴─────────────────────────────────┐
│ TransformerDecoder × 6 层 ── 查询组织 (DAC + presence)          │
│  查询初始化: 200 个 object query = 100 o2o + 100 o2m (DAC对半)   │
│    + 各自带 reference box (4-d embedding, 可学习)                │
│    + 1 个 presence token (可学习 embedding, 位置编码置零)         │
│  每层:                                                          │
│   ① self-attn: 只允许 [presence token + 100 个 o2o] 参与         │
│      （o2m 查询跳过自注意力，靠后续交叉注意力获取信息）              │
│   ② text cross-attn:  所有 query → 读 text_memory               │
│   ③ image cross-attn: 所有 query → 读 encoder memory            │
│      + boxRPB("log"): query 当前 reference box 与图像 token 坐标 │
│        相对位置经 MLP(2→8头) 生成逐头 attention bias             │
│   ④ FFN                                                         │
│   ⑤ box refine: bbox_embed (MLP 256→256→4) 预测相对偏移          │
│      → 逐层迭代更新 reference box（最后一层权重零初始化）           │
│   每层输出都保留 (return_intermediate，供辅助损失)                │
└──────────────────────────────┬─────────────────────────────────┘
                               ▼
┌─ 输出头 ─────────────────────────────────────────────────────────┐
│  框:   bbox_embed → sigmoid cxcywh (200 候选)                    │
│  存在性: presence_token_head (MLP 3层→1) → presence logit         │
│         (clamp ±10；"图里根本没有这个概念"→整体抑制)               │
│  得分: DotProductScoring: query嵌入·prompt_mlp(文本嵌入)          │
│         (MLP 256→2048→256+LN) → 开放词汇相似度，无固定类别头       │
│  掩码: UniversalSegmentationHead                                 │
│         PixelDecoder(3级上采样, 吃 neck 多尺度特征)               │
│         + cross_attend_prompt(8头MHA: query attend prompt)        │
│         → query 的 mask 嵌入 ⊙ 像素特征 → 每实例一张 mask          │
└──────────────────────────────────────────────────────────────────┘
```

三个关键设计：

1. **DAC 双查询**：200 查询对半分 o2o（一对一匹配训练 → 推理高精度）与 o2m（一对多 → 高召回，且不参与自注意力防"凑答案"）。
2. **presence token**：每层加入 self-attn（排在 o2o 前、位置编码为零），像"全场巡查员"——图像 memory 里是否还有与概念匹配的内容，最终由它独自给出 1 个 logit；概念不存在时抑制所有输出。
3. **开放词汇落点**：分类不靠 `Linear(256, num_classes)`，而是 query 嵌入与**任意文本嵌入**点积——训练后换短语即可检新类别；几何 prompt 只是往 prompt token 序列多拼几个 token，与文本同一条融合通路。

**text_memory 的 3 个消费者**（全部在检测分支）：① EncoderFusion 每层 cross-attn 的 KV；② Decoder 每层 `ca_text` 的 KV；③ DotProductScoring 的点积打分。跟踪分支**完全不消费文本**。

### 4.3 跟踪分支（Tracker）与记忆库

```
════════════════ 记忆库（每个 obj_id 一套,跨帧持久）═════════════════════
  inference_state["tracker_inference_states"][obj_id] (SAM3原生)
  / inference_state["output_dict"] (SAM2-task)
  ┌────────────────────────────────────────────────────────────────────┐
  │ ├─ cond_frame_outputs: {帧t → {                   【条件帧记忆】    │
  │ │     maskmem_features: 64×(H/16)×(W/16)  空间记忆(64维压缩)         │
  │ │     maskmem_pos_enc:   对应位置编码                                 │
  │ │     obj_ptr:           256-d 对象指针                              │
  │ │     pred_masks / object_score_logits  ...}}                        │
  │ │   ※ 条件帧 = 用户提示帧 / 精修帧 / S4新轨迹的初始化帧               │
  │ │     —— 权重最高、锚点保护、重传播时不被覆盖                          │
  │ │                                                                    │
  │ └─ non_cond_frame_outputs: {帧t → {同上}}          【非条件帧记忆】   │
  │     ※ 纯传播帧的记忆，只保留滑窗内最近 6 帧                           │
  └────────────────────────────────────────────────────────────────────┘
        │读                                             ▲写
        ▼                                               │
  _prepare_memory_conditioned_features:                  │
  ① 条件帧记忆: select_closest_cond_frames 选最近        │
     max_cond_frames_in_attn=4 个 (t_pos=0)              │
  ② 非条件帧记忆: 滑窗最近 num_maskmem-1=6 帧            │
     (t_pos=1..6, maskmem_tpos_enc 区分远近)             │
  ③ 对象指针: 最近 16 帧的 obj_ptr(256-d);               │
     因 mem_dim=64<C=256 拆成 4 个 token,                │
     带相对时间 sine 编码                                 │
        │                                                │
        ▼ 拼接成 KV 序列                                  │
┌─ MemoryAttention ×4层 ─────────────────────────────────┐│
│  RoPE self-attn:  当前帧 token ↔ 当前帧 token            ││
│  RoPE cross-attn: Q=当前帧token(72×72×256,              ││
│                     即"SAM2路特征")                      ││
│                   KV=记忆序列(64-d压缩)                  ││
│  FFN 2048, pre-norm                                     ││
└──────────────────┬──────────────────────────────────────┘│
                   ▼ pix_feat_with_mem (72×72×256)          │
┌─ SAM头（每个obj独立执行一次）──────────────────────────────┐│
│  PromptEncoder: 传播帧无点/框输入 → 只有空sparse token     ││
│  MaskDecoder(TwoWayTransformer×2):                      ││
│    KV: pix_feat_with_mem + backbone_fpn高分辨率          ││
│  → 3候选mask + IoU头选优 + obj_score头                   ││
└────┬────────────────────────────┬────────────────────────┘│
     ▼                            ▼                         │
 低分辨率mask(288×288)      obj_ptr_proj(MLP×3)→256-d      │
     │                            │                         │
     ▼                            ▼                         │
┌─ 记忆写入（SimpleMaskEncoder）─────────────────────────────┐│
│  mask: sigmoid(logits×20-10) ⊕ pix_feat(SAM2路特征)       ││
│  → SimpleMaskDownSampler(conv stride2→1152对齐)           ││
│  → CXBlock×2 fuser (ConvNeXt dwconv)                     ││
│  → 64-d maskmem_features + pos_enc ──► 写入记忆库         ││
│  obj_ptr ─────────────────────────────► 写入记忆库        ││
└──────────────────────────────────────────────────────────┘
```

记忆库要点：

- **形态**：不是固定张量，而是**按 obj_id × 帧号索引的字典**，每条记录 = 64 维空间特征图 + 256 维对象指针。64 维压缩让 cross-attn KV 通道数减到 1/4，记忆条目多时显存线性可控。
- **读**：条件帧只取时间最近 4 个（语义="可靠锚点"）；非条件帧滑窗最近 6 帧（语义="短期运动连续性"）；obj_ptr 最近 16 帧累积（"目标的语义身份证"）。
- **写（SAM3 原生）**：刻意延迟——S2 预测的 mask 是"未经确认的猜测"，要等 S3 关联规划裁决后才编码入记忆，防止一次传播误差污染记忆库造成误差滚雪球。
- **SAM2 路特征**：SAM2 Neck 3 尺度 → conv_s0/s1 对齐 → `feature_cache[t]`，双消费者：MemoryAttention 的 src（Q 侧）+ MaskDecoder 高分辨率特征。

### 4.4 缝合平面：单帧五步流水线（`_det_track_one_frame`）

| 步 | 方法 | 干什么 |
|---|---|---|
| S1 | `run_backbone_and_detection` | 文本特征缓存 → detector 骨干+DETR 解码+NMS 产出 `det_out`（新目标框/mask/score），同时把 SAM2 路骨干特征存入 `feature_cache` 供 tracker 复用 |
| S2 | `run_tracker_propagation` | SAM2 记忆注意力预测已有 tracklet 的 mask（**不写记忆**，延迟到 S4），all-gather 全局 masks |
| S3 | `run_tracker_update_planning_phase` | rank 0 上 det↔trk IoU 匈牙利匹配、hotstart 去重、对象上限裁剪、GPU 负载均衡 → `tracker_update_plan` 广播各 GPU |
| S4 | `run_tracker_update_execution_phase` | 按计划增删 tracklet + 全局记忆编码（新轨迹初始化帧写 cond、高置信度时 recondition 刷新 mask、非重叠约束） |
| S5 | `build_outputs` | 合并检测+跟踪 → `{obj_id: mask, score}`，后处理（非重叠/未确认隐藏/坐标还原）→ yield |

### 4.5 完整结构图（一图流）

```
════════════════════════════ ① 输入层（一次性/每帧）════════════════════════════

 文本 "person"                  几何提示 box/point/mask              视频帧 t
      │                              │                          1008×1008×3
      ▼                              │                                │
 SimpleTokenizer                     │                                │
 (BPE, ≤77 token)                    │                                │
      ▼                              ▼                                │
 VETextEncoder            SequenceGeometryEncoder                     │
 (TextTransformer          (3层transformer+CLS,                       │
  24层×1024×16头)           box拆2 point)                              │
      │ text_memory          │ prompt嵌入(256-d)                      │
      │ (≤77×256, 缓存复用)   │                                        │
      │                      │                                        │
══════│═════════════════════│════ ② 共享骨干（S1内执行）═══════════════│══════
      │                      │                                ▼
      │                      │              ┌─ ViT trunk（共享）──────────────┐
      │                      │              │ Patch14 → 72×72=5184 token     │
      │                      │              │ ×1024-d, 32层: 28层窗口attn    │
      │                      │              │ +4层全局attn(块7/15/23/31)     │
      │                      │              │ RoPE(336→1008插值)             │
      │                      │              └───────────────┬─────────────────┘
      │                      │                              │ 72×72×1024
      │                      │            ┌─────────────────┴─────────────────┐
      │                      │            ▼                                   ▼
      │                      │  ┌─SAM3 Neck(检测路)─┐          ┌─SAM2 Neck(跟踪路)──┐
      │                      │  │ SimpleFPN 4尺度   │          │ deepcopy权重,3尺度: │
      │                      │  │ →288²/144²/72²/36²│          │ fpn_0 288×288×256 │
      │                      │  │ 每尺度1×1+3×3→256 │          │ fpn_1 144×144×256 │
      │                      │  └─────────┬─────────┘          │ fpn_2 72×72×256   │
      │                      │            │ 4尺度×256           └─────────┬─────────┘
      │                      │            │                               │
══════│═════════════════════│═══ ③ 检测分支（无状态,逐帧独立）═══════════│══════
      │                      │            │                               │
      │    ┌─────────────────┴──────────┐ │                               │
      │    │ TransformerEncoderFusion   │ │                               │
      │ ┌─►│  ×6层: ①self-attn(图像)    │ │                               │
      │ │KV│  ②cross-attn(Q=图像token,  │◄┼── prompt token序列 =           │
      │ │  │   KV=拼接prompt) ③FFN      │ │   text_memory+几何嵌入         │
      │ │  └──────────┬─────────────────┘ │                               │
      │ │             ▼ 融合文本语义memory  │                               │
      │ │  ┌─ TransformerDecoder ×6层 ────────────────────────┐           │
      │ │  │ 查询: 200 query(100 o2o+100 o2m DAC)              │           │
      │ │  │       +1 presence token +reference box            │           │
      │ │  │ 每层: ①self-attn(仅presence+o2o)                  │           │
      │ ├─►│  ②text cross-attn(Q=query)                        │           │
      │ │  │  ③image cross-attn(RoPE+boxRPB"log")              │           │
      │ │  │  ④FFN ⑤box_refine逐层精修框                       │           │
      │ │  └──────────────────────┬────────────────────────────┘           │
      │ │                         ▼                                        │
      │ │  ┌─ 输出头 ────────────────────────────────────────┐             │
      │ │  │ 框: bbox_embed / 存在性: presence_token_head     │             │
      │ ├─►│ 得分: DotProductScoring(点积=开放词汇)           │             │
      │ │  │ 掩码: UniversalSegmentationHead                 │             │
      │ │  │   (PixelDecoder 3级上采样+query attend prompt)   │             │
      │ │  └──────────────────────┬───────────────────────────┘             │
      │ │                         ▼                                        │
      │ │       det_out{boxes,masks,scores}──阈值+NMS                       │
      │ │                         │                                        │
      │ │    SAM2路特征同时存入 feature_cache[t]:                           │
      │ │     fpn_0──conv_s0──┐                                              │
      │ │     fpn_1──conv_s1──┼─►backbone_fpn(高分辨率2级)                   │
      │ │     fpn_2───────────┴─►vision_features(72×72×256)                 │
      │ │                         │                                        │
══════│═════════════════════│═══ ④ 跟踪分支（有状态,每obj一套记忆）══════│══════
      │ │                         │                                        │
      │ │  ┌─ 记忆库（per obj_id, 持久）─────────────────────────────────┐  │
      │ │  │ cond_frame_outputs{t→条目}: 条件帧(提示/精修/新轨迹          │  │
      │ │  │   初始化帧), 锚点保护, 读取时取最近4帧                       │  │
      │ │  │ non_cond_frame_outputs{t→条目}: 传播帧,滑窗最近6帧           │  │
      │ │  │ 条目 = {maskmem_features 64ch, maskmem_pos_enc,             │  │
      │ │  │         obj_ptr 256-d, pred_masks, obj_score}               │  │
      │ │  │ obj_ptr链: 最近16帧累积(非滑窗)                             │  │
      │ │  └───────┬──────────────────────────────────▲──────────────────┘  │
      │ │          │读(S2)                             │写(S4,确认后)        │
      │ │          ▼                                   │                    │
      │ │  _prepare_memory_conditioned_features        │                    │
      │ │    拼接KV序列: 条件帧4×5184(t_pos=0)          │                    │
      │ │    +非条件6×5184(t_pos=1..6) + obj_ptr 16帧   │                    │
      │ │    ×4token(64d拆分,带相对时间sine编码)         │                    │
      │ │          │                                   │                    │
      │ │          ▼                                   │                    │
      │ │  ┌─ MemoryAttention ×4层 ────────────────────┐│                    │
      │ └─►│ RoPE self-attn: 当前帧token               ││                    │
      │    │   (=vision_features,无文本输入!)           ││                    │
      │    │ RoPE cross-attn: Q=当前帧token,            ││                    │
      │    │   KV=记忆序列(64-d)                        ││                    │
      │    └──────────────────┬────────────────────────┘│                    │
      │                       ▼ pix_feat_with_mem       │                    │
      │    ┌─ SAM头(每obj独立) ─────────────────────────┐│                    │
      │    │ PromptEncoder: 传播帧无点框→空sparse token  ││                    │
      │    │ MaskDecoder(TwoWayTransformer×2):          ││                    │
      │    │   KV=pix_feat_with_mem+backbone_fpn高分辨率 ││                    │
      │    │ →3候选mask+IoU选优+obj_score头             ││                    │
      │    └────┬──────────────────────────┬────────────┘│                    │
      │         ▼                          ▼             │                    │
      │   低分辨率mask(288×288)    obj_ptr_proj(MLP×3)   │                    │
      │         │                          │(暂存,不写库) │                    │
      │         │ ══ S2到此为止,不写记忆 ══  │             │                    │
      │         ▼                          ▼             │                    │
═════════════════════ ⑤ 缝合平面（sam3_video_base, 模型外）════════│════════════
      │                                                    │                    │
      │   ┌─ S3 关联规划(rank0) ────────────────────────────┐                    │
      │   │ det_out ↔ 传播masks: mask IoU+匈牙利匹配(perflib)│                    │
      │   │ 检测命中轨迹→确认 / 检测无匹配→新obj_id          │                    │
      │   │ 轨迹N帧无命中→keep-alive到期删除 / hotstart去重  │                    │
      │   │ →tracker_update_plan ──► 广播各GPU              │                    │
      │   └──────────────────────┬──────────────────────────┘                    │
      │                          ▼                                              │
      │   ┌─ S4 执行更新+写记忆 ──────────────────────────────┐                  │
      │   │ 新轨迹: det的mask为输入初始化(该帧→cond)           │                  │
      │   │ 既有轨迹: recondition(高置信度时刷新mask)          │                  │
      │   │ 写入: sigmoid(mask)×20-10 ⊕ vision_features      │                  │
      │   │  →MaskDownSampler→CXBlock×2→64-d ────────────────┼──────────────────┘
      │   │ obj_ptr───────────────────────────────────────────┼──────────────────┘
      │   │ 普通传播帧→non_cond(滑窗淘汰) / 全局非重叠约束     │
      │   └──────────────────────┬───────────────────────────┘
      │                          ▼                                              │
      │   ┌─ S5 build_outputs ─────────────────────────────────┐                │
      │   │ 合并检测+跟踪→{obj_id:mask, obj_id:score}          │                │
      │   │ →postprocess → yield(frame_idx,masks)              │                │
      │   │   ──► 下一帧回到②(记忆已更新,闭环)                  │                │
      │   └────────────────────────────────────────────────────┘                │
```

读图速查：

| 看什么 | 在哪 |
|---|---|
| text_memory 的 3 条线 | EncoderFusion cross-attn KV、Decoder 每层 ca_text、DotProductScoring 点积——全在检测分支，跟踪分支零文本输入 |
| SAM2 路特征 | SAM2 Neck 3 尺度 → conv_s0/s1 对齐 → `feature_cache[t]`，双消费者：MemoryAttention 的 src + MaskDecoder 高分辨率特征 |
| 记忆读 | S2 拼接 KV：4 条件帧（t_pos=0）+ 6 非条件帧（t_pos=1..6）+ 16 帧 obj_ptr（每个拆 4 token） |
| 记忆写 | S4 才写：S3 确认后的 mask 经 SimpleMaskEncoder 压到 64-d；新轨迹初始化帧写 cond、传播帧写 non_cond |
| 解耦边界 | ③ 检测无状态逐帧独立 ↔ ④ 跟踪有状态跨帧闭环；⑤ 的 S3 是两者唯一"翻译官" |
| 共享部分 | 仅 ViT trunk（Neck 起双路权重独立）——② 的分叉点 |

一句话心法：**文本决定"检什么"（只在检测路），记忆决定"跟什么"（只在跟踪路），ViT 决定"看什么"（两路共享），S3 决定"谁是谁"（左右缝合）。**

---

## 5. SAM2-task 模式完整模型结构

### 5.1 模型组装方式（[sam2_service.py](app/server/sam2_service.py#L84-L110)）

```python
video_model = build_sam3_video_model(apply_temporal_disambiguation=False)
tracker = video_model.tracker              # 取 sam3.pt 的 tracker 子模块
tracker.backbone = video_model.detector.backbone  # 借用 detector 的 ViT 主干
tracker = tracker.cuda().eval()
del video_model                            # 丢弃 detector（含检测头）
```

- 权重来自同一个 `sam3.pt`，**不需要 SAM2 checkpoint**；
- `SAM3VLBackbone` 双 Neck 都会前向计算，但只取 `sam2_backbone_out`（[sam3_tracker_base.py `forward_image`](sam3/model/sam3_tracker_base.py#L442-L463)：fpn[0/1] 预过 conv_s0/s1，scalp=1 丢弃最低分辨率）——**检测路的卷积是纯开销**；
- text encoder 随模型加载但从不被调用。

### 5.2 完整结构图

```
══════════════════════ ① 输入层 ══════════════════════

  点/框提示 (obj_id 由用户指定!)         视频帧 1008×1008
  纯 point / 纯 box / 组合均可             │
  ※ 没有文本输入——text encoder            │
    随模型加载但从不被调用                  │
       │                                  │
       │  传播开始后新 obj_id → 409        │
       │  (必须先 reset_tracking 重放)     │
       │                                  │
═══════│═══════════════════ ② 骨干（借用的）═══════════│══════
       │                                  ▼
       │                  ┌─ ViT trunk（与SAM3检测共享同一套权重）─┐
       │                  │ patch14 → 72×72×1024, 32层            │
       │                  │ 28层窗口attn + 4层全局attn             │
       │                  └────────────────┬───────────────────────┘
       │                                   │
       │                   ┌─ Sam3DualViTDetNeck（双路都计算）─┐
       │                   │ SAM3路(检测用): 算完…丢弃 ✗        │
       │                   │ SAM2路(跟踪用): scalp=1丢最低分辨率  │
       │                   │  → 3级: 288²/144²/72² ×256        │
       │                   └────────────────┬───────────────────┘
       │                                    │ sam2_backbone_out
       │                    forward_image:  │
       │                    fpn[0]→conv_s0 ─┤
       │                    fpn[1]→conv_s1 ─┼─► backbone_fpn(3级)
       │                    fpn[2]──────────┘
       │                                    │
       │                       cached_features[frame_idx]
       │                                    │
═══════│════════════════ ③ 提示提交（交互路径,增量）═════════════
       ▼                                    │
  add_new_points_or_box ──► 该帧升级为【条件帧】
   (无reset_state! 无replay!)               │
       │                                    │
       │   ┌─ SAM头（每obj_id独立）─────────┐│
       └──►│ PromptEncoder:                ││
           │   点→sparse token(带正负label)  ││
           │   框→2个角点token               ││
           │   (点击帧同时有真实点/框输入)    ││
           │ MaskDecoder:                   ││
           │   KV=pix_feat_with_mem(若有记忆) ││
           │      +backbone_fpn高分辨率  ◄───┘│
           │   首次点击: 无记忆→no_mem_embed  │
           │   → 预览mask(3候选+IoU选优)      │
           └───────────┬─────────────────────┘
                       │ ※ run_mem_encoder=False
                       │   点击时【不写记忆】,仅临时存储
══════════════════════ ④ 传播（propagate_in_video）══════════════════

  preflight: SAM3强制 add_all_frames_to_correct_as_cond=True
    → 精修帧正式升级条件帧 + 写入记忆库
  方向"both" = 前向(最早提示帧→片尾)
             + 后向(最晚提示帧→片头)
             + 条件帧补齐(backfill)
                       │
  for 每帧(条件帧=锚点跳过):│
                       ▼
  ┌─ 记忆库（per obj_id, 与SAM3原生完全相同）────────────────┐
  │ cond_frame_outputs{t}: 提示帧/精修帧, t_pos=0, 最近≤4帧   │
  │ non_cond_frame_outputs{t}: 传播帧, 滑窗最近6帧            │
  │ obj_ptr链: ≤16帧×256-d                                   │
  └────────┬───────────────────────────────────▲────────────┘
           │读(每帧)                            │写(每帧立即!)
           ▼                                   │
  _prepare_memory_conditioned_features          │
    拼KV: 4×5184(条件) + 6×5184(非条件)          │
        + 64个obj_ptr token(16帧×4)              │
           │                                   │
           ▼                                   │
  ┌─ MemoryAttention ×4层 ──────────────────────┐
  │ RoPE self-attn: 当前帧token                 │
  │   (=cached_features的SAM2路,无文本!)         │
  │ RoPE cross-attn: Q=当前帧, KV=记忆(64-d)    │
  └──────────────────┬──────────────────────────┘
                     ▼ pix_feat_with_mem
  ┌─ SAM头(传播帧:无点框→空sparse token)─────────┐
  │ MaskDecoder → mask + obj_ptr ───────────────┼──► 记忆写入
  └──────────────────┬──────────────────────────┘    (无S3确认!)
                     ▼                              sigmoid(mask)⊕pix_feat
        yield(frame_idx, masks)                     →MaskDownSampler
        下一帧(记忆刚更新)                            →CXBlock×2→64-d
                                                    新轨迹首帧→cond
                                                    其余→non_cond
```

### 5.3 单目标跟踪生命周期（两模式共用的 tracker 机制）

```
诞生 ──► 预热 ──► 逐帧传播(读→融合→解码→写→输出 循环) ──► 精修(回到预热) ──► 消亡
(注册+   (确认入       每帧同一套 5 步循环              (升级锚点+       (remove_
 首帧)    记忆库)                                       全视频重算)       object)
```

1. **诞生**（`add_new_points_or_box`）：分配 `obj_idx` 槽位 → 记录提示 → 首帧分割（无记忆，走 `no_mem_embed`，等价纯 SAM 单图分割）→ 结果进临时仓库（`run_mem_encoder=False`，可撤销）。
2. **预热**（`propagate_in_video_preflight`）：临时结果正式收编——跑记忆编码器写入 `cond_frame_outputs`，该帧成为**条件帧（锚点）**；清除锚点周围过时的 non-cond 记忆；`tracking_has_started=True`（此后新 obj_id 拒收）。
3. **逐帧传播**：多目标打包成 batch（每目标一行），每帧 5 步——①读自己的记忆（4 cond + 6 non-cond + 16 obj_ptr）→ ②MemoryAttention 融合 → ③SAM 头解码出 mask + obj_score → ④立即写记忆（传播帧写 non_cond，滑窗淘汰）→ ⑤切回 per-obj 切片并 yield。条件帧被跳过（锚点保护）。
4. **精修**：任意帧加点/框——又是"只算不存"（带旧 mask logits + 新点在记忆上下文中修正）→ 下次 preflight 升级为**新条件帧** → 重传播时全视频非条件帧重算覆盖，锚点不动。
5. **消亡**：`remove_object` 切除槽位（其他目标原样保留）。

### 5.4 多目标隔离与相互影响

**完全隔离**：
- prompt 分账本存储（`point_inputs_per_obj[obj_idx]`），A 的框/点永不进入 B 的解码器；传播帧所有目标 `point_inputs=None`；
- 传播期注意力不跨 batch 行——A 行永远看不到 B 行的记忆特征；
- 交互精修只跑单目标切片。

**4 个真实影响点**（SAM2-task 模式，集中在"以帧为粒度的全局操作"）：

| # | 机制 | 后果 |
|---|---|---|
| ① | 条件帧记忆的**非重叠约束**（preflight 整合时逐像素取最高分目标，其余压到 logit≤-10 后重编码记忆） | 提示帧 mask 重叠时，低分目标的重叠区域从记忆里被抹掉，后续传播易缺角 |
| ② | `add_new_mask` 笔刷的**显式重叠擦除** | 给 A 刷 mask 会直接改写 B 同帧临时输出，扣掉重叠像素 |
| ③ | **条件帧选择全局共享**（`select_closest_cond_frames` 在所有目标条件帧并集里挑最近 4 帧） | 他目标的提示帧挤占自己的记忆窗口选择；晚出生目标被迫带早目标锚点节奏 |
| ④ | `clear_non_cond_mem_around_input` 对**所有目标**生效 | 构建时显式设 `clear_non_cond_mem_for_multi_obj=False` 守卫，多目标时禁用——隔离防御的证据 |

SAM3 原生模式下影响更剧烈：S3 全局匈牙利匹配（一对一竞争）、每帧全局非重叠记忆编码、hotstart 去重、`max_num_objects` 挤位、keep-alive 生存竞争、输出级非重叠+未确认隐藏。

### 5.5 与 SAM3 原生模式对比

```
                SAM3 原生模式                    SAM2-task 模式
              ┌─────────────────┐            ┌─────────────────┐
 文本输入      │ ✓ 3处消费        │            │ ✗ 加载不用       │
 点/框输入    │ 精修走tracker    │            │ ✓ 唯一入口       │
              │ text/box走检测   │            │   直接初始化     │
              └─────────────────┘            └─────────────────┘
 骨干         │ ViT共享,双路都被消费           │ ViT借用,双路都算
              │   检测路+跟踪路               │   只消费跟踪路
              │                               │   (检测路白算丢弃)
 检测分支     │ ✓ 每帧跑(6+6层DETR)           │ ✗ 整体不存在
              │   能发现新目标                 │   永不发现新目标
 obj_id       │ 模型自动分配                   │ 用户手动指定
 新目标       │ 传播中随时可加                 │ 传播前必须加完
              │  (检测器每帧找)                │  (之后409,需reset重放)
 记忆写入     │ S3关联确认后才写(延迟)          │ 每帧传播完立即写
              │   防误差滚雪球                 │   (无确认环节)
 关联规划S3   │ ✓ IoU匈牙利+hotstart           │ ✗ 无
              │   +keep-alive删轨迹            │   轨迹只增不删
              │   +GPU负载均衡                 │
 传播方向     │ processing_order双向           │ 前向+后向+backfill
 精修机制     │        └──── 同一套 tracker API ────┘
 记忆库结构   │        └──── 完全相同(7帧/4cond/16ptr/64-d) ────┘
 SAM头/MemoryAttention│      └──── 完全相同 ────┘
 权重来源     │        └──── 同一个 sam3.pt ────┘
```

要点解读：

1. **SAM2-task 是 SAM3 原生图的"右半裁剪"**：保留跟踪分支 + 借用骨干，砍掉检测分支和缝合平面（S3/S4 的确认-延迟写机制随之消失，退化为逐帧立即写记忆的朴素 SAM2 行为）。
2. **骨干微妙差异**：两模式跑的是同一个 `SAM3VLBackbone` 对象，双 Neck 都前向；SAM3 原生两路输出都被消费，SAM2-task 只取 `sam2_backbone_out`。
3. **精修路径两模式共用**：`add_tracker_new_points`（SAM3 原生）与 `add_new_points_or_box`（SAM2-task）最终都落在同一个 `Sam3TrackerPredictor` 上，"点击不写记忆 → preflight 升级条件帧 → 重传播锚点保护"完全同源。
4. **能力换稳定性**：SAM3 原生用检测分支换开放词汇与新目标发现，代价是 text/box 提交会 `reset_state`（服务层需重放对抗）；SAM2-task 用"无检测"换纯增量提交、用户掌控 obj_id、提示永不重置——标注场景（按目标精标）更看重后者。

---

## 附：两种模式完整对比表（算法/接口层）

| 维度 | SAM3 模式（D/E） | SAM2-task 模式（F） |
|------|------------------|---------------------|
| 加载的模型部分 | detector + tracker 全模型 | 仅 tracker 子模块 + 借用 ViT 主干 |
| 文本提示 | 支持（开放词汇，一次出全部实例） | **不支持** |
| 框提示语义 | 检测提示："框里是什么"→ 检出同类所有实例 | 跟踪初始化提示：框住哪个目标就跟踪哪个 |
| 点提示语义 | 对指定 obj_id 精修 | 对指定 obj_id 初始化/精修 |
| 对象 id 分配 | 文本/框路径自动分配 tracklet id | 始终由用户指定 obj_id |
| 单次提示产出 | 可一次产生多个对象 | 一次一个对象 |
| 提示的状态语义 | text/box 每次调用**重置** tracker 状态（需重放全部提示才能保持已确认目标） | **纯增量**，提示逐条累积，无需重放 |
| 传播中自动检测新实例 | 支持（dense tracking 启发式） | 不支持（纯交互式 VOS） |
| 传播后加新对象 | reset + 重放（服务层封装） | `reset_tracking` + 重放 prompt log |
| 提示撤销 | 需服务层自行实现（重放确认集） | 服务层 `undo_prompt`（回退单条提示） |
| 跨帧精修 | 点提示（任意帧、指定 obj_id） | 点/框提示（任意帧、指定 obj_id） |
| 传播方向 | forward / backward / both | 同左（both = 先 forward 后 backward） |
| 额外能力 | `encode_features` 预计算 ViT 特征加速后续提示 | `clear_all_points_in_video` 等 SAM2 兼容接口 |
| 显存/速度 | 全模型 + 检测路径重置重放，单次交互开销大 | 仅 tracker，交互轻量、无重放开销 |
| 适用场景 | "按概念批量标"（一句话标一类物体） | "按目标精标"（逐个目标手工框选/点选） |

一句话总结：**SAM3 模式是"概念驱动"的检测+跟踪，SAM2-task 模式是"提示驱动"的纯交互式跟踪**；前者赢在文本开放词汇和一次多实例，后者赢在增量状态、轻量交互和逐条撤销。
