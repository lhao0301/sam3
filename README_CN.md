# SAM 3 Video Annotation Tool

[English](README.md) | 简体中文

基于 [Meta SAM 3](https://github.com/facebookresearch/sam3) 视频跟踪引擎（SAM2-task 模式）构建的浏览器端**单目标视频标注工具**。

通过「框初始化 → 跨帧点精修 → 掩码传播」的交互流程，为视频中单个目标快速生成像素级分割标注；前端负责交互与可视化，FastAPI 后端负责推理与会话管理，掩码与进度经 WebSocket 实时流式回传。

> 本仓库同时包含完整的 SAM 3 模型代码（`sam3/`），上游官方文档见 [facebookresearch/sam3](https://github.com/facebookresearch/sam3)；模型结构深度解析见 [sam3.md](sam3.md)（中文）。

<p align="center">
  <img src="assets/annotation/workflow-demo.gif" alt="标注流程演示：画框 → 预览 → 确认 → 点精修 → 传播跟踪" width="880" />
</p>

---

## 功能维度总览

### 1. 视频接入与会话管理

| 功能 | 说明 |
|---|---|
| 本地视频上传 | 浏览器选择 MP4 等视频文件上传至服务器 |
| 服务器路径直连 | 直接输入服务器上的视频路径启动会话（免上传） |
| 自动抽帧 | 上传后按视频 FPS 抽帧，生成预览帧与缩略图条 |
| 会话生命周期 | 启动 / 关闭 / 删除会话；页面断开自动清空会话缓存 |
| GPU 自动选择 | 启动时自动选择显存最空闲的 GPU 卡，页面实时显示 GPU 状态 |

### 2. 交互式提示标注（两段式：绘制 → 预览 → 确认）

| 功能 | 说明 |
|---|---|
| 框提示 | 拖拽画框框住目标，可与点提示组合提交 |
| 正 / 负样本点 | 正点标记目标位置，负点排除误分割区域 |
| 点 / 框组合 | 同一提示内点与框自由组合，一次提交 |
| 撤销与清空 | 撤销最近一个点；一键清空全部待提交提示 |
| 掩码预览 | 提交前实时运行分割，掩码叠加显示在当前帧上 |
| 确认 / 撤销预览 | 预览满意后确认加入目标列表；不满意可整体撤销 |
| 用户指定 ID | 每个目标（tracklet）使用用户指定的 id，支持自定义类别名 |
| 增量提示 | 提示增量生效（无 reset / 重放），点与框可跨帧叠加精修 |

### 3. 掩码传播跟踪

| 功能 | 说明 |
|---|---|
| 三方向传播 | `forward` / `backward` / `both`，从标注帧向任意方向传播 |
| 流式进度 | WebSocket 逐帧回传掩码与进度条，边传播边预览 |
| 任务取消 | 传播过程中可随时取消 |
| 重置跟踪 | 清除跟踪结果但保留已确认提示，便于追加目标后重新传播 |
| 多目标并存 | 已确认 tracklet 列表统一管理，逐个传播 |

### 4. 帧导航与播放控制

| 功能 | 说明 |
|---|---|
| 逐帧切换 | 上一帧 / 下一帧按钮，快捷键 `←` / `→` |
| 播放 / 暂停 | 视频预览播放，快捷键 `空格` |
| 帧滑块 | 拖动滑块快速定位任意帧，显示当前帧号与总帧数 |
| 缩略图条 | 底部 filmstrip 缩略图快速跳转 |

### 5. 结果查看与导出

| 功能 | 说明 |
|---|---|
| 掩码叠加显示 | 传播结果掩码实时叠加在视频帧上 |
| 单帧掩码获取 | 按帧号获取 PNG 掩码，便于二次处理 |
| 标注视频导出 | 一键导出带掩码叠加的 MP4 标注视频 |
| 会话信息查询 | 查询会话帧数、FPS、已确认目标等元信息 |

掩码叠加效果：

![掩码叠加预览](assets/annotation/mask-overlay.png)

### 6. 可观测性与运维

| 功能 | 说明 |
|---|---|
| 前端日志终端 | 页面内嵌可折叠终端，实时展示操作与推理日志 |
| 日志服务端持久化 | 前端日志上报至服务器，按天写入 `logs/` 目录 |
| 错误模态框 | 上传校验失败、推理异常等错误以模态框明确提示 |
| 服务状态接口 | `/api/status` 暴露服务与 GPU 状态 |

---

## 系统架构

前后端分离：静态前端由 FastAPI 直接托管，也可独立部署（已启用 CORS）。

```
┌───────────────────────────────┐          ┌─────────────────────────────────────┐
│ 前端 app/frontend/            │  REST    │ 后端 app/server/                    │
│ ├─ ws-client.js     WS 客户端 │ ──────▶  │ ├─ app.py              FastAPI 入口 │
│ ├─ video-player.js  播放器    │ ◀──────  │ ├─ sam2_service.py     推理服务     │
│ ├─ prompt-canvas.js 提示绘制  │ WS 流式 │ ├─ session_manager.py  会话管理     │
│ ├─ mask-overlay.js  掩码叠加  │          │ ├─ ws_manager.py       WS 管理      │
│ ├─ log-terminal.js  日志终端  │          │ ├─ gpu_utils.py        GPU 选择     │
│ └─ app.js           状态编排  │          │ └─ frame/mask_utils.py 帧与掩码工具 │
└───────────────────────────────┘          └─────────────────────────────────────┘
                                                       │
                                          SAM 3 checkpoint
                                          （Sam3TrackerPredictor · SAM2-task 模式）
```

## 安装

### 前置要求

- Linux（或 macOS），NVIDIA GPU，CUDA 12.6+ 驱动
- Python ≥ 3.12、Conda、git

### 1. 安装 SAM 3 模型包

标注服务复用本仓库自带的 `sam3` 包，安装方式与上游模型完全一致：

```bash
# 1a. 创建隔离环境
conda create -n sam3 python=3.12
conda activate sam3

# 1b. 安装带 CUDA 支持的 PyTorch
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# 1c. 安装 sam3 包（在仓库根目录执行）
pip install -e .

# 1d. （可选）推理加速：FlashAttention-3 + 优化连通域
pip install einops ninja
pip install flash-attn-3 --no-deps --index-url https://download.pytorch.org/whl/cu128
pip install git+https://github.com/ronghanghu/cc_torch.git
```

Notebook 与开发所需的额外依赖组（`pip install -e ".[notebooks]"`、`pip install -e ".[dev,train]"`）见[上游 README](https://github.com/facebookresearch/sam3#installation)。

### 2. 准备 SAM 3 checkpoint

模型权重在 Hugging Face 上为 gated 资源——先申请 [facebook/sam3](https://huggingface.co/facebook/sam3)（或 [facebook/sam3.1](https://huggingface.co/facebook/sam3.1)）访问权限并完成认证：

```bash
hf auth login   # 粘贴 Hugging Face 访问令牌
```

然后可让服务在首次启动时自动下载权重，或提前放置到本地：

```bash
mkdir -p checkpoints/sam3
# 将 sam3.pt 权重下载到 checkpoints/sam3/
```

### 3. 安装标注服务依赖

```bash
pip install -r app/server/requirements.txt
```

将安装 FastAPI、uvicorn（含 WebSocket 支持）、python-multipart（视频上传）、aiofiles 与 websockets。

### 4. 启动服务

```bash
# SAM3_CHECKPOINT_PATH 可省略——未设置时权重自动从 Hugging Face 拉取（需认证）
SAM3_CHECKPOINT_PATH=checkpoints/sam3/sam3.pt \
uvicorn app.server.app:app --host 0.0.0.0 --port 8000
```

浏览器打开 `http://<server>:8000`，按「上传视频 → 画框/点提示 → 预览 → 确认 → 传播 → 导出」流程标注。

服务启动时自动绑定显存最空闲的 GPU；设置 `SAM3_GPU=<index>` 可指定卡号（外部已导出的 `CUDA_VISIBLE_DEVICES` 优先级最高）。

### 配置参考

服务端配置全部通过环境变量完成：

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `SAM3_CHECKPOINT_PATH` | *（未设置）* | 本地权重文件；未设置时自动从 Hugging Face 拉取 |
| `SAM3_GPU` | `auto` | 指定 GPU 卡号；`auto` 表示导入时选最空闲的卡 |
| `SAM3_MAX_SESSIONS` | `4` | 最大并发标注会话数 |
| `SAM3_MAX_INFERENCE` | `1` | 最大并发推理任务数 |
| `SAM3_LOG_DIR` | `logs` | 服务端日志持久化目录 |

生产环境建议使用进程管理器，或简单以 nohup 启动：

```bash
nohup uvicorn app.server.app:app --host 0.0.0.0 --port 8000 > sam3_server.log 2>&1 &
```

## 标注工作流

**① 接入视频** — 上传文件或输入服务器路径，服务自动抽帧并生成缩略图条：

![启动会话](assets/annotation/workflow-1-start.png)

**② 绘制提示** — 默认框工具拖拽框住目标，可叠加正/负点组合（`Delete` 删除选中提示），同时指定类别名与 tracklet id：

![框提示](assets/annotation/workflow-2-box-prompt.png)

**③ 确认目标** — 「分割预览」查看掩码，满意后「确认」加入目标列表，可继续标注下一个目标：

![确认目标](assets/annotation/workflow-3-confirm-tracklet.png)

**④ 跨帧精修** — 跳到任意帧补充点/框提示，掩码实时叠加预览；追加新目标前先「重置跟踪」：

![跨帧点精修](assets/annotation/workflow-4-point-refine.png)

**⑤ 传播与导出** — 选择正向/反向/双向传播到全视频，进度经 WebSocket 实时流式展示，完成后一键导出标注 MP4；操作与推理日志实时显示在页面终端：

![传播与终端日志](assets/annotation/workflow-5-terminal-logs.png)

## API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/status` | 服务与 GPU 状态 |
| POST | `/api/upload` | 上传视频 |
| POST | `/api/session/start` | 启动会话（含服务器路径模式） |
| GET | `/api/session/{id}/info` | 会话元信息 |
| POST | `/api/session/{id}/close` · `DELETE /api/session/{id}` | 关闭 / 删除会话 |
| POST | `/api/prompt/add` | 添加提示（点/框，可组合） |
| POST | `/api/prompt/confirm` | 预览确认，加入目标列表 |
| DELETE | `/api/prompt/{id}/pending` · `/{id}/{obj_id}` | 撤销待定提示 / 移除目标 |
| POST | `/api/session/{id}/reset-tracking` | 清除跟踪结果，保留提示 |
| GET | `/api/frame/{id}/{idx}` · `/api/thumb/{id}/{idx}` | 帧图 / 缩略图 |
| GET | `/api/mask/{id}/{idx}` | 单帧掩码 PNG |
| GET | `/api/export/{id}` | 导出标注 MP4 |
| POST | `/api/logs` | 前端日志上报（服务端持久化） |
| WS | `/ws/propagate/{id}` | 传播：`start(direction)` / `cancel` / `ping` |

## 测试

```bash
python -m pytest test/ -v               # Python：box 工作流、SAM2 模式、推理优化
node test/test_prompt_canvas_frame.js   # 前端：提示画布帧绑定逻辑
```

## 目录结构

```
app/                        视频标注工具（本 README 主体）
├── frontend/               静态前端（HTML/CSS/JS，无构建依赖）
└── server/                 FastAPI 后端（推理、会话、WS、GPU）
sam3/                       SAM 3 模型代码（上游完整保留）
examples/                   官方推理示例 Notebook
checkpoints/                模型权重（不入库）
test/                       标注工作流与前端测试
```

## 底层模型

推理运行于 SAM 3 checkpoint 内的 **Sam3TrackerPredictor**（SAM2-task 模式）：流式记忆跟踪，提示增量生效不重置，每个对象携带用户指定 id，点与框可组合。SAM 3 相较 SAM 2 引入概念提示与开放词汇检测；四代模型演进与结构详解见 [sam3.md](sam3.md)。

SAM 3 论文与更多资源：

[[`Paper`](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/)] [[`Project`](https://ai.meta.com/sam3)] [[`Demo`](https://segment-anything.com/)]

## 致谢

本项目构建于 Meta AI Research 开源的 **SAM 3** 模型、代码与权重之上——感谢 SAM 3 团队（[facebookresearch/sam3](https://github.com/facebookresearch/sam3)）开放模型与完整训练/推理代码。标注工具使用的跟踪引擎是 SAM 3 的 SAM2-task 模式，其流式记忆视频分割设计承自 SAM 2。可选的推理加速依赖 [FlashAttention-3](https://github.com/Dao-AILab/flash-attention) 与 [cc_torch](https://github.com/ronghanghu/cc_torch)。

## SAM 3 论文引用

如果在研究中使用了 SAM 3（或本工具），请引用：

```bibtex
@misc{carion2025sam3segmentconcepts,
      title={SAM 3: Segment Anything with Concepts},
      author={Nicolas Carion and Laura Gustafson and Yuan-Ting Hu and Shoubhik Debnath and Ronghang Hu and Didac Suris and Chaitanya Ryali and Kalyan Vasudev Alwala and Haitham Khedr and Andrew Huang and Jie Lei and Tengyu Ma and Baishan Guo and Arpit Kalla and Markus Marks and Joseph Greer and Meng Wang and Peize Sun and Roman Rädle and Triantafyllos Afouras and Effrosyni Mavroudi and Katherine Xu and Tsung-Han Wu and Yu Zhou and Liliane Momeni and Rishi Hazra and Shuangrui Ding and Sagar Vaze and Francois Porcher and Feng Li and Siyuan Li and Aishwarya Kamath and Ho Kei Cheng and Piotr Dollár and Nikhila Ravi and Kate Saenko and Pengchuan Zhang and Christoph Feichtenhofer},
      year={2025},
      eprint={2511.16719},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.16719},
}
```

## License

本仓库衍生自 Meta 的 [SAM 3](https://github.com/facebookresearch/sam3)，沿用其 [SAM License](LICENSE) 发布；`app/` 标注工具部分遵循同一许可。上游原始文档与贡献指南见[上游仓库](https://github.com/facebookresearch/sam3)与 [CONTRIBUTING.md](CONTRIBUTING.md)。
