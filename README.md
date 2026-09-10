# SAM 3 Video Annotation Tool

[English](README_EN.md) | 简体中文

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

## 快速开始

```bash
# 1. 安装后端依赖
pip install -r app/server/requirements.txt

# 2. 准备 SAM 3 checkpoint（checkpoints/ 目录，如 checkpoints/sam3/）

# 3. 启动服务（GPU 自动选择；SAM3_GPU=<index> 可指定）
uvicorn app.server.app:app --host 0.0.0.0 --port 8000
```

浏览器打开 `http://<server>:8000`，按「上传视频 → 画框/点提示 → 预览 → 确认 → 传播 → 导出」流程标注。

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
python -m pytest test/ -v          # Python：box 工作流、SAM2 模式、推理优化
node test/test_prompt_canvas_frame.js   # 前端：提示画布帧逻辑
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

## License

本仓库衍生自 Meta 的 [SAM 3](https://github.com/facebookresearch/sam3)，沿用其 [SAM License](LICENSE)；`app/` 标注工具部分遵循同一许可发布。上游原始 README 与贡献指南见 [CONTRIBUTING.md](CONTRIBUTING.md)。
