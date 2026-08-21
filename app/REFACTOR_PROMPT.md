# SAM3 视频标注系统前端重构任务 Prompt

> 本文档是自包含的任务规范，供后续会话/agent 无缝继续执行。
> 当前进度：**重构与测试已全部完成**（见第九节实测结果）。

## 一、任务目标

重构 `/data/luohao/project/sam3` 项目中 `app/` 目录下的前后端标注应用，实现四大功能：

1. **视频播放控制**：支持播放/暂停预览视频，支持上一帧、下一帧切换
2. **多类型 Prompt 标注**：支持 text、正/负 point、box 三种 prompt 类型；
   人工选择 prompt 类型，其中 point 与 box 互斥；添加 prompt 后立即分割并
   预览当前帧分割效果；人工确认时填写类别名（class name）和 tracklet id
   （等价于 SAM3 中的 obj_id / tracklet id 概念）
3. **方向可选的跟踪**：增加跟踪按钮，支持选择 前向 / 后向 / 双向 三种
   分割+跟踪模式
4. **帧导航缩略图条（Filmstrip）**：在播放进度条（帧滑块）下方显示一排
   缩略预览帧；拖动进度条或切换帧时，当前选中帧所对应的缩略图四周显示
   蓝色边框高亮；同时清晰显示当前选中帧的帧号

## 二、项目背景与约束

- 工作区：`/data/luohao/project/sam3`
- **严禁修改 `sam3/` 原始库代码**；所有改动仅限 `app/` 目录：
  - 后端：`app/server/`（FastAPI：app.py、sam3_service.py、session_manager.py、
    ws_manager.py、frame_utils.py、mask_utils.py）
  - 前端：`app/frontend/`（index.html、css/style.css、js/*.js，纯静态无构建工具，
    由后端挂载于 `/static`）
- 启动方式：`conda run -n sam3 uvicorn app.server.app:app --host 0.0.0.0 --port 8000`
  （conda 位于 `/root/miniconda3/bin/conda`）
- 现有功能必须保留：视频上传、会话管理、feat编码（`POST /api/encode/{session_id}`）、
  mask 缓存与逐帧显示、WebSocket 流式传播、导出视频
- 代码风格与现有代码保持一致（中文 UI 文案、英文注释/日志）

## 三、已确认的设计决策（用户已确认，勿更改）

| 决策点 | 结论 |
|---|---|
| 正/负点输入方式 | 工具栏「正点/负点」切换按钮，选中后左键点击画布打点 |
| 预览与确认流程 | **预览即提交**：「分割预览」时即用用户填写的 tracklet id 调 `add_prompt` 并显示当前帧 mask；「确认」= 保留该 obj；「撤销」= 调 `remove_object` 删除。只调一次模型 |
| text 模式的 tracklet id | 自动分配（text 检测可能返回多实例），在目标列表中展示，类别名默认取输入文本、可修改 |
| 类别持久化 | 暂不导出，类别名仅维护在前端 tracklet 列表中显示 |
| 每次提交目标数 | 每次提交针对一个目标：box 模式画布同一时间仅保留一个待提交框（画新框替换旧框）；point 模式一组正负点描述同一目标 |
| 缩略图条实现 | 后端新增缩略图端点（cv2 缩放，省带宽）；从视频均匀采样 12 张缩略图；每张缩略图左下角角标显示其代表的帧号；点击缩略图跳转到对应帧；当前帧落入的缩略图区间项加 `.active` 类（2px 蓝色边框）高亮；滑块旁保留「帧: X / Y」实时帧号显示 |

## 四、SAM3 后端能力调研结论（已验证，无需改动 sam3/ 库）

- `handle_request` 的 `add_prompt` 请求原生支持字段：`text`、`points` +
  `point_labels`（1=正 0=负）、`bounding_boxes` + `bounding_box_labels`
  （box label 即 obj_id/tracklet id）、`obj_id`（point prompt 指定 id）、
  `rel_coordinates`；返回 `{"frame_index", "outputs": {out_obj_ids,
  out_binary_masks, out_boxes_xywh}}`，即提交后立即可渲染当前帧分割结果
- `handle_stream_request` 的 `propagate_in_video` 原生支持
  `propagation_direction ∈ {"both", "forward", "backward"}`（forward =
  提示帧→末尾，backward = 提示帧→开头，both = 先 forward 再 backward）
- `remove_object` 请求支持按 `obj_id` 删除目标，返回剩余目标在当前帧的 masks
- 参见 `sam3/model/sam3_base_predictor.py` L159-225（add_prompt）、
  L274-334（propagate_in_video 方向校验与分发）

## 五、交互流程（状态机）

```
上传视频 → 会话启动 → (可选 feat编码)
→ 关键帧导航（播放/暂停、上一帧/下一帧、滑块、键盘：空格=播放暂停、←/→=切帧）
   - 滑块下方渲染缩略图条：会话启动后均匀采样 12 帧（含首帧/尾帧），
     通过缩略图端点懒加载，缓存 Map<thumbIdx, dataURL>
   - 交互联动：
     · 拖动滑块 / 点击缩略图 / 上一帧下一帧 / 播放中 → 实时计算当前帧
       落入的缩略图区间，该项加蓝色边框高亮并滚动到可见
     · 点击某缩略图 → player.loadFrame(该缩略图代表帧号)
   - 帧号显示：滑块旁「帧: X / Y」随切帧实时更新；每张缩略图左下角
     固定角标显示其代表帧号
→ 选择 Prompt 模式（radio：文本/点/框，三选一互斥，切换时清空 pending 内容）
→ 输入 prompt：
   - 文本模式：文本输入框（如 "the red car"）
   - 点模式：正/负点类型切换 + 画布点击打点（正点绿实心圆、负点红叉），
     支持撤销上一点、清空
   - 框模式：拖拽画框（同时仅一个待提交框，画新框替换旧框）
→ 填写 类别名 + tracklet id（自动建议 max(已有id)+1；text 模式隐藏 id 输入）
→ 点击「分割预览」→ POST /api/prompt/add → 当前帧叠加显示 mask
   ├─ 「确认」→ 目标加入 tracklet 列表（id、类别、颜色、prompt 类型、删除按钮）
   └─ 「撤销」→ DELETE /api/prompt/{sid}/{obj_id} → 移除该 obj，刷新当前帧 mask
→ 可换帧/换模式继续累积多个 tracklet
→ 选择跟踪方向（前向/后向/双向，默认双向）→「开始跟踪」→ WebSocket 流式
   接收逐帧 mask → 完成后可播放预览（逐帧叠加 mask）
```

## 六、文件级改动清单

### 后端（app/server/）

1. **`sam3_service.py`** 【已完成，勿重复改】
   - `add_prompt` 已扩展：text / points / point_labels / obj_id 参数并透传
   - 已新增 `remove_object(session_id, obj_id, frame_idx)` 方法
   - `propagate_in_video` 已增加 `direction` 参数（校验后透传
     `propagation_direction`）
2. **`app.py`**
   - 重写 `POST /api/prompt/add`：
     - 请求体：`{session_id, frame_idx, prompt_type: "box"|"point"|"text",
       boxes, box_labels, points, point_labels, obj_id, text}`（按
       prompt_type 校验必填字段）
     - 响应：调用 `render_masks_only`（mask_utils.py 已有）渲染当前帧
       mask，返回 `{session_id, frame_idx, obj_ids, boxes_xywh, mask_png}`
       （base64 PNG），不再用 `_to_jsonable` 序列化巨大 mask 数组
   - 新增 `DELETE /api/prompt/{session_id}/{obj_id}?frame_idx=N`：透传
     remove_object，响应同样返回剩余目标的 `{obj_ids, mask_png}`
   - 新增 `GET /api/thumb/{session_id}/{frame_idx}?w=160`（放在
     `GET /api/frame/...` 端点旁）：
     - 读取该帧 JPEG（路径规则与 `/api/frame` 一致：`{frame_idx:05d}.jpg`），
       用 cv2 等比缩放至宽度 w（默认 160，上限 320），JPEG 编码后返回
       `Response(media_type="image/jpeg")`
     - 会话不存在或帧不存在时返回 404（与 /api/frame 一致）
     - 调用 `info.touch()` 保持会话活跃
   - 更新 `/manifest` 端点与前端文件列表：移除 `js/box-drawer.js`，
     加入 `js/prompt-canvas.js`
3. **`ws_manager.py`**
   - `handle_connection` 解析 start 消息中的 `direction`（默认 "both"，
     校验合法值），传入 `_run_propagation`
   - `_run_propagation(websocket, session_id, sam3_service, session_manager,
     direction="both")`：透传给 `sam3_service.propagate_in_video`；
     `propagation_started` 消息携带 `direction`；进度 progress 保持
     `sent_count / total_frames`（可 clamp 至 1.0）

### 前端（app/frontend/）

4. **`js/prompt-canvas.js`**【新建，替代并删除 `js/box-drawer.js`】
   - 类 `PromptCanvas(drawCanvas, videoPlayer)`，参考原 BoxDrawer 的
     事件绑定/归一化坐标方式
   - `mode: "box"|"point"|"text"`，`setMode()` 切换时清空 pending；
     text 模式画布不响应鼠标
   - `pointType: 1|0`（`setPointType()` 工具栏切换）；`undoLastPoint()`、
     `clearPending()`、`hasPending()`
   - `getPending()` 按模式返回归一化数据：
     `{type:"box", boxes, boxLabels}` / `{type:"point", points, pointLabels}`
     / `{type:"text"}`（text 内容由 app.js 从输入框获取）
   - 渲染：pending 框黄色虚线；正点绿实心圆；负点红叉/空心圆；
     `onChange` 回调；`enable()/disable()`
   - 挂载 `window.PromptCanvas`
5. **`index.html`** —— 侧栏面板重构为：
   1. 上传视频（不变）；2. 特征编码（不变）；3. 帧导航（新增 ▶/⏸ 播放暂停
      按钮、保留 ◀上一帧/下一帧▶/滑块、速度下拉移入此面板、快捷键提示文案；
      `frame-slider` 下方新增 `<div id="filmstrip" class="filmstrip"></div>`
      缩略图条容器）；
      4. 标注 Prompt（模式 radio 文本/点/框；模式相关控件：文本输入 /
      正负点切换+撤销+清空 / 画框提示；类别名输入 + tracklet id 输入 +
      「分割预览」「确认」「撤销」按钮；状态文字）；
      5. 跟踪传播（方向 radio 前向/后向/双向默认双向 + 开始/取消 + 进度条）
   - 删除原「5. 预览结果」面板（播放控制已并入帧导航）
   - 右侧栏：tracklet 目标列表（替代原 object-list，含 id/类别/颜色/
     prompt 类型/删除按钮）+ result-info 状态
   - 底部 `<script>` 顺序：ws-client → video-player → prompt-canvas →
     mask-overlay → app
6. **`js/app.js`** —— 核心重写：
   - 状态：`tracklets: Map<id, {id, className, color, promptType, frameIdx}>`、
     `pendingObjIds`（最近一次预览返回的待确认 obj ids）
   - 分割预览：校验 → POST /api/prompt/add → `maskOverlay.storeMask +
     displayMask` 当前帧 → 激活确认/撤销按钮
   - 确认：text 模式将返回的所有 obj_ids 逐个加入 tracklets（className=
     文本，可改）；point/box 模式加入用户指定 id 的单个目标；清 pending；
     启用跟踪按钮；tracklet id 建议值 = max(已有id)+1
   - 撤销：对 pendingObjIds 逐个 DELETE → 用响应刷新当前帧 mask → 清 pending
   - tracklet 列表渲染与单项删除（删除后调 DELETE 并刷新 mask）
   - 播放控制：整合原 `_startPreview/_stopPreview` 到帧导航按钮
     （`btn-play` 切换 ▶/⏸）；播放时 `promptCanvas.disable()`，暂停恢复；
     键盘监听（空格/←/→，输入框聚焦时跳过）
   - 缩略图条：
     · `_buildFilmstrip()`：会话启动后计算采样帧号
       （`i * Math.floor((numFrames-1) / 11)`，i=0..11，含首尾帧），
       为每项创建 `<div class="film-thumb">`（内含 `<img>` 懒加载
       `/api/thumb/{sid}/{idx}` + 帧号角标），绑定点击跳转
     · `_updateFilmstripHighlight(frameIdx)`：在 `player.onFrameChange`
       回调中调用；计算 `thumbIdx = min(11, floor(frameIdx / step))`，
       切换 `.active` 类；若高亮项不在可视区域则 `scrollIntoView`
       （播放中节流，约每 200ms 一次，避免重排开销）
     · 会话切换/重置时清空 filmstrip 与缓存
   - 跟踪：方向 radio 读取 → `wsClient.startPropagation(direction)`；
     `propagation_started` 回调显示方向文案；其余 WS 回调逻辑沿用现有实现
7. **`js/ws-client.js`** —— `startPropagation(direction = "both")` 发送
   `{type: "start", direction}`
8. **`css/style.css`** —— 新增：
   - 模式选择器（.mode-selector 按钮组）、正/负点切换、确认区输入行
     （.confirm-row）、tracklet 列表项（.tracklet-item）、方向选择器
     （.direction-selector）、播放按钮等样式，与现有 panel/btn/状态色风格一致
   - 缩略图条：`.filmstrip`（横向 flex 布局、可横向滚动、gap 4px、上方与
     滑块间距 8px）；`.film-thumb`（相对定位、圆角 4px、overflow hidden、
     cursor pointer、缩略图等高约 54px 宽度按比例）；`.film-thumb img`
     填充显示；`.film-thumb .thumb-idx`（左下角角标，半透明黑底白字 11px）；
     `.film-thumb.active`（`outline: 2px solid #1e88e5; outline-offset: 1px`
     蓝色高亮边框，可加轻微放大 `transform: scale(1.04)`）；加载中占位样式
     （灰色背景 + 居占位符）

## 七、验证要求

- 后端：`python3 -m py_compile app/server/*.py` 通过；
  `/root/miniconda3/bin/conda run -n sam3 python -c "import app.server.app"`
  导入成功
- 前端：`node --check` 逐一通过所有 js 文件；确认 `box-drawer.js` 已删除、
  index.html 与 /manifest 均不再引用它
- 功能自检清单：
  - 上传→编码→三种模式（text/point/box）各走一遍「预览→确认/撤销」
  - 三种方向（前向/后向/双向）跟踪→播放暂停/逐帧切换
  - tracklet 列表单项删除后 mask 正确刷新
  - 会话启动后缩略图条渲染 12 张、含首尾帧、角标帧号正确
  - 拖动滑块/点击缩略图/键盘切帧/播放中：蓝色高亮实时跟随正确区间
  - 点击缩略图正确跳帧；`/api/thumb` 404 场景（无效帧号）不崩溃
  - 窗口 resize 后布局不错乱
- 完成后清理 `__pycache__`

## 八、执行边界

- 若发现 SAM3 add_prompt 对 obj_id/box_labels 的实际行为与上述调研结论不符
  （如 obj_id 未生效），在 app/server 层适配，**不得修改 sam3/ 库**
- 不新增第三方前端依赖，不引入构建工具
- 遇到与本 prompt 冲突的旧代码逻辑（如旧 boxes/labels 协议），以本 prompt 为准

## 九、实现与验证结果（实测，2026-08-20）

### 测试结果

- 静态：`py_compile` 全部通过；`node --check` 全部 JS 通过；无 box-drawer 残留引用
- REST 端到端（bedroom.mp4, 200 帧）：**32/32 通过**
- WebSocket 方向传播：**7/7 通过**（forward 帧 100→102 递增；backward 帧 99→97
  递减；direction 字段回传；cancel 正常；非法方向返回 error）

### 实测发现的 SAM3 (v1) 行为与适配（重要）

1. **检测路径每次 add_prompt 无条件 reset_state**（text/box 提示会清空全部
   tracker 状态），因此新增 `app/server/prompt_history.py`（PromptHistory）
   维护已确认 prompt 集合，每次提交/撤销/删除时全量重放（replay）；
   reset 会清除 point 目标，重放时逐一恢复；`reset_session` 保留 ViT 缓存
2. **box_labels 语义为正/负提示**（0=负框会抑制检出，必须传 1）
3. **检测目标 obj id 由模型自动分配**（从 0 递增），box_labels/obj_id 不会
   映射为输出 id → tracklet id 仅 point 模式可人工指定；box/text 的 id
   为自动分配（前端在确认后展示实际 id）
4. **SAM3 v1 每会话仅支持单个 box visual prompt**（reset 后首个调用多 box
   会 RuntimeError）→ box 采用"单活跃、后确认替换"语义（与 text 一致）；
   point 可多个（obj_id 增量累积）
5. numpy/tensor 的 `or` 真值陷阱：`out_obj_ids or []` 在单元素值为 0 的
   tensor 上会错误走空分支，必须用 `is not None` 判断
6. asyncio 取消陷阱：被取消的 task 内不能再 `await send_json`（会立即
   再抛 CancelledError）→ cancelled 消息改由连接主循环发送

### 最终 API 形态（与第六节的差异）

- `POST /api/prompt/add`：预览即提交（含重放），响应含 `new_obj_ids`
  （本次新增 id，前端 pending 集合用）
- `POST /api/prompt/confirm`：把 pending 落入已确认集合（无模型调用）
- `DELETE /api/prompt/{sid}/pending`：撤销预览（重放恢复）
- `DELETE /api/prompt/{sid}/{obj_id}`：删除已确认 tracklet（重放恢复）
- `GET /api/thumb/...`、`/api/encode/...`、WS direction 均与第六节一致
