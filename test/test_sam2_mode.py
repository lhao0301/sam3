"""Automated end-to-end test for the SAM2-task-mode annotation backend.

Covers the three-stage workflow:
  1. object selection: point / box / point+box, any frame, any number
  2. full-video propagation (forward + backward "both")
  3. preview + correction: frame mask fetch, mid-video refinement,
     re-propagation, post-propagation new-object rejection (409),
     reset-tracking, undo, remove
"""
import base64
import io
import json
import sys
import time

import requests
import websocket

BASE = "http://localhost:8000"
WS_BASE = "ws://localhost:8000"
VIDEO = "/data/luohao/project/sam3/assets/videos/bedroom.mp4"

PASSED = 0


def check(name, cond, detail=""):
    global PASSED
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        print("TEST ABORTED")
        sys.exit(1)
    PASSED += 1


def mask_pixels(mask_png):
    """Count non-transparent pixels in a base64 mask PNG."""
    if not mask_png:
        return 0
    from PIL import Image
    import numpy as np
    img = Image.open(io.BytesIO(base64.b64decode(mask_png)))
    arr = np.array(img)
    return int((arr[..., 3] > 0).sum()) if arr.shape[-1] == 4 else int((arr > 0).any(-1).sum())


def add_prompt(sid, frame_idx, obj_id, points=None, point_labels=None, box=None):
    payload = {"session_id": sid, "frame_idx": frame_idx, "obj_id": obj_id}
    if points:
        payload["points"] = points
        payload["point_labels"] = point_labels
    if box:
        payload["box"] = box
    return requests.post(f"{BASE}/api/prompt/add", json=payload, timeout=180)


def ws_propagate(sid, direction="both", expect_frames=True):
    """Run a WS propagation and collect per-frame results."""
    ws = websocket.create_connection(f"{WS_BASE}/ws/propagate/{sid}", timeout=600)
    ws.send(json.dumps({"type": "start", "direction": direction}))
    frames = {}
    msg = None
    while True:
        msg = json.loads(ws.recv())
        if msg["type"] == "frame_result":
            frames[msg["frame_index"]] = msg
        elif msg["type"] == "propagation_complete":
            break
        elif msg["type"] == "error":
            ws.close()
            return None, msg
    ws.close()
    return frames, msg


t0 = time.time()

# ── Stage 1: session + multi-frame / multi-object selection ──────────
r = requests.post(f"{BASE}/api/session/start",
                  json={"video_path": VIDEO}, timeout=180)
check("session/start", r.status_code == 200, f"({r.status_code}) {r.text[:120]}")
sid = r.json()["session_id"]
num_frames = r.json()["num_frames"]
check("session metadata", num_frames == 200 and r.json()["orig_height"] == 540)

# point-initialized object on frame 0
r = add_prompt(sid, 0, 1, points=[[0.35, 0.5]], point_labels=[1])
d = r.json()
check("add point obj1@f0", r.status_code == 200 and d.get("obj_ids") == [1]
      and d.get("new_obj_ids") == [1] and d.get("mask_png"),
      f"boxes={d.get('boxes_xywh')}")
check("obj1 mask non-empty", mask_pixels(d.get("mask_png")) > 500,
      f"px={mask_pixels(d.get('mask_png'))}")

# box-initialized object on the same frame
r = add_prompt(sid, 0, 2, box=[0.25, 0.25, 0.3, 0.3])
d = r.json()
check("add box obj2@f0", r.status_code == 200 and d.get("obj_ids") == [1, 2],
      f"obj_ids={d.get('obj_ids')}")
check("obj2 in combined mask", mask_pixels(d.get("mask_png"))
      > mask_pixels(None) and d.get("mask_png"))

# combined point+box object on a LATE frame (frame 100)
r = add_prompt(sid, 100, 3, points=[[0.5, 0.5]], point_labels=[1],
               box=[0.45, 0.4, 0.2, 0.2])
d = r.json()
check("add point+box obj3@f100", r.status_code == 200
      and d.get("obj_ids") == [1, 2, 3], f"obj_ids={d.get('obj_ids')}")

# confirm is pure bookkeeping
r = requests.post(f"{BASE}/api/prompt/confirm",
                  json={"session_id": sid}, timeout=30)
check("confirm", r.status_code == 200 and r.json().get("confirmed") is True)

# ── Stage 2: full-video propagation (both) ────────────────────────────
t1 = time.time()
frames, msg = ws_propagate(sid, "both")
check("ws propagate both", frames is not None
      and msg["type"] == "propagation_complete",
      f"frames={len(frames) if frames else 0}, {time.time()-t1:.1f}s")
check("all frames covered", len(frames) == num_frames,
      f"{len(frames)}/{num_frames}")

f0 = frames.get(0, {})
fmid = frames.get(100, {})
fend = frames.get(num_frames - 1, {})
check("frame 0 has all objects", sorted(f0.get("out_obj_ids", [])) == [1, 2, 3],
      f"{f0.get('out_obj_ids')}")
check("mid frame has all objects", sorted(fmid.get("out_obj_ids", [])) == [1, 2, 3],
      f"{fmid.get('out_obj_ids')}")
check("last frame has all objects", sorted(fend.get("out_obj_ids", [])) == [1, 2, 3],
      f"{fend.get('out_obj_ids')}")
check("frame 199 mask non-empty", mask_pixels(fend.get("mask_png")) > 500)

# ── Stage 3: preview + correction ─────────────────────────────────────
# REST mask fetch matches WS results
r = requests.get(f"{BASE}/api/mask/{sid}/50", timeout=60)
d = r.json()
check("GET mask f50", r.status_code == 200 and sorted(d.get("obj_ids", [])) == [1, 2, 3]
      and d.get("mask_png"))

# mid-video refinement on a bad frame (obj 1, frame 50): add a negative click
r = add_prompt(sid, 50, 1, points=[[0.35, 0.5], [0.38, 0.45]],
               point_labels=[1, 0])
check("refine obj1@f50", r.status_code == 200
      and r.json().get("obj_ids") == [1, 2, 3])

# re-propagate (full, as designed)
t2 = time.time()
frames2, msg = ws_propagate(sid, "both")
check("re-propagate after refine", frames2 is not None
      and len(frames2) == num_frames, f"{time.time()-t2:.1f}s")

# post-propagation: new object must be rejected with 409
r = add_prompt(sid, 10, 9, points=[[0.6, 0.6]], point_labels=[1])
check("new obj after propagate -> 409", r.status_code == 409,
      f"({r.status_code}) {r.text[:100]}")

# refinement of an EXISTING object after propagation is still allowed
r = add_prompt(sid, 60, 2, points=[[0.3, 0.35]], point_labels=[1])
check("refine existing obj after propagate ok", r.status_code == 200)

# tracking state is visible via session info (propagation has started)
r = requests.get(f"{BASE}/api/session/{sid}/info", timeout=30)
check("session info (tracking_started=True)", r.status_code == 200
      and r.json().get("tracking_started") is True)

# reset tracking -> new object accepted again
r = requests.post(f"{BASE}/api/session/{sid}/reset-tracking", timeout=180)
check("reset-tracking", r.status_code == 200
      and r.json().get("num_prompts_replayed") == 5,
      f"{r.text[:120]}")
r = add_prompt(sid, 10, 9, points=[[0.6, 0.6]], point_labels=[1])
check("new obj after reset ok", r.status_code == 200
      and r.json().get("obj_ids") == [1, 2, 3, 9])

# undo a pending (never-confirmed) new object on frame 120
r = add_prompt(sid, 120, 5, points=[[0.7, 0.3]], point_labels=[1])
check("add obj5@f120 (to undo)", r.status_code == 200)
r = requests.delete(f"{BASE}/api/prompt/{sid}/pending?obj_id=5&frame_idx=120",
                    timeout=120)
check("undo obj5@f120", r.status_code == 200)
r = requests.get(f"{BASE}/api/mask/{sid}/120", timeout=60)
check("obj5 gone after undo", 5 not in r.json().get("obj_ids", []),
      f"obj_ids={r.json().get('obj_ids')}")

# remove a confirmed object entirely
r = requests.delete(f"{BASE}/api/prompt/{sid}/2", timeout=120)
check("remove obj2", r.status_code == 200)
r = requests.get(f"{BASE}/api/mask/{sid}/0", timeout=60)
d = r.json()
check("obj2 gone after remove, obj1 visible via temp fallback",
      2 not in d.get("obj_ids", []) and 1 in d.get("obj_ids", []),
      f"obj_ids={d.get('obj_ids')}")

# after reset (and no propagation since), tracking is not started
r = requests.get(f"{BASE}/api/session/{sid}/info", timeout=30)
check("session info (tracking_started=False after reset)", r.status_code == 200
      and r.json().get("tracking_started") is False)

# close
r = requests.delete(f"{BASE}/api/session/{sid}", timeout=120)
check("close session", r.status_code == 200)

print(f"\nALL {PASSED} CHECKS PASSED in {time.time()-t0:.1f}s")
