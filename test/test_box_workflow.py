"""Box-focused end-to-end test for the SAM2-task-mode annotation backend.

Validates the three requested capabilities:
  1. box annotation on one or more frames, one or more boxes per frame,
     with per-frame preview (prompt-frame mask render)
  2. segmentation / tracking propagation built on those box prompts
  3. frame-targeted correction of the propagation results
     (box refinement, point refinement, re-propagation, reset+add)

Run: conda run -n sam3 python test/test_box_workflow.py
"""
import base64
import io
import json
import sys
import time
from pathlib import Path

import requests
import websocket

BASE = "http://localhost:8000"
WS_BASE = "ws://localhost:8000"
VIDEO = str(Path(__file__).resolve().parents[1] / "assets" / "videos" / "bedroom.mp4")

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


def get_mask_obj_ids(sid, frame_idx):
    r = requests.get(f"{BASE}/api/mask/{sid}/{frame_idx}", timeout=60)
    d = r.json()
    return r.status_code, d.get("obj_ids", []), d.get("mask_png")


def ws_propagate(sid, direction="both"):
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

# ════ Capability 1: box annotation (single frame / multi frame / multi box) ════
r = requests.post(f"{BASE}/api/session/start",
                  json={"video_path": VIDEO}, timeout=180)
check("C1: session/start", r.status_code == 200, f"({r.status_code}) {r.text[:120]}")
sid = r.json()["session_id"]
num_frames = r.json()["num_frames"]

# T1.1 single box on frame 20
r = add_prompt(sid, 20, 1, box=[0.28, 0.28, 0.24, 0.42])
d = r.json()
check("C1: single box obj1@f20", r.status_code == 200
      and d.get("new_obj_ids") == [1] and d.get("obj_ids") == [1]
      and d.get("boxes_xywh"), f"boxes={d.get('boxes_xywh')}")
p1 = mask_pixels(d.get("mask_png"))
check("C1: obj1 prompt-frame mask non-empty", p1 > 500, f"px={p1}")

# T1.2 second box on the SAME frame (obj2) -> combined per-frame preview
r = add_prompt(sid, 20, 2, box=[0.55, 0.30, 0.20, 0.35])
d = r.json()
check("C1: second box obj2@f20 (same frame)", r.status_code == 200
      and d.get("obj_ids") == [1, 2], f"obj_ids={d.get('obj_ids')}")
p2 = mask_pixels(d.get("mask_png"))
check("C1: f20 combined preview covers both", p2 >= p1, f"px {p1} -> {p2}")

# T1.3 box on ANOTHER frame (obj3 @ f100) -> multi-frame annotation.
# obj1/obj2 also get a prediction on f100 via shared memory attention,
# so the response covers ALL objects inferred on this frame.
r = add_prompt(sid, 100, 3, box=[0.42, 0.35, 0.22, 0.30])
d = r.json()
check("C1: box obj3@f100 (multi-frame)", r.status_code == 200
      and d.get("new_obj_ids") == [3] and d.get("obj_ids") == [1, 2, 3],
      f"obj_ids={d.get('obj_ids')}")

# T1.4 per-frame preview (GET /api/mask, the persisted store): a prompt
#      frame keeps only its OWN objects (obj1/obj2's transient f100
#      prediction from the add response is not persisted; propagation
#      will recompute it); a non-prompted frame has no mask yet
st, ids, png = get_mask_obj_ids(sid, 20)
check("C1: preview f20 -> [1, 2]", st == 200 and ids == [1, 2] and png,
      f"obj_ids={ids}")
st, ids, png = get_mask_obj_ids(sid, 100)
check("C1: preview f100 -> [3]", st == 200 and ids == [3] and png,
      f"obj_ids={ids}")
st, ids, _ = get_mask_obj_ids(sid, 50)
check("C1: preview f50 (not prompted) -> empty", st == 200 and ids == [],
      f"obj_ids={ids}")

# ════ Capability 2: propagation built on the box prompts ════
t1 = time.time()
frames, msg = ws_propagate(sid, "both")
check("C2: propagate both completes", frames is not None
      and msg["type"] == "propagation_complete",
      f"frames={len(frames) if frames else 0}, {time.time()-t1:.1f}s")
check("C2: all frames covered", len(frames) == num_frames,
      f"{len(frames)}/{num_frames}")
for fi in (0, 20, 60, 100, num_frames - 1):
    got = sorted(frames.get(fi, {}).get("out_obj_ids", []))
    check(f"C2: frame {fi} has all 3 objects", got == [1, 2, 3], f"{got}")
check("C2: last-frame mask non-empty",
      mask_pixels(frames.get(num_frames - 1, {}).get("mask_png")) > 500)

# ════ Capability 3: frame-targeted correction ════
# C3-a: brand-new object after propagation must be rejected (409)
r = add_prompt(sid, 10, 4, box=[0.6, 0.55, 0.15, 0.2])
check("C3: new box after propagate -> 409", r.status_code == 409,
      f"({r.status_code}) {r.text[:80]}")

# C3-b: box refinement of an EXISTING object on a chosen frame (obj1 @ f60)
r = add_prompt(sid, 60, 1, box=[0.25, 0.25, 0.30, 0.48])
d = r.json()
check("C3: box-refine obj1@f60", r.status_code == 200
      and d.get("obj_ids") == [1, 2, 3], f"obj_ids={d.get('obj_ids')}")

# C3-c: point refinement of an existing object (obj3 @ f120, +point/-point)
r = add_prompt(sid, 120, 3, points=[[0.5, 0.5], [0.55, 0.45]],
               point_labels=[1, 0])
check("C3: point-refine obj3@f120", r.status_code == 200
      and r.json().get("obj_ids") == [1, 2, 3])

# C3-d: re-propagation applies the corrections session-wide
t2 = time.time()
frames2, msg = ws_propagate(sid, "both")
check("C3: re-propagate after corrections", frames2 is not None
      and len(frames2) == num_frames, f"{time.time()-t2:.1f}s")
st, ids, png = get_mask_obj_ids(sid, 60)
check("C3: corrected frame f60 still consistent", st == 200
      and ids == [1, 2, 3] and png, f"obj_ids={ids}")

# C3-e: reset tracking keeps prompts, then a NEW box is accepted again
r = requests.post(f"{BASE}/api/session/{sid}/reset-tracking", timeout=180)
check("C3: reset-tracking replays prompts", r.status_code == 200
      and r.json().get("num_prompts_replayed") == 5,
      f"{r.text[:120]}")
r = add_prompt(sid, 10, 4, box=[0.6, 0.55, 0.15, 0.2])
check("C3: new box after reset ok", r.status_code == 200
      and r.json().get("obj_ids") == [1, 2, 3, 4],
      f"obj_ids={r.json().get('obj_ids')}")

# close
r = requests.delete(f"{BASE}/api/session/{sid}", timeout=120)
check("close session", r.status_code == 200)

print(f"\nALL {PASSED} CHECKS PASSED in {time.time()-t0:.1f}s")
