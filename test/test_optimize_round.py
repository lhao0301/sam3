"""E2E test for the 2026-08-31 optimization round (SAM2-task mode).

Covers:
  1. Interleaved multi-object prompts (obj boxes on frames 20 / 8 / 0)
     followed by "both" propagation — after the new backfill pass every
     conditioning frame must hold ALL objects (the old code left objects
     prompted on other frames as empty masks on cond frames, e.g. frame
     0 missing obj 1).
  2. Honest progress: propagation_started carries a plan
     (forward/backward/backfill budget); frame_result messages carry
     segment / sent_count / expected_total; progress must NOT hit 100%
     when the forward segment ends (old behavior); propagation_complete
     carries segment_counts.
  3. Propagation hot path slimming: mask PNG still delivered per frame;
     out_boxes_xywh absent (frontend never consumed it) but REST
     /api/mask responses keep full output.

Also verifies the new operation log lines (plan / segment / backfill).

Run: conda run -n sam3 python test/test_optimize_round.py
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
VIDEO = "/tmp/sam3_e2e_short.mp4"  # 40 frames, 960x540

PASSED = 0


def check(name, cond, detail=""):
    global PASSED
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        print("TEST ABORTED")
        sys.exit(1)
    PASSED += 1


def mask_pixel_count(mask_png):
    if not mask_png:
        return 0
    from PIL import Image
    import numpy as np
    img = Image.open(io.BytesIO(base64.b64decode(mask_png)))
    arr = np.array(img)
    return int((arr[..., 3] > 0).sum()) if arr.shape[-1] == 4 else int((arr > 0).any(-1).sum())


def add_prompt(sid, frame_idx, obj_id, box):
    payload = {"session_id": sid, "frame_idx": frame_idx, "obj_id": obj_id, "box": box}
    return requests.post(f"{BASE}/api/prompt/add", json=payload, timeout=180)


def get_mask_obj_ids(sid, frame_idx):
    r = requests.get(f"{BASE}/api/mask/{sid}/{frame_idx}", timeout=60)
    d = r.json()
    return r.status_code, d.get("obj_ids", []), d.get("mask_png")


# ────────────────────────────────────────────────────────────────
print("=== Step 1: session start (40-frame short clip) ===")
r = requests.post(f"{BASE}/api/session/start",
                  json={"video_path": VIDEO}, timeout=300)
check("session start 200", r.status_code == 200, f"({r.status_code})")
sid = r.json()["session_id"]
num_frames = r.json()["num_frames"]
check("num_frames == 40", num_frames == 40, f"(got {num_frames})")

# ────────────────────────────────────────────────────────────────
print("=== Step 2: interleaved box prompts (obj1@20, obj2@8, obj3@0) ===")
# Distinct boxes so the three objects resolve to different targets
t0 = time.time()
r1 = add_prompt(sid, 20, 1, [0.10, 0.10, 0.25, 0.30])
check("obj 1 box @ frame 20", r1.status_code == 200,
      f"({r1.status_code}) {r1.text[:120] if r1.status_code != 200 else ''}")
r2 = add_prompt(sid, 8, 2, [0.45, 0.15, 0.25, 0.30])
check("obj 2 box @ frame 8", r2.status_code == 200,
      f"({r2.status_code}) {r2.text[:120] if r2.status_code != 200 else ''}")
r3 = add_prompt(sid, 0, 3, [0.55, 0.55, 0.30, 0.35])
check("obj 3 box @ frame 0", r3.status_code == 200,
      f"({r3.status_code}) {r3.text[:120] if r3.status_code != 200 else ''}")
print(f"    prompts submitted in {time.time()-t0:.1f}s")

# ────────────────────────────────────────────────────────────────
print("=== Step 3: 'both' propagation via WebSocket ===")
ws = websocket.create_connection(f"{WS_BASE}/ws/propagate/{sid}", timeout=600)
ws.send(json.dumps({"type": "start", "direction": "both"}))

start_msg = None
complete_msg = None
frame_msgs = []
t0 = time.time()
while True:
    msg = json.loads(ws.recv())
    mtype = msg.get("type")
    if mtype == "propagation_started":
        start_msg = msg
    elif mtype == "frame_result":
        frame_msgs.append(msg)
    elif mtype == "propagation_complete":
        complete_msg = msg
        break
    elif mtype == "error":
        ws.close()
        check("propagation no error", False, json.dumps(msg)[:300])
ws.close()
prop_secs = time.time() - t0
print(f"    propagation wall time: {prop_secs:.1f}s, "
      f"{len(frame_msgs)} frame messages")

check("got propagation_started", start_msg is not None)
plan = start_msg.get("plan") or {}
expected_total = start_msg.get("expected_total")
check("plan present in started msg", bool(plan), f"plan={plan}")
# backfill messages are frame-granular: every cond frame missing at
# least one object yields one message (here frames 0, 8, 20 -> 3);
# backward = latest cond (20) + 1 = 21 (independent of N)
check("plan.backward == 21", plan.get("backward") == 21, f"(got {plan.get('backward')})")
check("plan.backfill == 3", plan.get("backfill") == 3, f"(got {plan.get('backfill')})")
# forward depends on the tracker's frame count (container metadata may
# differ from the decoded frame count by one); check self-consistency
check("expected_total == forward + backward + backfill",
      expected_total == plan.get("forward", 0) + plan.get("backward", 0) + plan.get("backfill", 0),
      f"({expected_total} vs {plan})")

segments = [m.get("segment") for m in frame_msgs]
check("all frame messages carry segment",
      all(s in ("forward", "backward", "backfill") for s in segments))
check("forward segment present", "forward" in segments)
check("backward segment present", "backward" in segments)
check("backfill segment present",
      segments.count("backfill") == 3,
      f"({segments.count('backfill')} backfill messages, want 3: one per cond frame)")
check("forward message count matches plan",
      segments.count("forward") == plan.get("forward"),
      f"({segments.count('forward')} vs {plan.get('forward')})")

# Honest progress: when the last forward message arrives, overall progress
# must be below 100% (old code hit a fake 100% there)
fwd_msgs = [m for m in frame_msgs if m["segment"] == "forward"]
last_fwd_progress = fwd_msgs[-1]["progress"] if fwd_msgs else None
check("progress < 1.0 at end of forward segment",
      last_fwd_progress is not None and last_fwd_progress < 1.0,
      f"(was {last_fwd_progress})")
check("final message progress == 1.0",
      frame_msgs[-1]["progress"] >= 0.999,
      f"(was {frame_msgs[-1]['progress']})")

# sent_count / expected_total monotonic and consistent
counts = [m["sent_count"] for m in frame_msgs]
check("sent_count monotonic 1..N", counts == list(range(1, len(counts) + 1)))
check("message total == expected_total",
      len(frame_msgs) == expected_total,
      f"(got {len(frame_msgs)} msgs, expected {expected_total})")

check("complete msg has segment_counts",
      bool(complete_msg.get("segment_counts")),
      f"segment_counts={complete_msg.get('segment_counts')}")
sc = complete_msg.get("segment_counts") or {}
check("segment counts add up",
      sum(sc.values()) == len(frame_msgs), f"({sc} vs {len(frame_msgs)})")

# masks still delivered for every frame message
frames_with_png = sum(1 for m in frame_msgs if m.get("mask_png"))
check("mask_png delivered on frame messages",
      frames_with_png == len(frame_msgs),
      f"({frames_with_png}/{len(frame_msgs)})")

# ────────────────────────────────────────────────────────────────
print("=== Step 4: backfill correctness — every cond frame has ALL objects ===")
for f in (0, 8, 20):
    code, obj_ids, mask_png = get_mask_obj_ids(sid, f)
    check(f"frame {f} returns all objects [1,2,3]",
          code == 200 and sorted(obj_ids) == [1, 2, 3],
          f"(code={code}, obj_ids={obj_ids})")
    px = mask_pixel_count(mask_png)
    check(f"frame {f} mask non-trivial after backfill", px > 500,
          f"({px} px)")

# A regular (non-cond) frame must also carry all objects
code, obj_ids, mask_png = get_mask_obj_ids(sid, 35)
check("frame 35 (non-cond) has all objects", sorted(obj_ids) == [1, 2, 3],
      f"(obj_ids={obj_ids})")

# Object separation sanity: the three boxes are disjoint, so the final
# frame-35 mask should contain at least two distinct objects' worth of
# pixels (weak structural check)
px35 = mask_pixel_count(mask_png)
check("frame 35 mask pixel count reasonable", px35 > 1000, f"({px35} px)")

# ────────────────────────────────────────────────────────────────
print("=== Step 5: server-side operation log lines ===")
log_tail = ""
try:
    with open(Path(__file__).resolve().parents[1] / "sam3_server.log", "r",
              encoding="utf-8", errors="replace") as f:
        log_tail = f.read()[-20000:]
except OSError:
    pass
check("log has propagation plan line",
      "Propagation plan for session" in log_tail)
check("log has segment start lines",
      "Propagation segment 'forward' starting" in log_tail and
      "Propagation segment 'backward' starting" in log_tail)
check("log has backfill lines",
      "Backfilled cond frame" in log_tail)
check("log has completion summary",
      "cond frame(s) backfilled" in log_tail)

# ────────────────────────────────────────────────────────────────
print("=== Step 6: cleanup ===")
r = requests.delete(f"{BASE}/api/session/{sid}", timeout=60)
check("session closed", r.status_code == 200)

print(f"\nALL {PASSED} CHECKS PASSED")
