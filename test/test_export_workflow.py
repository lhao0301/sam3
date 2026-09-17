"""End-to-end test for the server-side three-artifact export feature.

Workflow under test: box prompt on the right-side person (black clothes) in
test.mp4 frame 0 -> confirm -> both-direction propagation -> POST
/api/export/{sid} -> poll status -> assert the three artifacts under
output/<video>_<session8>/:
  1. masks/00000.png ...  grayscale PNGs whose pixel values equal tracklet id
  2. annotated.mp4             composited H.264 video (browser-playable)
  3. annotation.json           per-frame object entries with pixel xyxy boxes

Run: conda run -n sam3 python test/test_export_workflow.py
"""
import json
import sys
import time
from pathlib import Path

import requests
import websocket

BASE = "http://localhost:8000"
WS_BASE = "ws://localhost:8000"
VIDEO = str(Path(__file__).resolve().parents[1] / "test.mp4")
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "output"

# Right-side person in black clothes on frame 0 (normalized xywh)
PERSON_BOX = [0.72, 0.04, 0.20, 0.93]

PASSED = 0


def check(name, cond, detail=""):
    global PASSED
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        print("TEST ABORTED")
        sys.exit(1)
    PASSED += 1


def main():
    t0 = time.time()

    # ── Session start ────────────────────────────────────────────────
    r = requests.post(
        f"{BASE}/api/session/start",
        json={"video_path": VIDEO, "video_filename": "test.mp4"},
        timeout=300,
    )
    check("start session", r.status_code == 200, r.text[:200])
    sid = r.json()["session_id"]
    num_frames = r.json()["num_frames"]
    print(f"  session={sid[:8]} frames={num_frames}")

    try:
        # ── Box prompt (right-side person in black) on frame 0 ───────
        r = requests.post(
            f"{BASE}/api/prompt/add",
            json={
                "session_id": sid,
                "frame_idx": 0,
                "obj_id": 1,
                "box": PERSON_BOX,
                "className": "person",
            },
            timeout=180,
        )
        check("add box prompt", r.status_code == 200, r.text[:200])
        check(
            "prompt registered obj 1",
            1 in r.json().get("obj_ids", []),
            f"obj_ids={r.json().get('obj_ids')}",
        )

        r = requests.post(
            f"{BASE}/api/prompt/confirm",
            json={"session_id": sid, "obj_ids": [1], "className": "person"},
            timeout=30,
        )
        check("confirm prompt", r.status_code == 200)

        # ── Both-direction propagation over WebSocket ─────────────────
        ws = websocket.create_connection(f"{WS_BASE}/ws/propagate/{sid}", timeout=600)
        ws.send(json.dumps({"type": "start", "direction": "both"}))
        frame_results = 0
        while True:
            msg = json.loads(ws.recv())
            if msg["type"] == "frame_result":
                frame_results += 1
            elif msg["type"] == "propagation_complete":
                print(f"  propagation complete: {msg['sent_count']}/{msg['expected_total']}")
                break
            elif msg["type"] == "error":
                ws.close()
                check("propagation", False, msg["message"])
        ws.close()
        check("propagation streamed", frame_results > 0, f"frames={frame_results}")

        # ── Export: idempotent start + status polling ─────────────────
        r = requests.post(f"{BASE}/api/export/{sid}", timeout=30)
        check("start export", r.status_code == 200, r.text[:200])
        first = r.json()

        r2 = requests.post(f"{BASE}/api/export/{sid}", timeout=30)
        check(
            "export start idempotent",
            r2.status_code == 200 and not r2.json().get("started_new", False),
            "second call accepted while running",
        )

        deadline = time.time() + 300
        status = first
        while status.get("running") and time.time() < deadline:
            time.sleep(1.0)
            status = requests.get(
                f"{BASE}/api/export/{sid}/status", timeout=30
            ).json()
        check(
            "export finished",
            not status.get("running") and not status.get("last_error"),
            f"progress={status.get('progress')} err={status.get('last_error')}",
        )
        out_dir = Path(status["output_dir"])
        print(f"  output_dir={out_dir} ({time.time() - t0:.1f}s total)")

        # ── Artifact 1: mask PNG sequence, pixel value == tracklet id ─
        masks_dir = out_dir / "masks"
        pngs = sorted(masks_dir.glob("*.png"))
        check(
            "mask count matches frames",
            len(pngs) == num_frames,
            f"{len(pngs)} pngs vs {num_frames} frames",
        )

        import cv2
        import numpy as np

        f0 = cv2.imread(str(masks_dir / "00000.png"), cv2.IMREAD_UNCHANGED)
        f_mid = cv2.imread(str(masks_dir / f"{num_frames // 2:05d}.png"), cv2.IMREAD_UNCHANGED)
        check("mask dtype 8-bit", f0.dtype == np.uint8, f"dtype={f0.dtype}")
        vals0 = set(np.unique(f0).tolist())
        vals_mid = set(np.unique(f_mid).tolist())
        check("mask frame0 values == {0,1}", vals0 == {0, 1}, f"values={vals0}")
        check(
            "mask midframe values == {0,1}",
            vals_mid == {0, 1},
            f"frame {num_frames // 2} values={vals_mid}",
        )
        area0 = int((f0 == 1).sum())
        area_mid = int((f_mid == 1).sum())
        check("mask frame0 person area sane", 5000 < area0 < 2_000_000, f"area={area0}")
        check("mask midframe person present", area_mid > 5000, f"area={area_mid}")

        # mask box vs prompt box: tracked box should overlap the prompt box
        ys, xs = np.nonzero(f0)
        bx0, bx1 = xs.min() / f0.shape[1], (xs.max() + 1) / f0.shape[1]
        by0, by1 = ys.min() / f0.shape[0], (ys.max() + 1) / f0.shape[0]
        px0, py0, pw, ph = PERSON_BOX
        ox = max(0, min(bx1, px0 + pw) - max(bx0, px0))
        oy = max(0, min(by1, py0 + ph) - max(by0, py0))
        iou = (ox * oy) / max(
            1e-6, (bx1 - bx0) * (by1 - by0) + pw * ph - ox * oy
        )
        check("mask aligns with prompt box", iou > 0.5, f"iou={iou:.3f}")

        # ── Artifact 2: annotated video ───────────────────────────────
        cap = cv2.VideoCapture(str(out_dir / "annotated.mp4"))
        vframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        check("video frame count", vframes == num_frames, f"{vframes} vs {num_frames}")
        check("video dimensions", (vw, vh) == (1972, 1080), f"{vw}x{vh}")

        # Browsers (incl. the in-IDE preview) only decode H.264, not the
        # MPEG-4 Part 2 that cv2's "mp4v" fourcc produces
        import subprocess

        probe = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=codec_name",
                "-of", "csv=p=0", str(out_dir / "annotated.mp4"),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        check(
            "video codec is h264 (browser-playable)",
            probe.stdout.strip() == "h264",
            f"codec={probe.stdout.strip()!r}",
        )
        check(
            "video non-trivial size",
            (out_dir / "annotated.mp4").stat().st_size > 100_000,
        )

        # ── Artifact 3: annotation JSON ───────────────────────────────
        payload = json.loads((out_dir / "annotation.json").read_text(encoding="utf-8"))
        check("json video filename", payload["video_filename"] == "test.mp4")
        check("json num_frames", payload["num_frames"] == num_frames)
        check("json dims", (payload["width"], payload["height"]) == (1972, 1080))
        check("json fps", abs(payload["fps"] - 25.0) < 0.1, f"fps={payload['fps']}")
        check(
            "json objects list",
            payload["objects"] == [{"id": 1, "class": "person"}],
            f"objects={payload['objects']}",
        )
        check("json frames count", len(payload["frames"]) == num_frames)
        check(
            "json frames contiguous",
            [f["frame_idx"] for f in payload["frames"]] == list(range(num_frames)),
        )
        f0_entry = payload["frames"][0]["objects"]
        check("json frame0 has obj 1", len(f0_entry) == 1 and f0_entry[0]["id"] == 1)
        check(
            "json class name recorded",
            f0_entry[0]["class"] == "person",
            f"class={f0_entry[0]['class']}",
        )
        x0, y0, x1, y1 = f0_entry[0]["bbox_xyxy"]
        check(
            "json bbox in pixel coords near prompt box",
            abs(x0 - px0 * 1972) < 80 and abs(y0 - py0 * 1080) < 80,
            f"xyxy={[x0, y0, x1, y1]}",
        )
        check(
            "json entry has only id/class/bbox_xyxy",
            set(f0_entry[0].keys()) == {"id", "class", "bbox_xyxy"},
            f"keys={sorted(f0_entry[0].keys())}",
        )

        # The video draws the very same xyxy boxes: the middle of the bbox
        # top edge is a 3px line in the object's overlay color (obj 1 ->
        # palette[1] = pure green in BGR)
        cap = cv2.VideoCapture(str(out_dir / "annotated.mp4"))
        ok_read, vframe = cap.read()
        cap.release()
        check("video frame0 decoded", ok_read)
        cx = (x0 + x1) // 2
        px = vframe[min(y0 + 1, vframe.shape[0] - 1), cx].astype(int)
        check(
            "json bbox drawn on video (top edge in object color)",
            px[1] > 180 and px[0] < 90 and px[2] < 90,
            f"pixel@({cx},{y0 + 1})={px.tolist()}",
        )
        cy = (y0 + y1) // 2
        pxx = vframe[cy, min(x0 + 1, vframe.shape[1] - 1)].astype(int)
        check(
            "json bbox drawn on video (left edge in object color)",
            pxx[1] > 180 and pxx[0] < 90 and pxx[2] < 90,
            f"pixel@({x0 + 1},{cy})={pxx.tolist()}",
        )
        mid_entry = payload["frames"][num_frames // 2]["objects"]
        check("json midframe has obj 1", len(mid_entry) == 1 and mid_entry[0]["id"] == 1)

        # ── Re-export: outputs are replaced, no leftovers ──────────────
        r = requests.post(f"{BASE}/api/export/{sid}", timeout=30)
        deadline = time.time() + 300
        st = r.json()
        while st.get("running") and time.time() < deadline:
            time.sleep(1.0)
            st = requests.get(f"{BASE}/api/export/{sid}/status", timeout=30).json()
        check("re-export finished", not st.get("running") and not st.get("last_error"))
        leftovers = [p.name for p in OUTPUT_ROOT.iterdir() if p.name.startswith(".tmp_")]
        check("no tmp leftovers", not leftovers, f"{leftovers}")

        # ── JSON entries mirror the masks exactly (no zero boxes) ────
        # Register a second object prompted on a late frame (100), reset,
        # re-propagate forward, re-export. Whether an object is present in
        # a given frame is the tracker's call (a late-prompted object may
        # still get non-empty masks in earlier frames); the export contract
        # is: every frame's JSON ids == the mask PNG's non-zero values, and
        # absent objects get NO entry (never a [0,0,0,0] box)
        r = requests.post(
            f"{BASE}/api/session/{sid}/reset-tracking", timeout=120
        )
        check("reset tracking for second object", r.status_code == 200, r.text[:200])

        # obj 2 box: inset of obj 1's tracked box on frame 100 (the person)
        ref = payload["frames"][100]["objects"][0]["bbox_xyxy"]
        norm_box = [
            (ref[0] + 10) / 1972, (ref[1] + 10) / 1080,
            max(10, ref[2] - ref[0] - 20) / 1972,
            max(10, ref[3] - ref[1] - 20) / 1080,
        ]
        r = requests.post(
            f"{BASE}/api/prompt/add",
            json={
                "session_id": sid,
                "frame_idx": 100,
                "obj_id": 2,
                "box": norm_box,
                "className": "person2",
            },
            timeout=180,
        )
        check("add second object on late frame", r.status_code == 200, r.text[:200])
        requests.post(
            f"{BASE}/api/prompt/confirm",
            json={"session_id": sid, "obj_ids": [2], "className": "person2"},
            timeout=30,
        )

        ws = websocket.create_connection(f"{WS_BASE}/ws/propagate/{sid}", timeout=600)
        ws.send(json.dumps({"type": "start", "direction": "forward"}))
        msg = None
        while True:
            msg = json.loads(ws.recv())
            if msg["type"] in ("propagation_complete", "error"):
                break
        ws.close()
        check(
            "second propagation complete",
            msg["type"] == "propagation_complete",
            str(msg.get("message", ""))[:120],
        )

        r = requests.post(f"{BASE}/api/export/{sid}", timeout=30)
        deadline = time.time() + 300
        st = r.json()
        while st.get("running") and time.time() < deadline:
            time.sleep(1.0)
            st = requests.get(
                f"{BASE}/api/export/{sid}/status", timeout=30
            ).json()
        check(
            "third export finished",
            not st.get("running") and not st.get("last_error"),
            f"err={st.get('last_error')}",
        )

        payload2 = json.loads(
            (out_dir / "annotation.json").read_text(encoding="utf-8")
        )
        zero_boxes = [
            (f["frame_idx"], o["id"])
            for f in payload2["frames"] for o in f["objects"]
            if o["bbox_xyxy"] == [0, 0, 0, 0]
        ]
        check("no zero boxes in json", not zero_boxes, f"{zero_boxes[:5]}")

        # Frame-by-frame: JSON object ids must mirror the mask PNG's
        # non-zero pixel values, and each bbox must match the mask's
        # non-zero bounding box (this is the omission contract: absent
        # objects have no mask pixels AND no JSON entry). The PNG is a
        # merged label map where higher ids overwrite lower ones on
        # overlap, while the JSON bbox comes from the pre-merge mask, so
        # on overlap frames a lower id's PNG bbox may shrink (containment
        # only); the highest id is never overwritten (exact match).
        mismatches = []
        for fr in payload2["frames"]:
            idx = fr["frame_idx"]
            m = cv2.imread(
                str(masks_dir / f"{idx:05d}.png"), cv2.IMREAD_UNCHANGED
            )
            mask_ids = set(int(v) for v in np.unique(m) if v)
            json_ids = set(o["id"] for o in fr["objects"])
            overlap = (
                sum(int((m == v).sum()) for v in mask_ids)
                != int((m > 0).sum())
            )
            if overlap:
                if not mask_ids <= json_ids:
                    mismatches.append((idx, "ids", sorted(mask_ids), sorted(json_ids)))
            elif mask_ids != json_ids:
                mismatches.append((idx, "ids", sorted(mask_ids), sorted(json_ids)))
                continue
            max_id = max(json_ids) if json_ids else None
            for o in fr["objects"]:
                ys, xs = np.nonzero(m == o["id"])
                if len(ys) == 0:
                    continue  # fully overwritten by a higher id on overlap
                bbox = [
                    int(xs.min()), int(ys.min()),
                    int(xs.max()) + 1, int(ys.max()) + 1,
                ]
                if o["id"] == max_id:
                    if o["bbox_xyxy"] != bbox:
                        mismatches.append((idx, o["id"], o["bbox_xyxy"], bbox))
                else:
                    j = o["bbox_xyxy"]
                    if not (j[0] <= bbox[0] and j[1] <= bbox[1]
                            and j[2] >= bbox[2] and j[3] >= bbox[3]):
                        mismatches.append((idx, o["id"], o["bbox_xyxy"], bbox))
        check(
            "json mirrors masks per frame",
            not mismatches,
            f"{mismatches[:3]}",
        )

        ids100 = [o["id"] for o in payload2["frames"][100]["objects"]]
        check("obj 2 present at its prompt frame", 2 in ids100, f"ids@100={ids100}")
        coverage = {
            oid: sum(
                1 for f in payload2["frames"]
                if any(o["id"] == oid for o in f["objects"])
            )
            for oid in (1, 2)
        }
        print(f"  object coverage (frames present): {coverage}")
        check(
            "json objects header updated",
            sorted(o["id"] for o in payload2["objects"]) == [1, 2],
            f"objects={payload2['objects']}",
        )

        print(f"\nALL {PASSED} CHECKS PASSED ({time.time() - t0:.1f}s)")
    finally:
        requests.delete(f"{BASE}/api/session/{sid}", timeout=60)


if __name__ == "__main__":
    main()
