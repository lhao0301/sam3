# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Confirmed prompt bookkeeping for SAM3 sessions.

SAM3's detection-path prompts (text / box) reset ALL tracker state on every
``add_prompt`` call, while point prompts go through the incremental tracker
path with a user-specified obj_id. To accumulate multiple confirmed targets
across submissions we therefore replay the full confirmed prompt set on every
detection-path call, and replay the confirmed point prompts afterwards to
restore the tracker objects wiped by the reset.

Prompt formats (all coordinates normalized [0,1], xywh for boxes):
    text  -> {"type": "text",   "frame_idx": int, "text": str}
    box   -> {"type": "box",    "frame_idx": int, "boxes": [[x,y,w,h]]}
    point -> {"type": "point",  "frame_idx": int, "points": [[x,y], ...],
              "point_labels": [1|0, ...], "obj_id": int}
"""


class PromptHistory:
    """Tracks confirmed prompts and the currently previewed (pending) one."""

    def __init__(self):
        self.text = None        # active text prompt dict (SAM3 allows a single one)
        self.boxes = []         # confirmed box prompt dicts
        self.points = []        # confirmed point prompt dicts
        self.pending = None     # previewed but not yet confirmed prompt dict
        self.det_frame = 0      # frame the detection prompts are anchored on

    # ── Pending lifecycle ────────────────────────────────────────────

    def set_pending(self, prompt):
        self.pending = prompt

    def clear_pending(self):
        self.pending = None

    def confirm_pending(self):
        """Move the pending prompt into the confirmed set."""
        if self.pending is None:
            return
        p = self.pending
        if p["type"] == "text":
            # A newly confirmed text replaces the previous active text
            self.text = p
        elif p["type"] == "box":
            # SAM3 (v1) supports a single visual-prompt box per session:
            # a newly confirmed box replaces the previous one.
            self.boxes = [p]
        elif p["type"] == "point":
            self.points.append(p)
        self.pending = None

    # ── Removal ──────────────────────────────────────────────────────

    def remove_by_obj_id(self, obj_id):
        """Remove the confirmed prompt that produced the given tracklet id.

        Point prompts are addressable directly by their user-specified obj_id.
        Detection-path prompts (box / text) remember the ids the model
        assigned to them at confirm time ("assigned_ids"). Returns True if
        something was removed.
        """
        before = (len(self.boxes), len(self.points), self.text is not None)

        self.points = [p for p in self.points if p["obj_id"] != obj_id]

        if obj_id in self._detection_ids(self.text):
            self.text = None
        else:
            self.boxes = [
                b for b in self.boxes
                if obj_id not in self._detection_ids(b)
            ]

        after = (len(self.boxes), len(self.points), self.text is not None)
        return before != after

    def _detection_ids(self, prompt):
        """Ids assigned by the model to a box / text prompt at confirm time."""
        if not prompt:
            return set()
        return set(prompt.get("assigned_ids", []))

    # ── Replay planning ──────────────────────────────────────────────

    def detection_replay_payload(self):
        """Arguments for a single detection-path add_prompt call covering
        the active confirmed prompts (plus the pending one).

        SAM3 (v1) constraints: a single text prompt and a single box
        (visual prompt) per submission — a pending box/text replaces the
        confirmed one. Point prompts are handled separately (tracker path).

        Returns a dict of kwargs for Sam3Service.add_prompt, or None when
        there is nothing to replay on the detection path.
        """
        # Text: pending replaces confirmed
        if self.pending is not None and self.pending["type"] == "text":
            text = self.pending["text"]
        elif self.text is not None:
            text = self.text["text"]
        else:
            text = None

        # Box: pending replaces confirmed (single visual prompt)
        if self.pending is not None and self.pending["type"] == "box":
            box_prompt = self.pending
        elif self.boxes:
            box_prompt = self.boxes[-1]
        else:
            box_prompt = None

        if text is None and box_prompt is None:
            return None
        payload = {}
        if text is not None:
            payload["text"] = text
        if box_prompt is not None:
            payload["bounding_boxes"] = list(box_prompt["boxes"])
            payload["bounding_box_labels"] = list(
                box_prompt.get("box_labels", [1] * len(box_prompt["boxes"]))
            )
        return payload

    def point_replay_list(self):
        """Confirmed point prompts (plus the pending one) to replay through
        the incremental tracker path after a detection-path reset."""
        prompts = list(self.points)
        if self.pending is not None and self.pending["type"] == "point":
            prompts.append(self.pending)
        return prompts

    def record_detection_ids(self, obj_ids):
        """Record the model-assigned ids for the pending detection prompt.

        Called right after a preview: the ids belonging to the *newly added*
        prompt are the ones not already owned by confirmed prompts.
        """
        if self.pending is None or self.pending["type"] not in ("text", "box"):
            return
        known = set()
        if self.text is not None:
            known |= self._detection_ids(self.text)
        for b in self.boxes:
            known |= self._detection_ids(b)
        new_ids = [i for i in obj_ids if i not in known]
        self.pending.setdefault("assigned_ids", [])
        self.pending["assigned_ids"].extend(new_ids)

    @property
    def confirmed_obj_ids(self):
        """All tracklet ids owned by confirmed prompts (point ids directly,
        detection ids as assigned at confirm time)."""
        ids = set()
        if self.text is not None:
            ids |= self._detection_ids(self.text)
        for b in self.boxes:
            ids |= self._detection_ids(b)
        for p in self.points:
            ids.add(p["obj_id"])
        return ids
