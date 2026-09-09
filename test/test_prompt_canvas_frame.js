// Unit test for the pending-prompt frame binding in prompt-canvas.js
// (fix for: a box drawn on one frame "reappearing" under the cursor on
// another frame, and cross-frame submission landing on the wrong frame).
//
// Run: node test/test_prompt_canvas_frame.js
"use strict";

const fs = require("fs");
const path = require("path");

// Load prompt-canvas.js with a minimal window/DOM mock
const code = fs.readFileSync(
    path.join(__dirname, "..", "app", "frontend", "js", "prompt-canvas.js"),
    "utf8"
);
global.window = {};
global.document = { addEventListener() {} };
eval(code);
const PromptCanvas = global.window.PromptCanvas;

if (typeof PromptCanvas !== "function") {
    console.error("[FAIL] PromptCanvas not exported to window");
    process.exit(1);
}

// ── Mocks ───────────────────────────────────────────────────────
function makeCtx() {
    const calls = { strokeRect: 0, clearRect: 0 };
    const ctx = new Proxy({}, {
        get(_t, prop) {
            if (prop === "__calls") return calls;
            if (prop in calls) {
                return (...args) => { calls[prop] += 1; };
            }
            return () => {};
        },
        set() { return true; },
    });
    return ctx;
}

const ctx = makeCtx();
const canvas = {
    getContext: () => ctx,
    addEventListener() {},
    getBoundingClientRect: () => ({ left: 0, top: 0 }),
    style: {},
};

let curFrame = 0;
const player = {
    displayWidth: 640,
    displayHeight: 360,
    getCurrentFrame: () => curFrame,
    toNormalized: (x, y) => ({ x: x / 640, y: y / 360 }),
    fromNormalized: (nx, ny) => ({ x: nx * 640, y: ny * 360 }),
};

const pc = new PromptCanvas(canvas, player);

let PASSED = 0;
function check(name, cond, detail = "") {
    const status = cond ? "PASS" : "FAIL";
    console.log(`[${status}] ${name} ${detail}`);
    if (!cond) {
        console.log("TEST ABORTED");
        process.exit(1);
    }
    PASSED += 1;
}

function mouse(type, x, y) {
    pc[`_on${type}`]({ clientX: x, clientY: y });
}

// ── Scenario ────────────────────────────────────────────────────
// 1. Draw a box on frame 10
curFrame = 10;
mouse("MouseDown", 100, 100);
mouse("MouseMove", 180, 160);
mouse("MouseUp", 180, 160);
check("box pending after draw on frame 10", pc.pendingBox !== null);
check("pendingFrameIdx anchored to 10", pc.getPendingFrameIdx() === 10,
    `(got ${pc.getPendingFrameIdx()})`);
check("hasPending true", pc.hasPending());

// 2. Switch to frame 25: box must be invisible AND unhittable there
curFrame = 25;
pc.notifyFrameChanged();
check("hitTest on other frame returns null",
    pc._hitTest({ x: 140, y: 130 }) === null,
    "(box edge position on frame 25)");
// edge band of the box (100,100)-(180,160): point (140, 103) is on the
// top edge; on frame 10 it must hit, on frame 25 it must not
check("hitTest box edge works on anchor frame",
    (() => { curFrame = 10; const h = pc._hitTest({ x: 140, y: 103 }); curFrame = 25; return h && h.type === "box"; })());

// 3. Render on frame 25 must NOT draw the box; on frame 10 it must
ctx.__calls.clearRect = 0;
ctx.__calls.strokeRect = 0;
curFrame = 25;
pc._render();
const drewOn25 = ctx.__calls.strokeRect > 0;
check("box not rendered on frame 25", !drewOn25,
    `(${ctx.__calls.strokeRect} strokeRect calls)`);
curFrame = 10;
pc._render();
check("box rendered on anchor frame 10", ctx.__calls.strokeRect > 0);

// 4. Drawing a NEW box on frame 25 replaces the cross-frame pending set
//    (one submission cannot mix two frames)
curFrame = 25;
mouse("MouseDown", 300, 200);
mouse("MouseUp", 420, 300);
check("new box anchored to frame 25", pc.getPendingFrameIdx() === 25);
check("new box replaces old-frame pending",
    pc.pendingBox !== null && Math.abs(pc.pendingBox.x - 300 / 640) < 1e-9);
check("getPending still provides box", pc.getPending() !== null && pc.getPending().box !== null);

// 5. Points anchored per frame too: point on frame 30, then a point on
//    frame 5 must drop the frame-30 point
pc.mode = "point";
pc.pointType = 1;
curFrame = 30;
mouse("MouseDown", 50, 50);
check("point pending on frame 30",
    pc.pendingPoints.length === 1 && pc.getPendingFrameIdx() === 30);
curFrame = 5;
mouse("MouseDown", 400, 100);
check("point on new frame drops old-frame point",
    pc.pendingPoints.length === 1 && pc.getPendingFrameIdx() === 5);

// 6. Same frame: box + points still combine (unchanged behavior).
//    The frame-30/5 point switches above dropped the frame-25 box (by
//    design); draw a fresh box on the CURRENT frame 5 and append another
//    point — the combined pending set must carry both.
pc.mode = "box";
mouse("MouseDown", 10, 10);
mouse("MouseUp", 90, 90);
check("same-frame box drawn alongside point", pc.pendingBox !== null);
curFrame = 5;
pc.mode = "point";
mouse("MouseDown", 20, 20);
check("same-frame point appended", pc.pendingPoints.length === 2);
const pending = pc.getPending();
check("combined submission carries box+points",
    pending.box !== null && pending.points !== null &&
    pending.points.length === 2);

// 7. clearPending resets the anchor
pc.clearPending();
check("clearPending resets frame anchor", pc.getPendingFrameIdx() === null);
check("clearPending empties content", !pc.hasPending());

console.log(`\nALL ${PASSED} CHECKS PASSED`);
