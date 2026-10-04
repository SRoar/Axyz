/* Data sources for the UI.  Both emit the SAME snapshot shape (schema v1, documented in src/web.py;
   tests/test_web.py compares the key sets so they cannot drift).

     createFake({speed, at})   a JS port of fakes.py's scripted world + ReferenceStateMachine, so the UI is fully
                               functional with no Python running (open ui/index.html straight from disk).
     createLive(baseUrl)       EventSource on the Python bridge (python -m src.main --web).
     connect(opts)             live if reachable, otherwise demo data, and flips back to live if it appears.

   Nothing here touches the DOM at load time, so it also runs under Node (used by the schema-parity test). */
(function (root) {
  const I = (root.Illumin = root.Illumin || {});
  const PAGE_W_CM = 21.59, PAGE_H_CM = 27.94;
  const TM = { HOVER_S: 0.8, LOCK_STILL_S: 0.3, ANSWER_IDLE_S: 2.0, NAV_SPEECH_EVERY_S: 3.5, HAPTIC_REFRESH_S: 0.5, AUDIT_TIMEOUT_S: 8.0, READ_GRACE_S: 1.5 };
  const GUIDE_OR_WARN = ["GUIDE_LEFT", "GUIDE_RIGHT", "GUIDE_BOTH", "WARN"];
  const ONE_SHOT = { LOCK: 0.5, COMPLETE: 0.5 };

  // The demo sheet (demo/sheet_spec.py): three thick boxes, normalized to the page.
  const QUESTIONS = [
    { id: "q1", text: "Explain the first law of thermodynamics.", box_id: "box1" },
    { id: "q2", text: "Describe how the second law of thermodynamics relates to entropy.", box_id: "box2" },
    { id: "q3", text: "Why does ice float on liquid water?", box_id: "box3" },
  ];
  const BOXES = [
    { id: "box1", xmin: 0.0926, ymin: 0.179, xmax: 0.9074, ymax: 0.3364 },
    { id: "box2", xmin: 0.0926, ymin: 0.4796, xmax: 0.9074, ymax: 0.6371 },
    { id: "box3", xmin: 0.0926, ymin: 0.7802, xmax: 0.9074, ymax: 0.9377 },
  ];

  // [t0, t1, motion, penFrom, penTo, jitter]: Q1 nav + a drift OUT of the box, Q2 with the pen hidden
  // under the hand while writing (the pitch moment), answers end when the writing stops, Q3 skipped with a double tap.
  const S = "STILL", M = "MOVING", W = "WRITING", L = "LIFTED";
  const SEG = [
    [0, 4.5, S, [0.5, 0.07], [0.5, 0.07], 0],
    [4.5, 8, M, [0.5, 0.07], [0.42, 0.24], 0],
    [8, 9, S, [0.42, 0.24], [0.42, 0.24], 0],
    [9, 12, W, [0.42, 0.24], [0.8, 0.25], 0.004],
    [12, 13, W, [0.8, 0.25], [0.8, 0.14], 0.004],
    [13, 13.8, W, [0.8, 0.14], [0.62, 0.24], 0.004],
    [13.8, 17, W, [0.62, 0.24], [0.18, 0.3], 0.004],
    [17, 24.5, S, [0.18, 0.3], [0.18, 0.3], 0],
    [24.5, 25, L, [0.18, 0.3], [0.18, 0.3], 0],
    [25, 28.5, M, [0.18, 0.3], [0.55, 0.53], 0],
    [28.5, 29.5, S, [0.55, 0.53], [0.55, 0.53], 0],
    [29.5, 35, W, [0.55, 0.53], [0.2, 0.59], 0.004],
    [35, 1e9, S, [0.2, 0.59], [0.2, 0.59], 0],
  ];
  const OCCLUSIONS = [[0, 2.4], [31, 32.2]];     // pen not yet picked up at the start; later, hidden under the hand while writing
  const TAPS = [[41.5, 2]];
  const COMPLETE_HOLD_S = 8;

  const r = (x, n) => { const k = Math.pow(10, n == null ? 3 : n); return Math.round(x * k) / k; };

  function worldAt(ts) {
    const seg = SEG.find((s) => s[0] <= ts && ts < s[1]) || SEG[SEG.length - 1];
    const occluded = OCCLUSIONS.some(([a, b]) => a <= ts && ts < b);
    const [t0, t1, motion, p0, p1, jit] = seg;
    const u = t1 - t0 > 1e6 ? 0 : (ts - t0) / (t1 - t0);
    const x = p0[0] + (p1[0] - p0[0]) * u + jit * Math.sin(ts * 40);
    const y = p0[1] + (p1[1] - p0[1]) * u + jit * Math.cos(ts * 53);
    return { motion, pen: occluded ? null : { x, y } };
  }

  // ---------------------------------------------------------------- guidance (port of FakeGuidanceEngine)
  function guide(pen, box) {
    if (!pen || !box) return { cmd: null, speech: null, dx_cm: 0, dy_cm: 0, dist_cm: 0, in_box: false, write_status: "UNKNOWN", pen_visible: !!pen };
    const dx = pen.x < box.xmin ? box.xmin - pen.x : pen.x > box.xmax ? box.xmax - pen.x : 0;
    const dy = pen.y < box.ymin ? box.ymin - pen.y : pen.y > box.ymax ? box.ymax - pen.y : 0;
    const dxc = dx * PAGE_W_CM, dyc = dy * PAGE_H_CM, dist = Math.hypot(dxc, dyc);
    const inBox = pen.x >= box.xmin && pen.x <= box.xmax && pen.y >= box.ymin && pen.y <= box.ymax;
    if (inBox) return { cmd: null, speech: "You are in the answer box.", dx_cm: 0, dy_cm: 0, dist_cm: 0, in_box: true, write_status: "INSIDE", pen_visible: true };
    const cmd = Math.abs(dxc) > 1.0 ? (dxc > 0 ? "GUIDE_RIGHT" : "GUIDE_LEFT") : "GUIDE_BOTH";
    let dir, n;
    if (Math.abs(dyc) >= Math.abs(dxc)) { dir = dyc > 0 ? "Down" : "Up"; n = Math.abs(dyc); } else { dir = dxc > 0 ? "Right" : "Left"; n = Math.abs(dxc); }
    const inches = Math.max(1, Math.round(n / 2.54));
    return { cmd, speech: `Move ${inches} ${inches === 1 ? "inch" : "inches"} ${dir}.`, dx_cm: dxc, dy_cm: dyc, dist_cm: dist, in_box: false, write_status: "OUTSIDE", pen_visible: true };
  }

  // ---------------------------------------------------------------- brain (port of ReferenceStateMachine)
  class Brain {
    constructor(questions, boxes) { this.qs = questions; this.boxes = {}; boxes.forEach((b) => (this.boxes[b.id] = b)); this.reset(); }
    reset() {
      this.index = 0; this.phase = "IDLE"; this.phaseT = null; this.stillSince = null; this.nonwritingSince = null;
      this.lastCmd = "OFF"; this.lastCmdT = -1e9; this.lastNavSpeech = -1e9; this.locked = false; this.heard = false; this.completeT = null;
    }
    get question() { return this.qs[this.index] || null; }
    enter(phase, now) { this.phase = phase; this.phaseT = now; this.heard = false; this.nonwritingSince = null; this.locked = false; }
    stillFor(now) { return this.stillSince == null ? 0 : now - this.stillSince; }
    haptic(cmd, now, acts) {
      const refresh = GUIDE_OR_WARN.includes(cmd) && now - this.lastCmdT >= TM.HAPTIC_REFRESH_S;
      if (cmd !== this.lastCmd || refresh) { acts.push({ type: "haptic", cmd }); this.lastCmd = cmd; this.lastCmdT = now; }
    }
    advance(now, acts) {
      this.index += 1;
      if (!this.question) { acts.push({ type: "speak", text: "That was the last question. Test complete.", interrupt: false }); this.phase = "COMPLETE"; this.completeT = now; }
      else this.enter("IDLE", now);
    }
    update(inp) {
      const now = inp.t, acts = [];
      if (this.phaseT == null) this.phaseT = now;
      if (inp.motion === S) { if (this.stillSince == null) this.stillSince = now; } else this.stillSince = null;
      const taps = inp.taps.slice();
      const q = this.question;
      if (!q || this.phase === "COMPLETE") return acts;
      this["on" + this.phase](inp, q, taps, acts);
      return acts;
    }
    onIDLE(inp, q, g, acts) {
      const now = inp.t;
      if (g.includes(2)) { acts.push({ type: "speak", text: "Skipping question.", interrupt: true }); return this.advance(now, acts); }
      const hovered = this.stillFor(now) >= TM.HOVER_S && now - this.phaseT >= TM.HOVER_S;
      if (g.includes(1) || hovered) {
        acts.push({ type: "setbox", box: this.boxes[q.box_id] }, { type: "speak", text: `Question ${this.index + 1}. ${q.text}`, interrupt: true });
        this.enter("READING", now);
      }
    }
    onREADING(inp, q, g, acts) {
      const now = inp.t;
      if (g.includes(2)) { acts.push({ type: "speak", text: "Skipping question.", interrupt: true }); return this.advance(now, acts); }
      if (g.includes(1) && now - this.phaseT > 0.5) { acts.push({ type: "speak", text: `Question ${this.index + 1}. ${q.text}`, interrupt: true }); this.phaseT = now; this.heard = false; return; }
      if (inp.speaking) this.heard = true;
      else if (this.heard || now - this.phaseT > TM.READ_GRACE_S) this.enter("NAVIGATING", now);
    }
    onNAVIGATING(inp, q, g, acts) {
      const now = inp.t, gd = inp.guidance;
      if (g.includes(2)) { acts.push({ type: "speak", text: "Skipping question.", interrupt: true }); return this.advance(now, acts); }
      if (g.includes(1)) acts.push({ type: "speak", text: `Question ${this.index + 1}. ${q.text}`, interrupt: true });
      if (inp.motion === W) { acts.push({ type: "silence" }); this.enter("WRITING", now); return; }
      if (inp.motion === M) {
        this.locked = false;
        const cmd = gd && gd.cmd && !gd.in_box ? gd.cmd : "OFF";
        this.haptic(cmd, now, acts);
        if (gd && !gd.in_box && gd.speech && !inp.speaking && now - this.lastNavSpeech >= TM.NAV_SPEECH_EVERY_S) {
          acts.push({ type: "speak", text: gd.speech, interrupt: false }); this.lastNavSpeech = now;
        }
      } else {
        this.haptic("OFF", now, acts);
        if (inp.motion === S && gd && gd.in_box && !this.locked && this.stillFor(now) >= TM.LOCK_STILL_S) {
          this.locked = true;
          acts.push({ type: "haptic", cmd: "LOCK" }, { type: "speak", text: "In the answer box. Start writing.", interrupt: false });
          this.lastCmd = "OFF";
        }
      }
    }
    onWRITING(inp, q, g, acts) {
      const now = inp.t, gd = inp.guidance;
      if (inp.motion === W) this.nonwritingSince = null; else if (this.nonwritingSince == null) this.nonwritingSince = now;
      const done = g.includes(3) || (this.nonwritingSince != null && now - this.nonwritingSince >= TM.ANSWER_IDLE_S);
      if (done) { this.haptic("OFF", now, acts); acts.push({ type: "snapshot", question_id: q.id }); this.enter("AUDITING", now); return; }
      const status = gd ? gd.write_status : "UNKNOWN";
      if (inp.motion === W && status === "OUTSIDE") this.haptic("WARN", now, acts); else this.haptic("OFF", now, acts);
    }
    onAUDITING(inp, q, g, acts) {
      const now = inp.t, res = inp.audit;
      let msg;
      if (res && res.question_id === q.id) {
        if (!res.ink_present) { acts.push({ type: "speak", text: "I did not find writing in the box. Move to the box and try again.", interrupt: false }); return this.enter("NAVIGATING", now); }
        msg = "Answer recorded." + (res.ink_outside ? " Some writing went outside the box." : "");
      } else if (now - this.phaseT > TM.AUDIT_TIMEOUT_S) msg = "I could not check your answer. Moving on.";
      else return;
      acts.push({ type: "speak", text: msg, interrupt: false }, { type: "haptic", cmd: "COMPLETE" });
      this.lastCmd = "OFF";
      this.advance(now, acts);
    }
  }

  // ---------------------------------------------------------------- the fake system
  function createFake(opts) {
    opts = opts || {};
    const speed = opts.speed || 1;
    const DT = 0.02;
    const brain = new Brain(QUESTIONS, BOXES);
    const sys = {
      simT: 0, worldStart: 0, tapsDone: 0, override: null, pendingTaps: [], voiceFreeAt: 0, voiceText: "",
      currentBox: null, log: [], seq: 0, haptic: { t: -1e9, cmd: "OFF" }, hseq: 0, tapsShown: [],
      status: {}, prevIndex: 0, lastAudit: null, pendingAudit: [], samples: [], nextSample: 0, ink: [], inkLast: null, lastPen: null,
      lastMotion: S, speakingNow: false, lastPhase: null, all: [],
    };
    QUESTIONS.forEach((q) => (sys.status[q.id] = "pending"));
    let rng = 7;
    const rand = () => { rng = (rng * 16807) % 2147483647; return (rng - 1) / 2147483646; };
    const gauss = (sd) => (rand() + rand() + rand() + rand() - 2) * sd * 1.7;

    function log(kind, text) {
      sys.seq += 1;
      const e = { seq: sys.seq, t: r(sys.simT), kind, text };
      sys.log.push(e);
      sys.all.push(e);                                   // untruncated history (parity tests, debugging)
      if (sys.log.length > 40) sys.log.shift();
      if (sys.all.length > 4000) sys.all.shift();
    }
    function speak(text, interrupt) {
      const now = sys.simT;
      const start = interrupt ? now : Math.max(now, sys.voiceFreeAt);
      sys.voiceFreeAt = start + 0.3 + 0.045 * text.length;
      sys.voiceText = text;
      log("SPEAK", text);
      const prev = QUESTIONS[sys.prevIndex] && QUESTIONS[sys.prevIndex].id;
      if (text.startsWith("Skipping question") && prev) sys.status[prev] = "skipped";
      if (text.startsWith("I could not check") && prev) sys.status[prev] = "unchecked";
    }
    function sample(t, m) {
      if (m === S) return [gauss(0.003), gauss(0.003), 1 + gauss(0.003)];
      if (m === M) return [0.15 * Math.sin(t * 3) + gauss(0.01), 0.1 * Math.cos(t * 2) + gauss(0.01), 1 + gauss(0.01)];
      if (m === W) return [0.05 * Math.sin(t * 60) + gauss(0.01), 0.05 * Math.cos(t * 75) + gauss(0.01), 1 + 0.03 * Math.sin(t * 50)];
      return [gauss(0.05), gauss(0.05), 1.8 + gauss(0.1)];
    }

    function restart() {
      brain.reset(); sys.lastPhase = null; sys.worldStart = sys.simT; sys.tapsDone = 0; sys.currentBox = null; sys.override = null;
      sys.ink = []; sys.inkLast = null; sys.lastAudit = null; sys.pendingAudit = []; sys.prevIndex = 0; sys.voiceFreeAt = 0;
      QUESTIONS.forEach((q) => (sys.status[q.id] = "pending"));
    }

    function step() {
      const now = (sys.simT += DT);
      const ts = now - sys.worldStart;
      const w = worldAt(ts);
      const events = [];
      const motion = sys.override || w.motion;
      if (motion !== sys.lastMotion) { sys.lastMotion = motion; }
      while (sys.tapsDone < TAPS.length && TAPS[sys.tapsDone][0] <= ts) {
        const n = TAPS[sys.tapsDone][1]; sys.tapsDone++; events.push(n);
      }
      sys.pendingTaps.splice(0).forEach((n) => events.push(n));
      events.forEach((n) => sys.tapsShown.push({ t: now, n }));
      while (sys.nextSample <= now) { sys.samples.push(sample(sys.nextSample, motion)); if (sys.samples.length > 80) sys.samples.shift(); sys.nextSample += 0.02; }

      const pen = w.pen;
      if (pen) sys.lastPen = pen;
      // ink: the pen only leaves ink while the IMU says WRITING (and it is visible)
      if (motion === W && pen) {
        if (sys.inkLast) sys.ink.push([sys.inkLast.x, sys.inkLast.y, pen.x, pen.y]);
        sys.inkLast = pen;
      } else if (motion !== W) sys.inkLast = null;

      const box = sys.currentBox;
      const gd = guide(pen, box);
      const res = sys.pendingAudit.length && sys.pendingAudit[0].at <= now ? sys.pendingAudit.shift().result : null;
      if (res) {
        sys.lastAudit = { t: now, result: res };
        log("AUDIT", `${res.question_id}: ink=${res.ink_present ? "True" : "False"} outside=${res.ink_outside ? "True" : "False"} conf=${res.confidence.toFixed(2)} ${res.note}`);
        if (res.ink_present) sys.status[res.question_id] = "answered";
      }
      const acts = brain.update({ t: now, motion, taps: events, speaking: now < sys.voiceFreeAt, guidance: gd, audit: res });
      acts.forEach((a) => {
        if (a.type === "speak") speak(a.text, a.interrupt);
        else if (a.type === "silence") { sys.voiceFreeAt = now; log("SILENCE", ""); }
        else if (a.type === "haptic") { sys.haptic = { t: now, cmd: a.cmd }; sys.hseq += 1; log("HAPTIC", a.cmd); }
        else if (a.type === "setbox") { sys.currentBox = a.box; log("BOX", a.box.id); }
        else if (a.type === "snapshot") {
          log("SNAPSHOT", a.question_id);
          sys.pendingAudit.push({ at: now + 1.0, result: { question_id: a.question_id, ink_present: true, ink_outside: sys.ink.length > 0 && a.question_id === "q1", confidence: 0.93, note: "Handwriting inside the box." } });
        }
      });
      if (brain.phase !== sys.lastPhase) { sys.lastPhase = brain.phase; log("PHASE", brain.phase); }   // like System.step()
      sys.prevIndex = brain.index;
      sys.cur = { pen, motion, gd, now };
      if (brain.phase === "COMPLETE" && brain.completeT != null && now - brain.completeT > COMPLETE_HOLD_S) restart();
    }

    function build() {
      const now = sys.simT, c = sys.cur || { pen: null, motion: S, gd: guide(null, null) };
      const q = brain.question;
      const phase = brain.phase;
      const progress = QUESTIONS.map((qq) => ({ id: qq.id, status: sys.status[qq.id] === "pending" && q && qq.id === q.id && phase !== "COMPLETE" ? "active" : sys.status[qq.id] }));
      const hAge = now - sys.haptic.t;
      const active = ONE_SHOT[sys.haptic.cmd] != null ? hAge < ONE_SHOT[sys.haptic.cmd] : GUIDE_OR_WARN.includes(sys.haptic.cmd) ? hAge < 1.5 : false;
      const gd = c.gd;
      const la = sys.lastAudit;
      return {
        v: 1, source: "fake", t: r(now), phase, motion: c.motion,
        question: q ? { id: q.id, text: q.text, index: brain.index, total: QUESTIONS.length, box_id: q.box_id } : null,
        progress,
        boxes: BOXES.map((b, i) => ({ id: b.id, label: `Answer ${i + 1}`, question_id: QUESTIONS[i].id, xmin: b.xmin, ymin: b.ymin, xmax: b.xmax, ymax: b.ymax, active: !!(sys.currentBox && sys.currentBox.id === b.id) })),
        pen: c.pen ? { x: r(c.pen.x, 4), y: r(c.pen.y, 4), confidence: 1 } : null,
        guidance: { cmd: gd.cmd, speech: gd.speech, dx_cm: r(gd.dx_cm, 2), dy_cm: r(gd.dy_cm, 2), dist_cm: r(gd.dist_cm, 2), in_box: gd.in_box, write_status: gd.write_status, pen_visible: gd.pen_visible },
        haptic: { cmd: sys.haptic.cmd, seq: sys.hseq, age_s: r(Math.min(hAge, 999), 2), active },
        voice: { speaking: now < sys.voiceFreeAt, level: null, level_source: "synthetic", text: sys.voiceText },
        imu: { samples: sys.samples.map((s) => s.map((v) => r(v))), light: c.motion === W ? 400 : 700 },
        taps: sys.tapsShown.filter((t) => now - t.t < 1.6).map((t) => ({ age_s: r(now - t.t), n: t.n })),
        audit: la ? { question_id: la.result.question_id, ink_present: la.result.ink_present, ink_outside: la.result.ink_outside, confidence: la.result.confidence, note: la.result.note, age_s: r(now - la.t, 2) } : null,
        events: sys.log.slice(),
        health: { errors: {}, last_error: "" },
        debug: { enabled: true, override: sys.override },
        camera: { has_frame: false, label: "Demo camera", rectified: true },
      };
    }

    // advance the simulation to (wall-driven) time `to`, in fixed 20 ms steps like the Python loop
    let timer = null, lastReal = null, listeners = [];
    function advanceTo(to) { while (sys.simT + DT <= to) step(); }
    if (opts.at) advanceTo(opts.at);                     // ?at=12.5 : start mid-demo (screenshots, rehearsal)

    return {
      kind: "fake",
      get ink() { return sys.ink; },
      get history() { return sys.all; },
      snapshot: () => { if (!sys.cur) step(); return build(); },
      onSnapshot(fn) { listeners.push(fn); },
      key(k) {
        const motions = { 1: S, 2: M, 3: W, 4: L };
        if (motions[k]) sys.override = motions[k];
        else if (k === "0") sys.override = null;
        else if (k === "t") sys.pendingTaps.push(2);
        else if (k === "r") sys.pendingTaps.push(1);
        else if (k === "n") sys.pendingTaps.push(2);
        else return false;
        return true;
      },
      start() {
        lastReal = performance.now();
        timer = setInterval(() => {
          const real = performance.now();
          advanceTo(sys.simT + ((real - lastReal) / 1000) * speed + 0.0005);
          lastReal = real;
          const snap = build();
          listeners.forEach((fn) => fn(snap));
        }, 50);
      },
      stop() { clearInterval(timer); timer = null; },
    };
  }

  // ---------------------------------------------------------------- live source (python -m src.main --web)
  function createLive(base) {
    let es = null, listeners = [], statusFns = [], lastMsg = 0, state = "connecting";
    const setState = (s) => { if (s !== state) { state = s; statusFns.forEach((f) => f(s)); } };
    return {
      kind: "live", base,
      get state() { return state; },
      get lastMessage() { return lastMsg; },
      onSnapshot(fn) { listeners.push(fn); },
      onState(fn) { statusFns.push(fn); },
      start() {
        es = new EventSource(base + "/events");
        es.onmessage = (e) => {
          lastMsg = performance.now();
          setState("open");
          let snap; try { snap = JSON.parse(e.data); } catch (_) { return; }
          listeners.forEach((fn) => fn(snap));
        };
        es.onerror = () => setState("error");
      },
      stop() { if (es) es.close(); es = null; },
      key(k) {
        return fetch(base + "/api/key", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ key: k }) })
          .then((res) => res.json().then((j) => ({ ok: res.ok && j.ok, status: res.status, error: j.error })))
          .catch(() => ({ ok: false, status: 0, error: "unreachable" }));
      },
    };
  }

  // ---------------------------------------------------------------- connect: live if reachable, else demo
  function connect(handlers) {
    const q = new URLSearchParams((root.location && root.location.search) || "");
    const want = q.get("source");
    const speed = parseFloat(q.get("speed")) || 1, at = parseFloat(q.get("at")) || 0;
    const server = q.get("server") || (root.location && /^https?:$/.test(root.location.protocol) ? root.location.origin : null);
    let active = null, live = null, fake = null;

    function use(src) {
      if (active === src) return;
      if (active) active.stop();
      active = src;
      src.start();
      handlers.onSource && handlers.onSource(src.kind, src);
    }
    function ensureFake(reason) {
      if (!fake) { fake = createFake({ speed, at }); fake.onSnapshot((s) => active === fake && handlers.onSnapshot(s)); }
      use(fake);
      if (reason) handlers.onNotice && handlers.onNotice(reason);
    }

    if (want === "fake" || !server) { ensureFake(); return controller(); }

    live = createLive(server);
    live.onSnapshot((s) => {
      if (active !== live) use(live);        // live appeared (or came back): switch over
      handlers.onSnapshot(s);
    });
    live.onState((st) => { if (st === "error" && active === live && handlers.onNotice) handlers.onNotice("Lost the pen system. Reconnecting."); });
    live.start();
    active = live; handlers.onSource && handlers.onSource("live", live);
    if (want !== "live") {
      setTimeout(() => { if (!live.lastMessage && active === live) ensureFake("Can't reach the pen system. Showing demo data."); }, 2200);
    }
    function controller() {
      return {
        get source() { return active; },
        key(k) { return active ? active.key(k) : false; },
      };
    }
    return { get source() { return active; }, key(k) { return active.key(k); } };
  }

  I.sources = { createFake, createLive, connect, QUESTIONS, BOXES, SEG, TAPS, OCCLUSIONS, guide, Brain, PAGE_W_CM, PAGE_H_CM };
})(typeof window !== "undefined" ? window : globalThis);
