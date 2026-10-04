/* Illumin UI renderer.
   - snapshots (live bridge or demo data) arrive ~20 Hz -> render() updates the DOM, using the transitions.dev
     motion for state CHANGES only (phase, question, status, panels). Fast-changing values (guidance numbers,
     pen position, waveforms) are updated instantly: no animation on high-frequency updates.
   - gsap.ticker drives everything continuous: pen glide, trail, voice waveform, IMU trace, buzzer lamps. */
(function () {
  const I = (window.Illumin = window.Illumin || {});
  const M = I.motion, G = I.gestures;
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));

  // sRGB equivalents of the OKLCH tokens in tokens.css (canvas cannot read CSS variables).
  const C = { pen: "#4cc0ff", moving: "#f8c153", writing: "#58da98", lifted: "#d29ee2", still: "#a5b2c4", warn: "#ff6f6b", voice: "#f2ebd1", ink950: "#070a0f", line: "#383e46", text3: "#8f969f", paper: "#f1f4f7", paperInk: "#1a263f" };
  const MOTION_HEX = { STILL: C.still, MOVING: C.moving, WRITING: C.writing, LIFTED: C.lifted };
  const MOTION_LABEL = { STILL: "Still", MOVING: "Moving", WRITING: "Writing", LIFTED: "Lifted" };

  const PHASE = {
    IDLE: ["Idle", "Hold the pen still to hear the next question."],
    READING: ["Reading", "Reading the question aloud."],
    NAVIGATING: ["Navigating", "Guiding the pen to the answer box."],
    WRITING: ["Writing", "Voice is silent while the student writes."],
    OOB: ["Out of bounds", "Writing outside the box. The warning buzzer is on."],
    AUDITING: ["Auditing", "Checking the answer."],
    COMPLETE: ["Complete", "All questions answered."],
  };
  const HAPTIC_LABEL = { GUIDE_LEFT: "Left buzzer", GUIDE_RIGHT: "Right buzzer", GUIDE_BOTH: "Both buzzers", WARN: "Warning", LOCK: "Locked in", COMPLETE: "Answer recorded", OFF: "Silent" };
  const STATUS_LABEL = { pending: "Up next", active: "In progress", answered: "Answered", skipped: "Skipped", unchecked: "Not checked" };
  const STATUS_ICONS = ["pending", "active", "answered", "skipped", "unchecked"];
  const COMP_NAME = { imu: "Pen link", tracker: "Camera", voice: "Voice", guidance: "Guidance", auditor: "Auditor", brain: "Brain" };
  const COMP_HINT = {
    imu: "Check the USB cable and the serial port.", tracker: "Check the camera cable and CAMERA_INDEX.",
    voice: "Check the audio output and the ElevenLabs key.", guidance: "Restart the system.",
    auditor: "Check the Gemini key. The pixel check still runs.", brain: "Restart the system.",
  };
  const KEYS = ["1", "2", "3", "4", "0", "t", "r", "n"];
  const WAVE_BARS = 48;

  const el = {};
  const S = {
    snap: null, source: null, ctl: null, t0: null, pw: 1, ph: 1, dpr: 1,
    lastPen: null, penHas: false, trail: [], lostSince: null, penPlaced: false,
    hseq: null, hActive: false, hCmd: "OFF", hSince: 0, warnActive: false,
    tapSeen: new Set(), eventSeq: 0, said: [], buzz: [], audit: { state: "empty", qid: null, data: null, key: "" },
    errSeen: {}, cache: {}, imuDirty: false, auditNew: false, wasSpeaking: false, speakOffAt: 0, wave: { hist: new Array(WAVE_BARS).fill(0), last: 0, level: 0, setters: [] },
    imgFailed: false, errorLogged: false,
  };
  I.app = { state: S };            // I.app.frame / I.app.render are exposed for profiling and tests

  const fmtTime = (t) => { const s = Math.max(0, t - (S.t0 || 0)); return Math.floor(s / 60) + ":" + String(Math.floor(s % 60)).padStart(2, "0"); };
  const signed = (v) => (v >= 0 ? "+" : "\u2212") + Math.abs(v).toFixed(1);
  const setInstant = (node, text) => { if (node.dataset.txt !== text) { node.dataset.txt = text; node.dataset.ready = "1"; node.textContent = text; } };
  const once = (key, val, fn) => { if (S.cache[key] !== val) { S.cache[key] = val; fn(val); } };

  function icon(k) {
    const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    s.setAttribute("class", "icon t-icon");
    s.dataset.k = k;
    s.innerHTML = `<use href="#i-${k}"/>`;
    return s;
  }
  function iconSlot() { const s = document.createElement("span"); s.className = "t-icon-slot"; STATUS_ICONS.forEach((k) => s.appendChild(icon(k))); return s; }

  /* ================================================================== render (per snapshot) */
  function render(snap) {
    S.snap = snap;
    if (S.t0 == null) S.t0 = snap.t;
    const app = el.app;
    app.dataset.debug = String(!!snap.debug.enabled);
    renderPhase(snap);
    renderQuestion(snap);
    renderSteps(snap);
    renderBoxes(snap);
    renderPen(snap);
    renderEvents(snap);
    renderHaptics(snap);
    renderVoice(snap);
    renderImu(snap);
    renderGuidance(snap);
    renderAudit(snap);
    renderHealth(snap);
    renderCamera(snap);
    renderMenu(snap);
    if (app.classList.contains("is-booting")) requestAnimationFrame(() => requestAnimationFrame(() => app.classList.remove("is-booting")));
  }

  function renderPhase(snap) {
    const gd = snap.guidance;
    const oob = snap.phase === "WRITING" && snap.motion === "WRITING" && gd.write_status === "OUTSIDE";
    const key = oob ? "OOB" : snap.phase;
    el.app.dataset.phase = snap.phase;
    el.app.dataset.oob = String(oob);
    const [name, sub0] = PHASE[key];
    M.textSwap(el.phaseName, name);
    const navSub = key === "NAVIGATING" && gd.speech ? gd.speech : sub0;
    if (key === "NAVIGATING" && S.cache.phaseKey === "NAVIGATING") setInstant(el.phaseSub, navSub);   // fast-changing: no animation
    else M.textSwap(el.phaseSub, navSub);
    S.cache.phaseKey = key;
  }

  function renderQuestion(snap) {
    const q = snap.question, total = snap.progress.length;
    once("qTotal", total, (v) => (el.qTotal.textContent = v));
    if (q) { M.digits(el.qNum, q.index + 1); M.textSwap(el.qText, q.text); }
    else { M.digits(el.qNum, total || 1); M.textSwap(el.qText, snap.phase === "COMPLETE" ? "All questions answered." : "Waiting for the first question."); }
  }

  function renderSteps(snap) {
    const key = snap.progress.map((p) => p.id).join();
    if (S.cache.stepsKey !== key) {
      S.cache.stepsKey = key;
      el.steps.innerHTML = "";
      snap.progress.forEach((p, i) => {
        const li = document.createElement("li");
        li.className = "step";
        li.appendChild(iconSlot());
        li.insertAdjacentHTML("beforeend", `<span class="step-name">Answer ${i + 1}</span><span class="step-status"><span class="t-text"></span></span>`);
        el.steps.appendChild(li);
      });
    }
    snap.progress.forEach((p, i) => {
      const li = el.steps.children[i];
      if (li && li.dataset.status !== p.status) {
        li.dataset.status = p.status;
        M.iconSwap($(".t-icon-slot", li), p.status);
        M.textSwap($(".t-text", li), STATUS_LABEL[p.status]);
      }
    });
  }

  function renderBoxes(snap) {
    const key = snap.boxes.map((b) => [b.id, b.xmin, b.ymin, b.xmax, b.ymax].join(":")).join();
    if (S.cache.boxesKey !== key) {
      S.cache.boxesKey = key;
      el.boxes.innerHTML = "";
      snap.boxes.forEach((b) => {
        const d = document.createElement("div");
        d.className = "box";
        d.dataset.id = b.id;
        d.style.cssText = `left:${b.xmin * 100}%;top:${b.ymin * 100}%;width:${(b.xmax - b.xmin) * 100}%;height:${(b.ymax - b.ymin) * 100}%`;
        const lab = document.createElement("span");
        lab.className = "box-label";
        lab.appendChild(iconSlot());
        const t = document.createElement("span");
        t.textContent = b.label;
        lab.appendChild(t);
        d.appendChild(lab);
        el.boxes.appendChild(d);
      });
    }
    snap.boxes.forEach((b, i) => {
      const d = el.boxes.children[i];
      if (!d) return;
      const st = (snap.progress.find((p) => p.id === b.question_id) || {}).status || "pending";
      const isActive = b.active && (st === "active" || st === "pending");
      d.classList.toggle("is-active", isActive);
      d.classList.toggle("is-answered", st === "answered");
      d.classList.toggle("is-skipped", st === "skipped" || st === "unchecked");
      M.iconSwap($(".t-icon-slot", d), isActive ? "active" : st);
    });
    // margin alarm: the box turns red + shakes when the WARN buzzer starts, and calms when it stops
    const warn = snap.haptic.cmd === "WARN" && snap.haptic.active;
    const act = $(".box.is-active", el.boxes);
    if (warn && !S.warnActive && act) { act.classList.add("is-error"); M.shake(act); }
    if (!warn && S.warnActive) $$(".box.is-error", el.boxes).forEach((b) => b.classList.remove("is-error"));
    if (warn && act) act.classList.add("is-error");
    S.warnActive = warn;
  }

  function renderPen(snap) {
    const p = snap.pen, now = performance.now();
    if (p) {
      S.lastPen = { x: p.x, y: p.y }; S.penHas = true; S.lostSince = null;
      S.trail.push({ t: now, x: p.x, y: p.y });
      if (S.trail.length > 90) S.trail.shift();
    } else {
      S.penHas = false;
      if (S.lostSince == null) S.lostSince = now;
    }
    el.pen.dataset.state = p ? (p.confidence < 0.5 ? "low" : "ok") : S.lastPen ? "lost" : "none";
  }

  function renderEvents(snap) {
    const fresh = snap.events.filter((e) => e.seq > S.eventSeq);
    if (!fresh.length) return;
    S.eventSeq = fresh[fresh.length - 1].seq;
    fresh.forEach((e) => { if (e.t < S.t0) S.t0 = e.t; });          // clock origin = earliest thing we have seen
    let saidChanged = false, buzzChanged = false;
    fresh.forEach((e) => {
      if (e.kind === "SPEAK") { S.said.unshift({ t: e.t, text: e.text }); saidChanged = true; }
      else if (e.kind === "HAPTIC") {
        const top = S.buzz[0];
        if (top && top.cmd === e.text && e.t - top.lastT < 2.5 && e.text.startsWith("GUIDE")) top.lastT = e.t;   // collapse the 0.5 s refresh
        else S.buzz.unshift({ t: e.t, lastT: e.t, cmd: e.text });
        buzzChanged = true;
      } else if (e.kind === "SNAPSHOT") setAudit({ state: "checking", qid: e.text, data: null, key: "c" + e.seq });
      else if (e.kind === "AUDIT") S.auditNew = true;           // the structured result is in snap.audit (same snapshot)
    });
    if (saidChanged) {
      S.said = S.said.slice(0, 5);
      el.said.innerHTML = S.said.map((s) => `<li><time>${fmtTime(s.t)}</time><p></p></li>`).join("");
      $$("p", el.said).forEach((p, i) => (p.textContent = S.said[i].text));
    }
    if (buzzChanged) {
      S.buzz = S.buzz.slice(0, 4);
      el.buzzLog.innerHTML = S.buzz.map((b) => `<li><time>${fmtTime(b.t)}</time><span>${HAPTIC_LABEL[b.cmd] || b.cmd}</span></li>`).join("");
    }
  }

  function renderHaptics(snap) {
    const h = snap.haptic, gd = snap.guidance, now = performance.now();
    if (S.hseq === null) S.hseq = h.seq;               // never replay history when (re)connecting
    let kind = null;
    if (h.active) kind = { GUIDE_LEFT: "left", GUIDE_RIGHT: "right", GUIDE_BOTH: gd.dy_cm >= 0 ? "both-down" : "both-up", WARN: "warn" }[h.cmd] || null;
    G.set(kind);
    if (h.seq !== S.hseq) {
      S.hseq = h.seq;
      if (h.cmd === "LOCK") G.once("lock");
      else if (h.cmd === "COMPLETE") G.once("done");
      S.hSince = now;
    }
    if (h.active && (!S.hActive || h.cmd !== S.hCmd)) S.hSince = now;
    S.hActive = h.active; S.hCmd = h.cmd;
    M.textSwap(el.buzzCmd, h.active ? HAPTIC_LABEL[h.cmd] || h.cmd : "Silent");
    el.buzzSec.dataset.kind = h.active ? (h.cmd === "WARN" ? "warn" : h.cmd === "LOCK" || h.cmd === "COMPLETE" ? "good" : "") : "";

    // taps: badge at the pen + highlight in the panel
    (snap.taps || []).forEach((t) => {
      const key = Math.round((snap.t - t.age_s) * 50) + "-" + t.n;
      if (!S.tapSeen.has(key)) { S.tapSeen.add(key); if (t.age_s < 0.6) G.tap(t.n); }
    });
    $$("li", el.taps).forEach((li) => li.classList.toggle("is-hit", (snap.taps || []).some((t) => String(t.n) === li.dataset.n)));
  }

  function renderVoice(snap) {
    const v = snap.voice, now = performance.now();
    // Keep the dock open for 380 ms after speech ends, so a short gap between two phrases does not flap it.
    // Only after real speech: a first snapshot that is silent must not open it.
    if (v.speaking) { S.wasSpeaking = true; S.speakOffAt = 0; }
    else if (S.wasSpeaking && !S.speakOffAt) S.speakOffAt = now;
    const open = v.speaking || (S.wasSpeaking && now - S.speakOffAt < 380);
    if (!open) S.wasSpeaking = false;
    M.panel(el.dock, open);
    if (v.text) M.textSwap(el.dockText, v.text);
  }

  function renderImu(snap) {
    const m = snap.motion;
    el.imuSec.dataset.motion = m;
    M.iconSwap(el.imuIcon, m);
    M.textSwap(el.imuLabel, MOTION_LABEL[m] || m);
    once("light", snap.imu.light, (v) => (el.lightVal.textContent = v));
    S.imuDirty = true;
  }

  function renderGuidance(snap) {
    const gd = snap.guidance;
    const idle = snap.phase === "COMPLETE" ? "Test complete." : snap.phase === "AUDITING" ? "Checking the answer." : snap.pen ? "No active box." : "Waiting for the pen.";
    const quiet = snap.phase === "COMPLETE" || snap.phase === "AUDITING";
    setInstant(el.guideSay, !quiet && gd.speech ? gd.speech : idle);
    el.gDx.textContent = signed(gd.dx_cm);
    el.gDy.textContent = signed(gd.dy_cm);
    const label = { INSIDE: "Inside the box", OUTSIDE: "Outside the box", UNKNOWN: "No pen or box" }[gd.write_status];
    el.gState.dataset.s = gd.write_status;
    setInstant(el.gState, label);
  }

  /* ---- audit: empty -> checking -> result (or failed if the brain times out); the card resizes between them ---- */
  function setAudit(next) { if (S.audit.key !== next.key) { S.audit = next; drawAudit(); } }
  function renderAudit(snap) {
    const a = snap.audit;
    if (a && (S.auditNew || S.audit.state === "empty")) {
      S.auditNew = false;
      setAudit({ state: "result", qid: a.question_id, data: a, key: "r" + a.question_id + "@" + S.eventSeq });
    } else if (S.audit.state === "checking" && snap.phase !== "AUDITING") {
      setAudit({ state: "failed", qid: S.audit.qid, data: null, key: "f" + S.audit.key });
    }
  }
  function drawAudit(initial) {
    const A = S.audit, snap = S.snap;
    const n = snap ? Math.max(1, snap.progress.findIndex((p) => p.id === A.qid) + 1) : 1;
    let mark = null;
    M.resize(el.auditBox, () => {
      const inner = el.auditIn;
      if (A.state === "empty") { inner.innerHTML = '<p class="audit-empty">Nothing checked yet.</p>'; return; }
      if (A.state === "checking") { inner.innerHTML = `<p class="audit-empty">Checking answer ${n}.</p>`; return; }
      if (A.state === "failed") { inner.innerHTML = `<p class="audit-empty">Could not check answer ${n}. The test moved on.</p>`; return; }
      const d = A.data, ok = d.ink_present;
      inner.innerHTML =
        `<div class="audit-head ${ok ? "audit-ok" : "audit-bad"}"><span class="audit-mark"></span><span class="audit-title"></span></div>` +
        `<p class="audit-meta"></p><p class="audit-note"></p>`;
      $(".audit-title", inner).textContent = ok ? (d.ink_outside ? "Ink in the box, some outside" : "Ink in the box") : "No ink in the box";
      $(".audit-meta", inner).textContent = `Answer ${n}, ${Math.round(d.confidence * 100)}% confident`;
      $(".audit-note", inner).textContent = (d.note || "").slice(0, 160);
      const host = $(".audit-mark", inner);
      if (ok) { mark = M.makeCheck(); host.appendChild(mark); }
      else {
        host.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true" class="t-shake"><path d="M6.5 6.5l11 11M17.5 6.5l-11 11"/></svg>';
        mark = host.firstElementChild;
      }
    });
    if (mark) { if (mark.classList.contains("t-check")) M.success(mark); else M.shake(mark); }
  }

  function renderHealth(snap) {
    const errs = (snap.health && snap.health.errors) || {};
    const key = Object.keys(errs).sort().map((k) => k + errs[k]).join();
    once("health", key, () => {
      el.health.innerHTML = "";
      Object.keys(errs).filter((k) => errs[k] > 0).forEach((k) => {
        const c = document.createElement("span");
        c.className = "chip";
        c.textContent = `${COMP_NAME[k] || k} error`;
        el.health.appendChild(c);
      });
    });
    Object.keys(errs).forEach((k) => {
      if (errs[k] > 0 && !S.errSeen[k]) { S.errSeen[k] = 1; M.toast(`${COMP_NAME[k] || k} had an error. ${COMP_HINT[k] || ""}`, "warn", 7000); }
      if (!errs[k]) S.errSeen[k] = 0;
    });
  }

  function renderCamera(snap) {
    const cam = snap.camera || {};
    once("camLabel", cam.label, (v) => (el.camLabel.textContent = v || "Overhead camera"));
    const useImg = !!(S.source && S.source.kind === "live" && cam.has_frame && !S.imgFailed);
    once("useImg", useImg, (v) => {
      el.feedImg.hidden = !v; el.feedCanvas.hidden = v;
      if (v) el.feedImg.src = S.source.base + "/video.mjpg?" + Date.now(); else el.feedImg.removeAttribute("src");
    });
  }

  function renderMenu(snap) {
    const live = S.source && S.source.kind === "live";
    const canKey = !live || snap.debug.enabled;
    $$("[data-key]", el.menu).forEach((b) => {
      b.disabled = !canKey;
      if (["1", "2", "3", "4"].includes(b.dataset.key)) b.setAttribute("aria-pressed", String(snap.debug.override === { 1: "STILL", 2: "MOVING", 3: "WRITING", 4: "LIFTED" }[b.dataset.key]));
    });
    once("menuNote", canKey + "", () => {
      el.menuNote.textContent = canKey ? "Keys 1 to 4 set the pen state. 0 hands control back to the sensor. r, n and t send taps." : "Start the system with --debug-keys to send keys from here.";
    });
  }

  /* ================================================================== continuous loop (gsap.ticker) */
  const penSet = {};
  function frame() {
    const now = performance.now();
    const snap = S.snap;
    if (!snap || !S.pw) return;

    // pen glide (smoothing only; the data is already page coordinates)
    if (S.lastPen) {
      const x = S.lastPen.x * S.pw, y = S.lastPen.y * S.ph;
      if (!S.penPlaced) { gsap.set(el.pen, { x, y }); S.penPlaced = true; }
      else { penSet.x(x); penSet.y(y); }
      G.follow(x, y);
    }
    drawTrail(now);

    // camera-lost + pick-up cues (debounced so a 100 ms occlusion does not flash a banner)
    const lostFor = S.lostSince == null ? 0 : now - S.lostSince;
    const writingLost = lostFor > 250 && snap.motion === "WRITING";
    const reach = lostFor > 700 && (snap.phase === "IDLE" || snap.phase === "READING") && (snap.motion === "STILL" || snap.motion === "LIFTED");
    const lostOn = lostFor > 250 && !reach;
    once("lostText", lostOn ? (writingLost ? "w" : "h") : "", (v) => {
      if (v === "w") el.lost.innerHTML = 'Camera lost the pen. The pen says it is still <b>writing</b>.';
      else if (v === "h") el.lost.textContent = "Camera lost the pen. Holding the last cue.";
    });
    M.badge(el.lost, lostOn);
    M.badge(el.reach, reach);

    if (!el.feedCanvas.hidden) Cam.draw(snap);
    if (S.imuDirty) { drawImu(snap); S.imuDirty = false; }
    drawLamps(now);
    driveWave(snap, now);
  }

  function drawTrail(now) {
    const c = el.trail, ctx = S.trailCtx;
    if (!ctx) return;
    while (S.trail.length && now - S.trail[0].t > 2000) S.trail.shift();
    if (S.trail.length < 2 && !S.trailDirty) return;
    S.trailDirty = S.trail.length > 1;
    ctx.clearRect(0, 0, c.width, c.height);
    const k = S.dpr;
    for (let i = 1; i < S.trail.length; i++) {
      const a = S.trail[i - 1], b = S.trail[i];
      const age = Math.min(1, (now - b.t) / 2000);
      ctx.globalAlpha = (1 - age) * 0.8;
      ctx.lineWidth = (1 + 3.2 * (1 - age)) * k;
      ctx.strokeStyle = C.pen;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(a.x * S.pw * k, a.y * S.ph * k);
      ctx.lineTo(b.x * S.pw * k, b.y * S.ph * k);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  function drawImu(snap) {
    const c = el.imuWave, ctx = S.imuCtx;
    if (!ctx || !c.width) return;
    const w = c.width, h = c.height, s = snap.imu.samples;
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = C.line; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();
    if (!s || s.length < 2) return;
    const abs = [];
    s.forEach((v) => { abs.push(Math.abs(v[0]), Math.abs(v[1]), Math.abs(v[2] - 1)); });
    abs.sort((a, b) => a - b);
    const scale = Math.max(0.2, abs[Math.floor(abs.length * 0.97)] * 1.3);
    const col = MOTION_HEX[snap.motion] || C.still;
    [[0, 1, 1.6], [1, 0.55, 1.2], [2, 0.3, 1.2]].forEach(([ch, alpha, lw]) => {
      ctx.globalAlpha = alpha; ctx.strokeStyle = col; ctx.lineWidth = lw * S.dpr; ctx.lineJoin = "round";
      ctx.beginPath();
      s.forEach((v, i) => {
        const val = ch === 2 ? v[2] - 1 : v[ch];
        const x = (i / (s.length - 1)) * (w - 1), y = h / 2 - Math.max(-1, Math.min(1, val / scale)) * (h / 2 - 5 * S.dpr);
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      });
      ctx.stroke();
    });
    ctx.globalAlpha = 1;
  }

  // buzzer lamps replay the firmware patterns (PLAN.md section 3) so the room sees the rhythm
  function drawLamps(now) {
    let L = false, R = false, col = C.moving;
    if (S.hActive) {
      const t = (now - S.hSince) / 1000, on = (p, a) => t % p < a;
      switch (S.hCmd) {
        case "GUIDE_LEFT": L = on(0.25, 0.1); break;
        case "GUIDE_RIGHT": R = on(0.25, 0.1); break;
        case "GUIDE_BOTH": L = R = on(0.5, 0.1); break;
        case "WARN": col = C.warn; L = Math.floor(t / 0.1) % 2 === 0; R = !L; break;
        case "LOCK": col = C.writing; L = R = t < 0.08 || (t > 0.16 && t < 0.24); break;
        case "COMPLETE": col = C.writing; L = R = t < 0.36; break;
      }
    }
    el.lampL.style.setProperty("--lamp-color", col); el.lampR.style.setProperty("--lamp-color", col);
    el.lampL.classList.toggle("is-on", L); el.lampR.classList.toggle("is-on", R);
  }

  // voice waveform: real level when the voice engine provides one, otherwise a speech-shaped envelope
  function driveWave(snap, now) {
    const W = S.wave, v = snap.voice;
    let target = 0;
    if (v.speaking) {
      if (v.level != null) target = v.level;
      else {
        const t = now / 1000;
        const syl = Math.abs(Math.sin(t * Math.PI * 3.4)), phr = Math.abs(Math.sin(t * Math.PI * 1.1 + 1.1));
        target = Math.min(1, 0.1 + (0.55 * syl + 0.28 * phr) * (0.72 + 0.28 * Math.random()));
      }
    }
    W.level += (target - W.level) * 0.4;
    if (now - W.last > 42) {
      W.last = now;
      W.hist.push(W.level); W.hist.shift();
      if (el.dock.dataset.open === "true" || W.level > 0.02) W.setters.forEach((fn, i) => fn(Math.max(0.07, W.hist[i])));
    }
  }

  /* ================================================================== demo camera (paper, printed sheet, ink, hand shadow) */
  const Cam = {
    paper: null, ink: null, inkCtx: null, drawn: 0, w: 0, h: 0, ctx: null, sheetFor: null, dirty: true, handKey: "",
    resize(w, h, dpr) {
      this.w = Math.round(w * dpr); this.h = Math.round(h * dpr);
      const fc = el.feedCanvas; fc.width = this.w; fc.height = this.h; this.ctx = fc.getContext("2d");
      this.ink = document.createElement("canvas"); this.ink.width = this.w; this.ink.height = this.h; this.inkCtx = this.ink.getContext("2d");
      this.drawn = 0; this.paper = null; this.dirty = true;
    },
    buildPaper(snap) {
      const { w, h } = this, src = S.source;
      const c = document.createElement("canvas"); c.width = w; c.height = h;
      const x = c.getContext("2d");
      x.fillStyle = C.paper; x.fillRect(0, 0, w, h);
      // paper grain
      const g = x.createImageData(w, h), d = g.data;
      for (let i = 0; i < d.length; i += 4) { const n = (Math.random() - 0.5) * 7; d[i] = d[i + 1] = d[i + 2] = 128 + n; d[i + 3] = 22; }
      const gc = document.createElement("canvas"); gc.width = w; gc.height = h; gc.getContext("2d").putImageData(g, 0, 0);
      x.drawImage(gc, 0, 0);
      const demo = src && src.kind === "fake";
      const u = h / 792;                                   // 1 pt on a US Letter page
      x.fillStyle = "#1c2230"; x.textBaseline = "alphabetic";
      if (demo) {
        x.font = `600 ${13 * u}px "Instrument Sans", sans-serif`; x.fillText("Illumin demo test  -  free response", 2.0 / 21.59 * w, 1.6 / 27.94 * h);
        x.font = `400 ${9 * u}px "Instrument Sans", sans-serif`; x.textAlign = "right"; x.fillText("Name: ______________________", (21.59 - 2.0) / 21.59 * w, 1.6 / 27.94 * h); x.textAlign = "left";
        const rows = [3.6, 11.7, 20.1];
        I.sources.QUESTIONS.forEach((q, i) => { x.font = `400 ${11 * u}px "Instrument Sans", sans-serif`; x.fillText(`${i + 1}. ${q.text}`, 2.0 / 21.59 * w, rows[i] / 27.94 * h); });
      } else {
        x.font = `500 ${15 * u}px "Instrument Sans", sans-serif`; x.fillStyle = "#6b7482"; x.textAlign = "center";
        x.fillText("Waiting for the camera", w / 2, h * 0.5); x.textAlign = "left";
      }
      // the printed answer boxes: thick, dark, 4 pt
      x.strokeStyle = "#0d1119"; x.lineWidth = 4 * u; x.lineJoin = "miter";
      (snap.boxes || []).forEach((b) => x.strokeRect(b.xmin * w, b.ymin * h, (b.xmax - b.xmin) * w, (b.ymax - b.ymin) * h));
      // camera feel: soft light falloff
      const rg = x.createRadialGradient(w * 0.5, h * 0.45, h * 0.2, w * 0.5, h * 0.5, h * 0.78);
      rg.addColorStop(0, "rgba(255,255,255,0)"); rg.addColorStop(1, "rgba(20,28,45,0.10)");
      x.fillStyle = rg; x.fillRect(0, 0, w, h);
      this.paper = c; this.sheetFor = (snap.boxes || []).length + ":" + (demo ? 1 : 0);
    },
    draw(snap) {
      if (!this.ctx || !this.w) return;
      const demo = S.source && S.source.kind === "fake";
      const sheet = (snap.boxes || []).length + ":" + (demo ? 1 : 0);
      if (!this.paper || this.sheetFor !== sheet) { this.buildPaper(snap); this.dirty = true; }
      const ink = demo ? S.source.ink || [] : [];
      const hand = demo && !snap.pen && S.lastPen && snap.motion !== "STILL" ? "h" : "";
      // The canvas changes only when ink grows, the hand shadow toggles, or the size changes.
      // Redrawing it every frame made a rounded-clip repaint per frame (400+ ms stalls on software GPUs).
      if (!this.dirty && ink.length === this.drawn && hand === this.handKey) return;
      this.dirty = false; this.handKey = hand;
      const ctx = this.ctx, w = this.w, h = this.h;
      ctx.drawImage(this.paper, 0, 0);
      if (demo) {
        if (ink.length < this.drawn) { this.inkCtx.clearRect(0, 0, w, h); this.drawn = 0; }     // demo loop restarted
        const ic = this.inkCtx;
        ic.strokeStyle = "rgba(24,36,104,0.9)"; ic.lineWidth = 0.0042 * w; ic.lineCap = "round";
        ic.beginPath();
        for (let i = this.drawn; i < ink.length; i++) { const sg = ink[i]; ic.moveTo(sg[0] * w, sg[1] * h); ic.lineTo(sg[2] * w, sg[3] * h); }
        ic.stroke();
        this.drawn = ink.length;
        ctx.drawImage(this.ink, 0, 0);
        if (hand) {                                                          // a hand covering the pen
          const hx = (S.lastPen.x + 0.035) * w, hy = (S.lastPen.y + 0.045) * h;
          const rg = ctx.createRadialGradient(hx, hy, 0, hx, hy, 0.15 * w);
          rg.addColorStop(0, "rgba(28,26,34,0.62)"); rg.addColorStop(0.7, "rgba(28,26,34,0.38)"); rg.addColorStop(1, "rgba(28,26,34,0)");
          ctx.fillStyle = rg; ctx.fillRect(hx - 0.2 * w, hy - 0.2 * w, 0.4 * w, 0.4 * w);
        }
      }
    },
  };

  /* ================================================================== sizing */
  function measure() {
    const r = el.page.getBoundingClientRect();
    S.pw = r.width; S.ph = r.height; S.dpr = Math.min(2, window.devicePixelRatio || 1);
    [el.trail].forEach((c) => { c.width = Math.round(r.width * S.dpr); c.height = Math.round(r.height * S.dpr); });
    S.trailCtx = el.trail.getContext("2d");
    Cam.resize(r.width, r.height, Math.min(S.dpr, 1.5));
    const ir = el.imuWave.getBoundingClientRect();
    el.imuWave.width = Math.round(ir.width * S.dpr); el.imuWave.height = Math.round(ir.height * S.dpr);
    S.imuCtx = el.imuWave.getContext("2d");
    S.imuDirty = true;
    S.penPlaced = false;
  }

  /* ================================================================== operator menu + keys */
  function setupMenu() {
    const btn = el.connBtn, menu = el.menu;
    const set = (open) => { M.dropdown(menu, open); btn.setAttribute("aria-expanded", String(open)); };
    btn.addEventListener("click", (e) => { e.stopPropagation(); set(menu.dataset.open !== "true"); });
    document.addEventListener("click", (e) => { if (menu.dataset.open === "true" && !menu.contains(e.target)) set(false); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && menu.dataset.open === "true") { set(false); btn.focus(); } });
    $$("[data-key]", menu).forEach((b) => b.addEventListener("click", () => sendKey(b.dataset.key)));
    $$("[data-source-pick]", menu).forEach((b) => b.addEventListener("click", () => {
      const q = new URLSearchParams(location.search); q.set("source", b.dataset.sourcePick); location.search = q.toString();
    }));
    window.addEventListener("keydown", (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey || e.repeat) return;
      if (e.target.closest && e.target.closest("input, textarea, select")) return;
      const k = e.key.toLowerCase();
      if (KEYS.includes(k)) sendKey(k);
    });
  }
  async function sendKey(k) {
    if (!S.ctl) return;
    const res = await S.ctl.key(k);
    if (res && res.ok === false && res.status === 409) M.toast("Start the system with --debug-keys to send keys from here.", "warn");
    else if (res === false) return;
  }

  /* ================================================================== boot */
  function boot() {
    [["app", "#app"], ["phaseName", "#phaseName"], ["phaseSub", "#phaseSub"], ["qNum", "#qNum"], ["qTotal", "#qTotal"], ["qText", "#qText"],
      ["steps", "#steps"], ["said", "#said"], ["page", "#page"], ["feedImg", "#feedImg"], ["feedCanvas", "#feedCanvas"], ["boxes", "#boxes"],
      ["trail", "#trail"], ["pen", "#pen"], ["lost", "#lost"], ["reach", "#reach"], ["camLabel", "#camLabel"], ["dock", "#dock"],
      ["dockText", "#dockText"], ["wave", "#wave"], ["imuSec", "#imuSec"], ["imuIcon", "#imuIcon"], ["imuLabel", "#imuLabel"], ["imuWave", "#imuWave"],
      ["lightVal", "#lightVal"], ["taps", "#taps"], ["buzzSec", "#buzzSec"], ["lampL", "#lampL"], ["lampR", "#lampR"], ["buzzCmd", "#buzzCmd"],
      ["buzzLog", "#buzzLog"], ["guideSay", "#guideSay"], ["gDx", "#gDx"], ["gDy", "#gDy"], ["gState", "#gState"], ["auditBox", "#auditBox"],
      ["auditIn", "#auditIn"], ["health", "#health"], ["connBtn", "#connBtn"], ["connLabel", "#connLabel"], ["menu", "#connMenu"], ["menuNote", "#menuNote"],
    ].forEach(([k, s]) => (el[k] = $(s)));

    G.mount();
    penSet.x = gsap.quickTo(el.pen, "x", { duration: 0.12, ease: "power3.out" });
    penSet.y = gsap.quickTo(el.pen, "y", { duration: 0.12, ease: "power3.out" });
    for (let i = 0; i < WAVE_BARS; i++) { const b = document.createElement("i"); el.wave.appendChild(b); S.wave.setters.push(gsap.quickTo(b, "scaleY", { duration: 0.14, ease: "power2.out" })); }
    drawAudit(true);
    setupMenu();
    measure();
    new ResizeObserver(measure).observe(el.page);
    new ResizeObserver(() => { const ir = el.imuWave.getBoundingClientRect(); if (Math.round(ir.width * S.dpr) !== el.imuWave.width) measure(); }).observe(el.imuWave);
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { Cam.paper = null; Cam.dirty = true; });
    el.feedImg.addEventListener("error", () => { S.imgFailed = true; S.cache.useImg = undefined; setTimeout(() => (S.imgFailed = false), 2500); });

    S.ctl = I.sources.connect({
      onSnapshot(snap) { try { render(snap); } catch (e) { if (!S.errorLogged) { S.errorLogged = true; console.error("[illumin-ui] render failed", e); } } },
      onSource(kind, src) {
        S.source = src;
        el.app.dataset.source = kind;
        el.connLabel.textContent = kind === "live" ? "Live" : "Demo data";
        $$("[data-source-pick]", el.menu).forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.sourcePick === kind)));
        // a different source means different history: start clean
        S.hseq = null; S.eventSeq = 0; S.said = []; S.buzz = []; S.tapSeen.clear(); S.trail = []; S.lastPen = null; S.penPlaced = false;
        S.cache.useImg = undefined; Cam.paper = null; Cam.dirty = true; S.audit = { state: "empty", qid: null, data: null, key: "" };
        el.said.innerHTML = ""; el.buzzLog.innerHTML = ""; drawAudit(true);
      },
      onNotice(msg) { M.toast(msg, "warn", 5200); },
    });
    I.app.frame = frame; I.app.render = render;
    gsap.ticker.add(frame);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
})();
