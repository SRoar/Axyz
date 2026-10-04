/* Gesture layer: minimal line glyphs that appear at the pen when a buzzer fires, so the room can SEE what the
   student FEELS.  The one place this UI spends its boldness; everything else stays quiet.

     G.set(kind)   continuous cue while the buzzers are sounding:
                   'left' | 'right' | 'both-down' | 'both-up' | 'warn' | null
     G.once(kind)  one-shot:  'lock' (two pulses) | 'done' (three rising bars, like 600/900/1200 Hz)
     G.tap(n)      tap badge (1 repeat, 2 skip, 3 done)
     G.follow(x,y) glide the layer to the pen (px inside .page)

   GSAP rules followed (gsap-skills): one timeline per cue (no chained delays), defaults on the timeline,
   transforms and opacity only, kill before rebuild, gsap.matchMedia() for prefers-reduced-motion. */
(function () {
  const I = (window.Illumin = window.Illumin || {});
  const G = (I.gestures = {});
  const $ = (s) => document.querySelector(s);

  let layer, caption, capWrap, tapBadge, tapDots, tapWord;
  let groups = {};
  let tl = null, current = null, onceTl = null, tapTimer = null, capTimer = null;
  let followX, followY, reduce = false;

  const TAP_WORD = { 1: "repeat", 2: "skip", 3: "done" };
  const CAPTION = { lock: "Locked in", warn: "Out of bounds", done: "Answer recorded" };

  G.mount = function () {
    layer = $("#gestures");
    caption = $("#gCaption");
    capWrap = caption.parentElement;           // the pill; .is-on lives here, the text swap lives on the span
    tapBadge = $("#tapBadge");
    tapDots = $("#tapDots");
    tapWord = $("#tapWord");
    ["Left", "Right", "Down", "Up", "Sides", "Lock", "Warn", "Done"].forEach((k) => (groups[k] = $("#g" + k)));
    gsap.set(layer, { x: -999, y: -999 });
    followX = gsap.quickTo(layer, "x", { duration: 0.16, ease: "power3.out" });
    followY = gsap.quickTo(layer, "y", { duration: 0.16, ease: "power3.out" });
    const mm = gsap.matchMedia();
    mm.add("(prefers-reduced-motion: reduce)", () => { reduce = true; return () => { reduce = false; }; });
    G.hideAll();
  };

  G.follow = function (x, y) {
    if (!layer) return;
    if (layer._placed) { followX(x); followY(y); }
    else { gsap.set(layer, { x, y }); layer._placed = true; }
  };

  const allGroups = () => Object.values(groups);
  G.hideAll = function () {
    if (tl) { tl.kill(); tl = null; }
    if (onceTl) { onceTl.kill(); onceTl = null; }
    gsap.set(allGroups(), { opacity: 0 });
    gsap.set(layer.querySelectorAll(".gc, .gs, .gr, .ga, .gb"), { clearProps: "all" });
    current = null;
    setCaption("");
  };

  function tone(t) { layer.dataset.tone = t || "pen"; layer.style.color = t === "warn" ? "var(--warn)" : t === "good" ? "var(--good)" : "var(--pen)"; }

  function setCaption(text, hold) {
    clearTimeout(capTimer);
    if (!text) { capWrap.classList.remove("is-on"); return; }
    I.motion.textSwap(caption, text);
    capWrap.classList.add("is-on");
    if (hold) capTimer = setTimeout(() => capWrap.classList.remove("is-on"), hold);
  }

  function fade(group, to, d) { return gsap.to(group, { opacity: to, duration: d == null ? 0.2 : d, overwrite: "auto" }); }

  /* ---------------------------------------------------------------- continuous cues */
  const SWEEP = { left: ["Left", "x", -1], right: ["Right", "x", 1], "both-down": ["Down", "y", 1], "both-up": ["Up", "y", -1] };

  G.set = function (kind) {
    if (kind === current) return;
    if (onceTl && kind && kind !== "warn") { onceTl.kill(); onceTl = null; }
    if (tl) { tl.kill(); tl = null; }
    const prev = current;
    current = kind;

    if (!kind) {                                           // buzzers stopped: fade whatever was showing
      if (prev === "warn") setCaption("");
      allGroups().forEach((g) => { if (!onceTl || (g !== groups.Lock && g !== groups.Done)) fade(g, 0, 0.18); });
      return;
    }
    allGroups().forEach((g) => { if (!onceTl || (g !== groups.Lock && g !== groups.Done)) gsap.set(g, { opacity: 0 }); });

    if (kind === "warn") return warn();

    const [name, axis, dir] = SWEEP[kind];
    const g = groups[name];
    const chevs = g.querySelectorAll(".gc");
    tone("pen");
    setCaption("");
    gsap.set(g, { opacity: 1 });
    if (kind.startsWith("both")) { gsap.set(groups.Sides, { opacity: 1 }); }

    if (reduce) {                                          // static glyph, no loop
      chevs.forEach((c, i) => gsap.set(c, { [axis]: dir * (22 + i * 14), opacity: 1 - i * 0.28 }));
      return;
    }
    tl = gsap.timeline({ defaults: { ease: "power2.out" } });
    // one sweep of three chevrons travelling away from the pen, repeated while the buzzer sounds
    tl.fromTo(chevs,
      { [axis]: dir * 14, opacity: 0 },
      { keyframes: { [axis]: [dir * 14, dir * 40, dir * 66], opacity: [0, 1, 0], easeEach: "power1.inOut" }, duration: 0.85, stagger: 0.14, repeat: -1, repeatDelay: 0.18 }, 0);
    if (kind.startsWith("both")) {
      // both buzzers: 100 ms on / 400 ms off, shown on the two side dots
      const sides = groups.Sides.querySelectorAll(".gs");
      tl.fromTo(sides, { opacity: 0.18 }, { opacity: 1, duration: 0.1, ease: "none", repeat: -1, repeatDelay: 0.4, yoyo: true, yoyoEase: "none" }, 0);
    }
  };

  function warn() {
    tone("warn");
    setCaption(CAPTION.warn);
    gsap.set(groups.Warn, { opacity: 1 });
    const l = groups.Warn.querySelector(".ga-l"), r = groups.Warn.querySelector(".ga-r");
    if (reduce) { gsap.set([l, r], { opacity: 1 }); return; }
    // harsh alternating buzzers every 100 ms, exactly like the firmware pattern
    tl = gsap.timeline({ repeat: -1, defaults: { ease: "none", duration: 0.001 } });
    tl.set(l, { opacity: 1 }, 0).set(r, { opacity: 0.12 }, 0)
      .set(l, { opacity: 0.12 }, 0.1).set(r, { opacity: 1 }, 0.1)
      .to({}, { duration: 0.1 }, 0.1);
  }

  /* ---------------------------------------------------------------- one-shots */
  G.once = function (kind) {
    if (onceTl) { onceTl.kill(); onceTl = null; }
    if (kind === "lock") return lock();
    if (kind === "done") return done();
  };

  function lock() {
    if (tl && current !== "warn") { tl.kill(); tl = null; current = null; }
    tone("good");
    setCaption(CAPTION.lock, 1500);
    const rings = groups.Lock.querySelectorAll(".gr");
    gsap.set(groups.Lock, { opacity: 1 });
    if (reduce) { gsap.set(rings, { opacity: 1, scale: 1, svgOrigin: "0 0" }); onceTl = gsap.timeline().to(groups.Lock, { opacity: 0, duration: 0.2 }, 1.4); return; }
    onceTl = gsap.timeline({ defaults: { ease: "power3.out" }, onComplete: () => { onceTl = null; if (!current) tone("pen"); } });
    // two 80 ms pulses (1000 Hz then 1500 Hz): the rings contract onto the pen twice, then let go
    onceTl.fromTo(rings, { scale: 1.9, opacity: 0, svgOrigin: "0 0" }, { scale: 1, opacity: 1, duration: 0.22, stagger: 0.08 }, 0)
      .fromTo(rings, { scale: 1.9, opacity: 0.0, svgOrigin: "0 0" }, { scale: 1, opacity: 1, duration: 0.22, stagger: 0.08 }, 0.34)
      .to(rings, { opacity: 0, duration: 0.4, ease: "power2.in", stagger: 0.05 }, 1.1)
      .to(groups.Lock, { opacity: 0, duration: 0.01 }, 1.6);
  }

  function done() {
    tone("good");
    setCaption(CAPTION.done, 1800);
    const bars = groups.Done.querySelectorAll(".gb");
    gsap.set(groups.Done, { opacity: 1 });
    if (reduce) { gsap.set(bars, { scaleY: 1, transformOrigin: "50% 100%" }); onceTl = gsap.timeline().to(groups.Done, { opacity: 0, duration: 0.2 }, 1.6); return; }
    onceTl = gsap.timeline({ defaults: { ease: "power3.out" }, onComplete: () => { onceTl = null; if (!current) tone("pen"); } });
    // ascending 600 / 900 / 1200 Hz, 120 ms each
    onceTl.fromTo(bars, { scaleY: 0, transformOrigin: "50% 100%" }, { scaleY: 1, duration: 0.24, stagger: 0.12 }, 0)
      .to(groups.Done, { opacity: 0, duration: 0.35, ease: "power2.in" }, 1.5);
  }

  /* ---------------------------------------------------------------- tap badge (notification-badge transition) */
  G.tap = function (n) {
    tapDots.innerHTML = "<i></i>".repeat(n);
    tapWord.textContent = TAP_WORD[n] || "";
    I.motion.badge(tapBadge, true);
    clearTimeout(tapTimer);
    tapTimer = setTimeout(() => I.motion.badge(tapBadge, false), 1400);
  };
})();
