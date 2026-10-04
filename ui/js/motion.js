/* transitions.dev orchestration.  The CSS lives in app.css (t-* classes); durations are read from the
   :root tokens so they never drift from tokens.css.  Rules kept from the skill:
   - reflow (void el.offsetWidth) between removing and re-adding a class, or the animation will not replay
   - clean up .is-closing with a timer, or the next open starts from the closing scale
   - interruptible: a newer change always wins over a pending older one. */
(function () {
  const I = (window.Illumin = window.Illumin || {});
  const M = (I.motion = {});

  const root = () => document.documentElement;
  const token = (name) => getComputedStyle(root()).getPropertyValue(name).trim();
  const ms = (name) => { const v = token(name); return v.endsWith("ms") ? parseFloat(v) : parseFloat(v) * 1000; };
  const reflow = (el) => void el.offsetWidth;
  const reduced = () => window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;
  const booting = () => !!document.querySelector(".is-booting");
  M.reduced = reduced;
  M.ms = ms;

  /* Text states swap: old text leaves up with a blur, new text arrives from below. */
  M.textSwap = function (el, text) {
    text = String(text == null ? "" : text);
    if (el.dataset.txt === text) return;
    el.dataset.txt = text;
    const mine = (el._swap = (el._swap || 0) + 1);
    if (!el.dataset.ready || booting() || reduced()) {
      el.textContent = text;
      el.dataset.ready = "1";
      el.classList.remove("is-out", "is-in-start");
      return;
    }
    el.classList.add("is-out");
    clearTimeout(el._swapTimer);
    el._swapTimer = setTimeout(() => {
      if (el._swap !== mine) return;
      el.classList.remove("is-out");
      el.classList.add("is-in-start");
      el.textContent = text;
      reflow(el);
      el.classList.remove("is-in-start");
    }, ms("--text-swap-dur"));
  };

  /* Number pop-in: every digit re-enters with a staggered blurred slide. */
  M.digits = function (el, text) {
    text = String(text);
    if (el.dataset.txt === text) return;
    const first = !el.dataset.ready;
    el.dataset.txt = text;
    el.dataset.ready = "1";
    el.setAttribute("aria-label", text);
    el.textContent = "";
    const stagger = ms("--digit-stagger");
    [...text].forEach((ch, i) => {
      const s = document.createElement("span");
      s.className = "t-digit";
      s.setAttribute("aria-hidden", "true");
      s.textContent = ch;
      if (first || booting() || reduced()) s.style.animation = "none";
      else s.style.animationDelay = i * stagger + "ms";
      el.appendChild(s);
    });
  };

  /* Panel reveal. */
  M.panel = function (el, open) {
    const v = String(!!open);
    if (el.dataset.open === v) return;
    el.dataset.open = v;
    el.setAttribute("aria-hidden", String(!open));
  };

  /* Menu dropdown (origin-aware). Keeps .is-closing for the close duration, then clears it. */
  M.dropdown = function (el, open) {
    const v = String(!!open);
    if (el.dataset.open === v) return;
    clearTimeout(el._closing);
    if (open) {
      el.classList.remove("is-closing");
      el.dataset.open = "true";
    } else {
      el.classList.add("is-closing");
      el.dataset.open = "false";
      el._closing = setTimeout(() => el.classList.remove("is-closing"), ms("--dropdown-close-dur"));
    }
  };

  /* Icon swap: icons in one slot, the one whose data-k matches is active. */
  M.iconSwap = function (slot, key) {
    if (slot.dataset.k === key) return;
    slot.dataset.k = key;
    slot.querySelectorAll(".t-icon").forEach((ic) => ic.classList.toggle("is-active", ic.dataset.k === key));
  };

  /* Build the success-check markup (three wrappers so rotate / bob / blur each own a transform). */
  M.makeCheck = function (cls) {
    const el = document.createElement("span");
    el.className = "t-check " + (cls || "");
    el.innerHTML =
      '<span class="t-check-rot"><span class="t-check-bob">' +
      '<svg viewBox="0 0 24 24" aria-hidden="true"><path class="t-check-path" d="M5 12.5l4.5 4.5L19 7.5"/></svg>' +
      "</span></span>";
    return el;
  };
  /* Success check: dash length is measured from the real path (never the 20 placeholder). */
  M.success = function (el) {
    const path = el.querySelector(".t-check-path");
    if (path && path.getTotalLength) el.style.setProperty("--len", Math.ceil(path.getTotalLength()) + 1);
    el.classList.remove("is-playing");
    if (booting() || reduced()) { el.classList.add("is-playing"); return; }
    reflow(el);
    el.classList.add("is-playing");
  };

  /* Error shake: .is-error (colour) and .is-shaking (motion) stay independent so the shake can replay. */
  M.shake = function (el) {
    if (reduced()) return;
    el.classList.add("t-shake");
    el.classList.remove("is-shaking");
    reflow(el);
    el.classList.add("is-shaking");
    clearTimeout(el._shakeTimer);
    el._shakeTimer = setTimeout(() => el.classList.remove("is-shaking"), ms("--shake-dur-a") * 3 + ms("--shake-dur-b") * 2 + 40);
  };

  /* Notification badge slide. */
  M.badge = function (el, on) { el.classList.toggle("is-on", !!on); };

  /* Card resize: tween height between two content states. `mutate` swaps the content. */
  M.resize = function (box, mutate) {
    if (!box.dataset.ready || booting() || reduced()) {
      mutate();
      box.dataset.ready = "1";
      box.style.height = "";
      return;
    }
    const from = box.offsetHeight;
    box.style.height = from + "px";
    mutate();
    const inner = box.firstElementChild;
    const to = inner ? inner.offsetHeight : from;
    reflow(box);
    box.style.height = to + "px";
    clearTimeout(box._rz);
    box._rz = setTimeout(() => { box.style.height = ""; }, ms("--resize-dur") + 40);
  };

  /* Toast: rises with fade + blur, slower in than out. Same message within 6 s is not repeated. */
  M.toast = function (message, kind, ttl) {
    const host = document.getElementById("toasts");
    if (!host) return;
    const now = Date.now();
    M._seen = M._seen || {};
    if (M._seen[message] && now - M._seen[message] < 6000) return;
    M._seen[message] = now;
    while (host.children.length >= 3) host.firstElementChild.remove();
    const t = document.createElement("div");
    t.className = "toast t-toast";
    t.dataset.kind = kind || "info";
    t.textContent = message;
    host.appendChild(t);
    reflow(t);
    t.classList.add("is-on");
    setTimeout(() => {
      t.classList.remove("is-on");
      setTimeout(() => t.remove(), ms("--toast-close-dur") + 60);
    }, ttl || 4200);
  };
})();
