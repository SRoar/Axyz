# Illumin browser UI

One screen that combines everything the project can show: the overhead camera with labeled answer boxes, the voice
waveform (pops up while the voice speaks), the pen probe (IMU state, accelerometer trace, taps), the buzzers, guidance,
the answer audit, and minimal gesture glyphs that appear at the pen when a buzzer fires.

It does not replace the cv2 HUD (`src/hud.py`). Run both, or either.

## Run it

```bash
# 1. Demo data, no Python needed: just open the file (works offline, GSAP and fonts are vendored)
open ui/index.html                      # or double-click it

# 2. Live: the UI served by your Dev 4 process, wired to the real System
python -m src.main --web                              # all fakes (scripted world), http://localhost:8765
python -m src.main --web --real all --debug-keys      # full system, keyboard rescue from the UI too
python -m src.main --headless --web                   # no cv2 window; serves until Ctrl-C
python -m src.main --web 9000 --web-host 0.0.0.0      # reachable from a phone/tablet on the same Wi-Fi
```
WSL2: open `http://localhost:8765` in the Windows browser. If the page cannot reach Python it shows demo data and says so
(top right: **Demo data** / **Live**). It switches to live by itself when the bridge appears.

URL parameters: `?source=fake|live` force a source, `?server=http://host:8765` point a file-opened UI at a bridge,
`?speed=2` demo speed, `?at=12.5` start the demo 12.5 s in (screenshots, rehearsal).

Operator keys (same as the cv2 HUD, work anywhere on the page, and from the dropdown top right):
`1` still, `2` moving, `3` writing, `4` lifted, `0` hand control back to the sensor, `r` repeat tap, `n` (or `t`) skip tap.
Live mode needs `--debug-keys`, otherwise the UI tells you.

## How it plugs in

```
System.step() --> Snapshotter (src/web.py) --> /events (SSE, 20 Hz) --> ui/js/sources.js --> ui/js/app.js
tracker frame --> /video.mjpg                                          demo world (JS) ------^
```
Both the bridge and the JS demo world emit the same JSON snapshot (schema v1, documented at the top of `src/web.py`).
`tests/test_web.py` runs the demo world under Node and fails if its keys differ from the bridge's, and if its brain port
stops following the reference brain. To add a field: add it in `Snapshotter.build()`, in `createFake().build()`, run pytest.

## Where each motion comes from

transitions.dev (`css/app.css` t-* classes, `js/motion.js`, tokens in `css/tokens.css`):

| Where | Transition |
|---|---|
| Phase name, question, status labels, buzzer command, IMU state label, spoken line | Text states swap |
| Question number | Number pop-in |
| IMU state icon, answer-status icons (list and box labels) | Icon swap |
| Voice dock under the camera | Panel reveal |
| Operator menu | Menu dropdown |
| "Camera lost the pen", "Pick up the pen", tap badge at the pen | Notification badge slide |
| Audit card changing between empty, checking, result | Card resize |
| Ink in the box | Success check |
| No ink in the box, box turning red on WARN | Error state shake |
| Status messages | Toast (derived from the panel tokens; transitions.dev does not publish its toast tokens) |

Not animated on purpose (updates many times a second): guidance numbers, the navigating subtitle, pen position.

GSAP (`js/gestures.js`, `js/app.js`): gesture timelines at the pen, pen glide (`quickTo`), voice waveform bars,
all driven from `gsap.ticker`. `gsap.matchMedia` switches the gestures to static glyphs under reduced motion.

Deviations from transitions.dev, deliberately: three overshoot easings (number digits, badge pop, check bob) are
flattened to its own exponential-out curve, because Impeccable treats bounce easing as an anti-pattern. The
per-transition reference CSS could not be fetched when this was built, so each transition was written from the published
tokens, hooks and decision rules; run `transitions refine` (or `npx transitions-dev add <name>`) and diff if you want
byte-for-byte parity.

## Needs from the logic side (everything works without these)

| Owner | Ask | Until then |
|---|---|---|
| Dev 3 | `VoiceEngine.level() -> float` 0..1 (loudness of audio playing now) | waveform is a speech-shaped envelope while `is_speaking()` |
| Dev 3 | `brain.results = {question_id: "answered" \| "skipped" \| "unchecked"}` | answer list infers skips from the spoken phrases; breaks if `speech.py` rewords them |
| Dev 3 | decide whether the brain speaks "Pick up the pen" when the pen is not visible at the start | the UI shows the cue, the student hears nothing |
| Dev 2 | `TrackerReading.frame` must be PAGE-RECTIFIED (boxes are drawn in page coordinates); optional `tracker.source_name` for the camera label (for example "iPhone") | boxes misalign if the frame is raw |
| Dev 1 | none. Lamps replay the firmware patterns from PLAN.md section 3; if the patterns change, edit `drawLamps()` in `app.js` | |

## Files

```
ui/index.html            structure + icon sprite
ui/css/tokens.css        OKLCH palette, type, concentric radii, transitions.dev :root block (+3 overrides)
ui/css/app.css           layout, components, t-* transitions, reduced-motion guards
ui/js/motion.js          transitions.dev orchestration (text swap, panel, dropdown, icon swap, success, shake, ...)
ui/js/sources.js         demo world (port of fakes.py + the reference brain), live SSE source, auto connect + fallback
ui/js/gestures.js        GSAP gesture glyphs at the pen
ui/js/app.js             snapshot renderer, demo camera, ticker loop, operator menu
ui/vendor, ui/fonts      GSAP 3.15 (standard no-charge license), Instrument Sans + Newsreader (OFL)
ui/tests/                Node helper used by tests/test_web.py
```
