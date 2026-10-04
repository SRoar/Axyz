# Illumin UI: design record (Impeccable-style DESIGN.md)

**Product.** Assistive exam tool. The student is blind; this screen is for the people watching (judges, operator, teachers).
Its job: make the invisible visible, so a room can see what the pen feels and the voice says.

**One bold thing.** Gesture glyphs drawn on the paper at the pen, echoing each buzzer. Everything else is quiet.

## Color (OKLCH, neutrals tinted toward pen-ink hue 255; no pure black, no pure grey)
| Role | Token | Value |
|---|---|---|
| Desk / app / raised | `--ink-950` / `--ink-900` / `--ink-850` | oklch(0.145 0.012 255) / (0.18 0.013 255) / (0.21 0.014 255) |
| Text | `--text-1` / `-2` / `-3` | oklch(0.965 0.006 255) / (0.80 0.014 255) / (0.67 0.016 255, 5.0:1 on app) |
| Where the pen is (camera) | `--pen` | oklch(0.77 0.14 238) |
| What the pen is doing (IMU) | `--still` `--moving` `--writing` `--lifted` | 0.76 0.03 255 / 0.84 0.14 82 / 0.80 0.15 158 / 0.77 0.11 318 |
| Out of bounds | `--warn` | oklch(0.72 0.18 24) |
| The voice | `--voice` | oklch(0.94 0.035 95) |
Color is functional only: one hue per thing the system knows. Canvas drawing uses sRGB equivalents (top of `app.js`).

## Type
Instrument Sans for the interface (tabular figures on every number), Newsreader for text the student hears (question,
spoken line). Phase name `clamp(2.25rem, 4.2vw, 4rem)` so it reads from 2 m. Sentence case everywhere, no all-caps labels,
`text-wrap: balance` on headings and `pretty` on prose.

## Layout
```
+-----------------------------------------------------------------------------+
| Illumin   * Phase name + what is happening                      [Source v]  |
+---------------+---------------------------------------+---------------------+
| Question      |           CAMERA (the page)           | Pen probe           |
| Answers       |   boxes, pen, trail, gestures         | Buzzers             |
| Said aloud    |   [voice dock rises here when talking]| Guidance, Audit     |
+---------------+---------------------------------------+---------------------+
```
Rails are plain columns separated by hairlines, not cards. The page keeps its real aspect (21.59 x 27.94) and sits on a
darker desk. The voice dock has its own reserved strip, so it never covers a box.

## Rules followed
- Impeccable anti-patterns avoided: purple gradients, bounce easing, glows, side-tab borders, nested cards, gray on
  color, Inter, small touch targets, eyebrow labels, monospace data labels.
- make-interfaces-feel-better: concentric radii (6 / 12 / 20), shadows for elevation and borders only for structure or
  state, `scale(0.96)` on press, 40 to 44 px hit areas, never `transition: all`, tabular numbers, one `currentColor`
  outline icon set (fill only for the active state), 1px pure-white 10% outline on the camera image, nothing animates on
  first paint, no custom motion on high-frequency updates, every animated change also has a static cue (color, icon, label).
- Quality floor: visible focus, keyboard operable (Esc closes the menu and returns focus), reduced motion honored in CSS
  and in GSAP, stacks under 1100 px.
