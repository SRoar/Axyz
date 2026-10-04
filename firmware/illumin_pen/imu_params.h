#pragma once
// Single source of truth for the pen's motion / tap / haptic logic.
// Read by the firmware AND parsed by tools/imu_logic.py (the Python mirror used for tests),
// so keep each constant on one line in the form:   constexpr <type> NAME = value;

// ---- tuned by tools/tune_thresholds.py (rewrites exactly these three lines) -------------------
constexpr float T_ACTIVE = 0.115f;      // window motion-spread >= this: pen is active (not STILL)
constexpr float T_WRITE = 0.240f;       // window motion-spread >= this: WRITING, below it MOVING
constexpr bool EMIT_WRITING = true;     // false = plan section 8 kill switch: report STILL/MOVING only

// ---- motion classifier (runs on the emitted ~33 Hz samples, the same data it was tuned on) ----
constexpr int CLS_WINDOW = 16;          // samples per window (~0.5 s at 33 Hz)
constexpr int CLS_DWELL = 8;            // new state must hold this many samples (~0.24 s) before it is reported: 3 gave 5.5 false MOVING blips/min at rest, 8 gives 2.5
constexpr float CLS_HYST = 0.85f;       // leaving a state needs the score to fall to this fraction of its threshold

// ---- tap detector (BURST based; tuned on one person's 20 recorded taps, see tools/verify_taps.py) --------
// A real pen tap is not one clean spike: it rings for 0.1-0.5 s. A single tap is a short burst of |a|
// deviation, a double tap is a longer one (or two bursts). The detector times bursts instead of counting spikes.
constexpr float TAP_ACT_G = 0.25f;      // |a| deviation from its slow baseline that counts as activity
constexpr float TAP_PEAK_G = 0.30f;     // a burst must reach this height to count as a tap (smaller = a bump)
constexpr int TAP_PRE_QUIET_MS = 400;   // no activity this long before a burst, else it is motion
constexpr int TAP_NOT_WRITING_MS = 1000; // taps are ignored while WRITING and for this long after it (a tap's wind-up reads as MOVING, so STILL is too strict)
constexpr int TAP_BURST_GAP_MS = 200;   // this long without activity ends a burst
constexpr int TAP_ONE_MAX_MS = 250;     // burst span up to this = 1 tap, longer = 2 taps
constexpr int TAP_BURST_MAX_MS = 700;   // a burst longer than this is motion, not a tap
constexpr int TAP_MAX = 2;              // taps per gesture: 1 = read question, 2 = next question; 3 or more cancels
constexpr int TAP_GROUP_MS = 600;       // after a burst, wait this long for a second burst before reporting
constexpr int TAP_FREEZE_MS = 700;      // hold the motion state this long after tap activity (tap energy is not motion)
constexpr float TAP_EMA_ALPHA = 0.05f;  // speed of the |a| baseline

// ---- haptics ------------------------------------------------------------------------------
constexpr int HAPTIC_DEADMAN_MS = 1500; // GUIDE_* / WARN stop if not refreshed within this
