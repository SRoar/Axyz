#pragma once
// 1- or 2-tap gestures from BURSTS of |a| activity. Pure logic, mirrored by TapDetector in tools/imu_logic.py.
//
// A pen tap rings for 0.1-0.5 s, so one tap is a short burst and a double tap a longer one (or two bursts).
// Feed it EVERY internal sample. A burst starts at the first sample whose |a| deviates TAP_ACT_G from its slow
// baseline, provided `allowed` (the pen is not writing) and there was no activity for TAP_PRE_QUIET_MS; it ends
// after TAP_BURST_GAP_MS without activity. Span <= TAP_ONE_MAX_MS counts as one tap, a longer span up to
// TAP_BURST_MAX_MS as two; anything longer is motion and cancels the whole gesture. A gesture is reported
// TAP_GROUP_MS after the last burst; more than TAP_MAX taps in total cancel it.
// Tuned on 20 recorded taps from one person (tools/verify_taps.py, tools/eval_taps.py).
#include <math.h>
#include <stdint.h>
#include "imu_params.h"

class TapDetector {
 public:
  // True while the motion classifier should hold its state (tap energy is not motion).
  bool frozen(uint32_t now) const { return haveFreeze_ && (int32_t)(now - freezeUntil_) < 0; }

  // Returns 1 or 2 when a tap gesture completes on this sample, else 0.
  uint8_t update(uint32_t now, float ax, float ay, float az, bool allowed) {
    float mag = sqrtf(ax * ax + ay * ay + az * az);
    if (!haveEma_) { ema_ = mag; haveEma_ = true; }
    float dev = fabsf(mag - ema_);
    bool act = dev >= TAP_ACT_G;
    if (!act) ema_ += TAP_EMA_ALPHA * (mag - ema_);  // the baseline follows quiet only, never a tap
    uint8_t out = 0;

    if (state_ == IDLE) {
      if (act && allowed && (!haveAct_ || (now - lastAct_) >= (uint32_t)TAP_PRE_QUIET_MS)) {
        beginBurst(now, dev);
        total_ = 0;
      }
    } else if (state_ == IN_BURST) {
      if (act) {
        last_ = now;
        if (dev > peak_) peak_ = dev;
        freezeUntil_ = now + TAP_FREEZE_MS;
        haveFreeze_ = true;
        if ((last_ - start_) > (uint32_t)TAP_BURST_MAX_MS) cancel();  // too long: this is motion
      } else if ((now - last_) >= (uint32_t)TAP_BURST_GAP_MS) {
        uint32_t span = last_ - start_;
        if (peak_ < TAP_PEAK_G || span > (uint32_t)TAP_BURST_MAX_MS) {
          cancel();  // a small bump, not a tap
        } else {
          total_ += (span <= (uint32_t)TAP_ONE_MAX_MS) ? 1 : 2;
          if (total_ > TAP_MAX) cancel(); else state_ = WAIT;
        }
      }
    } else {  // WAIT: a second burst may follow
      if (act) {
        beginBurst(now, dev);  // its own span decides whether it adds one tap or two
      } else if ((now - last_) >= (uint32_t)TAP_GROUP_MS) {
        out = total_;
        state_ = IDLE;
        total_ = 0;
      }
    }

    if (act) { lastAct_ = now; haveAct_ = true; }
    return out;
  }

 private:
  enum State : uint8_t { IDLE, IN_BURST, WAIT };
  State state_ = IDLE;
  float ema_ = 0;
  bool haveEma_ = false;
  uint32_t lastAct_ = 0;
  bool haveAct_ = false;
  uint32_t start_ = 0, last_ = 0;
  float peak_ = 0;
  uint8_t total_ = 0;
  uint32_t freezeUntil_ = 0;
  bool haveFreeze_ = false;

  void beginBurst(uint32_t now, float dev) {
    state_ = IN_BURST;
    start_ = last_ = now;
    peak_ = dev;
    freezeUntil_ = now + TAP_FREEZE_MS;
    haveFreeze_ = true;
  }
  void cancel() {
    state_ = IDLE;
    total_ = 0;
    haveFreeze_ = false;
  }
};
