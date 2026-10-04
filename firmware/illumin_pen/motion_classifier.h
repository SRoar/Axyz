#pragma once
// STILL / MOVING / WRITING from a sliding window of accelerometer samples.
// Pure logic, no Arduino calls. Mirrored line for line by MotionClassifier in tools/imu_logic.py.
//
// Feed it every EMITTED sample (the ~33 Hz stream): that is the data the thresholds were tuned on.
// A new state is reported only after it has held for CLS_DWELL samples, and leaving a state needs the
// score to cross a slightly lower (hysteresis) threshold than entering it.
#include <math.h>
#include <stdint.h>
#include "imu_params.h"

enum MotionState : uint8_t { MS_STILL = 0, MS_MOVING = 1, MS_WRITING = 2 };

inline const char *motionStateName(MotionState s) {
  switch (s) {
    case MS_MOVING:  return "MOVING";
    case MS_WRITING: return "WRITING";
    default:         return "STILL";
  }
}

class MotionClassifier {
 public:
  MotionState state() const { return state_; }
  float score() const { return score_; }

  // Returns true if the reported state changed on this sample. `frozen` = a tap just happened:
  // its energy is not motion, so hold the current state.
  bool update(float ax, float ay, float az, bool frozen) {
    buf_[head_][0] = ax; buf_[head_][1] = ay; buf_[head_][2] = az;
    head_ = (head_ + 1) % CLS_WINDOW;
    if (count_ < CLS_WINDOW) count_++;
    if (count_ < CLS_WINDOW) return false;

    score_ = spread();
    if (frozen) { cand_ = state_; dwell_ = 0; return false; }

    MotionState tgt = target(score_);
    if (tgt == state_) { cand_ = tgt; dwell_ = 0; return false; }
    if (tgt == cand_) dwell_++; else { cand_ = tgt; dwell_ = 1; }
    if (dwell_ >= CLS_DWELL) { state_ = tgt; cand_ = tgt; dwell_ = 0; return true; }
    return false;
  }

 private:
  float buf_[CLS_WINDOW][3] = {};
  int head_ = 0;
  int count_ = 0;
  MotionState state_ = MS_STILL;
  MotionState cand_ = MS_STILL;
  int dwell_ = 0;
  float score_ = 0;

  // Sum over the three axes of the population standard deviation.
  float spread() const {
    float total = 0;
    for (int k = 0; k < 3; k++) {
      float mean = 0;
      for (int i = 0; i < CLS_WINDOW; i++) mean += buf_[i][k];
      mean /= CLS_WINDOW;
      float var = 0;
      for (int i = 0; i < CLS_WINDOW; i++) { float d = buf_[i][k] - mean; var += d * d; }
      total += sqrtf(var / CLS_WINDOW);
    }
    return total;
  }

  MotionState target(float score) const {
    MotionState tgt;
    if (state_ == MS_STILL) {
      if (score < T_ACTIVE) tgt = MS_STILL;
      else if (score >= T_WRITE) tgt = MS_WRITING;
      else tgt = MS_MOVING;
    } else if (state_ == MS_MOVING) {
      if (score < T_ACTIVE * CLS_HYST) tgt = MS_STILL;
      else if (score >= T_WRITE) tgt = MS_WRITING;
      else tgt = MS_MOVING;
    } else {  // WRITING
      if (score < T_ACTIVE * CLS_HYST) tgt = MS_STILL;
      else if (score < T_WRITE * CLS_HYST) tgt = MS_MOVING;
      else tgt = MS_WRITING;
    }
    if (tgt == MS_WRITING && !EMIT_WRITING) tgt = MS_MOVING;  // kill switch: never report WRITING
    return tgt;
  }
};
