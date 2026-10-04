#pragma once
// Haptic pattern player. Pure logic, no pin or tone() calls: tick() just says what frequency each
// buzzer should make right now, and the sketch applies that. Wiring to the real buzzers is left for
// integration. Mirrored by HapticEngine in tools/imu_logic.py.
//
// LOCK / COMPLETE are one-shot and play to the end. GUIDE_* and WARN repeat until replaced and stop on
// their own HAPTIC_DEADMAN_MS after the last refresh (dead-man switch: if the PC dies the buzzers go quiet).
// OFF stops everything. D3 = left buzzer, D4 = right buzzer. 0 Hz = silent.
#include <stdint.h>
#include <string.h>
#include "imu_params.h"

enum HapticCmd : uint8_t {
  HC_NONE = 0, HC_OFF, HC_LOCK, HC_COMPLETE, HC_GUIDE_LEFT, HC_GUIDE_RIGHT, HC_GUIDE_BOTH, HC_WARN
};

struct HapticStep { uint16_t ms; uint16_t left; uint16_t right; };
struct HapticPattern { const HapticStep *steps; uint8_t n; bool oneShot; };

static const HapticStep STEPS_LOCK[] = {{80, 1000, 1000}, {40, 0, 0}, {80, 1500, 1500}};
static const HapticStep STEPS_COMPLETE[] = {{120, 600, 600}, {120, 900, 900}, {120, 1200, 1200}};
static const HapticStep STEPS_GUIDE_LEFT[] = {{100, 1000, 0}, {150, 0, 0}};
static const HapticStep STEPS_GUIDE_RIGHT[] = {{100, 0, 1000}, {150, 0, 0}};
static const HapticStep STEPS_GUIDE_BOTH[] = {{100, 800, 800}, {400, 0, 0}};
static const HapticStep STEPS_WARN[] = {{100, 2500, 0}, {100, 0, 2500}};

inline bool hapticPattern(HapticCmd c, HapticPattern &p) {
  switch (c) {
    case HC_LOCK:        p = {STEPS_LOCK, 3, true}; return true;
    case HC_COMPLETE:    p = {STEPS_COMPLETE, 3, true}; return true;
    case HC_GUIDE_LEFT:  p = {STEPS_GUIDE_LEFT, 2, false}; return true;
    case HC_GUIDE_RIGHT: p = {STEPS_GUIDE_RIGHT, 2, false}; return true;
    case HC_GUIDE_BOTH:  p = {STEPS_GUIDE_BOTH, 2, false}; return true;
    case HC_WARN:        p = {STEPS_WARN, 2, false}; return true;
    default: return false;
  }
}

// "LOCK" -> HC_LOCK etc. Returns HC_NONE for anything unknown.
inline HapticCmd parseHapticCmd(const char *s) {
  if (!strcmp(s, "OFF")) return HC_OFF;
  if (!strcmp(s, "LOCK")) return HC_LOCK;
  if (!strcmp(s, "COMPLETE")) return HC_COMPLETE;
  if (!strcmp(s, "GUIDE_LEFT")) return HC_GUIDE_LEFT;
  if (!strcmp(s, "GUIDE_RIGHT")) return HC_GUIDE_RIGHT;
  if (!strcmp(s, "GUIDE_BOTH")) return HC_GUIDE_BOTH;
  if (!strcmp(s, "WARN")) return HC_WARN;
  return HC_NONE;
}

class HapticEngine {
 public:
  void command(HapticCmd c, uint32_t now) {
    HapticPattern p;
    if (c == HC_OFF) { stop(); pendingValid_ = false; return; }
    if (!hapticPattern(c, p)) return;
    if (p.oneShot) {
      if (!(active_ && oneShot_ && cmd_ == c)) start(c, p, now);  // re-sent while playing: let it finish
    } else {
      if (active_ && oneShot_) { pending_ = c; pendingRefresh_ = now; pendingValid_ = true; }  // after the one-shot
      else if (active_ && cmd_ == c) lastRefresh_ = now;
      else start(c, p, now);
    }
  }

  // Fills left/right with the frequency each buzzer should play at `now`.
  void tick(uint32_t now, uint16_t &left, uint16_t &right) {
    left = right = 0;
    if (!active_) return;
    if (!oneShot_ && (now - lastRefresh_) > (uint32_t)HAPTIC_DEADMAN_MS) { stop(); return; }
    while (active_ && (now - stepStart_) >= pat_.steps[idx_].ms) {
      stepStart_ += pat_.steps[idx_].ms;
      idx_++;
      if (idx_ >= pat_.n) {
        if (oneShot_) {
          stop();
          if (pendingValid_ && (now - pendingRefresh_) <= (uint32_t)HAPTIC_DEADMAN_MS) {
            HapticPattern np;
            if (hapticPattern(pending_, np)) { start(pending_, np, stepStart_); lastRefresh_ = pendingRefresh_; }
          }
          pendingValid_ = false;
        } else {
          idx_ = 0;
        }
      }
    }
    if (!active_) return;
    left = pat_.steps[idx_].left;
    right = pat_.steps[idx_].right;
  }

  bool active() const { return active_; }
  HapticCmd current() const { return active_ ? cmd_ : HC_NONE; }

 private:
  bool active_ = false;
  bool oneShot_ = false;
  HapticCmd cmd_ = HC_NONE;
  HapticPattern pat_ = {nullptr, 0, false};
  uint8_t idx_ = 0;
  uint32_t stepStart_ = 0;
  uint32_t lastRefresh_ = 0;
  HapticCmd pending_ = HC_NONE;
  uint32_t pendingRefresh_ = 0;
  bool pendingValid_ = false;

  void start(HapticCmd c, const HapticPattern &p, uint32_t now) {
    cmd_ = c; pat_ = p; oneShot_ = p.oneShot; idx_ = 0;
    stepStart_ = now; lastRefresh_ = now; active_ = true;
  }
  void stop() { active_ = false; oneShot_ = false; idx_ = 0; cmd_ = HC_NONE; }
};
