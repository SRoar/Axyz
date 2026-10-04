// TactileReader pen probe firmware (Arduino UNO Q).
// MMA7660 accelerometer @ I2C 0x4C. Accelerometer only: no light or UV sensor.
//
// Serial protocol (115200, '\n'-terminated), see PLAN.md section 3:
//   Arduino -> PC:  READY                                         at boot, and in reply to PING
//                   S,<ms>,<ax_g>,<ay_g>,<az_g>                   ~33 Hz raw sample
//                   M,<STILL|MOVING|WRITING>                      only when the motion state changes
//                   T,<1|2>                                       tap gesture (1 = read question, 2 = next)
//   PC -> Arduino:  PING                                          answered with READY and the current M state
//                   H,<LOCK|WARN|COMPLETE|GUIDE_LEFT|GUIDE_RIGHT|GUIDE_BOTH|OFF>
//
// STREAMING IS KEEPALIVE-GATED: the board streams S/M/T for the first STREAM_GRACE_MS after boot (so a Serial
// Monitor works) and afterwards only while the PC keeps sending something (PING about once a second, or H,
// commands). Without this, whatever nobody reads piles up on the board side (measured: 14 minutes, 27000 lines)
// and the next connection has to wade through it.
//
// The logic lives in headers so it can be mirrored and tested on the PC (tools/imu_logic.py):
//   imu_params.h         every threshold (T_ACTIVE / T_WRITE are written by tools/tune_thresholds.py)
//   motion_classifier.h  STILL / MOVING / WRITING
//   tap_detector.h       1 / 2 / 3 taps
//   haptics.h            buzzer pattern player (pure logic)
//
// BUZZERS: the pattern logic is done, the pin output below (applyBuzzers) is NOT tested on the UNO Q yet.
// Left for integration: whether tone() can drive D3 and D4 at the same time on this board.
#include <Wire.h>
#include "imu_params.h"
#include "motion_classifier.h"
#include "tap_detector.h"
#include "haptics.h"

const byte ACCEL_ADDR = 0x4C;
const byte BUZZER_LEFT_PIN = 3;   // D3 = LEFT
const byte BUZZER_RIGHT_PIN = 4;  // D4 = RIGHT

const unsigned long SAMPLE_PERIOD_MS = 10;  // internal sample clock (the tap detector sees every sample)
const byte EMIT_EVERY = 2;                  // every 2nd sample is sent and fed to the classifier
const float G_PER_COUNT = 0.046875f;        // MMA7660: 6-bit, 1 count = 3/64 g
const byte MAX_READ_FAILS = 5;
const unsigned long REINIT_PERIOD_MS = 1000;
const unsigned long STREAM_GRACE_MS = 20000;  // stream freely this long after boot
const unsigned long HOST_TIMEOUT_MS = 3000;   // then only while the host has spoken within this long

bool sensorOk = false;
byte failCount = 0;
byte sampleCount = 0;
unsigned long nextSample = 0;
unsigned long lastReinit = 0;
unsigned long lastHostMs = 0;
bool hostSeen = false;
unsigned long lastWritingMs = 0;
bool wasWriting = false;

char cmdBuf[24];
byte cmdLen = 0;

MotionClassifier classifier;
TapDetector taps;
HapticEngine haptics;
uint16_t curLeftHz = 0, curRightHz = 0;

void writeReg(byte reg, byte val) {
  Wire.beginTransmission(ACCEL_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

bool initAccel() {
  Wire.beginTransmission(ACCEL_ADDR);
  if (Wire.endTransmission() != 0) return false;

  writeReg(0x07, 0x00);  // MODE: standby, so config registers can be written
  writeReg(0x08, 0x00);  // SR: 120 samples/s
  writeReg(0x07, 0x01);  // MODE: active
  delay(10);
  return true;
}

// Raw 6-bit signed counts. Returns false on I2C failure or if the alert flag never clears.
bool readAccelRaw(int8_t &vx, int8_t &vy, int8_t &vz) {
  for (byte attempt = 0; attempt < 5; attempt++) {
    Wire.beginTransmission(ACCEL_ADDR);
    Wire.write(0x00);  // XOUT, then YOUT, ZOUT
    if (Wire.endTransmission() != 0) return false;

    Wire.requestFrom((int)ACCEL_ADDR, 3);
    if (Wire.available() < 3) return false;

    byte rx = Wire.read();
    byte ry = Wire.read();
    byte rz = Wire.read();

    // Bit 6 (0x40) is the alert flag: set while the sensor updates a value, so retry
    if ((rx | ry | rz) & 0x40) continue;

    vx = rx & 0x3F; if (vx > 31) vx -= 64;
    vy = ry & 0x3F; if (vy > 31) vy -= 64;
    vz = rz & 0x3F; if (vz > 31) vz -= 64;
    return true;
  }
  return false;
}

// Append a decimal number to buf. No printf / float formatting, so it is cheap.
byte appendUInt(char *buf, byte pos, unsigned long v) {
  char tmp[11];
  byte n = 0;
  do { tmp[n++] = '0' + (v % 10); v /= 10; } while (v > 0);
  while (n > 0) buf[pos++] = tmp[--n];
  return pos;
}

// counts -> g with 3 decimals, e.g. 20 -> "0.938". 1 count = 46.875 mg.
byte appendG(char *buf, byte pos, int8_t counts) {
  long milli = ((long)counts * 375L) / 8L;
  if (milli < 0) { buf[pos++] = '-'; milli = -milli; }
  pos = appendUInt(buf, pos, milli / 1000);
  buf[pos++] = '.';
  int frac = milli % 1000;  // up to 999: must not be a byte (max 255)
  buf[pos++] = '0' + (frac / 100);
  buf[pos++] = '0' + ((frac / 10) % 10);
  buf[pos++] = '0' + (frac % 10);
  return pos;
}

// Every line is built in a buffer and written with ONE call: on the UNO Q each Serial call costs
// ~5 ms of fixed overhead plus ~96 us per byte, so several small prints per line are far too slow.
void sendLine(const char *s) {
  char line[40];
  byte n = 0;
  while (s[n] && n < sizeof(line) - 2) { line[n] = s[n]; n++; }
  line[n++] = '\n';
  Serial.write((const uint8_t *)line, n);
}

void emitSample(unsigned long ms, int8_t vx, int8_t vy, int8_t vz) {
  char line[64];
  byte n = 0;
  line[n++] = 'S'; line[n++] = ',';
  n = appendUInt(line, n, ms);   line[n++] = ',';
  n = appendG(line, n, vx);      line[n++] = ',';
  n = appendG(line, n, vy);      line[n++] = ',';
  n = appendG(line, n, vz);
  line[n++] = '\n';
  Serial.write((const uint8_t *)line, n);
}

// Applies the engine's output to the pins, only when it changes.
// UNTESTED on the UNO Q (integration): tone() on two pins at once is not verified.
void applyBuzzers(uint16_t left, uint16_t right) {
  if (left != curLeftHz) {
    if (left) tone(BUZZER_LEFT_PIN, left); else noTone(BUZZER_LEFT_PIN);
    curLeftHz = left;
  }
  if (right != curRightHz) {
    if (right) tone(BUZZER_RIGHT_PIN, right); else noTone(BUZZER_RIGHT_PIN);
    curRightHz = right;
  }
}

// Non-blocking command reader: accumulate until '\n', never wait for input.
bool streaming(unsigned long now) {
  return now < STREAM_GRACE_MS || (hostSeen && (now - lastHostMs) < HOST_TIMEOUT_MS);
}

void handleCommand(const char *cmd) {
  lastHostMs = millis();  // any command counts as the host being alive
  hostSeen = true;
  if (strcmp(cmd, "PING") == 0) {
    sendLine("READY");
    // M lines are otherwise only sent on a change; a PC that just connected has no idea what the state is.
    char m[16] = "M,";
    strcpy(m + 2, motionStateName(classifier.state()));
    sendLine(m);
  } else if (cmd[0] == 'H' && cmd[1] == ',') {
    HapticCmd c = parseHapticCmd(cmd + 2);
    if (c != HC_NONE) haptics.command(c, millis());
  }
}

void pollCommands() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      cmdBuf[cmdLen] = 0;
      handleCommand(cmdBuf);
      cmdLen = 0;
    } else if (cmdLen < sizeof(cmdBuf) - 1) {
      cmdBuf[cmdLen++] = c;
    } else {
      cmdLen = 0;  // overlong line: drop it
    }
  }
}

void setup() {
  Serial.begin(115200);
  Wire.begin();
  delay(100);
  sensorOk = initAccel();
  sendLine("READY");
  if (!sensorOk) sendLine("ERR,accel_not_found");
}

void loop() {
  pollCommands();
  unsigned long now = millis();

  // Haptics tick first: patterns must keep time even if the sensor drops out.
  uint16_t left, right;
  haptics.tick(now, left, right);
  applyBuzzers(left, right);

  if (!sensorOk) {
    // Dropped out (or never found): retry quietly so a loose cable self-heals
    if (now - lastReinit >= REINIT_PERIOD_MS) {
      lastReinit = now;
      if (initAccel()) {
        sensorOk = true;
        failCount = 0;
        sendLine("READY");
      }
    }
    return;
  }

  // Fixed schedule (not "now + period"): loop() calls are ~1.1 ms apart on the UNO Q and a line
  // write blocks ~8 ms, so re-arming from "now" drifts. If we fall behind, skip ahead.
  if ((long)(now - nextSample) < 0) return;
  nextSample += SAMPLE_PERIOD_MS;
  if ((long)(now - nextSample) >= 0) nextSample = now + SAMPLE_PERIOD_MS;

  int8_t vx, vy, vz;
  if (!readAccelRaw(vx, vy, vz)) {
    if (++failCount >= MAX_READ_FAILS) {
      sensorOk = false;
      sendLine("ERR,accel_lost");
    }
    return;
  }
  failCount = 0;

  float ax = vx * G_PER_COUNT, ay = vy * G_PER_COUNT, az = vz * G_PER_COUNT;

  // Taps look at every internal sample, and count unless the pen is writing (or was, a moment ago).
  // They are NOT limited to STILL: the wind-up before a tap reads as MOVING.
  bool live = streaming(now);
  if (classifier.state() == MS_WRITING) { lastWritingMs = now; wasWriting = true; }
  bool tapsAllowed = classifier.state() != MS_WRITING && (!wasWriting || (now - lastWritingMs) >= (unsigned long)TAP_NOT_WRITING_MS);
  uint8_t tapCount = taps.update(now, ax, ay, az, tapsAllowed);
  if (tapCount && live) {
    char t[] = "T,0";
    t[2] = '0' + tapCount;
    sendLine(t);
  }

  if (++sampleCount >= EMIT_EVERY) {
    sampleCount = 0;
    bool changed = classifier.update(ax, ay, az, taps.frozen(now));
    if (live) emitSample(now, vx, vy, vz);
    if (changed && live) {
      char m[16] = "M,";
      strcpy(m + 2, motionStateName(classifier.state()));
      sendLine(m);
    }
  }
}
