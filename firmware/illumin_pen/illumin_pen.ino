// TactileReader pen probe firmware (Arduino UNO Q).
// Task 1.1 skeleton: MMA7660 accelerometer @ I2C 0x4C + light sensor on A0.
//
// Serial protocol (115200, '\n'-terminated), see PLAN.md section 3:
//   Arduino -> PC:  READY
//                   S,<ms>,<ax_g>,<ay_g>,<az_g>,<light_raw>     ~50 Hz
//   PC -> Arduino:  PING   (answered with READY)
//   (M, T lines and H, haptic commands come in tasks 1.3 - 1.5)
#include <Wire.h>

const byte ACCEL_ADDR = 0x4C;
const byte LIGHT_PIN = A0;

const unsigned long SAMPLE_PERIOD_MS = 10;  // ~100 Hz internal sample clock
const byte EMIT_EVERY = 2;                  // emit every 2nd sample -> ~50 Hz S lines
const byte MAX_READ_FAILS = 5;
const unsigned long REINIT_PERIOD_MS = 1000;

bool sensorOk = false;
byte failCount = 0;
byte sampleCount = 0;
unsigned long lastSample = 0;
unsigned long lastReinit = 0;

char cmdBuf[24];
byte cmdLen = 0;

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

// Raw 6-bit signed counts, ~21.33 counts per g. Returns false on I2C failure
// or if the sensor never gives a valid sample (alert flag stuck).
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

// counts -> g with 3 decimals, e.g. 20 -> "0.938". 1 count = 46.875 mg (1/21.33 g).
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

// Build the whole line in a buffer and write it with ONE call. Several small
// Serial.print() calls per line were measured at ~130 ms per line on the UNO Q.
void emitSample(unsigned long ms, int8_t vx, int8_t vy, int8_t vz) {
  char line[64];
  byte n = 0;
  line[n++] = 'S'; line[n++] = ',';
  n = appendUInt(line, n, ms);   line[n++] = ',';
  n = appendG(line, n, vx);      line[n++] = ',';
  n = appendG(line, n, vy);      line[n++] = ',';
  n = appendG(line, n, vz);      line[n++] = ',';
  n = appendUInt(line, n, analogRead(LIGHT_PIN));
  line[n++] = '\n';
  Serial.write((const uint8_t *)line, n);
}

// Non-blocking command reader: accumulate until '\n', never wait for input.
void pollCommands() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      cmdBuf[cmdLen] = 0;
      if (strcmp(cmdBuf, "PING") == 0) Serial.println("READY");
      // "H,<CMD>" haptic commands are handled in task 1.5
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
  Serial.println("READY");
  if (!sensorOk) Serial.println("ERR,accel_not_found");
}

void loop() {
  pollCommands();
  unsigned long now = millis();

  if (!sensorOk) {
    // Dropped out (or never found): retry quietly so a loose cable self-heals
    if (now - lastReinit >= REINIT_PERIOD_MS) {
      lastReinit = now;
      if (initAccel()) {
        sensorOk = true;
        failCount = 0;
        Serial.println("READY");
      }
    }
    return;
  }

  if (now - lastSample < SAMPLE_PERIOD_MS) return;
  lastSample = now;

  int8_t vx, vy, vz;
  if (!readAccelRaw(vx, vy, vz)) {
    if (++failCount >= MAX_READ_FAILS) {
      sensorOk = false;
      Serial.println("ERR,accel_lost");
    }
    return;
  }
  failCount = 0;

  if (++sampleCount >= EMIT_EVERY) {
    sampleCount = 0;
    emitSample(now, vx, vy, vz);
  }
}
