// Illumin pen firmware (Arduino 101 + LIS3DH on I2C).
//
// Arduino -> PC (115200 baud, newline-terminated, see src/contracts.py):
//   READY                          once at boot, and in reply to PING
//   S,<ms>,<ax>,<ay>,<az>,<light>  raw sample, ~50 Hz (g)
//   T,<1|2|3>                      tap gesture (1 = read question, 2 = next question)
//
// Serial Monitor helpers: "s" = I2C scan, "18"/"19" = accel address,
// "debug" = toggle printing the tap signal (dev in g) instead of S lines.

#include <Wire.h>

// ---- tap tuning ---------------------------------------------------------
const float TAP_THRESHOLD_G = 0.45;    // jolt above resting gravity that counts as a tap
const unsigned long TAP_MAX_MS = 80;   // spike longer than this = pen movement, not a tap
const unsigned long TAP_REFRACT_MS = 120;  // ignore ringing right after a tap
const unsigned long TAP_GROUP_MS = 450;    // max gap between taps of the same gesture
const float GRAVITY_ALPHA = 0.02;      // how fast the resting-gravity estimate follows tilt

const unsigned long SAMPLE_EVERY_MS = 20;  // S line rate (50 Hz)
const int BUZZER_PIN = 3;

byte accelAddr = 0x18;
bool debugTap = false;

float gx = 0, gy = 0, gz = 1;   // resting gravity estimate
bool gravityReady = false;

bool inSpike = false;
unsigned long spikeStart = 0;
unsigned long lastTapAt = 0;
int tapCount = 0;
unsigned long lastSampleAt = 0;

void writeReg(byte reg, byte val) {
  Wire.beginTransmission(accelAddr);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

void initAccel() {
  writeReg(0x20, 0x77);  // CTRL_REG1: 400 Hz, all axes on (fast enough to catch a tap)
  writeReg(0x23, 0x88);  // CTRL_REG4: block data update, +/-2 g, high-resolution
}

bool readAccel(float &x, float &y, float &z) {
  Wire.beginTransmission(accelAddr);
  Wire.write(0x28 | 0x80);  // OUT_X_L with auto-increment
  if (Wire.endTransmission() != 0) return false;
  Wire.requestFrom((int)accelAddr, 6);
  if (Wire.available() < 6) return false;
  int16_t rx = (int16_t)(Wire.read() | (Wire.read() << 8));
  int16_t ry = (int16_t)(Wire.read() | (Wire.read() << 8));
  int16_t rz = (int16_t)(Wire.read() | (Wire.read() << 8));
  x = (rx >> 4) / 1000.0;  // 12-bit, 1 mg/digit at +/-2 g
  y = (ry >> 4) / 1000.0;
  z = (rz >> 4) / 1000.0;
  return true;
}

void runI2CScanner() {
  Serial.println("# I2C scan:");
  for (byte address = 1; address < 127; address++) {
    Wire.beginTransmission(address);
    if (Wire.endTransmission() == 0) {
      Serial.print("#   found 0x");
      if (address < 16) Serial.print("0");
      Serial.print(address, HEX);
      Serial.println((address == 0x18 || address == 0x19) ? "  <-- LIS3DH" : "");
    }
  }
}

void emitTaps(unsigned long now) {
  int n = tapCount > 3 ? 3 : tapCount;
  Serial.print("T,");
  Serial.println(n);
  tapCount = 0;
  for (int i = 0; i < n; i++) {  // confirmation clicks so the user knows it registered
    tone(BUZZER_PIN, 1800, 30);
    delay(70);
  }
}

void updateTap(float x, float y, float z, unsigned long now) {
  if (!gravityReady) {
    gx = x; gy = y; gz = z;
    gravityReady = true;
    return;
  }
  float dx = x - gx, dy = y - gy, dz = z - gz;
  float dev = sqrt(dx * dx + dy * dy + dz * dz);

  if (debugTap) {
    Serial.print("# dev ");
    Serial.println(dev, 3);
  }

  if (!inSpike) {
    gx += GRAVITY_ALPHA * dx;
    gy += GRAVITY_ALPHA * dy;
    gz += GRAVITY_ALPHA * dz;
    if (dev > TAP_THRESHOLD_G && now - lastTapAt > TAP_REFRACT_MS) {
      inSpike = true;
      spikeStart = now;
    }
  } else if (dev < TAP_THRESHOLD_G * 0.5) {
    inSpike = false;
    if (now - spikeStart <= TAP_MAX_MS) {
      tapCount++;
      lastTapAt = now;
    }
  } else if (now - spikeStart > TAP_MAX_MS) {
    // Long jolt: the pen is being moved, not tapped. Wait for it to settle.
    tapCount = 0;
  }

  if (inSpike && now - spikeStart > 1000) inSpike = false;  // never get stuck

  if (tapCount > 0 && !inSpike && now - lastTapAt > TAP_GROUP_MS) emitTaps(now);
}

void handleSerial() {
  if (Serial.available() <= 0) return;
  String input = Serial.readStringUntil('\n');
  input.trim();
  if (input.equalsIgnoreCase("PING")) {
    Serial.println("READY");
  } else if (input.equalsIgnoreCase("s") || input.equalsIgnoreCase("scan")) {
    runI2CScanner();
  } else if (input.equals("18") || input.equals("19")) {
    accelAddr = input.equals("18") ? 0x18 : 0x19;
    initAccel();
    gravityReady = false;
  } else if (input.equalsIgnoreCase("debug")) {
    debugTap = !debugTap;
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(A0, INPUT);

  Wire.begin();
  delay(100);
  initAccel();

  tone(BUZZER_PIN, 1000, 100);
  Serial.println("READY");
}

void loop() {
  handleSerial();

  float x, y, z;
  if (!readAccel(x, y, z)) return;
  unsigned long now = millis();
  updateTap(x, y, z, now);

  if (!debugTap && now - lastSampleAt >= SAMPLE_EVERY_MS) {
    lastSampleAt = now;
    Serial.print("S,");
    Serial.print(now);
    Serial.print(",");
    Serial.print(x, 3);
    Serial.print(",");
    Serial.print(y, 3);
    Serial.print(",");
    Serial.print(z, 3);
    Serial.print(",");
    Serial.println(analogRead(A0));
  }
}
