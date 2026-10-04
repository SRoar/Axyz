#include <Wire.h>

// MMA7660FC accelerometer streamer (Grove 3-Axis Digital Accelerometer, I2C 0x4C).
// One CSV line per sample: "A,<ms>,<x_g>,<y_g>,<z_g>"
const byte ACCEL_ADDR = 0x4C;
const unsigned long SAMPLE_INTERVAL_MS = 20; // 50 Hz
const float COUNTS_PER_G = 21.33;            // 6-bit output, +/-1.5 g

unsigned long lastSample = 0;
unsigned long lastRetry = 0;
bool ready = false;

void writeReg(byte reg, byte val) {
  Wire.beginTransmission(ACCEL_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

bool initAccel() {
  Wire.beginTransmission(ACCEL_ADDR);
  if (Wire.endTransmission() != 0) return false;

  writeReg(0x07, 0x00); // MODE: standby, so config registers can be written
  writeReg(0x08, 0x00); // SR: 120 samples/s
  writeReg(0x07, 0x01); // MODE: active
  delay(10);
  return true;
}

// Returns false on I2C failure or if the sensor never gives a valid sample.
bool readAccel(float &x, float &y, float &z) {
  for (byte attempt = 0; attempt < 5; attempt++) {
    Wire.beginTransmission(ACCEL_ADDR);
    Wire.write(0x00); // XOUT, then YOUT, ZOUT
    if (Wire.endTransmission() != 0) return false;

    Wire.requestFrom((int)ACCEL_ADDR, 3);
    if (Wire.available() < 3) return false;

    byte rx = Wire.read();
    byte ry = Wire.read();
    byte rz = Wire.read();

    // Bit 6 (0x40) is the alert flag: set while the sensor updates a value, so retry
    if ((rx | ry | rz) & 0x40) continue;

    // 6-bit two's complement
    int8_t vx = rx & 0x3F; if (vx > 31) vx -= 64;
    int8_t vy = ry & 0x3F; if (vy > 31) vy -= 64;
    int8_t vz = rz & 0x3F; if (vz > 31) vz -= 64;

    x = vx / COUNTS_PER_G;
    y = vy / COUNTS_PER_G;
    z = vz / COUNTS_PER_G;
    return true;
  }
  return false;
}

void setup() {
  Serial.begin(115200);
  Wire.begin();
  delay(100);

  if (initAccel()) {
    ready = true;
    Serial.println("ACCEL_READY addr=0x4C");
  } else {
    Serial.println("ERR,no_mma7660_at_0x4C");
  }
}

void loop() {
  unsigned long now = millis();

  if (!ready) {
    // Retry every 2 s so you can fix wiring without re-uploading
    if (now - lastRetry >= 2000) {
      lastRetry = now;
      if (initAccel()) {
        ready = true;
        Serial.println("ACCEL_READY addr=0x4C");
      } else {
        Serial.println("ERR,no_mma7660_at_0x4C");
      }
    }
    return;
  }

  if (now - lastSample < SAMPLE_INTERVAL_MS) return;
  lastSample = now;

  float x, y, z;
  if (readAccel(x, y, z)) {
    Serial.print("A,");
    Serial.print(now);
    Serial.print(",");
    Serial.print(x, 3);
    Serial.print(",");
    Serial.print(y, 3);
    Serial.print(",");
    Serial.println(z, 3);
  } else {
    Serial.println("ERR,accel_read_failed");
  }
}
