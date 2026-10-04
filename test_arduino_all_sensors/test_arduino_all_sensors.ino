#include <Wire.h>

byte accelAddr = 0x18; 
unsigned long lastPrintTime = 0;
const unsigned long PRINT_INTERVAL = 5000; // 5-second telemetry loop

void runI2CScanner() {
  Serial.println("\n==========================================");
  Serial.println("   RUNNING INTEL CURIE I2C BUS SCANNER    ");
  Serial.println("==========================================");
  byte error, address;
  int nDevices = 0;

  for (address = 1; address < 127; address++) {
    Wire.beginTransmission(address);
    error = Wire.endTransmission();

    if (error == 0) {
      Serial.print("[FOUND] Device detected at I2C address: 0x");
      if (address < 16) Serial.print("0");
      Serial.print(address, HEX);

      if (address == 0x18 || address == 0x19) {
        Serial.println(" <-- LIS3DH Accelerometer!");
      } else {
        Serial.println();
      }
      nDevices++;
    }
  }

  if (nDevices == 0) {
    Serial.println("[WARNING] No I2C devices responded!");
    Serial.println(" -> Check that your Grove Shield VCC switch is set to 5V / 3.3V.");
  } else {
    Serial.println("------------------------------------------");
    Serial.println("Scan complete! Send '18' or '19' to set target address.");
  }
  Serial.println("==========================================\n");
}

void setup() {
  Serial.begin(115200);

  // Pin definitions
  pinMode(3, OUTPUT); // Buzzer 1 (D3)
  pinMode(4, OUTPUT); // Buzzer 2 (D4)
  pinMode(A0, INPUT); // Light Sensor (A0)
  pinMode(A1, INPUT); // UV Sensor (A1)

  // Initial startup boot chime
  tone(3, 1000, 100);
  delay(150);
  tone(4, 1000, 100);

  // Initialize native Curie hardware I2C bus (SDA/SCL)
  Wire.begin();
  delay(100);

  Serial.println("TactileReader System Ready!");
  Serial.println("Type 's' or 'scan' in Serial Monitor to scan the bus.");

  runI2CScanner();
}

void loop() {
  // 1. Process instant user serial commands
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    if (input.equalsIgnoreCase("s") || input.equalsIgnoreCase("scan")) {
      runI2CScanner();
    } else if (input.equals("18")) {
      accelAddr = 0x18;
      Serial.println("\n Switched target address to 0x18\n");
    } else if (input.equals("19")) {
      accelAddr = 0x19;
      Serial.println("\n Switched target address to 0x19\n");
    }
  }

  // 2. Continuous real-time haptic feedback
  int lightVal = analogRead(A0);
  
  // Haptic Feedback Rule: Trigger Buzzer 1 (D3) when over a dark line or ink mark
  if (lightVal < 500) { 
    tone(3, 800); 
  } else {
    noTone(3);    
  }

  // 3. Telemetry reporting (throttled to once every 5 seconds)
  if (millis() - lastPrintTime >= PRINT_INTERVAL) {
    lastPrintTime = millis();

    int uvVal = analogRead(A1);

    int16_t rawX = 0, rawY = 0, rawZ = 0;

    // Wake up LIS3DH register 0x20 (CTRL_REG1) -> 50Hz normal mode, all axes enabled
    Wire.beginTransmission(accelAddr);
    Wire.write(0x20);
    Wire.write(0x57); 
    Wire.endTransmission();

    // Read 6 bytes starting at register 0x28 (OUT_X_L with auto-increment bit 0x80)
    Wire.beginTransmission(accelAddr);
    Wire.write(0x28 | 0x80);
    if (Wire.endTransmission() == 0) {
      Wire.requestFrom((int)accelAddr, 6);
      if (Wire.available() >= 6) {
        rawX = (int16_t)(Wire.read() | (Wire.read() << 8));
        rawY = (int16_t)(Wire.read() | (Wire.read() << 8));
        rawZ = (int16_t)(Wire.read() | (Wire.read() << 8));
      }
    }

    // Convert raw 12-bit counts to approximate g-force values
    float x = (rawX >> 4) / 1000.0;
    float y = (rawY >> 4) / 1000.0;
    float z = (rawZ >> 4) / 1000.0;

    // Print telemetry output stream
    Serial.print("LIGHT: "); Serial.print(lightVal);
    Serial.print(" | UV: ");   Serial.print(uvVal);
    Serial.print(" | X: ");    Serial.print(x, 2);
    Serial.print(" | Y: ");    Serial.print(y, 2);
    Serial.print(" | Z: ");    Serial.println(z, 2);
  }
}