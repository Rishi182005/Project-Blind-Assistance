#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>

// ============================================================
// ESP32 FULL HARDWARE FIRMWARE
// HC-SR04 FRONT + REAR + PCA9685 + ULN2003 + 4 HAPTIC MOTORS
//
// This combines the previously working ultrasonic UDP firmware
// with the haptic-controller firmware so ONE ESP32 can do both.
//
// HC-SR04 wiring (your current setup):
//   FRONT: TRIG GPIO5,  ECHO GPIO18 (through your divider)
//   REAR:  TRIG GPIO19, ECHO GPIO23 (through your divider)
//
// PCA9685 I2C:
//   SDA GPIO21, SCL GPIO22, address 0x40
//
// Haptic mapping:
//   PWM0 / column 1 -> Motor 1 -> FRONT
//   PWM1 / column 2 -> Motor 2 -> FRONT-LEFT
//   PWM2 / column 3 -> Motor 3 -> FRONT-RIGHT
//   PWM3 / column 4 -> Motor 4 -> BACK
//
// Serial commands from Python:
//   HAPTIC,LOW,FRONT
//   HAPTIC,MEDIUM,FRONT_LEFT
//   HAPTIC,HIGH,FRONT_RIGHT
//   HAPTIC,CRITICAL,BACK
//   HAPTIC,OFF,FRONT
//
// Ultrasonic UDP packet:
//   front_cm:123.4,rear_cm:234.5,ts:123456
// ============================================================

// ---------------- Wi-Fi / UDP ----------------
const char* WIFI_SSID = "Rishi's Network";
const char* WIFI_PASSWORD = "Devi@1002";

// Keep the same laptop IP used by your working ultrasonic sketch.
const char* LAPTOP_IP = "192.168.29.85";
const uint16_t LAPTOP_PORT = 4210;

WiFiUDP udp;

// ---------------- HC-SR04 ----------------
const int FRONT_TRIG_PIN = 5;
const int FRONT_ECHO_PIN = 18;
const int REAR_TRIG_PIN  = 19;
const int REAR_ECHO_PIN  = 23;

// ---------------- PCA9685 ----------------
#define PCA9685_ADDR 0x40
#define SDA_PIN 21
#define SCL_PIN 22

#define MODE1      0x00
#define PRESCALE   0xFE
#define LED0_ON_L  0x06

// Bench-test starting levels. These can be changed later after
// you tune the pulse patterns on the actual headband.
const uint16_t PWM_LOW      = 1229;  // 30%
const uint16_t PWM_MEDIUM   = 2048;  // 50%
const uint16_t PWM_HIGH     = 3072;  // 75%
const uint16_t PWM_CRITICAL = 4095;  // 100%

String serialBuffer;
String currentMode = "OFF";
String currentDirection = "FRONT";
unsigned long lastPatternMs = 0;

// ---------------- PCA helpers ----------------
void writeRegister(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(PCA9685_ADDR);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

void setPWM(uint8_t channel, uint16_t on, uint16_t off) {
  uint8_t reg = LED0_ON_L + 4 * channel;

  Wire.beginTransmission(PCA9685_ADDR);
  Wire.write(reg);
  Wire.write(on & 0xFF);
  Wire.write(on >> 8);
  Wire.write(off & 0xFF);
  Wire.write(off >> 8);
  Wire.endTransmission();
}

void allMotorsOff() {
  for (uint8_t ch = 0; ch < 4; ch++) {
    setPWM(ch, 0, 0);
  }
}

void configurePCA9685() {
  Wire.begin(SDA_PIN, SCL_PIN);

  writeRegister(MODE1, 0x10);   // sleep
  writeRegister(PRESCALE, 29);  // about 200 Hz
  writeRegister(MODE1, 0x00);   // wake
  delay(10);
  writeRegister(MODE1, 0xA1);   // restart + auto-increment

  allMotorsOff();
}

int directionChannel(const String &direction) {
  if (direction == "FRONT") return 0;
  if (direction == "FRONT_LEFT") return 1;
  if (direction == "FRONT_RIGHT") return 2;
  if (direction == "BACK") return 3;
  return 0;
}

void applyPattern() {
  unsigned long now = millis();

  if (currentMode == "OFF") {
    allMotorsOff();
    return;
  }

  uint16_t duty = 0;
  unsigned long onOffPeriod = 1000;

  if (currentMode == "LOW") {
    duty = PWM_LOW;
    onOffPeriod = 900;
  } else if (currentMode == "MEDIUM") {
    duty = PWM_MEDIUM;
    onOffPeriod = 500;
  } else if (currentMode == "HIGH") {
    duty = PWM_HIGH;
    onOffPeriod = 250;
  } else if (currentMode == "CRITICAL") {
    duty = PWM_CRITICAL;
    onOffPeriod = 120;
  } else {
    allMotorsOff();
    return;
  }

  bool outputOn = ((now / onOffPeriod) % 2 == 0);
  int activeChannel = directionChannel(currentDirection);

  for (uint8_t ch = 0; ch < 4; ch++) {
    if (ch == activeChannel && outputOn) {
      if (currentMode == "CRITICAL") {
        setPWM(ch, 4096, 0);  // full ON
      } else {
        setPWM(ch, 0, duty);
      }
    } else {
      setPWM(ch, 0, 0);
    }
  }
}

void setHapticCommand(String mode, String direction) {
  mode.trim();
  direction.trim();
  mode.toUpperCase();
  direction.toUpperCase();

  direction.replace('-', '_');
  direction.replace(' ', '_');

  if (direction == "AHEAD" || direction == "CENTER") direction = "FRONT";
  if (direction == "LEFT") direction = "FRONT_LEFT";
  if (direction == "RIGHT") direction = "FRONT_RIGHT";
  if (direction == "REAR") direction = "BACK";

  if (mode != "LOW" && mode != "MEDIUM" && mode != "HIGH" &&
      mode != "CRITICAL" && mode != "OFF") {
    mode = "OFF";
  }

  currentMode = mode;
  currentDirection = direction;

  Serial.print("HAPTIC ACK: ");
  Serial.print(currentMode);
  Serial.print(",");
  Serial.println(currentDirection);

  if (currentMode == "OFF") {
    allMotorsOff();
  }
}

void parseSerialCommand(String line) {
  line.trim();
  if (!line.length()) return;

  int p1 = line.indexOf(',');
  int p2 = line.indexOf(',', p1 + 1);
  if (p1 < 0 || p2 < 0) return;

  String prefix = line.substring(0, p1);
  String mode = line.substring(p1 + 1, p2);
  String direction = line.substring(p2 + 1);

  prefix.trim();
  prefix.toUpperCase();

  if (prefix == "HAPTIC") {
    setHapticCommand(mode, direction);
  }
}

void handleSerial() {
  while (Serial.available()) {
    char c = static_cast<char>(Serial.read());

    if (c == '\n' || c == '\r') {
      if (serialBuffer.length() > 0) {
        parseSerialCommand(serialBuffer);
        serialBuffer = "";
      }
    } else {
      serialBuffer += c;
      if (serialBuffer.length() > 100) {
        serialBuffer = "";
      }
    }
  }
}

// ---------------- Ultrasonic helpers ----------------
float readDistanceCm(int trigPin, int echoPin) {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);

  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  unsigned long duration = pulseIn(echoPin, HIGH, 30000UL);

  if (duration == 0) {
    return -1.0f;
  }

  return (float)duration * 0.0343f / 2.0f;
}

void connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.print("Wi-Fi connected. ESP32 IP: ");
  Serial.println(WiFi.localIP());
  Serial.print("Sending UDP to: ");
  Serial.print(LAPTOP_IP);
  Serial.print(":");
  Serial.println(LAPTOP_PORT);
}

unsigned long lastUltrasonicMs = 0;
const unsigned long ULTRASONIC_INTERVAL_MS = 100;

void sendUltrasonicPacket() {
  unsigned long now = millis();
  if (now - lastUltrasonicMs < ULTRASONIC_INTERVAL_MS) return;
  lastUltrasonicMs = now;

  // Trigger separately to reduce cross-talk.
  float frontCm = readDistanceCm(FRONT_TRIG_PIN, FRONT_ECHO_PIN);
  delay(60);
  float rearCm = readDistanceCm(REAR_TRIG_PIN, REAR_ECHO_PIN);

  char packet[128];
  snprintf(packet, sizeof(packet),
           "front_cm:%.1f,rear_cm:%.1f,ts:%lu",
           frontCm, rearCm, millis());

  udp.beginPacket(LAPTOP_IP, LAPTOP_PORT);
  udp.print(packet);
  udp.endPacket();

  Serial.print("FRONT: ");
  if (frontCm < 0) Serial.print("No reading");
  else {
    Serial.print(frontCm, 1);
    Serial.print(" cm");
  }

  Serial.print("    REAR: ");
  if (rearCm < 0) Serial.println("No reading");
  else {
    Serial.print(rearCm, 1);
    Serial.println(" cm");
  }
}

void setup() {
  Serial.begin(115200);
  delay(300);

  // HC-SR04
  pinMode(FRONT_TRIG_PIN, OUTPUT);
  pinMode(FRONT_ECHO_PIN, INPUT);
  pinMode(REAR_TRIG_PIN, OUTPUT);
  pinMode(REAR_ECHO_PIN, INPUT);

  digitalWrite(FRONT_TRIG_PIN, LOW);
  digitalWrite(REAR_TRIG_PIN, LOW);

  // PCA9685 + haptics
  configurePCA9685();

  Serial.println("ESP32 Full Ultrasonic + Haptic Controller Ready");
  Serial.println("HC-SR04: FRONT T=GPIO5 E=GPIO18 | REAR T=GPIO19 E=GPIO23");
  Serial.println("PCA9685: SDA=GPIO21 SCL=GPIO22 ADDR=0x40");
  Serial.println("HAPTIC: PWM0=Front PWM1=FrontLeft PWM2=FrontRight PWM3=Back");

  connectWiFi();
}

void loop() {
  handleSerial();

  if (WiFi.status() != WL_CONNECTED) {
    connectWiFi();
  }

  // Local haptic pattern, non-blocking apart from the two ultrasonic
  // measurements that were already used by the working firmware.
  if (millis() - lastPatternMs >= 20) {
    lastPatternMs = millis();
    applyPattern();
  }

  sendUltrasonicPacket();
}
