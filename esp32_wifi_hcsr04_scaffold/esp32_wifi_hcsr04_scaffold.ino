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
// Haptic UDP commands from Python (ESP32 listens on UDP port 4211):
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
const uint16_t HAPTIC_UDP_PORT = 4211;

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
String udpBuffer;
String currentMode = "OFF";
String currentDirection = "FRONT";
unsigned long lastPatternMs = 0;
unsigned long patternStartMs = 0;

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
  unsigned long elapsed = millis() - patternStartMs;

  if (currentMode == "OFF") {
    allMotorsOff();
    return;
  }

  // Risk-dependent haptic design:
  // LOW      = one short pulse, long gap
  // MEDIUM   = two short pulses, then a gap
  // HIGH     = three rapid pulses, then a gap
  // CRITICAL = rapid repeated pulses, strongest level
  //
  // The pulse pattern carries urgency, so LOW/MEDIUM are not continuous.
  uint16_t duty = 0;
  unsigned long cycleMs = 1000;
  bool outputOn = false;

  if (currentMode == "LOW") {
    // 30% PWM, 120 ms ON every 1500 ms.
    duty = PWM_LOW;
    cycleMs = 1500;
    unsigned long phase = elapsed % cycleMs;
    outputOn = (phase < 120);
  }
  else if (currentMode == "MEDIUM") {
    // 50% PWM: 2 x 150 ms pulses with 250 ms between them,
    // followed by a long pause.
    duty = PWM_MEDIUM;
    cycleMs = 1750;
    unsigned long phase = elapsed % cycleMs;
    outputOn = (phase < 150) || (phase >= 400 && phase < 550);
  }
  else if (currentMode == "HIGH") {
    // 75% PWM: 3 rapid pulses, then a pause.
    duty = PWM_HIGH;
    cycleMs = 1260;
    unsigned long phase = elapsed % cycleMs;
    outputOn = (phase < 180)
            || (phase >= 360 && phase < 540)
            || (phase >= 720 && phase < 900);
  }
  else if (currentMode == "CRITICAL") {
    // 100% PWM: 4 rapid, unmistakable pulses, then a short pause.
    duty = PWM_CRITICAL;
    cycleMs = 850;
    unsigned long phase = elapsed % cycleMs;
    outputOn = (phase < 150)
            || (phase >= 250 && phase < 400)
            || (phase >= 500 && phase < 650)
            || (phase >= 750 && phase < 850);
  }
  else {
    allMotorsOff();
    return;
  }

  int activeChannel = directionChannel(currentDirection);

  // Only the selected directional motor is driven.
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

  bool changed = (currentMode != mode) || (currentDirection != direction);
  currentMode = mode;
  currentDirection = direction;
  if (changed) {
    patternStartMs = millis();
    allMotorsOff();
  }

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

// ---------------- Haptic UDP input ----------------
void handleHapticUdp() {
  int packetSize = udp.parsePacket();
  while (packetSize > 0) {
    char packet[128];
    int len = udp.read(packet, sizeof(packet) - 1);
    if (len < 0) len = 0;
    packet[len] = '\0';

    String line = String(packet);
    line.trim();
    if (line.length() > 0) {
      parseSerialCommand(line);
    }

    packetSize = udp.parsePacket();
  }
}

// ---------------- Ultrasonic helpers ----------------
const unsigned long ULTRASONIC_TIMEOUT_US = 25000UL;  // ~4.3 m maximum echo wait
const unsigned long SENSOR_SEPARATION_MS = 60;         // reduce front/rear cross-talk
const unsigned long REAR_RETRY_DELAY_MS = 6;           // quick recovery after a missed rear echo

float readDistanceCm(int trigPin, int echoPin) {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);

  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  // Keep the timeout bounded so a missed echo cannot stall the whole loop.
  unsigned long duration = pulseIn(echoPin, HIGH, ULTRASONIC_TIMEOUT_US);

  if (duration == 0) {
    return -1.0f;
  }

  float distanceCm = (float)duration * 0.0343f / 2.0f;

  // Reject physically invalid/clearly noisy HC-SR04 results.
  if (distanceCm < 2.0f || distanceCm > 430.0f) {
    return -1.0f;
  }

  return distanceCm;
}

float readRearDistanceReliable() {
  float rearCm = readDistanceCm(REAR_TRIG_PIN, REAR_ECHO_PIN);

  // A single missed rear echo is common with HC-SR04s. Retry once quickly
  // instead of allowing one missed echo to look like several seconds of loss.
  if (rearCm < 0.0f) {
    delay(REAR_RETRY_DELAY_MS);
    rearCm = readDistanceCm(REAR_TRIG_PIN, REAR_ECHO_PIN);
  }

  return rearCm;
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
  delay(SENSOR_SEPARATION_MS);
  float rearCm = readRearDistanceReliable();

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
  udp.begin(HAPTIC_UDP_PORT);
  Serial.print("HAPTIC UDP listening on port: ");
  Serial.println(HAPTIC_UDP_PORT);
}

void loop() {
  handleSerial();
  handleHapticUdp();

  if (WiFi.status() != WL_CONNECTED) {
    connectWiFi();
  }

  // Local haptic pattern. Ultrasonic reads are bounded and the rear sensor
  // gets one quick retry after a missed echo, so a temporary rear timeout
  // does not turn into a prolonged apparent sensor outage.
  if (millis() - lastPatternMs >= 20) {
    lastPatternMs = millis();
    applyPattern();
  }

  sendUltrasonicPacket();
}
