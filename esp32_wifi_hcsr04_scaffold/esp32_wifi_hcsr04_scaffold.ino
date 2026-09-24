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
// Keep the same laptop IP used by your working ultrasonic sketch. const 
char* LAPTOP_IP = "192.168.29.85";
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

// Direction-aware haptic intensity tuned for the cardboard wearable.
// Front/Back motors are physically stronger, so their scale starts at 60%.
// Left/Right motors are physically weaker, so their scale starts at 75%.
// The four risk levels are linearly spread from the directional starting
// level up to 100% at CRITICAL.
//
// FRONT/BACK: LOW 60%, MEDIUM 73%, HIGH 87%, CRITICAL 100%
// LEFT/RIGHT: LOW 75%, MEDIUM 83%, HIGH 92%, CRITICAL 100%
const uint16_t PWM_FB_LOW      = 2458;  // 60%
const uint16_t PWM_FB_MEDIUM   = 3004;  // 73.3%
const uint16_t PWM_FB_HIGH     = 3549;  // 86.7%
const uint16_t PWM_LR_LOW      = 3072;  // 75%
const uint16_t PWM_LR_MEDIUM   = 3413;  // 83.3%
const uint16_t PWM_LR_HIGH     = 3755;  // 91.7%
const uint16_t PWM_CRITICAL    = 4095;  // 100%

String serialBuffer;
String udpBuffer;
String currentMode = "OFF";
String currentDirection = "FRONT";
unsigned long lastPatternMs = 0;
unsigned long patternStartMs = 0;

// ------------------------------------------------------------
// HAPTIC DIRECTION RHYTHM
// ------------------------------------------------------------
// Direction is encoded by rhythm rather than relying only on
// where the motor physically sits. Every event has a fixed slot
// with a long silent tail, so repeated detections cannot blend
// into one long sequence and look like another direction.
//
// FRONT       = 1 short pulse
// FRONT_LEFT  = 2 close pulses
// FRONT_RIGHT = 2 pulses with a longer gap
// BACK        = 3 close pulses
//
// Risk level controls intensity; direction controls rhythm.
const unsigned long HAPTIC_EVENT_CYCLE_MS = 1500;
const unsigned long HAPTIC_PULSE_ON_MS = 110;
const unsigned long HAPTIC_GAP_CLOSE_MS = 120;
const unsigned long HAPTIC_GAP_RIGHT_MS = 280;

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

uint16_t modeDuty() {
  if (currentMode == "CRITICAL") return PWM_CRITICAL;

  const bool leftRight = (currentDirection == "FRONT_LEFT" ||
                          currentDirection == "FRONT_RIGHT");

  if (currentMode == "LOW") {
    return leftRight ? PWM_LR_LOW : PWM_FB_LOW;
  }
  if (currentMode == "MEDIUM") {
    return leftRight ? PWM_LR_MEDIUM : PWM_FB_MEDIUM;
  }
  if (currentMode == "HIGH") {
    return leftRight ? PWM_LR_HIGH : PWM_FB_HIGH;
  }
  return 0;
}

bool directionPulseIsOn(unsigned long phase) {
  // FRONT: one pulse at the start of the 1.5 s event slot.
  if (currentDirection == "FRONT") {
    return phase < HAPTIC_PULSE_ON_MS;
  }

  // FRONT_LEFT: two close pulses.
  if (currentDirection == "FRONT_LEFT") {
    const unsigned long p1Start = 0;
    const unsigned long p1End = HAPTIC_PULSE_ON_MS;
    const unsigned long p2Start = HAPTIC_PULSE_ON_MS + HAPTIC_GAP_CLOSE_MS;
    const unsigned long p2End = p2Start + HAPTIC_PULSE_ON_MS;
    return (phase >= p1Start && phase < p1End) ||
           (phase >= p2Start && phase < p2End);
  }

  // FRONT_RIGHT: two pulses separated by a noticeably longer gap.
  if (currentDirection == "FRONT_RIGHT") {
    const unsigned long p1Start = 0;
    const unsigned long p1End = HAPTIC_PULSE_ON_MS;
    const unsigned long p2Start = HAPTIC_PULSE_ON_MS + HAPTIC_GAP_RIGHT_MS;
    const unsigned long p2End = p2Start + HAPTIC_PULSE_ON_MS;
    return (phase >= p1Start && phase < p1End) ||
           (phase >= p2Start && phase < p2End);
  }

  // BACK: three close pulses.
  if (currentDirection == "BACK") {
    const unsigned long step = HAPTIC_PULSE_ON_MS + HAPTIC_GAP_CLOSE_MS;
    const unsigned long p1Start = 0;
    const unsigned long p2Start = step;
    const unsigned long p3Start = 2 * step;
    return (phase >= p1Start && phase < p1Start + HAPTIC_PULSE_ON_MS) ||
           (phase >= p2Start && phase < p2Start + HAPTIC_PULSE_ON_MS) ||
           (phase >= p3Start && phase < p3Start + HAPTIC_PULSE_ON_MS);
  }

  return false;
}

void applyPattern() {
  if (currentMode == "OFF") {
    allMotorsOff();
    return;
  }

  const uint16_t duty = modeDuty();
  if (duty == 0) {
    allMotorsOff();
    return;
  }

  // Fixed event slot. If Python repeats the same command, we do
  // not queue another pulse train; the current pattern continues
  // and naturally repeats after its silent tail.
  const unsigned long elapsed = millis() - patternStartMs;
  const unsigned long phase = elapsed % HAPTIC_EVENT_CYCLE_MS;
  const bool outputOn = directionPulseIsOn(phase);
  const int activeChannel = directionChannel(currentDirection);

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
// FRONT and REAR are measured concurrently.
// pulseIn() is deliberately NOT used because it blocks on one sensor before
// the other sensor can be read.
//
// Both HC-SR04 trigger lines are driven together. Each ECHO line is
// captured independently by an ISR.
// Therefore an echo from FRONT never makes the code wait before capturing REAR.

const unsigned long ULTRASONIC_TIMEOUT_US = 25000UL;  // ~4.3 m maximum
volatile uint32_t frontEchoRiseUs = 0;
volatile uint32_t rearEchoRiseUs = 0;
volatile uint32_t frontEchoDurationUs = 0;
volatile uint32_t rearEchoDurationUs = 0;
volatile bool frontEchoDone = false;
volatile bool rearEchoDone = false;

void IRAM_ATTR frontEchoISR() {
  uint32_t now = micros();

  if (digitalRead(FRONT_ECHO_PIN)) {
    frontEchoRiseUs = now;
  } else if (frontEchoRiseUs != 0) {
    frontEchoDurationUs = now - frontEchoRiseUs;
    frontEchoRiseUs = 0;
    frontEchoDone = true;
  }
}

void IRAM_ATTR rearEchoISR() {
  uint32_t now = micros();

  if (digitalRead(REAR_ECHO_PIN)) {
    rearEchoRiseUs = now;
  } else if (rearEchoRiseUs != 0) {
    rearEchoDurationUs = now - rearEchoRiseUs;
    rearEchoRiseUs = 0;
    rearEchoDone = true;
  }
}

void triggerBothUltrasonicSensors() {
  noInterrupts();

  frontEchoRiseUs = 0;
  rearEchoRiseUs = 0;
  frontEchoDurationUs = 0;
  rearEchoDurationUs = 0;
  frontEchoDone = false;
  rearEchoDone = false;

  // Trigger both HC-SR04 sensors together.
  // digitalWrite() is used for compatibility with the installed
  // Arduino-ESP32 core; there is no 60 ms front/rear separation.
  digitalWrite(FRONT_TRIG_PIN, LOW);
  digitalWrite(REAR_TRIG_PIN, LOW);

  delayMicroseconds(2);

  digitalWrite(FRONT_TRIG_PIN, HIGH);
  digitalWrite(REAR_TRIG_PIN, HIGH);

  delayMicroseconds(10);

  digitalWrite(FRONT_TRIG_PIN, LOW);
  digitalWrite(REAR_TRIG_PIN, LOW);

  interrupts();
}

float durationToDistanceCm(uint32_t durationUs) {
  if (durationUs == 0 || durationUs > ULTRASONIC_TIMEOUT_US) {
    return -1.0f;
  }

  float distanceCm = (float)durationUs * 0.0343f / 2.0f;

  if (distanceCm < 2.0f || distanceCm > 430.0f) {
    return -1.0f;
  }

  return distanceCm;
}

void readBothUltrasonicSimultaneously(float &frontCm, float &rearCm) {
  triggerBothUltrasonicSensors();

  // Wait for BOTH echo ISRs. There is no intentional front/rear delay.
  uint32_t waitStart = micros();

  while (true) {
    bool frontDone;
    bool rearDone;

    noInterrupts();
    frontDone = frontEchoDone;
    rearDone = rearEchoDone;
    uint32_t frontDuration = frontEchoDurationUs;
    uint32_t rearDuration = rearEchoDurationUs;
    interrupts();

    uint32_t elapsed = micros() - waitStart;

    if ((frontDone && rearDone) || elapsed >= ULTRASONIC_TIMEOUT_US) {
      frontCm = durationToDistanceCm(frontDuration);
      rearCm = durationToDistanceCm(rearDuration);
      return;
    }

    // Yield to Wi-Fi/FreeRTOS while waiting for the echo edges.
    // This does NOT introduce a sensor-to-sensor delay.
    delayMicroseconds(20);
  }
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

  // FRONT + REAR are triggered at the same instant and their ECHO pulses
  // are captured independently by interrupts.
  float frontCm = -1.0f;
  float rearCm = -1.0f;
  readBothUltrasonicSimultaneously(frontCm, rearCm);

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

  // Capture both echo signals independently. No pulseIn() blocking.
  attachInterrupt(digitalPinToInterrupt(FRONT_ECHO_PIN), frontEchoISR, CHANGE);
  attachInterrupt(digitalPinToInterrupt(REAR_ECHO_PIN), rearEchoISR, CHANGE);

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

  // Local haptic pattern. Ultrasonic FRONT + REAR are captured concurrently,
  // so one sensor never waits for the other sensor to finish.
  if (millis() - lastPatternMs >= 20) {
    lastPatternMs = millis();
    applyPattern();
  }

  sendUltrasonicPacket();
}
