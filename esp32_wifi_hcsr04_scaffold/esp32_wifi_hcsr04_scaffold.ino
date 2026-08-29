/*
  ESP32 WiFi + HC-SR04 Scaffold — Wearable Navigation System
  ============================================================

  WHAT THIS DOES:
  1. Connects the ESP32 to your home WiFi
  2. Reads distance from an HC-SR04 ultrasonic sensor
  3. Sends each reading to your laptop over WiFi (UDP) as a simple text message

  HOW TO USE THIS TODAY (before HC-SR04 arrives Saturday):
  - Fill in your WiFi name/password below (WIFI_SSID, WIFI_PASSWORD)
  - Flash this to the ESP32 (see "HOW TO FLASH" section at the bottom)
  - Open Arduino IDE's Serial Monitor (Tools > Serial Monitor, set to 115200 baud)
  - You should see it connect to WiFi and print its own IP address
  - It will print "HC-SR04 not connected or no echo" repeatedly — that's expected,
    since the sensor isn't wired in yet. This still proves WiFi works.

  ONCE HC-SR04 ARRIVES SATURDAY:
  - Wire it: VCC -> ESP32 3.3V (or 5V if your board has a 5V pin), GND -> GND,
    TRIG -> GPIO 5, ECHO -> GPIO 18 (see WIRING NOTES below for why these pins)
  - No code changes needed — it'll start printing real distances automatically

  ============================================================
*/

#include <WiFi.h>
#include <WiFiUdp.h>

// ---------- FILL THESE IN ----------
const char* WIFI_SSID     = "Rishi's Network";
const char* WIFI_PASSWORD = "Devi@1002";

// Your laptop's IP address on the same WiFi network.
// HOW TO FIND IT:
//   Windows: open Command Prompt, type "ipconfig", look for "IPv4 Address"
//            under your WiFi adapter (usually starts with 192.168.x.x)
//   Mac/Linux: open Terminal, type "ifconfig" or "ip addr", look for
//              inet address under your WiFi interface (en0 / wlan0)
const char* LAPTOP_IP = "192.168.29.85";  // <-- CHANGE THIS to your laptop's actual IP
const int   LAPTOP_PORT = 4210;           // arbitrary port, just needs to match on laptop side

// ---------- HC-SR04 PIN WIRING ----------
// WIRING NOTES:
//   TRIG and ECHO can technically use most ESP32 GPIO pins, but avoid these:
//     GPIO 0, 2, 15 (boot mode pins - can cause flashing issues if something
//     is wired to them during power-up)
//     GPIO 6-11 (connected to the ESP32's internal flash memory - never use these)
//   GPIO 5 and 18 are safe, commonly-used choices with no such conflicts.
//   IMPORTANT: HC-SR04's ECHO pin outputs 5V, but ESP32 GPIO pins are only
//   3.3V-safe. If your HC-SR04 is powered from 5V, put a simple voltage
//   divider (two resistors, e.g. 1kΩ and 2kΩ) between ECHO and GPIO 18 to
//   avoid risking damage to the ESP32. If you power the HC-SR04 from the
//   ESP32's 3.3V pin instead, this usually isn't necessary but check your
//   specific sensor's datasheet — some HC-SR04 clones need 5V to work at all.
const int TRIG_PIN = 5;
const int ECHO_PIN = 18;

// ---------- Timing ----------
const unsigned long READ_INTERVAL_MS = 100;  // ~10 readings per second

WiFiUDP udp;
unsigned long lastReadTime = 0;

void connectToWiFi() {
  Serial.print("Connecting to WiFi: ");
  Serial.println(WIFI_SSID);

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 40) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println();
    Serial.println("WiFi connected!");
    Serial.print("ESP32 IP address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println();
    Serial.println("WiFi connection FAILED. Check SSID/password and try again.");
    Serial.println("Retrying in 5 seconds...");
    delay(5000);
    connectToWiFi();  // try again
  }
}

// Returns distance in cm, or -1.0 if no valid echo received
// (e.g. sensor not connected yet, or nothing in range)
float readDistanceCM() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  // pulseIn waits for ECHO to go HIGH, times how long it stays HIGH,
  // times out after 30000 microseconds (~5m range, matches HC-SR04 spec)
  long duration = pulseIn(ECHO_PIN, HIGH, 30000);

  if (duration == 0) {
    return -1.0;  // no echo received (sensor unplugged, or out of range)
  }

  // Speed of sound ~343 m/s = 0.0343 cm/microsecond.
  // Divide by 2 because duration covers the round trip (there and back).
  float distanceCM = (duration * 0.0343) / 2.0;
  return distanceCM;
}

void sendReadingOverWiFi(float distanceCM) {
  if (WiFi.status() != WL_CONNECTED) return;  // skip silently if WiFi dropped

  char message[64];
  snprintf(message, sizeof(message), "distance_cm:%.1f,ts:%lu", distanceCM, millis());

  udp.beginPacket(LAPTOP_IP, LAPTOP_PORT);
  udp.print(message);
  udp.endPacket();
}

void setup() {
  Serial.begin(115200);
  delay(1000);  // give Serial Monitor time to connect

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  digitalWrite(TRIG_PIN, LOW);

  connectToWiFi();
}

void loop() {
  // Reconnect automatically if WiFi drops
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi disconnected, reconnecting...");
    connectToWiFi();
  }

  unsigned long now = millis();
  if (now - lastReadTime >= READ_INTERVAL_MS) {
    lastReadTime = now;

    float distance = readDistanceCM();

    if (distance < 0) {
      Serial.println("HC-SR04 not connected or no echo (this is expected until Saturday)");
    } else {
      Serial.print("Distance: ");
      Serial.print(distance);
      Serial.println(" cm");
      sendReadingOverWiFi(distance);
    }
  }
}

/*
  ============================================================
  HOW TO FLASH THIS TO YOUR ESP32
  ============================================================

  ONE-TIME SETUP (do this once, before your first flash):
  1. Install Arduino IDE (free): https://www.arduino.cc/en/software
  2. Open Arduino IDE > File > Preferences
  3. In "Additional Board Manager URLs", paste:
     https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
  4. Go to Tools > Board > Boards Manager, search "esp32", install the
     "esp32 by Espressif Systems" package (this takes a few minutes)

  EVERY TIME YOU FLASH:
  1. Plug the ESP32 into your laptop via USB cable
  2. In Arduino IDE: Tools > Board > select "ESP32 Dev Module"
     (or the specific board name if it appears, e.g. "ESP32-WROOM-DA Module")
  3. Tools > Port > select the COM port that appeared when you plugged it in
     (Windows: something like COM3, COM4. Mac: /dev/cu.usbserial-xxxx)
  4. Paste this code into a new sketch, fill in WIFI_SSID/WIFI_PASSWORD/LAPTOP_IP
  5. Click the Upload button (right-arrow icon, top-left)
  6. Wait for "Done uploading" — if it fails with a timeout, hold the "BOOT"
     button on the ESP32 board while upload is in progress (some boards need this)
  7. Open Tools > Serial Monitor, set baud rate to 115200 (bottom-right dropdown)
     to see the connection status and readings

  ============================================================
*/
