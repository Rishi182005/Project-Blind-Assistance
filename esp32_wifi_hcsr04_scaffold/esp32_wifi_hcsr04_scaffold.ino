#include <WiFi.h>
#include <WiFiUdp.h>

// Keep the SAME Wi-Fi credentials from your current working sketch.
const char* WIFI_SSID = "Rishi's Network";
const char* WIFI_PASSWORD = "Devi@1002";

const char* LAPTOP_IP = "192.168.29.85";
const uint16_t LAPTOP_PORT = 4210;

// Sensor 1: FRONT
const int FRONT_TRIG_PIN = 5;
const int FRONT_ECHO_PIN = 18;

// Sensor 2: REAR
const int REAR_TRIG_PIN = 19;
const int REAR_ECHO_PIN = 23;

WiFiUDP udp;

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

void setup() {
  Serial.begin(115200);

  pinMode(FRONT_TRIG_PIN, OUTPUT);
  pinMode(FRONT_ECHO_PIN, INPUT);

  pinMode(REAR_TRIG_PIN, OUTPUT);
  pinMode(REAR_ECHO_PIN, INPUT);

  digitalWrite(FRONT_TRIG_PIN, LOW);
  digitalWrite(REAR_TRIG_PIN, LOW);

  connectWiFi();
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    connectWiFi();
  }

  // Trigger the two sensors separately so their ultrasonic pulses do not
  // overlap and cause cross-talk.
  float frontCm = readDistanceCm(FRONT_TRIG_PIN, FRONT_ECHO_PIN);
  delay(60);
  float rearCm = readDistanceCm(REAR_TRIG_PIN, REAR_ECHO_PIN);

  char packet[128];
  snprintf(
    packet,
    sizeof(packet),
    "front_cm:%.1f,rear_cm:%.1f,ts:%lu",
    frontCm,
    rearCm,
    millis()
  );

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

  delay(40);
}
