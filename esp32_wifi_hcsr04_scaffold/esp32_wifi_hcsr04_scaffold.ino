#include <Wire.h>

#define PCA9685_ADDR 0x40
#define SDA_PIN 21
#define SCL_PIN 22

#define MODE1 0x00
#define PRESCALE 0xFE
#define LED0_ON_L 0x06

void writeRegister(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(PCA9685_ADDR);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

void setPWM(uint8_t channel, uint16_t on, uint16_t off) {
  uint8_t reg = LED0_ON_L + (4 * channel);

  Wire.beginTransmission(PCA9685_ADDR);
  Wire.write(reg);

  Wire.write(on & 0xFF);
  Wire.write(on >> 8);

  Wire.write(off & 0xFF);
  Wire.write(off >> 8);

  Wire.endTransmission();
}

void setDutyPercent(uint8_t percent) {
  percent = constrain(percent, 0, 100);

  uint16_t offValue = (4095UL * percent) / 100UL;

  if (percent == 0) {
    setPWM(0, 0, 0);
  }
  else if (percent == 100) {
    setPWM(0, 4096, 0);
  }
  else {
    setPWM(0, 0, offValue);
  }
}

void setup() {
  Serial.begin(115200);

  Wire.begin(SDA_PIN, SCL_PIN);

  // Put PCA9685 into sleep mode
  writeRegister(MODE1, 0x10);

  // ~200 Hz PWM
  writeRegister(PRESCALE, 29);

  // Wake up
  writeRegister(MODE1, 0x00);

  delay(10);

  // Restart + auto increment
  writeRegister(MODE1, 0xA1);

  // Start with LED OFF
  setDutyPercent(0);

  Serial.println("PCA9685 PWM0 LED test");
}

void loop() {

  Serial.println("PWM0 = 0%");
  setDutyPercent(0);
  delay(2000);

  Serial.println("PWM0 = 25%");
  setDutyPercent(25);
  delay(2000);

  Serial.println("PWM0 = 50%");
  setDutyPercent(50);
  delay(2000);

  Serial.println("PWM0 = 75%");
  setDutyPercent(75);
  delay(2000);

  Serial.println("PWM0 = 100%");
  setDutyPercent(100);
  delay(2000);
}