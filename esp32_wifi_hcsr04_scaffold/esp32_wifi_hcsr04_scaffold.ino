#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

void setup() {
  Serial.begin(115200);

  Wire.begin(21, 22);

  pwm.begin();
  pwm.setPWMFreq(1000);

  Serial.println("PCA9685 test started");
}

void loop() {
  // LED ON
  pwm.setPWM(0, 0, 4095);
  Serial.println("LED ON");
  delay(3000);

  // LED OFF
  pwm.setPWM(0, 4095, 4095);
  Serial.println("LED OFF");
  delay(3000);
}