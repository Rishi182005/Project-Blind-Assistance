# AI-Driven Wearable Navigation System with Adaptive Feedback

> **For Visually Impaired Navigation — Predictive Risk Sensing with Personalized, Low-Irritation Guidance**

A wearable navigation assistance prototype that combines computer vision, monocular depth estimation, temporal risk prediction, ultrasonic sensing, directional haptic feedback, and voice guidance.

The system is designed to move beyond simple obstacle detection by considering **distance, closing speed, Time-to-Collision (TTC), object type, and temporal context** before deciding how an obstacle should be communicated to the wearer.

---

## Project Overview

The prototype uses a head-mounted cardboard wearable with four vibration motors distributed around the head:

- **Motor 1 — Front Center**
- **Motor 2 — Front-Left**
- **Motor 3 — Front-Right**
- **Motor 4 — Back Center**

Two HC-SR04 ultrasonic sensors provide front and rear range sensing. An ESP32 handles the sensor interface and haptic controller communication. A PCA9685 PWM driver and ULN2003 transistor array drive the four vibration motors.

The main navigation pipeline runs in Python and combines:

1. **YOLOv8n** for known-object detection
2. **Multi-object tracking** for maintaining obstacle identities across frames
3. **Empirical distance calibration** for metric distance estimation
4. **Closing-speed estimation**
5. **Time-to-Collision (TTC)** estimation and lowest-TTC obstacle selection
6. **GRU-based temporal risk prediction**
7. **MiDaS Small** for unknown-obstacle/depth-based detection
8. **HC-SR04 + ESP32** sensing as an additional range/failsafe layer
9. **Voice guidance**
10. **Directional haptic feedback**

Safety decisions remain deterministic and sensor/risk-model driven; language generation is used for natural-language guidance rather than as the primary safety decision mechanism.

---

## Key Features

### AI Perception

- Real-time YOLOv8n object detection
- Multi-object tracking
- Known-object semantic labels
- MiDaS-based depth structure for unknown obstacles
- OpenVINO acceleration for YOLO and MiDaS

### Predictive Risk Estimation

Instead of selecting an obstacle only because it is closest, the system evaluates temporal behavior.

The risk model uses a sequence containing:

- Distance
- Closing speed
- Object class
- Assumed agent speed

The GRU produces a continuous risk score in the range **0–1**, which is converted into risk buckets.

### Time-to-Collision

For tracked objects, closing speed is estimated over time and TTC is used to prioritize the obstacle requiring attention.

This is important because a farther object that is closing rapidly can become more urgent than a closer object that is not approaching.

### Unknown-Obstacle Handling

YOLO can only identify objects represented by its trained classes. The MiDaS layer provides an additional depth-based mechanism for identifying obstacle structure that is not represented by a known YOLO class.

Temporal confirmation is used to reduce one-frame depth noise.

### Adaptive Feedback

The final feedback layer uses both speech and vibration.

Risk levels:

- **LOW**
- **MEDIUM**
- **HIGH**
- **CRITICAL**

Directional vibration patterns:

- **FRONT** → 1 short pulse
- **FRONT-LEFT** → 2 close pulses
- **FRONT-RIGHT** → 2 pulses with a longer gap
- **BACK** → 3 close pulses

The haptic controller uses fixed event slots so repeated detections do not unintentionally merge into another direction pattern.

The final implementation also compensates for the physically weaker perception of the left/right motors by using a higher directional vibration intensity for those positions.

Moving objects at **HIGH** or **CRITICAL** risk also trigger haptic feedback.

---

## System Architecture

```text
                    ┌─────────────────────┐
                    │   Phone Camera      │
                    │   Wearable Video    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Frame Acquisition   │
                    │ Latest-frame buffer  │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
        ┌─────────────────┐        ┌─────────────────┐
        │    YOLOv8n      │        │   MiDaS Small   │
        │ Known Objects   │        │ Depth / Unknown │
        └────────┬────────┘        └────────┬────────┘
                 │                           │
                 ▼                           ▼
        ┌─────────────────┐        ┌─────────────────┐
        │ Multi-Object    │        │ Unknown-Object  │
        │ Tracking        │        │ Confirmation    │
        └────────┬────────┘        └────────┬────────┘
                 │                           │
                 └─────────────┬─────────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Distance + Closing  │
                    │ Speed + TTC         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   GRU Risk Model    │
                    │      Risk 0–1       │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
        ┌─────────────────┐        ┌─────────────────┐
        │ Deterministic   │        │ Voice Guidance  │
        │ Safety Logic    │        │ / Natural Text  │
        └────────┬────────┘        └─────────────────┘
                 │
                 ▼
        ┌───────────────────────────┐
        │ ESP32 / UDP Haptic Layer  │
        └─────────────┬─────────────┘
                      ▼
          ┌───────────────────────────┐
          │ PCA9685 → ULN2003 → Motors│
          └───────────────────────────┘
```

---

## Hardware

### Main Components

| Component | Quantity | Purpose |
|---|---:|---|
| ESP32 DevKit / NodeMCU-style board | 1 | Main wearable controller |
| HC-SR04 ultrasonic sensor | 2 | Front and rear distance sensing |
| PCA9685 16-channel PWM driver | 1 | PWM control for vibration channels |
| ULN2003 | 1 | Transistor driver stage for motors |
| Coin vibration motors | 4 | Directional haptic feedback |
| Breadboard | 1 | Prototype wiring |
| 10,000 mAh power bank | 1 | Portable power |
| Phone camera | 1 | Main vision input |
| Bluetooth earbuds | 1 pair | Voice output |

---

## ESP32 Pin Mapping

### Ultrasonic Sensors

| Function | ESP32 GPIO |
|---|---:|
| Front HC-SR04 TRIG | GPIO 5 |
| Front HC-SR04 ECHO | GPIO 18 |
| Rear HC-SR04 TRIG | GPIO 19 |
| Rear HC-SR04 ECHO | GPIO 23 |
| PCA9685 SDA | GPIO 21 |
| PCA9685 SCL | GPIO 22 |

The HC-SR04 ECHO signals are reduced using resistor voltage dividers before entering ESP32 GPIO pins.

### PCA9685 → ULN2003 → Motor Mapping

| PCA9685 | ULN2003 Input | ULN2003 Output | Motor |
|---|---:|---:|---|
| PWM0 / CH0 | Pin 1 | Pin 16 | Motor 1 — Front |
| PWM1 / CH1 | Pin 2 | Pin 15 | Motor 2 — Front-Left |
| PWM2 / CH2 | Pin 3 | Pin 14 | Motor 3 — Front-Right |
| PWM3 / CH3 | Pin 4 | Pin 13 | Motor 4 — Back |

**Important:** The vibration motors are driven through the ULN2003 stage rather than directly from PCA9685 outputs.

---

## Motor Feedback Design

The system separates **direction** from **risk intensity**.

### Direction

| Direction | Pattern |
|---|---|
| Front | ● |
| Front-Left | ● ● |
| Front-Right | ● &nbsp;&nbsp;&nbsp; ● |
| Back | ● ● ● |

### Risk

The vibration intensity increases with the severity of the selected risk state.

The final wearable was physically tuned because the left and right motors were perceived as weaker than the front and back motors. The software therefore uses stronger left/right drive levels while preserving the same risk ordering.

Moving-object behavior was also updated so that:

- LOW + moving → speech only
- MEDIUM + moving → speech only
- HIGH + moving → speech + vibration
- CRITICAL + moving → speech + vibration

---

## Distance Estimation

The system uses empirical calibration rather than assuming a simple inverse-square-root relationship between bounding-box area and distance.

The fitted form used during development is:

```text
distance = C × area^(-p)
```

A multi-point calibration procedure was used to fit the constants for the camera/setup.

This provides a project-specific metric-distance estimate over the intended near-field operating range.

---

## Risk Prediction

The temporal model receives four features per timestep:

```text
[distance, closing_speed, object_class, agent_speed]
```

A sequence of recent observations is passed to the GRU.

The final risk decision combines:

- GRU predictive risk
- Distance/proximity safety logic
- Motion direction
- TTC
- Unknown-obstacle safety information
- Ultrasonic sensing where applicable

A tracked-object selection step prioritizes the object with the **lowest valid TTC** instead of simply choosing the nearest object.

---

## Unknown-Obstacle Layer

The MiDaS Small model provides monocular depth information.

The depth layer is used to identify candidate obstacle regions not represented by the known-object detector. Temporal confirmation reduces false triggers caused by isolated depth-map noise.

During development, MiDaS Small with OpenVINO GPU acceleration was selected over the tested alternative because it provided the latency required for repeated depth checks in the real-time loop.

---

## Software Stack

- Python
- OpenCV
- Ultralytics YOLO
- YOLOv8n
- OpenVINO
- MiDaS Small
- TensorFlow / Keras
- GRU risk model
- NumPy
- Socket/UDP communication
- ESP32 Arduino firmware
- PCA9685 I2C control
- Piper / speech output
- Groq-based language generation for concise guidance

---

## Project Results

The Review 2 evaluation reported the following held-out real-session results:

- **Overall bucketed accuracy:** 90.95%
- **Macro accuracy:** 88.07%
- **Mean Squared Error:** 0.0375
- **Held-out test sessions:** 5
- **Training sessions:** 25
- **Validation sessions:** 5
- **Test windows:** 412
- **Test timesteps:** 12,360
- **HIGH → LOW safety misses:** 0

The Review 2 presentation also reported that MiDaS Small with OpenVINO GPU processing was approximately **18× faster** than the compared Depth Anything V2 Small setup in the tested benchmark, enabling more frequent depth checks.

These numbers are development/evaluation results from this project and should not be interpreted as clinical validation or a guarantee of safety in uncontrolled environments.

---

## Project Structure

A practical repository layout is:

```text
AI-Driven-Wearable-Navigation-System/
│
├── README.md
├── python/
│   └── final_navigation.py
│
├── esp32/
│   └── final_haptic_ultrasonic.ino
│
├── models/
│   ├── YOLOv8n/
│   ├── MiDaS/
│   └── risk_gru_model_final.keras
│
├── docs/
│   ├── architecture/
│   ├── hardware/
│   ├── testing/
│   └── report/
│
└── media/
    ├── prototype/
    ├── wiring/
    └── results/
```

> Model files and large generated assets can be kept outside the Git repository or provided through Git LFS/releases when necessary.

---

## Running the System

### 1. Prepare the Python environment

Install the libraries required by the final Python source.

Example:

```bash
pip install numpy opencv-python ultralytics openvino tensorflow groq pywin32
```

Additional speech/model dependencies may be required depending on the local installation.

### 2. Prepare the ESP32

1. Open the final ESP32 `.ino` sketch in Arduino IDE.
2. Select the ESP32 board.
3. Verify the PCA9685 address and ESP32 pin mapping.
4. Upload the firmware.
5. Power the wearable from the portable power source.

### 3. Start the Python navigator

Run the final Python navigation program on the computer hosting the vision/risk pipeline.

The ESP32 communicates haptic/sensor information using UDP.

### 4. Verify the pipeline

Before a live test, confirm:

- Camera frames are updating
- YOLO detections are visible
- Distance values are reasonable
- TTC values are stable for tracked objects
- Ultrasonic readings are present
- ESP32 is receiving haptic commands
- Each directional motor responds correctly
- Speech output is functioning

---

## Testing

The system was developed through incremental testing rather than a single final build.

Important tests included:

- Individual motor testing
- Four-motor sequence testing
- ULN2003 output-path testing
- Left/right motor wire troubleshooting
- Ultrasonic front/rear reliability testing
- Distance calibration
- Multi-object tracking
- Closing-speed estimation
- TTC selection
- GRU risk prediction
- MiDaS unknown-obstacle detection
- Direction-specific haptic patterns
- Moving-object feedback
- End-to-end wearable testing

A key hardware debugging lesson was that an apparently failed motor can actually be caused by an intermittent wire or breadboard/output connection. Each motor was therefore tested independently before final integration.

---

## Current Prototype

The physical prototype uses a cardboard headband with distributed components rather than concentrating all electronics in one location.

The completed wearable includes:

- Front ultrasonic sensor
- Rear ultrasonic sensor
- Four internally mounted vibration motors
- ESP32 controller
- PCA9685 PWM driver
- ULN2003 motor driver
- Breadboard-based prototype wiring
- Secured wire routing
- External power-bank connection

The phone camera is used as the vision source, while audio guidance is delivered through Bluetooth earbuds.

---

## Limitations

The current prototype is a research/academic prototype and has several limitations.

- The real-session dataset is limited compared with a large-scale deployment dataset.
- MEDIUM-risk cases remain more ambiguous than clear LOW/HIGH cases.
- GRU inference on the development Windows environment is CPU constrained.
- Monocular depth remains sensitive to scene and camera conditions.
- The cardboard wearable is suitable for prototyping but is not a production enclosure.
- Battery life, weather resistance, thermal behavior, long-term durability, and extensive field validation require additional work.
- The final system should not be treated as a replacement for established mobility training or professional assistive guidance.

---

## Future Work

Potential extensions include:

- Larger and more diverse real-navigation datasets
- More extensive user-centered evaluation
- Improved embedded deployment
- On-device temporal risk inference
- Better calibration across different cameras and environments
- More robust sensor-fusion timing
- Improved unknown-obstacle validation
- Long-duration habituation studies
- Personalization of alert sensitivity
- Smaller and more durable wearable hardware
- Lower-power operation
- Lightweight on-device language guidance

---

## Research References

The literature review for this project includes work covering:

- Wearable assistive navigation
- Stereo and monocular vision
- Ultrasonic obstacle detection
- Object detection
- Monocular depth estimation
- Time-to-Collision prediction
- Pedestrian trajectory prediction
- Haptic feedback
- LLM-assisted navigation

The project report contains the complete reviewed reference list and APA-formatted bibliographic entries.

---

## Disclaimer

This repository documents an academic research prototype. It has been developed for experimentation, evaluation, and demonstration. Real-world deployment for safety-critical mobility requires substantially broader validation, hardware engineering, user testing, and regulatory/safety assessment.
