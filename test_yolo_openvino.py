import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov


SOURCE = "http://192.168.29.115:8080/video"

MODEL_PATH = Path(
    "yolov8n_openvino_model/yolov8n.xml"
)

INPUT_SIZE = 256

core = ov.Core()

print("Available devices:", core.available_devices)

model = core.read_model(MODEL_PATH)

compiled = core.compile_model(
    model,
    "GPU",
)

input_layer = compiled.input(0)
output_layer = compiled.output(0)

print("Compiled device: GPU")
print("Input shape:", input_layer.shape)
print("Output shape:", output_layer.shape)

cap = cv2.VideoCapture(SOURCE)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open stream: {SOURCE}"
    )

print("\nCamera opened.")
print("Press 'q' to quit.")

# ------------------------------------------------------------
# Read first frame
# ------------------------------------------------------------
ok, frame = cap.read()

if not ok:
    cap.release()
    raise RuntimeError("Could not read first frame.")


def preprocess(frame):
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    rgb = cv2.resize(
        rgb,
        (INPUT_SIZE, INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )

    rgb = rgb.astype(np.float32) / 255.0

    # YOLO OpenVINO export expects NCHW float input.
    tensor = np.transpose(
        rgb,
        (2, 0, 1),
    )[None, ...]

    return tensor.astype(np.float32)


# ------------------------------------------------------------
# Warmup
# ------------------------------------------------------------
tensor = preprocess(frame)

print("\nWarming up GPU...")

for _ in range(20):
    compiled([tensor])

print("Warm-up complete.")


# ------------------------------------------------------------
# Live benchmark
# ------------------------------------------------------------
latencies = []
frame_count = 0

fps = 0.0
last_time = time.perf_counter()

while frame_count < 200:

    ok, frame = cap.read()

    if not ok:
        continue

    tensor = preprocess(frame)

    start = time.perf_counter()

    result = compiled([tensor])

    # Force output access.
    raw_output = result[output_layer]

    inference_time = (
        time.perf_counter() - start
    )

    latency_ms = inference_time * 1000.0

    latencies.append(latency_ms)

    frame_count += 1

    # FPS for the complete processing loop.
    now = time.perf_counter()
    dt = now - last_time
    last_time = now

    if dt > 0:
        current_fps = 1.0 / dt

        fps = (
            current_fps
            if fps == 0
            else 0.15 * current_fps
            + 0.85 * fps
        )

    cv2.putText(
        frame,
        f"YOLO OpenVINO GPU | "
        f"{latency_ms:.1f} ms",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"Loop FPS: {fps:.1f}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    cv2.imshow(
        "YOLOv8n OpenVINO GPU Benchmark",
        frame,
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------
if latencies:

    latencies = np.asarray(
        latencies,
        dtype=np.float64,
    )

    print("\n" + "=" * 60)
    print("YOLOv8n + OPENVINO GPU BENCHMARK")
    print("=" * 60)

    print(
        f"Successful inferences : "
        f"{len(latencies)}"
    )

    print(
        f"Average latency       : "
        f"{latencies.mean():.2f} ms"
    )

    print(
        f"Median latency        : "
        f"{np.median(latencies):.2f} ms"
    )

    print(
        f"Best latency          : "
        f"{latencies.min():.2f} ms"
    )

    print(
        f"Worst latency         : "
        f"{latencies.max():.2f} ms"
    )

    print(
        f"Theoretical inference FPS : "
        f"{1000.0 / latencies.mean():.2f}"
    )

    print("=" * 60)