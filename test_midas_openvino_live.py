import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
SOURCE = "http://192.168.29.115:8080/video"

MODEL_PATH = Path(
    "MiDaS/weights/openvino/openvino_midas_v21_small_256.xml"
)

INPUT_SIZE = 256

WARMUP = 10
BENCHMARK_ITERATIONS = 100


# ------------------------------------------------------------
# OpenVINO
# ------------------------------------------------------------
core = ov.Core()

print("Available devices:", core.available_devices)

if "GPU" not in core.available_devices:
    raise RuntimeError(
        f"OpenVINO GPU not available: {core.available_devices}"
    )

print("\nLoading MiDaS Small OpenVINO model...")

model = core.read_model(MODEL_PATH)

compiled_model = core.compile_model(
    model,
    "GPU",
)

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

print("Compiled device: GPU")
print("Input shape:", input_layer.shape)
print("Output shape:", output_layer.shape)


# ------------------------------------------------------------
# Camera
# ------------------------------------------------------------
cap = cv2.VideoCapture(SOURCE)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open camera stream: {SOURCE}"
    )

print("\nCamera stream opened.")
print("Press 'q' to quit.")


# ------------------------------------------------------------
# MiDaS preprocessing
# Official OpenVINO MiDaS Small uses ImageNet normalization.
# ------------------------------------------------------------
mean = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
).reshape(1, 1, 3)

std = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
).reshape(1, 1, 3)


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

    rgb = (rgb - mean) / std

    tensor = np.transpose(
        rgb,
        (2, 0, 1),
    )[None, ...]

    return tensor.astype(np.float32)


def depth_visualization(raw_depth, width, height):
    raw_depth = np.asarray(
        raw_depth,
        dtype=np.float32,
    ).squeeze()

    lo = float(np.percentile(raw_depth, 2))
    hi = float(np.percentile(raw_depth, 98))

    norm = np.clip(
        (raw_depth - lo) / max(hi - lo, 1e-6),
        0.0,
        1.0,
    )

    depth_u8 = (
        norm * 255.0
    ).astype(np.uint8)

    depth_vis = cv2.applyColorMap(
        depth_u8,
        cv2.COLORMAP_TURBO,
    )

    depth_vis = cv2.resize(
        depth_vis,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )

    return depth_vis


# ------------------------------------------------------------
# First frame
# ------------------------------------------------------------
ok, frame = cap.read()

if not ok:
    cap.release()
    raise RuntimeError(
        "Could not read first camera frame."
    )


# ------------------------------------------------------------
# Warm-up
# ------------------------------------------------------------
tensor = preprocess(frame)

print("\nWarming up GPU...")

for _ in range(WARMUP):
    compiled_model([tensor])

print("Warm-up complete.")
print(
    f"Benchmarking {BENCHMARK_ITERATIONS} "
    "MiDaS GPU inferences...\n"
)


# ------------------------------------------------------------
# Benchmark
# ------------------------------------------------------------
latencies = []
count = 0

loop_start = time.perf_counter()

while count < BENCHMARK_ITERATIONS:

    ok, frame = cap.read()

    if not ok:
        continue

    tensor = preprocess(frame)

    start = time.perf_counter()

    result = compiled_model([tensor])

    # Force output access.
    raw_depth = result[output_layer]

    elapsed = (
        time.perf_counter() - start
    )

    latency_ms = elapsed * 1000.0

    latencies.append(latency_ms)

    count += 1

    depth_vis = depth_visualization(
        raw_depth,
        frame.shape[1],
        frame.shape[0],
    )

    combined = np.hstack(
        (frame, depth_vis)
    )

    cv2.putText(
        combined,
        f"MiDaS Small | GPU | "
        f"{latency_ms:.1f} ms",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    cv2.imshow(
        "MiDaS Small - OpenVINO Intel GPU",
        combined,
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


total_time = (
    time.perf_counter() - loop_start
)

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

    average = latencies.mean()
    median = np.median(latencies)
    minimum = latencies.min()
    maximum = latencies.max()

    print("\n" + "=" * 60)
    print(
        "MiDaS Small + OpenVINO GPU Benchmark"
    )
    print("=" * 60)

    print(
        f"Successful inferences : {len(latencies)}"
    )

    print(
        f"Average latency       : {average:.2f} ms"
    )

    print(
        f"Median latency        : {median:.2f} ms"
    )

    print(
        f"Best latency          : {minimum:.2f} ms"
    )

    print(
        f"Worst latency         : {maximum:.2f} ms"
    )

    print(
        f"Depth FPS             : "
        f"{1000.0 / average:.2f}"
    )

    print(
        f"End-to-end FPS        : "
        f"{len(latencies) / total_time:.2f}"
    )

    print("=" * 60)