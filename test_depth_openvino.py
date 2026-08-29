"""
Live OpenVINO Depth Anything V2 GPU Benchmark
---------------------------------------------

Uses the existing phone IP-camera stream and runs the converted
Depth Anything V2 Small model on the Intel GPU through OpenVINO.

This is a BENCHMARK ONLY:
    Camera -> OpenVINO Depth Anything V2 -> Intel Iris Xe GPU

It does not run YOLO or the GRU.
"""

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
    "depth_openvino/depth_anything_v2_vits_fp32.xml"
)

INPUT_SIZE = 518

# Run continuously for a stable benchmark.
# Use 30 warm-up iterations and then benchmark this many iterations.
WARMUP = 10
BENCHMARK_ITERATIONS = 100


# ------------------------------------------------------------
# OpenVINO
# ------------------------------------------------------------
core = ov.Core()

print("Available devices:", core.available_devices)

if "GPU" not in core.available_devices:
    raise RuntimeError(
        "OpenVINO cannot see an Intel GPU. "
        f"Available devices: {core.available_devices}"
    )

print("\nLoading OpenVINO model...")
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
# Preprocessing
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
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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


def make_depth_vis(raw_depth, width, height):
    raw_depth = np.asarray(
        raw_depth,
        dtype=np.float32,
    ).squeeze()

    lo = float(np.percentile(raw_depth, 2))
    hi = float(np.percentile(raw_depth, 98))

    depth_norm = np.clip(
        (raw_depth - lo) / max(hi - lo, 1e-6),
        0.0,
        1.0,
    )

    depth_u8 = (depth_norm * 255.0).astype(np.uint8)

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
# Benchmark
# ------------------------------------------------------------
latencies = []
frame_count = 0
display_fps = 0.0
last_display_time = time.perf_counter()

# Read one frame first.
ok, frame = cap.read()

if not ok:
    cap.release()
    raise RuntimeError("Could not read the first camera frame.")

tensor = preprocess(frame)

print("\nWarming up OpenVINO GPU...")
for _ in range(WARMUP):
    compiled_model([tensor])

print("Warm-up complete.")
print(f"Benchmarking {BENCHMARK_ITERATIONS} GPU inferences...\n")


benchmark_start = time.perf_counter()

while frame_count < BENCHMARK_ITERATIONS:
    ok, frame = cap.read()

    if not ok:
        print("Camera frame read failed.")
        continue

    tensor = preprocess(frame)

    start = time.perf_counter()

    result = compiled_model([tensor])

    # Force access to the output so the benchmark includes output completion.
    raw_depth = result[output_layer]

    latency = (time.perf_counter() - start) * 1000.0
    latencies.append(latency)

    frame_count += 1

    # Visualize current depth.
    depth_vis = make_depth_vis(
        raw_depth,
        frame.shape[1],
        frame.shape[0],
    )

    combined = np.hstack(
        (frame, depth_vis)
    )

    # UI FPS measures the full camera/inference/display iteration.
    now = time.perf_counter()
    dt = now - last_display_time
    last_display_time = now

    if dt > 0:
        instant_fps = 1.0 / dt
        display_fps = (
            instant_fps
            if display_fps == 0
            else 0.15 * instant_fps + 0.85 * display_fps
        )

    cv2.putText(
        combined,
        f"OpenVINO GPU | inference: {latency:.1f} ms",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        combined,
        f"Loop FPS: {display_fps:.1f}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        combined,
        "DEPTH ANYTHING V2 - RELATIVE DEPTH",
        (frame.shape[1] + 10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    cv2.imshow(
        "Depth Anything V2 - OpenVINO Intel GPU Benchmark",
        combined,
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


benchmark_elapsed = time.perf_counter() - benchmark_start

cap.release()
cv2.destroyAllWindows()


if latencies:
    latencies_np = np.asarray(latencies, dtype=np.float64)

    print("\n" + "=" * 60)
    print("DEPTH ANYTHING V2 + OPENVINO GPU BENCHMARK")
    print("=" * 60)
    print(f"Successful GPU inferences : {len(latencies)}")
    print(f"Average inference latency : {latencies_np.mean():.2f} ms")
    print(f"Median inference latency  : {np.median(latencies_np):.2f} ms")
    print(f"Best inference latency    : {latencies_np.min():.2f} ms")
    print(f"Worst inference latency   : {latencies_np.max():.2f} ms")
    print(
        f"Depth inference FPS       : "
        f"{1000.0 / latencies_np.mean():.2f}"
    )
    print(
        f"End-to-end benchmark FPS  : "
        f"{len(latencies) / benchmark_elapsed:.2f}"
    )
    print("=" * 60)
