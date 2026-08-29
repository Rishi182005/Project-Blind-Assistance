
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import openvino as ov


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
SOURCE = "http://192.168.29.115:8080/video"

MIDAS_MODEL_FILE = (
    "MiDaS/weights/openvino/"
    "openvino_midas_v21_small_256.xml"
)

DEVICE = "GPU"
INPUT_SIZE = 256

# We will NOT use a fixed depth value. MiDaS is relative depth, so this
# is calculated from the current scene every frame.
NEAR_PERCENTILE = 70.0

# Only inspect the lower/forward part of the image.
ROI_TOP = 0.34
ROI_BOTTOM = 0.92

# Depth-island filtering.
MIN_COMPONENT_AREA = 500
MIN_COMPONENT_HEIGHT = 35
MIN_COMPONENT_FILL = 0.22
MIN_DEPTH_CONTRAST = 0.045
MAX_BOTTOM_TOUCH_FRACTION = 0.70

# Three navigation zones.
ZONES = [
    ("LEFT", 0.00, 0.34),
    ("CENTER", 0.34, 0.66),
    ("RIGHT", 0.66, 1.00),
]

# Ignore the extreme bottom edge because the floor/body can dominate depth.
BOTTOM_IGNORE = 0.05

# Minimum fraction of a zone that must be near before we report a candidate.
ZONE_CANDIDATE_FRACTION = 0.10
MIN_COMPONENT_AREA = 500
MIN_COMPONENT_HEIGHT = 35



# ---------------------------------------------------------------------
# SHORT TEMPORAL STABILITY FILTER
# ---------------------------------------------------------------------
# 3 of the last 5 frames must agree to confirm an obstacle.
STABILITY_WINDOW = 5
STABILITY_HITS_REQUIRED = 3

# A smoothed box reduces small frame-to-frame jitter.
BOX_SMOOTH_ALPHA = 0.65

# Keep a confirmed box for only one missed frame. This adds no perceptible
# delay while preventing one bad depth frame from making the box disappear.
MAX_MISSED_CONFIRMED_FRAMES = 1


# ---------------------------------------------------------------------
# LOW-LATENCY LATEST-FRAME CAMERA READER
# ---------------------------------------------------------------------
class LatestFrameCapture:
    """
    Decode the phone MJPEG stream continuously and retain only the newest
    frame. This prevents old frames from accumulating while MiDaS runs.
    """

    def __init__(self, source):
        import threading

        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera stream: {source}"
            )

        try:
            self.cap.set(
                cv2.CAP_PROP_BUFFERSIZE,
                1,
            )
        except Exception:
            pass

        self.lock = threading.Lock()
        self.latest_frame = None
        self.running = False
        self.thread = None

    def start(self):
        import threading

        self.running = True
        self.thread = threading.Thread(
            target=self._reader_loop,
            daemon=True,
        )
        self.thread.start()

    def _reader_loop(self):
        while self.running:
            ok, frame = self.cap.read()

            if not ok:
                time.sleep(0.002)
                continue

            if (
                frame is None
                or not isinstance(frame, np.ndarray)
                or frame.size == 0
                or frame.ndim < 2
                or frame.shape[0] <= 0
                or frame.shape[1] <= 0
            ):
                continue

            with self.lock:
                self.latest_frame = frame

    def read_latest(self):
        with self.lock:
            if self.latest_frame is None:
                return False, None

            return True, self.latest_frame.copy()

    def stop(self):
        self.running = False

        if self.thread is not None:
            self.thread.join(timeout=1.0)

        if self.cap is not None:
            self.camera_reader.stop()
            self.cap = None


# ---------------------------------------------------------------------
# MiDaS
# ---------------------------------------------------------------------
def load_midas():
    core = ov.Core()

    print("Available devices:", core.available_devices)

    model_path = Path(MIDAS_MODEL_FILE)

    if not model_path.exists():
        raise RuntimeError(
            f"MiDaS model not found:\n{model_path}"
        )

    model = core.read_model(model_path)

    compiled = core.compile_model(
        model,
        DEVICE,
    )

    inp = compiled.input(0)
    out = compiled.output(0)

    print("MiDaS device:", DEVICE)
    print("Input shape:", inp.shape)
    print("Output shape:", out.shape)

    return compiled, out


def midas_infer(compiled, output_layer, frame):
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    rgb = cv2.resize(
        rgb,
        (INPUT_SIZE, INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )

    # Same normalization used by the OpenVINO model wrapper in our
    # current pipeline: float32 NCHW, 0..1.
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(
        tensor,
        (2, 0, 1),
    )[None, ...]

    result = compiled([tensor])

    depth = np.asarray(
        result[output_layer],
        dtype=np.float32,
    )

    # Handle [1,H,W], [H,W], or [1,1,H,W].
    depth = np.squeeze(depth)

    if depth.ndim != 2:
        raise RuntimeError(
            f"Unexpected MiDaS output after squeeze: {depth.shape}"
        )

    depth = np.nan_to_num(
        depth,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # Normalize only for visualization / relative comparisons.
    dmin = float(depth.min())
    dmax = float(depth.max())

    if dmax > dmin:
        norm = (depth - dmin) / (dmax - dmin)
    else:
        norm = np.zeros_like(depth)

    return norm


# ---------------------------------------------------------------------
# Standalone obstacle analysis
# ---------------------------------------------------------------------


def analyze_depth(depth_norm, frame_shape):
    """
    V4: detect objects as local depth protrusions, not simply "near" pixels.

    Key idea:
      * A floor usually changes depth smoothly across the image.
      * A nearby object creates a local depth residual relative to its
        surrounding smooth surface.
      * We combine local residual, gradient/edge support, component geometry,
        and a modest near-depth requirement.

    This is still a diagnostic detector; no YOLO/GRU decision is made here.
    """
    h, w = frame_shape[:2]

    depth = cv2.resize(
        depth_norm,
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)

    # Forward/ground region.
    y0 = int(h * 0.32)
    y1 = int(h * 0.94)

    roi = depth[y0:y1, :]

    if roi.size == 0:
        return (
            np.zeros((h, w), dtype=np.uint8),
            [],
            0.0,
            {},
        )

    # Smooth surface model. Large-scale floor/background gradients remain
    # in the blur, while local protrusions survive in the residual.
    smooth = cv2.GaussianBlur(
        roi,
        (0, 0),
        sigmaX=18.0,
        sigmaY=18.0,
    )

    residual = roi - smooth

    # A second, smaller blur emphasizes coherent local protrusions rather than
    # one-pixel noise.
    residual_smooth = cv2.GaussianBlur(
        residual,
        (0, 0),
        sigmaX=4.0,
        sigmaY=4.0,
    )

    # Depth gradient gives supporting evidence for object boundaries.
    gx = cv2.Sobel(
        roi,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )
    gy = cv2.Sobel(
        roi,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )
    gradient = cv2.magnitude(gx, gy)

    # Normalize gradient robustly.
    grad_scale = float(
        np.percentile(
            gradient,
            90
        )
    )
    if grad_scale > 1e-6:
        gradient_n = np.clip(
            gradient / grad_scale,
            0.0,
            1.0,
        )
    else:
        gradient_n = np.zeros_like(gradient)

    # Current-frame near threshold is used only as supporting evidence.
    near_threshold = float(
        np.percentile(
            roi,
            70.0,
        )
    )
    near = roi >= near_threshold

    # Robust residual threshold.
    abs_residual = np.abs(residual_smooth)

    residual_med = float(
        np.median(abs_residual)
    )
    residual_mad = float(
        np.median(
            np.abs(
                abs_residual
                - residual_med
            )
        )
    )

    residual_threshold = max(
        0.035,
        residual_med
        + 3.0 * residual_mad,
    )

    # Higher depth = closer in our normalized representation, so keep only
    # positive residuals: locally nearer than the surrounding smooth surface.
    protrusion = (
        (residual_smooth >= residual_threshold)
        & near
        & (gradient_n >= 0.08)
    ).astype(np.uint8) * 255

    # Light morphology connects object interiors but avoids giant floor blobs.
    kernel = np.ones(
        (3, 3),
        np.uint8,
    )

    protrusion = cv2.morphologyEx(
        protrusion,
        cv2.MORPH_OPEN,
        kernel,
    )

    protrusion = cv2.morphologyEx(
        protrusion,
        cv2.MORPH_CLOSE,
        kernel,
    )

    full_mask = np.zeros(
        (h, w),
        dtype=np.uint8,
    )

    full_mask[y0:y1, :] = protrusion

    # Connected residual regions.
    n_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            protrusion,
            connectivity=8,
        )
    )

    candidates = []

    for label in range(1, n_labels):
        area = int(
            stats[label, cv2.CC_STAT_AREA]
        )

        bw = int(
            stats[label, cv2.CC_STAT_WIDTH]
        )

        bh = int(
            stats[label, cv2.CC_STAT_HEIGHT]
        )

        x = int(
            stats[label, cv2.CC_STAT_LEFT]
        )

        y = int(
            stats[label, cv2.CC_STAT_TOP]
        )

        if area < 250:
            continue

        if bh < int(h * 0.035):
            continue

        if bw < int(w * 0.03):
            continue

        fill = area / max(
            float(bw * bh),
            1.0,
        )

        if fill < 0.10:
            continue

        component = labels == label

        component_residual = residual_smooth[
            component
        ]

        component_gradient = gradient_n[
            component
        ]

        component_depth = roi[
            component
        ]

        if component_residual.size == 0:
            continue

        contrast = float(
            np.median(component_residual)
        )

        edge_support = float(
            np.mean(component_gradient)
        )

        depth_level = float(
            np.median(component_depth)
        )

        # Require the candidate to actually protrude from a smooth surface.
        if contrast < residual_threshold:
            continue

        # Ignore very flat strips, which are often table/floor boundaries.
        aspect = bh / max(
            float(bw),
            1.0,
        )

        if aspect < 0.12 and bw > int(w * 0.20):
            continue

        cx = x + bw / 2.0

        if cx < w * 0.34:
            zone = "LEFT"
        elif cx < w * 0.66:
            zone = "CENTER"
        else:
            zone = "RIGHT"

        score = (
            min(
                contrast / 0.16,
                1.0,
            ) * 0.50
            + min(
                edge_support / 0.50,
                1.0,
            ) * 0.20
            + min(
                fill / 0.70,
                1.0,
            ) * 0.15
            + depth_level * 0.15
        )

        candidates.append({
            "zone": zone,
            "box": (
                x,
                y + y0,
                bw,
                bh,
            ),
            "area": area,
            "component_fill": fill,
            "depth_contrast": contrast,
            "edge_support": edge_support,
            "depth_level": depth_level,
            "score": score,
        })

    candidates.sort(
        key=lambda c: c["score"],
        reverse=True,
    )

    zone_stats = {
        "threshold": residual_threshold,
        "mean_abs_residual": float(
            np.mean(abs_residual)
        ),
        "max_positive_residual": float(
            np.max(residual_smooth)
        ),
        "near_fraction": float(
            np.mean(near)
        ),
        "candidates": len(candidates),
    }

    return (
        full_mask,
        candidates,
        residual_threshold,
        zone_stats,
    )



# ---------------------------------------------------------------------
# TEMPORAL STABILITY
# ---------------------------------------------------------------------
def stabilize_candidates(candidates, state):
    """
    Convert flickering per-frame candidates into stable obstacle tracks.

    A zone is confirmed when a candidate is present in >=3 of the last 5
    frames. The box is exponentially smoothed, so it does not jump around.
    """
    by_zone = {}

    # Keep only the strongest candidate per navigation zone.
    for candidate in candidates:
        zone = candidate["zone"]

        if (
            zone not in by_zone
            or candidate["score"]
            > by_zone[zone]["score"]
        ):
            by_zone[zone] = candidate

    stable = []

    for zone in ("LEFT", "CENTER", "RIGHT"):
        s = state.setdefault(
            zone,
            {
                "history": deque(maxlen=STABILITY_WINDOW),
                "smooth_box": None,
                "confirmed": False,
                "missed": 0,
                "last_candidate": None,
            },
        )

        candidate = by_zone.get(zone)

        # Record whether this frame has valid evidence in this zone.
        s["history"].append(candidate is not None)

        if candidate is not None:
            new_box = candidate["box"]

            if s["smooth_box"] is None:
                smooth = tuple(
                    int(v)
                    for v in new_box
                )
            else:
                old = s["smooth_box"]

                smooth = tuple(
                    int(
                        BOX_SMOOTH_ALPHA * new
                        + (1.0 - BOX_SMOOTH_ALPHA) * prev
                    )
                    for new, prev in zip(
                        new_box,
                        old,
                    )
                )

            s["smooth_box"] = smooth
            s["last_candidate"] = candidate
            s["missed"] = 0

        else:
            s["missed"] += 1

        hits = sum(s["history"])

        if hits >= STABILITY_HITS_REQUIRED:
            s["confirmed"] = True
        elif (
            not s["confirmed"]
            and hits == 0
        ):
            s["confirmed"] = False

        # Once confirmed, tolerate only a very short gap.
        if (
            s["confirmed"]
            and candidate is None
            and s["missed"] > MAX_MISSED_CONFIRMED_FRAMES
        ):
            s["confirmed"] = False
            s["smooth_box"] = None
            s["last_candidate"] = None
            s["missed"] = 0

        if s["confirmed"] and s["last_candidate"] is not None:
            out = dict(s["last_candidate"])

            out["box"] = s["smooth_box"]
            out["stable_hits"] = hits
            out["stable_window"] = STABILITY_WINDOW
            out["confirmed"] = True
            out["stale"] = candidate is None

            stable.append(out)

    return stable


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
midas, output_layer = load_midas()

camera_reader = LatestFrameCapture(SOURCE)
camera_reader.start()

print("\nLow-latency latest-frame reader started.")
print("Press 'q' to quit.")
print("This program tests MiDaS obstacle extraction ONLY.")
print("Temporal stability: 3/5 frames")
print()

# Wait for the first valid frame.
frame = None

for _ in range(300):
    ok, frame = camera_reader.read_latest()

    if ok and frame is not None:
        break

    time.sleep(0.005)

if frame is None:
    camera_reader.stop()
    raise RuntimeError(
        "Camera opened but no valid frame arrived."
    )

# Warmup.
ok, frame = camera_reader.read_latest()

if not ok or frame is None:
    camera_reader.stop()
    raise RuntimeError(
        "Camera opened but first frame could not be read."
    )

print("Warming up MiDaS...")

for _ in range(20):
    midas_infer(
        midas,
        output_layer,
        frame,
    )

print("Warm-up complete.\n")

last = time.perf_counter()
fps = 0.0
stability_state = {}

while True:
    ok, frame = camera_reader.read_latest()

    if not ok or frame is None:
        continue

    depth = midas_infer(
        midas,
        output_layer,
        frame,
    )

    mask, candidates, threshold, stats = analyze_depth(
        depth,
        frame.shape,
    )
    candidates = stabilize_candidates(
        candidates,
        stability_state,
    )

    now = time.perf_counter()
    dt = now - last
    last = now

    if dt > 0:
        instant = 1.0 / dt
        fps = (
            instant
            if fps == 0
            else 0.15 * instant + 0.85 * fps
        )

    camera_view = frame.copy()

    # Draw candidates.
    for candidate in candidates:
        x, y, bw, bh = candidate["box"]

        cv2.rectangle(
            camera_view,
            (x, y),
            (x + bw, y + bh),
            (0, 255, 255),
            3,
        )

        cv2.putText(
            camera_view,
            (
                f"{candidate['zone']} "
                f"C={candidate['depth_contrast']:.2f} "
                f"fill={candidate.get('component_fill', 0):.0%}"
            ),
            (x, max(y - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )

    # Draw forward-path ROI.
    y0 = int(frame.shape[0] * ROI_TOP)
    y1 = int(frame.shape[0] * ROI_BOTTOM)

    cv2.rectangle(
        camera_view,
        (0, y0),
        (frame.shape[1] - 1, y1),
        (255, 255, 255),
        1,
    )

    zone_text = (
        f"L:{stats.get('LEFT', {}).get('near_fraction', 0):.0%} "
        f"C:{stats.get('CENTER', {}).get('near_fraction', 0):.0%} "
        f"R:{stats.get('RIGHT', {}).get('near_fraction', 0):.0%}"
    )

    status = (
        f"V5 STABLE:{len(candidates)}  "
        f"R:{threshold:.3f}  FPS:{fps:.1f}"
    )

    cv2.putText(
        camera_view,
        status,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        camera_view,
        zone_text
        + f"  near:{stats.get('near_fraction', 0.0):.0%}",
        (10, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    # Depth visualization.
    depth_u8 = np.clip(
        depth * 255.0,
        0,
        255,
    ).astype(np.uint8)

    depth_vis = cv2.applyColorMap(
        depth_u8,
        cv2.COLORMAP_TURBO,
    )

    depth_vis = cv2.resize(
        depth_vis,
        (frame.shape[1], frame.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    cv2.putText(
        depth_vis,
        "MiDaS RELATIVE DEPTH",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
    )

    # Near mask.
    mask_vis = cv2.cvtColor(
        mask,
        cv2.COLOR_GRAY2BGR,
    )

    cv2.putText(
        mask_vis,
        "DEPTH PROTRUSION MASK",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
    )

    combined_depth = np.hstack(
        (
            depth_vis,
            mask_vis,
        )
    )

    combined_depth = cv2.resize(
        combined_depth,
        (frame.shape[1], frame.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    combined = np.hstack(
        (
            camera_view,
            combined_depth,
        )
    )

    cv2.imshow(
        "MiDaS Standalone Obstacle Test",
        combined,
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera_reader.stop()
cv2.destroyAllWindows()
