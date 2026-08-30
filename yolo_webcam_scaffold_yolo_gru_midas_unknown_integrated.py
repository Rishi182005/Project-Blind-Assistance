"""
YOLOv8n webcam/video scaffold for the wearable nav system -- MULTI-OBJECT
TRACKING VERSION.

Difference from the single-nearest-object scaffold: instead of collapsing
to "whichever object is closest right now" every single frame, this keeps
a small set of tracked objects across frames (matched by centroid distance
+ class), computes closing speed PER OBJECT, estimates time-to-collision
(TTC) per object, and selects whichever tracked object currently has the
LOWEST TTC as "the" obstacle fed to the GRU.

Why this matters: a fast object far away (e.g. a car approaching quickly)
can be more urgent than a slow/stationary object that's physically closer
(e.g. a pole 2m away). Picking by raw nearest-distance misses this --
picking by TTC catches it.

Still 100% software -- no new hardware required. YOLO distance uses the
existing calibrated bbox-size heuristic. MiDaS-only distance uses raw MiDaS
inverse-depth with a separate online metric calibration learned from matched
YOLO+MiDaS objects.

pip install ultralytics opencv-python --break-system-packages
"""

import time
import threading
from collections import deque
import numpy as np
import cv2
from tensorflow import keras
import openvino as ov
from pathlib import Path


class LatestFrameReader:
    """Reads frames from a VideoCapture in a background thread, always
    keeping only the MOST RECENT frame. This prevents the delay-creep you
    get from cv2's internal buffer queuing up frames faster than the main
    loop (YOLO inference) can process them -- without this, every frame
    processed is older than the last, and the lag grows over time."""

    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open source {source}")
        self.lock = threading.Lock()
        self.latest_frame = None
        self.running = True
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def _reader_loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                continue
            with self.lock:
                self.latest_frame = frame

    def read(self):
        """Returns (ok, latest_frame) -- non-blocking, always the freshest
        frame available, never a stale queued one."""
        with self.lock:
            if self.latest_frame is None:
                return False, None
            return True, self.latest_frame.copy()

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()

# ---- config ----
SOURCE = "http://192.168.29.115:8080/video"   # phone IP-cam stream; 0 = default webcam
MODEL = "yolov8n.pt"  # kept for reference/documentation
YOLO_OPENVINO_MODEL = "yolov8n_openvino_model/yolov8n.xml"
YOLO_DEVICE = "GPU"
YOLO_INPUT_SIZE = 256

# Match the confidence/NMS style of a normal YOLO detection pipeline.
YOLO_CONF_THRESHOLD = 0.25
YOLO_NMS_IOU = 0.45
MAX_RANGE = 4.0
CLASS_MAP = {
    "person": "person",
    "car": "vehicle", "truck": "vehicle", "bus": "vehicle",
    "traffic light": "pole", "fire hydrant": "pole", "stop sign": "pole",
    "chair": "pole", "bench": "pole", "potted plant": "pole",
    "couch": "wall",
    "cell phone": "pole",
    "dining table": "wall",
    "bottle": "pole",
    "tv": "wall",
}
# Note: bucket, TV stand, air conditioner, table mat etc. aren't COCO
# classes at all -- YOLOv8n was trained on 80 fixed categories and none
# of these exist in it, so no CLASS_MAP entry can make them detectable.
# Only a custom-trained model would add them.

INFERENCE_IMGSZ = 256             # resolution YOLO actually runs inference at --
                                   # lower = faster, less accurate on small objects.
                                   # OpenVINO decoding rescales detected boxes back to the
                                   # ORIGINAL frame size automatically, so area_frac
                                   # (and CALIBRATION_K) stay valid at native res --
                                   # this only speeds up inference, doesn't touch
                                   # the resolution used for distance calibration.
SHOW_ALL_DETECTIONS = False        # set True to also draw every raw YOLO detection
                                   # (any class, dim gray) for debugging -- off by
                                   # default so the view only shows the nearest
                                   # MAX_TRACKS tracked objects
FEATURE_CLASSES = ["none", "person", "pole", "wall", "vehicle", "curb"]
AGENT_SPEED_ASSUMED = 1.2

# MiDaS has no semantic class label. For the existing trained 4-feature GRU,
# use the already-trained "curb" feature as the generic unknown-obstacle
# proxy. This keeps the GRU input shape/schema unchanged (30 x 4).
UNKNOWN_GRU_CLASS = "curb"

# ---- live GRU config ----
GRU_MODEL_FILE = "risk_gru_model_final.keras"
GRU_SEQ_LEN = 30
GRU_WARMUP_RISK = 0.0

# Keep these identical to the offline evaluation.
LOW_MED_BOUNDARY = 0.33
MED_HIGH_BOUNDARY = 0.66

# Live inference does not need to run the neural network on every camera
# frame. Running every N frames reduces CPU load while retaining a smooth
# risk display.
GRU_INFERENCE_EVERY_N_FRAMES = 3

# ---- MiDaS Small + OpenVINO ----
# MiDaS does NOT change the GRU inputs. Confirmed unknown protrusions are
# integrated afterward as a deterministic safety override.
MIDAS_MODEL_FILE = "MiDaS/weights/openvino/openvino_midas_v21_small_256.xml"
MIDAS_DEVICE = "GPU"
MIDAS_INPUT_SIZE = 256
MIDAS_INFERENCE_EVERY_N_FRAMES = 1

# ---- immediate proximity safety layer ----
# These are deliberately NOT fed back into the GRU. The GRU remains the
# trained 4-feature model; this deterministic layer handles objects that
# are already extremely close even when closing_speed is near zero.
PROXIMITY_CRITICAL_M = 0.35
PROXIMITY_HIGH_M = 0.45

# ---- unknown-obstacle risk integration ----
# MiDaS unknown obstacles are NOT fed into the GRU because the GRU was
# trained on the original 4-feature schema. Instead, a confirmed MiDaS
# protrusion acts as a deterministic safety layer on top of GRU/proximity.
UNKNOWN_OBSTACLE_MIN_RISK = LOW_MED_BOUNDARY + 0.02
UNKNOWN_OBSTACLE_HIGH_SCORE = 0.72
UNKNOWN_OBSTACLE_HIGH_DEPTH = 0.68
UNKNOWN_OBSTACLE_HIGH_BOTTOM_FRAC = 0.88

# Bottom-of-frame warning: an object extending into this fraction of the
# image height is likely very close to the camera/user.
BOTTOM_ZONE_START_FRAC = 0.78
BOTTOM_ZONE_CRITICAL_FRAC = 0.95

# ---- MiDaS object-region growth / temporal stability ----
# The protrusion mask often contains only the strongest depth edges of a
# real object. Grow each confirmed seed into the surrounding depth plateau
# so the displayed bbox covers the object rather than a tiny edge fragment.
# V14: MiDaS bbox geometry comes from the protrusion mask itself.
# Do NOT grow a seed through a depth plateau: that was causing large
# background/laptop/floor regions to become giant false boxes.
DEPTH_GROW_RADIUS_FRAC = 0.0
DEPTH_GROW_TOLERANCE = 0.0
DEPTH_GROW_MIN_COMPONENT_FRAC = 0.0
DEPTH_GROW_MAX_COMPONENT_FRAC = 0.08
BOX_SMOOTH_ALPHA = 0.22

# Retained for Track geometry bookkeeping. The V18 fusion pipeline no longer
# uses the old first-bbox reference-distance mechanism, but Track.update()
# still uses these guards for safe area calculations.
TRACK_REFERENCE_MIN_AREA_FRAC = 1e-4
TRACK_REFERENCE_MAX_AREA_FRAC = 0.50
TRACK_AREA_EMA_ALPHA = 0.20
# Fragment merging is intentionally conservative. Only edge fragments that
# are close in image space and overlap strongly in one axis may be combined.
MIDAS_MERGE_GAP_FRAC_V14 = 0.018
MIDAS_MERGE_MIN_OVERLAP_V14 = 0.30
MIDAS_MERGE_CENTER_FRAC_V14 = 0.07



EMA_ALPHA = 0.12
CLOSING_SPEED_EMA_ALPHA = 0.14
RISK_EMA_ALPHA = 0.16
SPEED_WINDOW = 8                 # frames of history kept per tracked object

# ---- distance fusion ----
# Keep the original calibrated YOLO bbox-area model as the metric anchor.
# MiDaS is used as a RELATIVE correction signal, not as a second absolute
# metre measurement. This avoids treating raw MiDaS values as metres.
#
# Frame-level fusion:
#     d_yolo = existing calibrated YOLO distance
#     r      = median(MiDaS depth inside object) /
#              median(MiDaS depth around object)
#     d_fused = d_yolo * correction(r)
#
# The fused value is then aggregated over a short temporal window before
# closing speed/TTC are calculated.
DIST_FUSION_ENABLED = True
MIDAS_CORRECTION_GAMMA = 0.65
MIDAS_CORRECTION_MIN = 0.70
MIDAS_CORRECTION_MAX = 1.30
MIDAS_RATIO_MIN = 0.70
MIDAS_RATIO_MAX = 1.60
DIST_FUSION_MEDIAN_WINDOW = 5
TRACK_DISTANCE_EMA_ALPHA = 0.22

# The old per-track reference-distance mechanism is disabled because a wrong
# first bbox can permanently anchor a track to the wrong absolute distance.
TRACK_REFERENCE_LOCK = False

SPEED_DEADBAND = 0.05
DIST_C = 0.4327                  # recalibrated from 12 fresh person-distance points (0.25m..3.00m)
DIST_EXPONENT = 0.7530

# Retained for MiDaS-only fallback objects. These values are NOT used for
# YOLO+MiDaS fused objects.
MIDAS_CALIBRATION_MIN_SAMPLES = 6
MIDAS_CALIBRATION_MAX_SAMPLES = 120
MIDAS_CALIBRATION_MIN_RAW_SPREAD = 1e-4
MIDAS_DISTANCE_MIN = 0.20
MIDAS_DISTANCE_MAX = MAX_RANGE
MIDAS_DISTANCE_FALLBACK = 4.0
MIDAS_DISTANCE_EMA_ALPHA = 0.18

ROTATE = False                   # keep native landscape orientation
PROCESS_WIDTH = None             # keep native resolution -- set to an int to force resize
DISPLAY_MAX_WIDTH = 960          # display window is capped to this width so it fits on
                                  # screen -- purely visual, does NOT affect detection/
                                  # calibration, which still run on the native frame

# ---- multi-object tracking config ----
MAX_TRACKS = 3                   # keep at most this many simultaneous tracks
                                  # (top-3 covers "several obstacles at once"
                                  # without unbounded cost per frame)
MATCH_MAX_DIST_FRAC = 0.25       # max centroid movement (as a fraction of frame
                                  # width) between frames to count as "the same
                                  # object" -- tune up if fast objects lose their
                                  # track ID, down if separate objects get merged
TRACK_TIMEOUT_S = 1.0            # drop a track if not matched for this long
                                  # (object left frame / occluded)
TTC_SAFE_VALUE = 999.0
UNKNOWN_REPLACEMENT_MARGIN_M = 0.15           # TTC assigned when an object isn't closing

# V11: geometry/fusion guards.  MiDaS mask fragments are treated as one
# physical obstacle when they are spatially close and have similar depth.
MIDAS_MERGE_GAP_FRAC = 0.035
MIDAS_MERGE_MIN_OVERLAP = 0.20
MIDAS_MERGE_CENTER_FRAC = 0.11
YOLO_MIDAS_CENTER_FRAC = 0.10
YOLO_MIDAS_MIN_CONTAINMENT = 0.18

                                  # in (moving away or stationary) -- effectively
                                  # "infinite time," so it never wins the
                                  # lowest-TTC selection over a real threat


def bbox_area_to_distance(box_area_frac):
    box_area_frac = max(box_area_frac, 1e-4)
    est = DIST_C * (box_area_frac ** (-DIST_EXPONENT))
    return float(np.clip(est, 0.2, MAX_RANGE))


class MidasMetricCalibrator:
    """Learn raw MiDaS inverse-depth -> metric distance from YOLO matches."""
    def __init__(self):
        self.samples = deque(maxlen=MIDAS_CALIBRATION_MAX_SAMPLES)
        self.a = None
        self.b = None
        self.last_prediction = None

    def add(self, raw_depth, metric_distance):
        raw_depth = float(raw_depth)
        metric_distance = float(metric_distance)
        if not np.isfinite(raw_depth) or not np.isfinite(metric_distance):
            return
        if metric_distance < MIDAS_DISTANCE_MIN or metric_distance > MIDAS_DISTANCE_MAX:
            return
        self.samples.append((raw_depth, metric_distance))
        self._fit()

    def _fit(self):
        if len(self.samples) < MIDAS_CALIBRATION_MIN_SAMPLES:
            return
        x = np.asarray([p[0] for p in self.samples], dtype=np.float64)
        y = np.asarray([1.0 / p[1] for p in self.samples], dtype=np.float64)
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            return
        if float(np.ptp(x)) < MIDAS_CALIBRATION_MIN_RAW_SPREAD:
            return
        try:
            a, b = np.polyfit(x, y, 1)
            pred = a * x + b
            resid = np.abs(pred - y)
            med = float(np.median(resid))
            mad = float(np.median(np.abs(resid - med)))
            keep = resid <= max(3.0 * mad, 0.015)
            if int(np.count_nonzero(keep)) >= MIDAS_CALIBRATION_MIN_SAMPLES:
                a, b = np.polyfit(x[keep], y[keep], 1)
            if np.isfinite(a) and np.isfinite(b) and a > 0:
                self.a = float(a)
                self.b = float(b)
        except Exception:
            pass

    def predict(self, raw_depth):
        raw_depth = float(raw_depth)
        if self.a is None or self.b is None or not np.isfinite(raw_depth):
            return MIDAS_DISTANCE_FALLBACK, False
        inv_d = self.a * raw_depth + self.b
        if not np.isfinite(inv_d) or inv_d <= 1e-6:
            return MIDAS_DISTANCE_FALLBACK, False
        dist = float(np.clip(1.0 / inv_d, MIDAS_DISTANCE_MIN, MIDAS_DISTANCE_MAX))
        if self.last_prediction is None:
            self.last_prediction = dist
        else:
            self.last_prediction = (
                MIDAS_DISTANCE_EMA_ALPHA * dist
                + (1.0 - MIDAS_DISTANCE_EMA_ALPHA) * self.last_prediction
            )
        return float(self.last_prediction), True

    @property
    def ready(self):
        return self.a is not None and self.b is not None


def resize_fixed(frame, width):
    if width is None:
        return frame
    h, w = frame.shape[:2]
    if w == width:
        return frame
    scale = width / w
    return cv2.resize(frame, (width, int(h * scale)))


def _depth_core_and_surrounding(raw_depth, box, frame_shape):
    """Return robust MiDaS depth statistics for an object and its surroundings.

    The comparison is intentionally relative within the same frame. This is
    useful because MiDaS is a relative/inverse-depth model and its absolute
    scale can drift between frames.
    """
    if (
        raw_depth is None
        or not isinstance(raw_depth, np.ndarray)
        or raw_depth.size == 0
    ):
        return float("nan"), float("nan")

    h, w = frame_shape[:2]

    depth = cv2.resize(
        raw_depth,
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)

    x1, y1, x2, y2 = map(int, box)

    x1 = max(0, min(w - 2, x1))
    y1 = max(0, min(h - 2, y1))
    x2 = max(x1 + 2, min(w, x2))
    y2 = max(y1 + 2, min(h, y2))

    bw = x2 - x1
    bh = y2 - y1

    # Inner object core: reduce contamination from the bbox boundary.
    ix1 = x1 + int(0.20 * bw)
    ix2 = x2 - int(0.20 * bw)
    iy1 = y1 + int(0.20 * bh)
    iy2 = y2 - int(0.20 * bh)

    if ix2 <= ix1 or iy2 <= iy1:
        ix1, iy1, ix2, iy2 = x1, y1, x2, y2

    core = depth[iy1:iy2, ix1:ix2]
    core_vals = core[np.isfinite(core)]

    # Surrounding ring: expand bbox, then exclude the original bbox.
    pad_x = max(8, int(0.55 * bw))
    pad_y = max(8, int(0.55 * bh))

    ox1 = max(0, x1 - pad_x)
    oy1 = max(0, y1 - pad_y)
    ox2 = min(w, x2 + pad_x)
    oy2 = min(h, y2 + pad_y)

    outer = depth[oy1:oy2, ox1:ox2].copy()

    inner_x1 = x1 - ox1
    inner_y1 = y1 - oy1
    inner_x2 = x2 - ox1
    inner_y2 = y2 - oy1

    ring_mask = np.ones(
        outer.shape,
        dtype=bool,
    )
    ring_mask[
        inner_y1:inner_y2,
        inner_x1:inner_x2,
    ] = False

    surround_vals = outer[
        ring_mask & np.isfinite(outer)
    ]

    if core_vals.size < 8 or surround_vals.size < 20:
        return float("nan"), float("nan")

    return (
        float(np.median(core_vals)),
        float(np.median(surround_vals)),
    )


def midas_relative_correction(raw_depth, box, frame_shape):
    """Convert MiDaS relative depth contrast into a bounded distance correction.

    A larger MiDaS depth in the object core than in the surrounding region
    means the object is locally closer. The correction is intentionally bounded
    so MiDaS cannot catastrophically override the calibrated YOLO anchor.
    """
    obj_depth, bg_depth = _depth_core_and_surrounding(
        raw_depth,
        box,
        frame_shape,
    )

    if (
        not np.isfinite(obj_depth)
        or not np.isfinite(bg_depth)
        or bg_depth <= 1e-6
    ):
        return 1.0, obj_depth, bg_depth, 0.0

    ratio = float(
        np.clip(
            obj_depth / bg_depth,
            MIDAS_RATIO_MIN,
            MIDAS_RATIO_MAX,
        )
    )

    # Stronger local protrusion => stronger correction, but still bounded.
    correction = float(
        np.clip(
            ratio ** (-MIDAS_CORRECTION_GAMMA),
            MIDAS_CORRECTION_MIN,
            MIDAS_CORRECTION_MAX,
        )
    )

    strength = float(
        np.clip(
            abs(np.log(max(ratio, 1e-6)))
            / abs(np.log(1.45)),
            0.0,
            1.0,
        )
    )

    return correction, obj_depth, bg_depth, strength


class Track:
    """One tracked object's identity + rolling history across frames."""
    _next_id = 1

    def __init__(self, cls_name, centroid, dist, now, box=None, frame_area=None):
        self.id = Track._next_id
        Track._next_id += 1
        self.cls_name = cls_name
        self.source = "YOLO"
        self.centroid = centroid
        self.box = box

        # Reference established from the first stable observations. The
        # calibrated initial distance remains the metric anchor; later
        # distance changes come from tracked apparent-size change.
        self.reference_area_frac = None
        self.reference_distance = None
        self.area_ema_frac = None
        self.reference_observations = []
        self.reference_locked = False

        if box is not None and frame_area:
            x1, y1, x2, y2 = box
            area_frac = max(
                ((x2 - x1) * (y2 - y1)) / max(frame_area, 1),
                TRACK_REFERENCE_MIN_AREA_FRAC,
            )
            if TRACK_REFERENCE_MIN_AREA_FRAC <= area_frac <= TRACK_REFERENCE_MAX_AREA_FRAC:
                self.reference_observations.append((area_frac, dist))
                self.area_ema_frac = area_frac

        self.smoothed_dist = float(dist)
        self.distance_history = deque(
            [float(dist)],
            maxlen=DIST_FUSION_MEDIAN_WINDOW,
        )
        self.history = [(now, float(dist))]
        self.filtered_closing_speed = 0.0
        self.last_seen = now
        self.last_yolo_distance = float(dist)
        self.last_midas_ratio = 1.0
        self.last_midas_correction = 1.0
        self.last_midas_strength = 0.0

    def _update_reference(self, area_frac, calibrated_dist):
        if self.reference_locked:
            return

        self.reference_observations.append(
            (area_frac, calibrated_dist)
        )

        if len(self.reference_observations) < TRACK_REFERENCE_FRAMES:
            return

        areas = np.asarray(
            [a for a, _ in self.reference_observations],
            dtype=np.float64,
        )
        dists = np.asarray(
            [d for _, d in self.reference_observations],
            dtype=np.float64,
        )

        self.reference_area_frac = float(np.median(areas))
        self.reference_distance = float(np.median(dists))
        self.reference_locked = True

    def _tracked_distance(self, area_frac, fallback_dist):
        if self.reference_locked and self.reference_area_frac is not None:
            # Same empirical exponent as the existing calibration, but applied
            # to the CHANGE in apparent area relative to this object's traced
            # reference box. This is the key difference from recalibrating a
            # fresh absolute distance from every noisy detection box.
            ratio = self.reference_area_frac / max(
                area_frac,
                TRACK_REFERENCE_MIN_AREA_FRAC,
            )
            estimated = self.reference_distance * (ratio ** DIST_EXPONENT)
            return float(
                np.clip(
                    estimated,
                    0.2,
                    MAX_RANGE,
                )
            )
        return float(fallback_dist)

    def update(self, cls_name, centroid, box, raw_dist, now, frame_area):
        self.cls_name = cls_name
        self.centroid = centroid

        if self.box is None:
            self.box = box
        else:
            px1, py1, px2, py2 = map(float, self.box)
            nx1, ny1, nx2, ny2 = map(float, box)
            pw, ph = max(px2-px1, 1.0), max(py2-py1, 1.0)
            nw, nh = max(nx2-nx1, 1.0), max(ny2-ny1, 1.0)
            nw = float(np.clip(nw, pw*0.78, pw*1.28))
            nh = float(np.clip(nh, ph*0.78, ph*1.28))
            ncx = 0.30*((nx1+nx2)/2.0) + 0.70*((px1+px2)/2.0)
            ncy = 0.30*((ny1+ny2)/2.0) + 0.70*((py1+py2)/2.0)
            self.box = (
                int(ncx - nw/2.0), int(ncy - nh/2.0),
                int(ncx + nw/2.0), int(ncy + nh/2.0),
            )

        x1, y1, x2, y2 = self.box
        area_frac = max(
            ((x2 - x1) * (y2 - y1)) / max(frame_area, 1),
            TRACK_REFERENCE_MIN_AREA_FRAC,
        )
        area_frac = float(
            np.clip(
                area_frac,
                TRACK_REFERENCE_MIN_AREA_FRAC,
                TRACK_REFERENCE_MAX_AREA_FRAC,
            )
        )

        if self.area_ema_frac is None:
            self.area_ema_frac = area_frac
        else:
            self.area_ema_frac = (
                TRACK_AREA_EMA_ALPHA * area_frac
                + (1.0 - TRACK_AREA_EMA_ALPHA) * self.area_ema_frac
            )

        # raw_dist is now the already-fused metric estimate. The old
        # first-bbox reference mechanism is deliberately disabled because it
        # could permanently anchor a track to an incorrect initial distance.
        fused_dist = float(raw_dist)
        self.last_yolo_distance = fused_dist

        self.distance_history.append(fused_dist)

        # Robust temporal aggregation first, then a light EMA.
        median_dist = float(
            np.median(
                np.asarray(
                    self.distance_history,
                    dtype=np.float64,
                )
            )
        )

        self.smoothed_dist = (
            TRACK_DISTANCE_EMA_ALPHA * median_dist
            + (1.0 - TRACK_DISTANCE_EMA_ALPHA) * self.smoothed_dist
        )
        self.history.append((now, self.smoothed_dist))
        if len(self.history) > SPEED_WINDOW:
            self.history.pop(0)
        self.last_seen = now

    def closing_speed(self):
        if len(self.history) < 2:
            return 0.0
        t0, d0 = self.history[0]
        t1, d1 = self.history[-1]
        dt = max(t1 - t0, 1e-3)
        speed = (d0 - d1) / dt   # positive = getting closer
        if abs(speed) < SPEED_DEADBAND:
            speed = 0.0
        # Low-pass the derivative. Distance is already EMA-smoothed, but a
        # derivative amplifies small frame-to-frame box jitter.
        self.filtered_closing_speed = (
            CLOSING_SPEED_EMA_ALPHA * speed
            + (1.0 - CLOSING_SPEED_EMA_ALPHA) * self.filtered_closing_speed
        )
        if abs(self.filtered_closing_speed) < SPEED_DEADBAND:
            return 0.0
        return self.filtered_closing_speed

    def ttc(self):
        """Time-to-collision estimate. Lower = more urgent.
        Objects not closing in get a large 'safe' value so they never
        outrank a genuine approaching threat."""
        speed = self.closing_speed()
        if speed <= 0:
            return TTC_SAFE_VALUE
        return self.smoothed_dist / speed


def match_detections_to_tracks(detections, tracks, frame_width, frame_height, now):
    """Greedy centroid+class matching: each detection claims the closest
    unclaimed track of the same class within MATCH_MAX_DIST_FRAC, else
    spawns a new track."""
    max_dist_px = MATCH_MAX_DIST_FRAC * frame_width
    unmatched_tracks = list(tracks)
    updated = []

    for cls_name, centroid, box, raw_dist in detections:
        best_track, best_dist = None, None
        for tr in unmatched_tracks:
            if tr.cls_name != cls_name:
                continue
            d = np.hypot(centroid[0] - tr.centroid[0], centroid[1] - tr.centroid[1])
            if d <= max_dist_px and (best_dist is None or d < best_dist):
                best_track, best_dist = tr, d

        if best_track is not None:
            best_track.update(cls_name, centroid, box, raw_dist, now, frame_width * frame_height)
            unmatched_tracks.remove(best_track)
            updated.append(best_track)
        else:
            new_track = Track(cls_name, centroid, raw_dist, now, box=box, frame_area=frame_width * frame_height)
            new_track.box = box
            updated.append(new_track)

    # keep still-alive-but-unmatched tracks too (object briefly occluded)
    for tr in unmatched_tracks:
        if now - tr.last_seen <= TRACK_TIMEOUT_S:
            updated.append(tr)

    # cap total tracks -- keep the ones with lowest current distance
    updated.sort(key=lambda t: t.smoothed_dist)
    return updated[:MAX_TRACKS]


def current_yolo_anchor_distance(track, frame_area):
    """Recover the original calibrated YOLO bbox-area estimate for diagnostics."""
    if track is None or track.box is None:
        return float(MAX_RANGE)
    x1, y1, x2, y2 = map(int, track.box)
    bw = max(0, x2 - x1)
    bh = max(0, y2 - y1)
    area_frac = (
        (bw * bh)
        / max(float(frame_area), 1.0)
    )
    return bbox_area_to_distance(area_frac)


def risk_bucket(risk):
    """Convert continuous GRU risk into the same three buckets used offline."""
    if risk < LOW_MED_BOUNDARY:
        return "LOW"
    if risk < MED_HIGH_BOUNDARY:
        return "MEDIUM"
    return "HIGH"



def proximity_override(tr, frame_height):
    """
    Deterministic immediate-proximity check.

    Returns:
        (level, reason)

    This is intentionally separate from GRU prediction:
      - VERY close distance can force HIGH
      - A box reaching the bottom of the image can raise proximity risk
    """
    if tr is None or tr.box is None:
        return "NONE", "no selected obstacle"

    x1, y1, x2, y2 = tr.box
    bottom_frac = float(y2) / max(frame_height, 1)
    dist = float(tr.smoothed_dist)

    if dist <= PROXIMITY_CRITICAL_M:
        return "CRITICAL", f"distance {dist:.2f}m"

    if dist <= PROXIMITY_HIGH_M:
        return "HIGH", f"distance {dist:.2f}m"

    if bottom_frac >= BOTTOM_ZONE_CRITICAL_FRAC:
        return "HIGH", f"box bottom {bottom_frac:.0%}"

    if bottom_frac >= BOTTOM_ZONE_START_FRAC:
        return "MEDIUM", f"box bottom {bottom_frac:.0%}"

    return "NONE", "outside immediate zone"


def combine_risk(gru_risk, proximity_level):
    """Combine predictive GRU risk with the deterministic proximity override."""
    if proximity_level == "CRITICAL":
        return 1.0, "CRITICAL"

    if proximity_level == "HIGH":
        return max(float(gru_risk), MED_HIGH_BOUNDARY), "HIGH"

    if proximity_level == "MEDIUM":
        # Do not force a full high risk, but don't let the GRU call an
        # immediately foregrounded object LOW.
        return max(float(gru_risk), LOW_MED_BOUNDARY + 0.02), "MEDIUM"

    return float(gru_risk), risk_bucket(gru_risk)

def unknown_obstacle_risk(unknown_candidates, frame_height):
    """Return a deterministic safety floor for confirmed MiDaS obstacles.

    Metric distance/TTC for MiDaS-only obstacles are computed from raw MiDaS
    depth using the separate online MiDaS metric calibrator. MiDaS bbox area is
    never passed through the YOLO distance calibration.
    """
    confirmed = [
        c for c in unknown_candidates
        if c.get("confirmed", c.get("stable_confirmed", False))
    ]
    if not confirmed:
        return 0.0, "NONE", "none", "no confirmed unknown obstacle"

    best = max(confirmed, key=lambda c: c.get("score", 0.0))
    x, y, bw, bh = best["box"]
    bottom_frac = (y + bh) / max(float(frame_height), 1.0)
    score = float(best.get("score", 0.0))
    depth_level = float(best.get("depth_level", 0.0))

    high = (
        score >= UNKNOWN_OBSTACLE_HIGH_SCORE
        or depth_level >= UNKNOWN_OBSTACLE_HIGH_DEPTH
        or bottom_frac >= UNKNOWN_OBSTACLE_HIGH_BOTTOM_FRAC
    )

    if high:
        return (
            MED_HIGH_BOUNDARY, "HIGH", best.get("zone", "UNKNOWN"),
            f"confirmed MiDaS protrusion score={score:.2f} "
            f"depth={depth_level:.2f} bottom={bottom_frac:.2f}",
        )

    return (
        UNKNOWN_OBSTACLE_MIN_RISK, "MEDIUM", best.get("zone", "UNKNOWN"),
        f"confirmed MiDaS protrusion score={score:.2f} "
        f"depth={depth_level:.2f} bottom={bottom_frac:.2f}",
    )


def load_gru_model():
    """Load the exact fine-tuned model selected during validation."""
    try:
        gru = keras.models.load_model(GRU_MODEL_FILE)
    except Exception as e:
        raise RuntimeError(
            f"Could not load GRU model '{GRU_MODEL_FILE}'. "
            "Make sure risk_gru_model_final.keras is in the same folder "
            f"as this script. Original error: {e}"
        ) from e

    # Confirm the expected input shape.
    expected = (None, GRU_SEQ_LEN, 4)
    if len(gru.input_shape) != 3 or gru.input_shape[1:] != expected[1:]:
        raise RuntimeError(
            f"GRU input shape is {gru.input_shape}, but this live pipeline "
            f"expects {expected}."
        )

    print(f"Loaded live GRU model: {GRU_MODEL_FILE}")
    print(f"GRU input shape: {gru.input_shape}")
    return gru


def predict_live_risk(gru_model, sequence_buffer):
    """
    Run one live GRU prediction.

    sequence_buffer contains exactly 30 feature vectors with the same
    four features used by the offline model:
        normalized distance
        closing speed
        normalized class index
        assumed agent speed
    """
    if len(sequence_buffer) < GRU_SEQ_LEN:
        return GRU_WARMUP_RISK

    x = np.asarray(sequence_buffer, dtype=np.float32)
    x = x[np.newaxis, ...]  # (1, 30, 4)

    pred = gru_model.predict(x, verbose=0)

    # Model output is (1, 30, 1); use the LAST timestep because the
    # live system needs the current risk.
    risk = float(np.asarray(pred)[0, -1, 0])
    return float(np.clip(risk, 0.0, 1.0))



COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]


def load_yolo_openvino():
    """Load the exported YOLOv8n OpenVINO model on the Intel GPU."""
    model_path = Path(YOLO_OPENVINO_MODEL)

    if not model_path.exists():
        raise RuntimeError(
            f"YOLO OpenVINO model not found:\n{model_path}\n"
            "Export yolov8n.pt to OpenVINO at 256x256 first."
        )

    core = ov.Core()

    if YOLO_DEVICE not in core.available_devices:
        raise RuntimeError(
            f"OpenVINO device '{YOLO_DEVICE}' is unavailable. "
            f"Available devices: {core.available_devices}"
        )

    model = core.read_model(model_path)
    compiled = core.compile_model(model, YOLO_DEVICE)

    input_layer = compiled.input(0)
    output_layer = compiled.output(0)

    print(f"Loaded YOLOv8n OpenVINO: {model_path}")
    print(f"YOLO device: {YOLO_DEVICE}")
    print(f"YOLO input shape: {input_layer.shape}")
    print(f"YOLO output shape: {output_layer.shape}")

    return compiled, input_layer, output_layer


def preprocess_yolo_openvino(frame):
    """BGR OpenCV frame -> normalized NCHW tensor."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(
        rgb,
        (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )
    rgb = rgb.astype(np.float32) / 255.0

    tensor = np.transpose(
        rgb,
        (2, 0, 1),
    )[None, ...]

    return tensor.astype(np.float32)


def infer_yolo_openvino(compiled_model, output_layer, frame):
    """
    Decode the standard YOLOv8 detect output:
        [1, 84, 1344]
    = 4 box coordinates + 80 class scores.

    Returns detections as:
        (raw_class_name, centroid, box_px, confidence)
    """
    result = compiled_model(
        [preprocess_yolo_openvino(frame)]
    )

    output = np.asarray(
        result[output_layer],
        dtype=np.float32,
    )

    if output.ndim != 3 or output.shape[1] < 5:
        raise RuntimeError(
            f"Unexpected YOLO OpenVINO output shape: {output.shape}"
        )

    # [1, 84, 1344] -> [1344, 84]
    predictions = output[0].T

    # First 4 values are cx, cy, w, h. Remaining values are class scores.
    boxes_cxcywh = predictions[:, :4]
    class_scores = predictions[:, 4:]

    class_ids = np.argmax(
        class_scores,
        axis=1,
    )
    confidences = class_scores[
        np.arange(class_scores.shape[0]),
        class_ids,
    ]

    keep = confidences >= YOLO_CONF_THRESHOLD

    boxes_cxcywh = boxes_cxcywh[keep]
    class_ids = class_ids[keep]
    confidences = confidences[keep]

    if len(boxes_cxcywh) == 0:
        return []

    h, w = frame.shape[:2]

    sx = w / float(YOLO_INPUT_SIZE)
    sy = h / float(YOLO_INPUT_SIZE)

    boxes = []
    score_list = []

    for (cx, cy, bw, bh), conf in zip(
        boxes_cxcywh,
        confidences,
    ):
        x1 = int((cx - bw / 2.0) * sx)
        y1 = int((cy - bh / 2.0) * sy)
        x2 = int((cx + bw / 2.0) * sx)
        y2 = int((cy + bh / 2.0) * sy)

        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        bw_px = max(0, x2 - x1)
        bh_px = max(0, y2 - y1)

        boxes.append([x1, y1, bw_px, bh_px])
        score_list.append(float(conf))

    indices = cv2.dnn.NMSBoxes(
        boxes,
        score_list,
        YOLO_CONF_THRESHOLD,
        YOLO_NMS_IOU,
    )

    if indices is None or len(indices) == 0:
        return []

    indices = np.asarray(
        indices,
        dtype=np.int32,
    ).reshape(-1)

    detections = []

    for idx in indices:
        x, y, bw_px, bh_px = boxes[int(idx)]
        x2 = x + bw_px
        y2 = y + bh_px

        cls_id = int(class_ids[int(idx)])
        cls_name = (
            COCO_NAMES[cls_id]
            if 0 <= cls_id < len(COCO_NAMES)
            else f"class_{cls_id}"
        )

        centroid = (
            (x + x2) / 2.0,
            (y + y2) / 2.0,
        )

        detections.append(
            (
                cls_name,
                centroid,
                (x, y, x2, y2),
                float(score_list[int(idx)]),
            )
        )

    return detections

def load_midas_openvino():
    """Load the official MiDaS Small OpenVINO model on Intel GPU."""
    model_path = Path(MIDAS_MODEL_FILE)

    if not model_path.exists():
        raise RuntimeError(
            f"MiDaS OpenVINO model not found:\n{model_path}"
        )

    core = ov.Core()

    if MIDAS_DEVICE not in core.available_devices:
        raise RuntimeError(
            f"OpenVINO device '{MIDAS_DEVICE}' is unavailable. "
            f"Available devices: {core.available_devices}"
        )

    model = core.read_model(model_path)
    compiled = core.compile_model(model, MIDAS_DEVICE)

    print(f"Loaded MiDaS Small: {model_path}")
    print(f"MiDaS device: {MIDAS_DEVICE}")
    print(f"MiDaS input shape: {compiled.input(0).shape}")
    print(f"MiDaS output shape: {compiled.output(0).shape}")

    return compiled


def preprocess_midas(frame):
    """Preprocess BGR OpenCV frame for MiDaS v2.1 Small 256."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(
        rgb,
        (MIDAS_INPUT_SIZE, MIDAS_INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )
    rgb = rgb.astype(np.float32) / 255.0

    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32,
    ).reshape(1, 1, 3)

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32,
    ).reshape(1, 1, 3)

    rgb = (rgb - mean) / std

    return np.transpose(
        rgb,
        (2, 0, 1),
    )[None, ...].astype(np.float32)


def midas_infer(compiled_model, frame):
    result = compiled_model([preprocess_midas(frame)])
    depth_raw = np.asarray(
        result[compiled_model.output(0)],
        dtype=np.float32,
    ).squeeze()

    # Per-frame normalization is only for visualization/detection. Preserve the
    # raw model output for metric calibration.
    lo = float(np.percentile(depth_raw, 2))
    hi = float(np.percentile(depth_raw, 98))
    depth_norm = np.clip(
        (depth_raw - lo) / max(hi - lo, 1e-6),
        0.0,
        1.0,
    )
    return depth_norm, depth_raw


def midas_visual(depth_norm, width, height):
    depth_u8 = (depth_norm * 255.0).astype(np.uint8)
    vis = cv2.applyColorMap(
        depth_u8,
        cv2.COLORMAP_TURBO,
    )
    return cv2.resize(
        vis,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )


class AsyncMidasWorker:
    """
    Runs MiDaS on a background thread and keeps ONLY the newest frame/result.

    The YOLO + GRU main loop never waits for MiDaS. If MiDaS is still busy,
    the main loop simply uses the most recently completed depth map.
    """

    def __init__(self, compiled_model):
        self.compiled_model = compiled_model

        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_depth = None
        self._latest_raw_depth = None
        self._result_id = 0
        self._consumed_result_id = 0

        self._new_frame = threading.Event()
        self._stop = threading.Event()

        self.thread = threading.Thread(
            target=self._run,
            name="MiDaSWorker",
            daemon=True,
        )

    def start(self):
        self.thread.start()

    def submit(self, frame):
        with self._lock:
            self._latest_frame = frame.copy()
        self._new_frame.set()

    def get_latest(self):
        """Return the latest completed depth map for display/inspection."""
        with self._lock:
            if self._latest_depth is None:
                return None
            return self._latest_depth.copy()

    def get_new_result(self):
        """Return (normalized_depth, raw_depth) once per MiDaS inference."""
        with self._lock:
            if self._latest_depth is None or self._latest_raw_depth is None:
                return None
            if self._result_id == self._consumed_result_id:
                return None
            self._consumed_result_id = self._result_id
            return self._latest_depth.copy(), self._latest_raw_depth.copy()

    def stop(self):
        self._stop.set()
        self._new_frame.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def _run(self):
        while not self._stop.is_set():
            self._new_frame.wait(timeout=0.1)
            self._new_frame.clear()

            if self._stop.is_set():
                break

            with self._lock:
                frame = self._latest_frame
                self._latest_frame = None

            if frame is None:
                continue

            try:
                depth_norm, depth_raw = midas_infer(
                    self.compiled_model,
                    frame,
                )

                with self._lock:
                    self._latest_depth = depth_norm
                    self._latest_raw_depth = depth_raw
                    self._result_id += 1

            except Exception as e:
                print(f"MiDaS worker error: {e}")


# ---- PROVEN MiDaS UNKNOWN-OBSTACLE DETECTOR ----

def grow_depth_region(depth, seed_box, seed_depth, frame_shape):
    """V14: disabled depth-plateau growth.

    The protrusion mask is the trusted geometry source. Growing a seed through
    a broad relative-depth plateau can absorb desks, laptops, walls or floors
    that happen to have similar MiDaS values. Return the actual mask component
    bbox unchanged; fragment grouping is handled separately by
    merge_midas_candidates().
    """
    x, y, bw, bh = [int(v) for v in seed_box]
    return (x, y, bw, bh), float((bw * bh) / max(float(frame_shape[0] * frame_shape[1]), 1.0))

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
        0.055,
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

    # V11: connect broken edge fragments belonging to the same physical
    # object.  This acts on the residual mask, not raw depth, so smooth floors
    # are not turned into obstacles merely by this operation.
    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    protrusion = cv2.morphologyEx(
        protrusion,
        cv2.MORPH_CLOSE,
        bridge_kernel,
        iterations=2,
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

        # V14: the connected protrusion component is the navigation geometry.
        # Do not expand it using a depth plateau. The black/white mask is the
        # most reliable boundary signal we currently have from MiDaS.
        final_box = (x, y + y0, bw, bh)
        grown_area_frac = (bw * bh) / max(float(w * h), 1.0)

        # Candidate quality score. This is the same scoring rule used by the
        # proven standalone MiDaS detector. V17 previously referenced `score`
        # before defining it, which caused every MiDaS candidate pass to fail.
        score = (
            min(contrast / 0.16, 1.0) * 0.50
            + min(edge_support / 0.50, 1.0) * 0.20
            + min(fill / 0.70, 1.0) * 0.15
            + depth_level * 0.15
        )

        # Reject tiny navigation boxes. The protrusion mask can contain a
        # mathematically valid but physically meaningless speck; those should
        # never become a tracked obstacle. Keep this conservative because YOLO
        # remains responsible for known classes and MiDaS is the fallback.
        fx, fy, fw, fh = final_box
        if fw < int(w * 0.030) or fh < int(h * 0.040):
            continue

        candidates.append({
            "zone": zone,
            "box": final_box,
            "seed_box": (x, y + y0, bw, bh),
            "area": area,
            "grown_area_frac": grown_area_frac,
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

# ---- TEMPORAL STABILITY ----

STABILITY_WINDOW = 5
STABILITY_HITS_REQUIRED = 3
MAX_MISSED_CONFIRMED_FRAMES = 2

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

            # Reject implausible one-frame bbox explosions/shrinks. MiDaS can
            # momentarily attach to a nearby depth edge; temporal stability
            # should not let that redefine the tracked object immediately.
            if s["smooth_box"] is not None:
                px, py, pw, ph = s["smooth_box"]
                nx, ny, nw, nh = new_box
                max_w = max(pw * 1.45, 40)
                max_h = max(ph * 1.45, 40)
                min_w = min(pw * 0.70, max(8, pw - 12))
                min_h = min(ph * 0.70, max(8, ph - 12))
                nx = int(np.clip(nx, px - max(pw * 0.22, 30), px + max(pw * 0.22, 30)))
                ny = int(np.clip(ny, py - max(ph * 0.22, 30), py + max(ph * 0.22, 30)))
                nw = int(np.clip(nw, min_w, max_w))
                nh = int(np.clip(nh, min_h, max_h))
                new_box = (nx, ny, nw, nh)

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

def raw_midas_depth_for_box(raw_depth, box, frame_shape):
    """Robust raw MiDaS depth statistic inside a native-frame bbox."""
    if raw_depth is None or not isinstance(raw_depth, np.ndarray) or raw_depth.size == 0:
        return float("nan")
    h, w = frame_shape[:2]
    depth_full = cv2.resize(raw_depth, (w, h), interpolation=cv2.INTER_LINEAR)
    x, y, bw, bh = [int(v) for v in box]
    x0 = max(0, min(w - 1, x)); y0 = max(0, min(h - 1, y))
    x1 = max(x0 + 1, min(w, x + bw)); y1 = max(y0 + 1, min(h, y + bh))
    patch = depth_full[y0:y1, x0:x1].astype(np.float32)
    if patch.size == 0:
        return float("nan")
    py0, py1 = int(patch.shape[0]*0.15), max(int(patch.shape[0]*0.85), 1)
    px0, px1 = int(patch.shape[1]*0.15), max(int(patch.shape[1]*0.85), 1)
    core = patch[py0:py1, px0:px1]
    vals = core[np.isfinite(core)]
    if vals.size < 8:
        vals = patch[np.isfinite(patch)]
    return float(np.median(vals)) if vals.size else float("nan")


def update_unknown_tracks(raw_candidates, stable_candidates, unknown_tracks, now, frame_area, raw_midas_depth, frame_shape, midas_calibrator):
    """Maintain one metric-distance/TTC track per MiDaS navigation zone.

    Raw MiDaS candidates update the distance history every depth result, so
    once temporal stability confirms an obstacle we already have a short
    distance history for closing-speed/TTC estimation. Only confirmed zones
    are returned as active unknown tracks.
    """
    stable_by_zone = {
        c.get("zone"): c
        for c in stable_candidates
        if c.get("confirmed", c.get("stable_confirmed", False))
    }
    stable_zones = set(stable_by_zone)

    for candidate in raw_candidates:
        zone = candidate.get("zone")
        if zone not in ("LEFT", "CENTER", "RIGHT"):
            continue

        # Once a MiDaS obstacle is temporally confirmed, use the SAME smoothed
        # box that is drawn on screen for its distance estimate. This prevents
        # the displayed box and the GRU distance from disagreeing.
        measurement = stable_by_zone.get(zone, candidate)
        x, y, bw, bh = measurement["box"]
        centroid = (x + bw / 2.0, y + bh / 2.0)
        raw_midas = raw_midas_depth_for_box(
            raw_midas_depth, (x, y, bw, bh), frame_shape
        )
        measured_dist, metric_ready = midas_calibrator.predict(raw_midas)

        tr = unknown_tracks.get(zone)
        if tr is None:
            tr = Track(UNKNOWN_GRU_CLASS, centroid, measured_dist, now)
            tr.source = "MiDaS"
            tr.box = (x, y, x + bw, y + bh)
            unknown_tracks[zone] = tr
        else:
            tr.update(
                UNKNOWN_GRU_CLASS,
                centroid,
                (x, y, x + bw, y + bh),
                measured_dist,
                now,
                frame_area,
            )
            tr.source = "MiDaS"

    active = []
    for zone, tr in list(unknown_tracks.items()):
        if zone in stable_zones and tr.box is not None:
            stable = next(
                (c for c in stable_candidates if c.get("zone") == zone),
                None,
            )
            if stable is not None:
                x, y, bw, bh = stable["box"]
                tr.box = (x, y, x + bw, y + bh)
            active.append(tr)
        elif now - tr.last_seen > TRACK_TIMEOUT_S:
            del unknown_tracks[zone]

    return active


def safe_imshow(window_name, image):
    if not isinstance(image, np.ndarray):
        return
    if image.ndim < 2 or image.size == 0:
        return
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return
    try:
        cv2.imshow(
            window_name,
            np.ascontiguousarray(image),
        )
    except cv2.error as exc:
        print(f"OpenCV display warning: {exc}")

def box_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    ab = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / max(aa + ab - inter, 1.0)


def boxes_same_object(a, b):
    """Conservative physical-object association for YOLO <-> MiDaS.

    IoU alone is unreliable because YOLO can see only a semantic fragment
    while MiDaS can recover a much larger depth silhouette.  We therefore use
    overlap/containment plus centroid proximity, but *not* a simple
    center-inside rule: that rule was responsible for unrelated floor/table
    regions being fused to nearby objects.
    """
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    iou = box_iou_xyxy(a, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(1.0, (ax2-ax1)*(ay2-ay1))
    area_b = max(1.0, (bx2-bx1)*(by2-by1))
    containment = max(inter/area_a, inter/area_b)

    acx, acy = (ax1+ax2)/2.0, (ay1+ay2)/2.0
    bcx, bcy = (bx1+bx2)/2.0, (by1+by2)/2.0
    aw, ah = ax2-ax1, ay2-ay1
    bw, bh = bx2-bx1, by2-by1
    diag = max(1.0, ((aw+ah+ bw+bh)/4.0))
    center_dist = ((acx-bcx)**2 + (acy-bcy)**2) ** 0.5

    x_overlap = max(0.0, min(ax2,bx2)-max(ax1,bx1)) / max(1.0, min(aw,bw))
    y_overlap = max(0.0, min(ay2,by2)-max(ay1,by1)) / max(1.0, min(ah,bh))

    return (
        iou >= 0.15
        or containment >= YOLO_MIDAS_MIN_CONTAINMENT
        or (x_overlap >= 0.45 and y_overlap >= 0.30)
        or (center_dist <= YOLO_MIDAS_CENTER_FRAC * max(aw,ah,bw,bh,diag)
            and x_overlap >= 0.18 and y_overlap >= 0.18)
    )


def union_box(a, b):
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    )


def merge_midas_candidates(candidates, frame_w, frame_h):
    """Merge disconnected MiDaS mask fragments into ONE physical obstacle.

    The protrusion mask is an edge-like representation, so a single object
    can produce a top edge, side edge and bottom edge as separate connected
    components.  We merge those components before tracking.  A merge needs
    overlap/proximity evidence *and* similar depth; this prevents nearby but
    separate floor/table regions from becoming one giant box.
    """
    if not candidates:
        return []

    work = [dict(c) for c in candidates]

    def xyxy(c):
        x,y,bw,bh=c["box"]
        return (float(x),float(y),float(x+bw),float(y+bh))

    def depth_similar(a,b):
        da=float(a.get("depth_level",0.0))
        db=float(b.get("depth_level",0.0))
        # Relative MiDaS values are not metric; compare locally.
        return abs(da-db) <= 0.16

    def should_merge(a,b):
        ab=xyxy(a); bb=xyxy(b)
        # Never allow fragment chaining to create a giant physical-object box.
        ux1=min(ab[0],bb[0]); uy1=min(ab[1],bb[1])
        ux2=max(ab[2],bb[2]); uy2=max(ab[3],bb[3])
        uw=ux2-ux1; uh=uy2-uy1
        if (uw > frame_w*0.58 or uh > frame_h*0.62 or
                uw*uh > frame_w*frame_h*0.20):
            return False
        if not depth_similar(a,b):
            return False
        iou=box_iou_xyxy(ab,bb)
        if iou >= 0.03:
            return True

        ax1,ay1,ax2,ay2=ab; bx1,by1,bx2,by2=bb
        xgap=max(0.0,max(bx1-ax2,ax1-bx2))
        ygap=max(0.0,max(by1-ay2,ay1-by2))
        xover=max(0.0,min(ax2,bx2)-max(ax1,bx1)) / max(1.0,min(ax2-ax1,bx2-bx1))
        yover=max(0.0,min(ay2,by2)-max(ay1,by1)) / max(1.0,min(ay2-ay1,by2-by1))

        # Pieces of the same object commonly touch/approach each other with
        # substantial overlap in one axis.
        if xgap <= frame_w*MIDAS_MERGE_GAP_FRAC_V14 and yover >= MIDAS_MERGE_MIN_OVERLAP_V14:
            return True
        if ygap <= frame_h*MIDAS_MERGE_GAP_FRAC_V14 and xover >= MIDAS_MERGE_MIN_OVERLAP_V14:
            return True

        acx=(ax1+ax2)/2; acy=(ay1+ay2)/2
        bcx=(bx1+bx2)/2; bcy=(by1+by2)/2
        center_dist=((acx-bcx)**2+(acy-bcy)**2)**0.5
        scale=max(ax2-ax1,ay2-ay1,bx2-bx1,by2-by1,1.0)
        return center_dist <= MIDAS_MERGE_CENTER_FRAC_V14*scale and (xover>=0.10 or yover>=0.10)

    changed=True
    while changed and len(work)>1:
        changed=False
        best_pair=None
        best_score=-1.0
        for i in range(len(work)):
            for j in range(i+1,len(work)):
                if not should_merge(work[i],work[j]):
                    continue
                score=box_iou_xyxy(xyxy(work[i]),xyxy(work[j])) + 0.01*max(float(work[i].get("score",0)),float(work[j].get("score",0)))
                if score>best_score:
                    best_score=score; best_pair=(i,j)
        if best_pair is None:
            break
        i,j=best_pair
        a=work[i]; b=work[j]
        u=union_box(xyxy(a),xyxy(b))
        ux1,uy1,ux2,uy2=u
        new_box = (int(ux1), int(uy1), int(ux2-ux1), int(uy2-uy1))
        a["box"] = new_box
        for k in ("score","depth_level","grown_area_frac","component_fill","depth_contrast","edge_support"):
            a[k]=max(float(a.get(k,0.0)),float(b.get(k,0.0)))
        a["area"]=int(a.get("area",0))+int(b.get("area",0))
        cx=(ux1+ux2)/2.0
        a["zone"]="LEFT" if cx < frame_w*0.34 else ("CENTER" if cx < frame_w*0.66 else "RIGHT")
        work.pop(j)
        changed=True

    result=[]
    for c in work:
        x,y,bw,bh=c["box"]
        # Reject thin strips and tiny specks after merging.
        if bw < frame_w*0.055 or bh < frame_h*0.065:
            continue
        if y+bh > frame_h*0.98 and bh < frame_h*0.12:
            continue
        result.append(c)

    result.sort(key=lambda c: float(c.get("score",0.0)), reverse=True)
    return result


def main():
    yolo_model, yolo_input, yolo_output = load_yolo_openvino()
    gru_model = load_gru_model()
    midas_model = load_midas_openvino()

    midas_worker = AsyncMidasWorker(midas_model)
    midas_worker.start()
    midas_calibrator = MidasMetricCalibrator()

    reader = LatestFrameReader(SOURCE)

    try:
        reader.cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1,
        )
    except Exception:
        pass

    tracks = []
    feature_buffer = deque(maxlen=GRU_SEQ_LEN)

    live_risk = GRU_WARMUP_RISK
    live_risk_target = GRU_WARMUP_RISK

    frame_counter = 0
    midas_counter = 0

    latest_midas_depth = None
    latest_midas_raw_depth = None
    latest_unknown_mask = None
    unknown_candidates = []
    raw_unknown_candidates = []
    unknown_threshold = 0.0

    obstacle_state = {}
    unknown_tracks = {}

    # Once a depth result exists, keep the display panels alive using the
    # last valid result instead of hiding them on intermittent worker timing.
    have_depth_result = False

    fps = 0.0
    last_loop_time = time.perf_counter()
    FPS_SMOOTHING = 0.08

    print("MiDaS unknown-obstacle detector: ENABLED")
    print("Detector: proven local depth protrusion")
    print("Temporal stability: 3/5 frames")
    print("Unknown obstacle: unified distance/TTC/GRU path + safety floor.")
    print("Depth/BW panel: persistent after first valid MiDaS result.")
    print("MiDaS bbox: protrusion-mask geometry only + balanced fragment merge.")
    print("Press 'q' to quit.")

    try:
        while True:
            ok, frame = reader.read()

            if not ok or frame is None:
                time.sleep(0.002)
                continue

            if (
                not isinstance(frame, np.ndarray)
                or frame.size == 0
                or frame.ndim < 2
                or frame.shape[0] <= 0
                or frame.shape[1] <= 0
            ):
                continue

            if ROTATE:
                frame = cv2.rotate(
                    frame,
                    cv2.ROTATE_90_CLOCKWISE,
                )

            frame = resize_fixed(
                frame,
                PROCESS_WIDTH,
            )

            if (
                frame is None
                or frame.size == 0
                or frame.shape[0] <= 0
                or frame.shape[1] <= 0
            ):
                continue

            h, w = frame.shape[:2]

            # ---------------------------------------------------------
            # Loop FPS
            # ---------------------------------------------------------
            loop_now = time.perf_counter()
            dt = loop_now - last_loop_time
            last_loop_time = loop_now

            if dt > 0:
                instant_fps = 1.0 / dt
                fps = (
                    instant_fps
                    if fps <= 0
                    else FPS_SMOOTHING * instant_fps
                    + (1.0 - FPS_SMOOTHING) * fps
                )

            # ---------------------------------------------------------
            # Async MiDaS
            # ---------------------------------------------------------
            midas_counter += 1

            if (
                midas_counter
                % MIDAS_INFERENCE_EVERY_N_FRAMES
                == 0
            ):
                midas_worker.submit(frame)

            depth_result = midas_worker.get_new_result()

            if depth_result is not None:
                latest_midas_depth, latest_midas_raw_depth = depth_result
                if (
                    not isinstance(latest_midas_depth, np.ndarray)
                    or not isinstance(latest_midas_raw_depth, np.ndarray)
                    or latest_midas_depth.size == 0
                    or latest_midas_raw_depth.size == 0
                ):
                    continue
                have_depth_result = True

                # Calculate/update the B/W mask every time we receive a new
                # valid depth map. Store the mask BEFORE stability logic so a
                # temporary stabilizer error cannot make the panel disappear.
                try:
                    (
                        new_mask,
                        raw_unknown_candidates,
                        new_threshold,
                        _unknown_stats,
                    ) = analyze_depth(
                        latest_midas_depth,
                        frame.shape,
                    )

                    latest_unknown_mask = new_mask
                    unknown_threshold = new_threshold

                    # IMPORTANT: the protrusion mask can split one physical
                    # object into several disconnected components. Collapse
                    # those fragments BEFORE stability/tracking and BEFORE
                    # YOLO-vs-MiDaS association.
                    raw_unknown_candidates = merge_midas_candidates(
                        raw_unknown_candidates,
                        frame.shape[1],
                        frame.shape[0],
                    )

                    # Temporal stability cannot invalidate the underlying
                    # mask. Only the yellow stable boxes depend on this step.
                    unknown_candidates = (
                        stabilize_candidates(
                            raw_unknown_candidates,
                            obstacle_state,
                        )
                    )

                except Exception as exc:
                    # Preserve the last valid depth/mask/candidates.
                    print(
                        f"MiDaS diagnostic warning: {exc}"
                    )

            # ---------------------------------------------------------
            # YOLO
            # ---------------------------------------------------------
            yolo_detections = infer_yolo_openvino(
                yolo_model,
                yolo_output,
                frame,
            )

            now = time.time()
            detections = []
            all_boxes = []

            for (
                cls_name_raw,
                centroid,
                box_px,
                confidence,
            ) in yolo_detections:
                all_boxes.append(
                    (
                        cls_name_raw,
                        box_px,
                        confidence,
                    )
                )

                mapped = CLASS_MAP.get(
                    cls_name_raw
                )

                if mapped is None:
                    continue

                x1, y1, x2, y2 = box_px

                area_frac = (
                    (
                        (x2 - x1)
                        * (y2 - y1)
                    )
                    / max(
                        w * h,
                        1,
                    )
                )

                yolo_dist = bbox_area_to_distance(
                    area_frac
                )

                # Fuse the calibrated YOLO metric anchor with local MiDaS
                # depth evidence. MiDaS is never interpreted as metres here.
                fused_dist = yolo_dist
                midas_corr = 1.0
                midas_ratio = 1.0
                midas_strength = 0.0

                if (
                    DIST_FUSION_ENABLED
                    and latest_midas_raw_depth is not None
                ):
                    (
                        midas_corr,
                        obj_depth,
                        bg_depth,
                        midas_strength,
                    ) = midas_relative_correction(
                        latest_midas_raw_depth,
                        box_px,
                        frame.shape,
                    )

                    if (
                        np.isfinite(obj_depth)
                        and np.isfinite(bg_depth)
                        and bg_depth > 1e-6
                    ):
                        midas_ratio = float(
                            np.clip(
                                obj_depth / bg_depth,
                                MIDAS_RATIO_MIN,
                                MIDAS_RATIO_MAX,
                            )
                        )

                        # Trust the correction only in proportion to local
                        # depth evidence; weak/ambiguous MiDaS contrast stays
                        # close to the original YOLO metric estimate.
                        effective_corr = (
                            1.0
                            + (
                                midas_corr - 1.0
                            ) * midas_strength
                        )

                        fused_dist = float(
                            np.clip(
                                yolo_dist * effective_corr,
                                0.2,
                                MAX_RANGE,
                            )
                        )

                detections.append(
                    (
                        mapped,
                        centroid,
                        box_px,
                        fused_dist,
                    )
                )

            # ---------------------------------------------------------
            # PRIORITY FUSION / TOP-3 POLICY
            # ---------------------------------------------------------
            # Policy:
            #   1) YOLO owns the three primary obstacle slots.
            #   2) MiDaS is allowed to improve the geometry of a YOLO box
            #      when both describe the same physical object.
            #   3) A confirmed MiDaS-only obstacle can replace ONLY the
            #      farthest YOLO slot, and only when it is genuinely closer.
            #   4) Never display separate YOLO + MiDaS boxes for one object.
            #
            # This is deliberately different from simply concatenating both
            # detector outputs. The final navigation layer always contains
            # at most MAX_TRACKS physical obstacles.
            fused_zones = set()

            # First: keep only the three nearest valid YOLO detections.
            detections.sort(key=lambda d: d[3])
            detections = detections[:MAX_TRACKS]

            # Second: use MiDaS only as geometry enhancement for those YOLO
            # objects. If a depth region overlaps/contains a YOLO box, it is
            # considered the SAME physical obstacle.
            if unknown_candidates:
                fused_detections = []
                used_midas = set()

                for mapped, centroid, box_px, raw_dist in detections:
                    best = None
                    best_score = -1.0

                    for idx, uc in enumerate(unknown_candidates):
                        if idx in used_midas:
                            continue

                        ub = (
                            int(uc["box"][0]),
                            int(uc["box"][1]),
                            int(uc["box"][0] + uc["box"][2]),
                            int(uc["box"][1] + uc["box"][3]),
                        )

                        if boxes_same_object(box_px, ub):
                            score = box_iou_xyxy(box_px, ub)
                            # Prefer the candidate with the strongest overlap;
                            # containment also counts through boxes_same_object.
                            if score > best_score:
                                best_score = score
                                best = (idx, uc, ub)

                    if best is not None:
                        idx, uc, ub = best
                        used_midas.add(idx)
                        fused_zones.add(uc["zone"])

                        # YOLO supplies semantic identity AND metric distance.
                        # MiDaS supplies geometry only for this fused object.
                        # Learn the separate MiDaS metric mapping from this pair.
                        if latest_midas_raw_depth is not None:
                            midas_raw = raw_midas_depth_for_box(
                                latest_midas_raw_depth, uc["box"], frame.shape
                            )
                            midas_calibrator.add(midas_raw, raw_dist)

                        box_px = union_box(box_px, ub)
                        x1, y1, x2, y2 = box_px
                        centroid = (
                            (x1 + x2) / 2.0,
                            (y1 + y2) / 2.0,
                        )

                    fused_detections.append(
                        (mapped, centroid, box_px, raw_dist)
                    )

                detections = fused_detections

            # Keep frame-local fusion diagnostics keyed by bbox.
            fusion_diag = {}
            for _mapped, _centroid, _box, _fused_dist in detections:
                _area_frac = (
                    ((_box[2] - _box[0]) * (_box[3] - _box[1]))
                    / max(float(w * h), 1.0)
                )
                _yolo_anchor = bbox_area_to_distance(_area_frac)
                _corr = 1.0
                _ratio = 1.0
                _strength = 0.0
                if (
                    DIST_FUSION_ENABLED
                    and latest_midas_raw_depth is not None
                ):
                    (
                        _corr_raw,
                        _obj_d,
                        _bg_d,
                        _strength,
                    ) = midas_relative_correction(
                        latest_midas_raw_depth,
                        _box,
                        frame.shape,
                    )
                    if np.isfinite(_obj_d) and np.isfinite(_bg_d) and _bg_d > 1e-6:
                        _ratio = float(
                            np.clip(
                                _obj_d / _bg_d,
                                MIDAS_RATIO_MIN,
                                MIDAS_RATIO_MAX,
                            )
                        )
                        _corr = float(
                            1.0
                            + (_corr_raw - 1.0) * _strength
                        )
                fusion_diag[_box] = (
                    _yolo_anchor,
                    _corr,
                    _ratio,
                    _strength,
                )

            tracks = match_detections_to_tracks(
                detections,
                tracks,
                w,
                h,
                now,
            )

            for tr in tracks:
                if tr.box is not None:
                    diag = fusion_diag.get(
                        tuple(map(int, tr.box))
                    )
                    if diag is not None:
                        (
                            tr.last_yolo_distance,
                            tr.last_midas_correction,
                            tr.last_midas_ratio,
                            tr.last_midas_strength,
                        ) = diag

            for tr in tracks:
                tr.source = "YOLO"

            # ---------------------------------------------------------
            # MiDaS-only fallback candidates.
            # ---------------------------------------------------------
            # Only candidates NOT fused with a YOLO obstacle can become
            # unknown obstacles. Require temporal confirmation here so a
            # one-frame protrusion never displaces a YOLO object.
            midas_only_candidates = [
                c for c in (raw_unknown_candidates if have_depth_result else [])
                if c.get("zone") not in fused_zones
            ]
            midas_only_stable = [
                c for c in unknown_candidates
                if c.get("zone") not in fused_zones
            ]

            unknown_tracks_list = update_unknown_tracks(
                midas_only_candidates,
                midas_only_stable,
                unknown_tracks,
                now,
                w * h,
                latest_midas_raw_depth,
                frame.shape,
                midas_calibrator,
            )

            # A final geometric duplicate check protects against cases where
            # the MiDaS candidate came from a neighboring zone but is still the
            # same physical object as a YOLO box.
            known_boxes = [
                tr.box for tr in tracks
                if tr.box is not None
            ]
            unknown_tracks_list = [
                tr for tr in unknown_tracks_list
                if tr.box is not None
                and not any(
                    boxes_same_object(tr.box, kb)
                    for kb in known_boxes
                )
            ]

            # ---------------------------------------------------------
            # REPLACEMENT RULE
            # ---------------------------------------------------------
            # YOLO gets the first three slots. MiDaS can occupy a slot only
            # when its confirmed obstacle is closer than the current farthest
            # YOLO obstacle. This gives the exact priority requested:
            #
            #       YOLO #1, #2, #3
            #               ↓
            #       MiDaS unknown closer?
            #          YES → replace #3
            #          NO  → keep YOLO #3
            # ---------------------------------------------------------
            yolo_tracks = sorted(
                tracks,
                key=lambda tr: tr.smoothed_dist,
            )[:MAX_TRACKS]

            confirmed_unknown_tracks = sorted(
                unknown_tracks_list,
                key=lambda tr: tr.smoothed_dist,
            )

            final_obstacles = list(yolo_tracks)

            for utr in confirmed_unknown_tracks:
                if len(final_obstacles) < MAX_TRACKS:
                    final_obstacles.append(utr)
                    continue

                farthest_yolo = max(
                    final_obstacles,
                    key=lambda tr: tr.smoothed_dist,
                )

                # MiDaS must be meaningfully closer before it is allowed to
                # displace a YOLO slot. The margin avoids rapid A/B swapping
                # when both distance estimates are almost identical.
                if utr.smoothed_dist < (
                    farthest_yolo.smoothed_dist - UNKNOWN_REPLACEMENT_MARGIN_M
                ):
                    final_obstacles.remove(farthest_yolo)
                    final_obstacles.append(utr)

            # ---------------------------------------------------------
            # FINAL PHYSICAL-OBJECT DEDUPLICATION
            # ---------------------------------------------------------
            # A tracker can still temporarily retain a MiDaS track after the
            # association step, especially when the MiDaS box changes shape.
            # Never allow two final slots to represent the same physical
            # object.  If there is a conflict, YOLO always wins.
            deduped = []
            for candidate in sorted(final_obstacles, key=lambda tr: tr.smoothed_dist):
                duplicate_idx = None
                for idx, kept in enumerate(deduped):
                    if kept.box is not None and candidate.box is not None and boxes_same_object(candidate.box, kept.box):
                        duplicate_idx = idx
                        break
                if duplicate_idx is None:
                    deduped.append(candidate)
                    continue

                kept = deduped[duplicate_idx]
                kept_is_yolo = getattr(kept, "source", "YOLO") == "YOLO"
                cand_is_yolo = getattr(candidate, "source", "YOLO") == "YOLO"

                # YOLO has absolute priority for the displayed physical box.
                # If both are MiDaS, retain the one with the larger geometry.
                if cand_is_yolo and not kept_is_yolo:
                    deduped[duplicate_idx] = candidate
                elif cand_is_yolo == kept_is_yolo and candidate.box is not None and kept.box is not None:
                    carea = max(1, (candidate.box[2]-candidate.box[0]) * (candidate.box[3]-candidate.box[1]))
                    karea = max(1, (kept.box[2]-kept.box[0]) * (kept.box[3]-kept.box[1]))
                    if carea > karea * 1.10:
                        deduped[duplicate_idx] = candidate

            final_obstacles = sorted(
                deduped,
                key=lambda tr: tr.smoothed_dist,
            )[:MAX_TRACKS]

            # ---------------------------------------------------------
            # Known-object display
            # ---------------------------------------------------------

            # ---------------------------------------------------------
            if SHOW_ALL_DETECTIONS:
                for (
                    cls_name_raw,
                    box_px,
                    confidence,
                ) in all_boxes:
                    cv2.rectangle(
                        frame,
                        box_px[:2],
                        box_px[2:],
                        (80, 80, 80),
                        1,
                    )


            # ---------------------------------------------------------
            # Unified obstacle selection from the FINAL top-3 set.
            # Selection for the GRU is based on TTC, while display priority
            # remains the nearest-three policy above.
            # ---------------------------------------------------------
            selected = (
                min(final_obstacles, key=lambda t: t.ttc())
                if final_obstacles else None
            )

            for tr in final_obstacles:
                if tr.box is None:
                    continue

                is_unknown = getattr(tr, "source", "YOLO") == "MiDaS"
                color = (0, 0, 255) if tr is selected else ((0, 255, 255) if is_unknown else (150, 150, 150))

                cv2.rectangle(frame, tr.box[:2], tr.box[2:], color, 3)

            if selected is not None:

                dist = selected.smoothed_dist
                closing_speed = selected.closing_speed()
                cls_name = selected.cls_name
                selected_is_unknown = (
                    selected is not None
                    and getattr(selected, "source", "YOLO") == "MiDaS"
                )
            else:
                dist = MAX_RANGE
                closing_speed = 0.0
                cls_name = "none"
                selected_is_unknown = False

            # ---------------------------------------------------------
            # Existing GRU / proximity path — unchanged except that `dist`
            # is now the fused/aggregated metric estimate.
            # ---------------------------------------------------------
            yolo_anchor_dist = (
                current_yolo_anchor_distance(
                    selected,
                    w * h,
                )
                if selected is not None
                else MAX_RANGE
            )

            # ---------------------------------------------------------
            # Existing GRU / proximity path — unchanged
            # ---------------------------------------------------------
            cls_idx = FEATURE_CLASSES.index(
                cls_name
            )

            feature_vec = [
                dist / MAX_RANGE,
                closing_speed,
                cls_idx / len(FEATURE_CLASSES),
                AGENT_SPEED_ASSUMED,
            ]

            feature_buffer.append(
                feature_vec
            )

            frame_counter += 1

            if (
                len(feature_buffer)
                >= GRU_SEQ_LEN
                and
                frame_counter
                % GRU_INFERENCE_EVERY_N_FRAMES
                == 0
            ):
                live_risk_target = predict_live_risk(
                    gru_model,
                    feature_buffer,
                )

            # Low-pass the neural risk display as well. The GRU itself is
            # unchanged; this only prevents one noisy input window from
            # making the displayed risk jump several buckets in one frame.
            live_risk = (
                RISK_EMA_ALPHA * live_risk_target
                + (1.0 - RISK_EMA_ALPHA) * live_risk
            )

            proximity_level, proximity_reason = (
                proximity_override(
                    selected,
                    h,
                )
            )

            final_risk, final_bucket = (
                combine_risk(
                    live_risk,
                    proximity_level,
                )
            )

            # ---------------------------------------------------------
            # MiDaS unknown-obstacle safety layer
            # ---------------------------------------------------------
            selected_unknown_candidates = []
            for tr in final_obstacles:
                if getattr(tr, "source", "YOLO") == "MiDaS":
                    selected_unknown_candidates.append({
                        "zone": "FINAL",
                        "box": (
                            tr.box[0], tr.box[1],
                            tr.box[2] - tr.box[0],
                            tr.box[3] - tr.box[1],
                        ),
                        "score": 1.0,
                        "depth_level": 1.0,
                        "confirmed": True,
                    })

            (
                unknown_risk_floor,
                unknown_level,
                unknown_zone,
                unknown_reason,
            ) = unknown_obstacle_risk(selected_unknown_candidates, h)

            if unknown_risk_floor > final_risk:
                final_risk = unknown_risk_floor
                final_bucket = risk_bucket(final_risk)

            # If the selected obstacle is the MiDaS unknown obstacle, the
            # GRU/proximity path above has ALREADY consumed its distance,
            # closing speed and class proxy. Keep the safety floor as an
            # additional guard, but do not overwrite those measurements.
            selected_source = "MiDaS UNKNOWN" if selected_is_unknown else (
                "YOLO" if selected is not None else "NONE"
            )

            # ---------------------------------------------------------
            # CLEAN THREE-PANEL DISPLAY
            # ---------------------------------------------------------
            # No telemetry text is drawn over the camera/depth/mask images.
            # Each panel gets its own dedicated information strip underneath.

            def _draw_status_panel(width, height, lines, bg=(28, 28, 28), fg=(235, 235, 235), title=None):
                strip = np.full((height, width, 3), bg, dtype=np.uint8)
                y = 25
                if title:
                    cv2.putText(
                        strip, title, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                        fg, 2, cv2.LINE_AA,
                    )
                    y += 28
                for text, color, scale, thickness in lines:
                    if y >= height - 8:
                        break
                    cv2.putText(
                        strip, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale,
                        color, thickness, cv2.LINE_AA,
                    )
                    y += 24
                return strip

            # Keep the displayed obstacle boxes, but put ALL descriptive text
            # below the corresponding image instead of over it.
            if selected is not None:
                dist = selected.smoothed_dist
                closing_speed = selected.closing_speed()
                cls_name = selected.cls_name
                selected_is_unknown = (
                    getattr(selected, "source", "YOLO") == "MiDaS"
                )
            else:
                dist = MAX_RANGE
                closing_speed = 0.0
                cls_name = "none"
                selected_is_unknown = False

            selected_source = (
                "MiDaS UNKNOWN" if selected_is_unknown
                else ("YOLO" if selected is not None else "NONE")
            )
            selected_ttc = (
                selected.ttc() if selected is not None else TTC_SAFE_VALUE
            )

            # ---------------- Camera panel ----------------
            camera_lines = [
                (
                    f"SELECTED: {selected_source} {cls_name} | distance {dist:.2f} m",
                    (0, 90, 255), 0.52, 2,
                ),
                (
                    f"Closing: {closing_speed:+.2f} m/s | TTC: {selected_ttc:.1f} s",
                    (0, 90, 255), 0.50, 2,
                ),
                (
                    f"GRU: {live_risk:.3f} | RISK: {risk_bucket(live_risk)} | buffer {len(feature_buffer)}/{GRU_SEQ_LEN}",
                    (0, 90, 255), 0.50, 2,
                ),
                (
                    f"GRU input: d={dist:.2f} m  close={closing_speed:+.2f}  class={cls_name}",
                    (235, 235, 235), 0.46, 1,
                ),
                (
                    f"Proximity: {proximity_level} | Final: {final_risk:.3f} {final_bucket}",
                    (0, 90, 255), 0.50, 2,
                ),
                (
                    f"Reason: {proximity_reason} | tracks={len(final_obstacles)}",
                    (235, 235, 235), 0.46, 1,
                ),
                (
                    f"FPS: {fps:.1f} | Distance mode: {'TRACK-REF' if selected is not None and getattr(selected, 'reference_locked', False) else 'CALIBRATING'}",
                    (235, 235, 235), 0.46, 1,
                ),
            ]

            # Unknown status is kept in the camera strip because it is a
            # navigation-system decision, while the raw mask itself remains
            # text-free.
            confirmed = [
                c for c in unknown_candidates
                if c.get("stable_confirmed", False)
            ]
            if confirmed:
                best_unknown = max(
                    confirmed,
                    key=lambda c: c.get("score", 0.0),
                )
                unknown_summary = (
                    f"Unknown obstacle: {best_unknown['zone']} "
                    f"{best_unknown.get('stable_hits', 0)}/{STABILITY_HITS_REQUIRED} "
                    f"| risk {unknown_level}"
                )
            elif unknown_candidates:
                best_unknown = unknown_candidates[0]
                unknown_summary = (
                    f"Unknown candidate: {best_unknown['zone']} "
                    f"{best_unknown.get('stable_hits', 0)}/{STABILITY_HITS_REQUIRED}"
                )
            else:
                unknown_summary = "Unknown obstacle: none"

            camera_lines.append(
                (unknown_summary, (0, 255, 255), 0.46, 1)
            )

            # ---------------- MiDaS panel ----------------
            metric_status = (
                f"Metric status: CALIBRATED ({len(midas_calibrator.samples)} samples)"
                if midas_calibrator.ready
                else f"Metric status: UNCALIBRATED ({len(midas_calibrator.samples)}/3 samples)"
            )
            yolo_anchor_text = (
                f"YOLO anchor: {yolo_anchor_dist:.2f} m | final: {dist:.2f} m"
                if selected is not None
                else "YOLO anchor: -- | final: --"
            )
            midas_corr = (
                float(getattr(selected, "last_midas_correction", 1.0))
                if selected is not None else 1.0
            )
            midas_lines = [
                ("MiDaS Small - RELATIVE DEPTH", (255, 255, 255), 0.58, 2),
                (metric_status, (0, 255, 255), 0.46, 1),
                (yolo_anchor_text, (255, 255, 0), 0.46, 1),
                (f"MiDaS correction factor: {midas_corr:.2f}", (255, 255, 0), 0.46, 1),
                ("Depth values are used as relative correction evidence.", (235, 235, 235), 0.44, 1),
                ("Final bbox geometry: fused physical obstacles only.", (235, 235, 235), 0.44, 1),
            ]

            # ---------------- Mask panel ----------------
            mask_lines = [
                ("DEPTH PROTRUSION MASK", (255, 255, 255), 0.58, 2),
                (f"Confirmed candidates: {len(confirmed)}", (0, 255, 255), 0.46, 1),
                (f"Raw candidates: {len(raw_unknown_candidates)}", (235, 235, 235), 0.46, 1),
                (f"Stable requirement: {STABILITY_HITS_REQUIRED}/{STABILITY_WINDOW} frames", (235, 235, 235), 0.46, 1),
                ("White regions = MiDaS protrusion evidence.", (235, 235, 235), 0.44, 1),
                ("Only confirmed candidates enter navigation fusion.", (235, 235, 235), 0.44, 1),
            ]

            # Draw only bounding boxes on the image itself.
            for tr in final_obstacles:
                if tr.box is None:
                    continue
                is_unknown = getattr(tr, "source", "YOLO") == "MiDaS"
                color = (
                    (0, 0, 255) if tr is selected
                    else ((0, 255, 255) if is_unknown else (150, 150, 150))
                )
                cv2.rectangle(
                    frame,
                    tr.box[:2],
                    tr.box[2:],
                    color,
                    3,
                )

            if SHOW_ALL_DETECTIONS:
                for cls_name_raw, box_px, confidence in all_boxes:
                    cv2.rectangle(
                        frame,
                        box_px[:2],
                        box_px[2:],
                        (80, 80, 80),
                        1,
                    )

            # Build depth image and mask image without text overlays.
            if have_depth_result and latest_midas_depth is not None:
                try:
                    depth_vis = midas_visual(
                        latest_midas_depth,
                        w,
                        h,
                    )

                    # Same final physical obstacles, no labels.
                    for tr in final_obstacles:
                        if tr.box is None:
                            continue
                        is_unknown = getattr(tr, "source", "YOLO") == "MiDaS"
                        color = (
                            (0, 0, 255) if tr is selected
                            else ((0, 255, 255) if is_unknown else (150, 150, 150))
                        )
                        x1, y1, x2, y2 = map(int, tr.box)
                        cv2.rectangle(
                            depth_vis,
                            (x1, y1), (x2, y2),
                            color, 3,
                        )

                    if latest_unknown_mask is not None and latest_unknown_mask.size > 0:
                        mask_vis = cv2.cvtColor(
                            latest_unknown_mask,
                            cv2.COLOR_GRAY2BGR,
                        )
                    else:
                        mask_vis = np.zeros(
                            (h, w, 3),
                            dtype=np.uint8,
                        )

                except Exception as exc:
                    print(f"MiDaS display warning: {exc}")
                    depth_vis = np.zeros_like(frame)
                    mask_vis = np.zeros_like(frame)
            else:
                depth_vis = np.zeros_like(frame)
                mask_vis = np.zeros_like(frame)

            status_h = 215
            camera_status = _draw_status_panel(
                w, status_h, camera_lines,
                bg=(22, 22, 22),
                fg=(235, 235, 235),
            )
            midas_status = _draw_status_panel(
                w, status_h, midas_lines,
                bg=(22, 22, 22),
                fg=(235, 235, 235),
            )
            mask_status = _draw_status_panel(
                w, status_h, mask_lines,
                bg=(22, 22, 22),
                fg=(235, 235, 235),
            )

            camera_panel = np.vstack((frame, camera_status))
            midas_panel = np.vstack((depth_vis, midas_status))
            mask_panel = np.vstack((mask_vis, mask_status))

            combined = np.hstack(
                (
                    camera_panel,
                    midas_panel,
                    mask_panel,
                )
            )

            safe_imshow(
                "wearable-nav + LIVE GRU + MiDaS",
                combined,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("c"):
                if selected is None:
                    print("MiDaS calibration: no selected obstacle.")
                elif latest_midas_raw_depth is None or selected.box is None:
                    print("MiDaS calibration: no raw MiDaS depth available.")
                else:
                    try:
                        raw_value = raw_midas_depth_for_box(
                            latest_midas_raw_depth,
                            selected.box,
                            frame.shape,
                        )
                        entered = input(
                            "\nTRUE distance for the selected object (metres): "
                        ).strip()
                        true_dist = float(entered)
                        if midas_calibrator.add(raw_value, true_dist):
                            print(
                                f"Captured MiDaS calibration: raw={raw_value:.5f} "
                                f"-> {true_dist:.3f}m"
                            )
                        else:
                            print("Invalid MiDaS calibration point; skipped.")
                    except ValueError:
                        print("Invalid distance; skipped.")
                    except Exception as exc:
                        print(f"MiDaS calibration warning: {exc}")

            elif key == ord("x"):
                midas_calibrator.clear()
                print("MiDaS metric calibration cleared.")

            elif key == ord("q"):
                break

    finally:
        midas_worker.stop()
        reader.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
