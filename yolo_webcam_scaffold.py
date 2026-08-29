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

Still 100% software -- no new hardware required. Distance is still the
same bbox-size heuristic as before (DIST_C / DIST_EXPONENT, fitted via
calibrate_distance.py), swap for real HC-SR04 readings later.

pip install ultralytics opencv-python --break-system-packages
"""

import time
import threading
from collections import deque
import numpy as np
import cv2
from ultralytics import YOLO
from tensorflow import keras


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
MODEL = "yolov8n.pt"
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

INFERENCE_IMGSZ = 256              # resolution YOLO actually runs inference at --
                                   # lower = faster, less accurate on small objects.
                                   # Ultralytics rescales detected boxes back to the
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

# ---- immediate proximity safety layer ----
# These are deliberately NOT fed back into the GRU. The GRU remains the
# trained 4-feature model; this deterministic layer handles objects that
# are already extremely close even when closing_speed is near zero.
PROXIMITY_CRITICAL_M = 0.35
PROXIMITY_HIGH_M = 0.45

# Bottom-of-frame warning: an object extending into this fraction of the
# image height is likely very close to the camera/user.
BOTTOM_ZONE_START_FRAC = 0.78
BOTTOM_ZONE_CRITICAL_FRAC = 0.95


EMA_ALPHA = 0.25
SPEED_WINDOW = 8                 # frames of history kept per tracked object
SPEED_DEADBAND = 0.05
DIST_C = 0.3364                  # from calibrate_distance.py power-law fit --
                                  # RE-CALIBRATE if camera/mount/resolution changes
DIST_EXPONENT = 0.7769           # fitted exponent -- do NOT assume 0.5 (naive
                                  # geometric model); lens distortion and YOLO box
                                  # behavior at close range pull this away from 0.5
                                  # in practice, confirmed by calibration data

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
TTC_SAFE_VALUE = 999.0           # TTC assigned when an object isn't closing
                                  # in (moving away or stationary) -- effectively
                                  # "infinite time," so it never wins the
                                  # lowest-TTC selection over a real threat


def bbox_area_to_distance(box_area_frac):
    box_area_frac = max(box_area_frac, 1e-4)
    est = DIST_C * (box_area_frac ** (-DIST_EXPONENT))
    return float(np.clip(est, 0.2, MAX_RANGE))


def resize_fixed(frame, width):
    if width is None:
        return frame
    h, w = frame.shape[:2]
    if w == width:
        return frame
    scale = width / w
    return cv2.resize(frame, (width, int(h * scale)))


class Track:
    """One tracked object's identity + rolling history across frames."""
    _next_id = 1

    def __init__(self, cls_name, centroid, dist, now):
        self.id = Track._next_id
        Track._next_id += 1
        self.cls_name = cls_name
        self.centroid = centroid
        self.box = None
        self.smoothed_dist = dist
        self.history = [(now, dist)]   # (timestamp, smoothed_dist), capped to SPEED_WINDOW
        self.last_seen = now

    def update(self, cls_name, centroid, box, raw_dist, now):
        self.cls_name = cls_name       # allow class to be re-confirmed each frame
        self.centroid = centroid
        self.box = box
        self.smoothed_dist = (
            EMA_ALPHA * raw_dist + (1 - EMA_ALPHA) * self.smoothed_dist
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
            return 0.0
        return speed

    def ttc(self):
        """Time-to-collision estimate. Lower = more urgent.
        Objects not closing in get a large 'safe' value so they never
        outrank a genuine approaching threat."""
        speed = self.closing_speed()
        if speed <= 0:
            return TTC_SAFE_VALUE
        return self.smoothed_dist / speed


def match_detections_to_tracks(detections, tracks, frame_width, now):
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
            best_track.update(cls_name, centroid, box, raw_dist, now)
            unmatched_tracks.remove(best_track)
            updated.append(best_track)
        else:
            new_track = Track(cls_name, centroid, raw_dist, now)
            new_track.box = box
            updated.append(new_track)

    # keep still-alive-but-unmatched tracks too (object briefly occluded)
    for tr in unmatched_tracks:
        if now - tr.last_seen <= TRACK_TIMEOUT_S:
            updated.append(tr)

    # cap total tracks -- keep the ones with lowest current distance
    updated.sort(key=lambda t: t.smoothed_dist)
    return updated[:MAX_TRACKS]


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

def main():
    model = YOLO(MODEL)
    gru_model = load_gru_model()
    reader = LatestFrameReader(SOURCE)

    tracks = []
    feature_buffer = deque(maxlen=GRU_SEQ_LEN)
    live_risk = GRU_WARMUP_RISK
    frame_counter = 0

    fps = 0.0
    last_time = time.perf_counter()
    FPS_SMOOTHING = 0.15

    print("Press 'q' to quit.")
    while True:
        ok, frame = reader.read()
        if not ok:
            time.sleep(0.01)   # reader thread hasn't gotten a frame yet -- wait briefly
            continue

        if ROTATE:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        frame = resize_fixed(frame, PROCESS_WIDTH)
        h, w = frame.shape[:2]

        now_time = time.perf_counter()
        dt = now_time - last_time
        last_time = now_time

        if dt > 0:
            current_fps = 1.0 / dt
            fps = (
                FPS_SMOOTHING * current_fps
                + (1.0 - FPS_SMOOTHING) * fps
            ) if fps > 0 else current_fps

        results = model.predict(frame, verbose=False, imgsz=INFERENCE_IMGSZ)[0]
        now = time.time()

        detections = []
        all_boxes_for_display = []   # (cls_name_raw, box) -- includes unmapped classes
        for box in results.boxes:
            cls_name_raw = model.names[int(box.cls[0])]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            box_px = (int(x1), int(y1), int(x2), int(y2))
            all_boxes_for_display.append((cls_name_raw, box_px))

            mapped = CLASS_MAP.get(cls_name_raw)
            if mapped is None:
                continue
            area_frac = ((x2 - x1) * (y2 - y1)) / (w * h)
            raw_dist = bbox_area_to_distance(area_frac)
            centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
            detections.append((mapped, centroid, box_px, raw_dist))

        tracks = match_detections_to_tracks(detections, tracks, w, now)

        # draw ALL raw YOLO detections dim (debug visibility), tracked ones on top
        if SHOW_ALL_DETECTIONS:
            for cls_name_raw, box_px in all_boxes_for_display:
                cv2.rectangle(frame, box_px[:2], box_px[2:], (80, 80, 80), 1)
                cv2.putText(frame, cls_name_raw, (box_px[0], max(box_px[1] - 4, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)

        # draw all tracked objects (gray), highlight the selected one (red)
        selected = None
        if tracks:
            selected = min(tracks, key=lambda t: t.ttc())

        for tr in tracks:
            if tr.box is None:
                continue
            color = (0, 0, 255) if tr is selected else (150, 150, 150)
            cv2.rectangle(frame, tr.box[:2], tr.box[2:], color, 3)
            label = f"#{tr.id} {tr.cls_name} d={tr.smoothed_dist:.1f}m ttc={tr.ttc():.1f}s"
            cv2.putText(frame, label, (tr.box[0], max(tr.box[1] - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if selected is not None:
            dist = selected.smoothed_dist
            closing_speed = selected.closing_speed()
            cls_name = selected.cls_name
        else:
            dist, closing_speed, cls_name = MAX_RANGE, 0.0, "none"

        cls_idx = FEATURE_CLASSES.index(cls_name)
        feature_vec = [
            dist / MAX_RANGE,
            closing_speed,
            cls_idx / len(FEATURE_CLASSES),
            AGENT_SPEED_ASSUMED,
        ]

        # Build the same 30-timestep feature sequence used during training.
        # The GRU remains "warming up" until 30 live feature vectors exist.
        feature_buffer.append(feature_vec)
        frame_counter += 1

        if (
            len(feature_buffer) >= GRU_SEQ_LEN
            and frame_counter % GRU_INFERENCE_EVERY_N_FRAMES == 0
        ):
            live_risk = predict_live_risk(gru_model, feature_buffer)

        # Immediate proximity is an independent safety layer. It can
        # override a low/medium GRU result when the object is already
        # extremely close or reaches the bottom of the camera view.
        proximity_level, proximity_reason = proximity_override(selected, h)
        final_risk, final_bucket = combine_risk(
            live_risk,
            proximity_level,
        )

        # ---- live diagnostic overlay ----
        cv2.putText(
            frame,
            f"SELECTED: {cls_name}  dist={dist:.2f}m  "
            f"closing={closing_speed:+.2f}m/s",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
        )

        cv2.putText(
            frame,
            f"GRU: {live_risk:.3f}  GRU RISK: {risk_bucket(live_risk)}  "
            f"buffer={len(feature_buffer)}/{GRU_SEQ_LEN}",
            (10, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 0, 255),
            2,
        )

        cv2.putText(
            frame,
            f"PROXIMITY: {proximity_level}  |  FINAL: {final_risk:.3f} "
            f"{final_bucket}",
            (10, 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 0, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Reason: {proximity_reason}  |  tracks={len(tracks)}",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 255),
            1,
        )

        # Visualize the bottom-of-frame proximity zones for tuning.
        y_warn = int(h * BOTTOM_ZONE_START_FRAC)
        y_critical = int(h * BOTTOM_ZONE_CRITICAL_FRAC)
        cv2.line(frame, (0, y_warn), (w, y_warn), (120, 120, 120), 1)
        cv2.line(frame, (0, y_critical), (w, y_critical), (120, 120, 120), 2)

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )
        cv2.imshow("wearable-nav detection + LIVE GRU + proximity", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    reader.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()