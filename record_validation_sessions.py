"""
Real-World Validation Session Recorder (s/e/c flow)
========================================================

WHY THIS EXISTS:
Your GRU was trained and evaluated ONLY on synthetic data -- fine for
development, but not something you can present to a panel as the whole
evaluation story. A panel can reasonably ask "how does it do on real
sensor readings?" This script answers that by recording REAL camera
sessions with REAL ground-truth risk, so you can evaluate the trained
model against data it never saw and that isn't just more synthetic data.

USES THE EXACT SAME PIPELINE AS yolo_multiobject_tracking.py:
Same CLASS_MAP, same DIST_C/DIST_EXPONENT calibration, same Track/
match_detections_to_tracks logic. This matters -- you're validating the
SAME feature computation that will run live, not a simplified stand-in.

THE s/e/c KEYBOARD FLOW:
  s -- START a new session. You'll be prompted in the terminal to type
       a short label (e.g. "dangerous1", "safe1", "ambiguous1") -- purely
       for your own organization, doesn't affect labeling logic.
  c -- mark a COLLISION / near-miss moment RIGHT NOW. Press this the
       instant you (or whoever's walking) would actually have hit the
       obstacle. You can press it more than once per session if there
       are multiple close calls.
  e -- END the current session and save it to disk.
  q -- quit the recorder entirely (auto-ends an open session first).

HOW GROUND TRUTH RISK IS COMPUTED (the important part):
For each recorded frame, if there's a 'c' marker LATER in the same
session:
    real_ttc = (time of that marker) - (this frame's timestamp)
    risk = clip(1.5 / (real_ttc + 0.3), 0, 1)
  -- same formula shape as your synthetic labels, but real_ttc here
  comes from an ACTUAL observed event, not from noisy estimated
  distance/speed. This is the strong, defensible ground truth.

For frames with no upcoming 'c' marker in that session (e.g. a "safe"
session, or after the last collision marker), ground truth falls back
to the SAME instantaneous distance/closing-speed formula your live
pipeline already computes -- consistent with how synthetic labels were
generated, just using real (noisy) sensor-derived numbers instead of
simulated ones.

SESSION SUGGESTIONS TO RECORD (do a few of each):
  - "safe": walk around, near objects, but never actually approach
    anything closing -- press 'e' only, no 'c'
  - "dangerous": walk directly and steadily toward an object until
    you'd actually collide -- press 'c' at that moment, then 'e'
  - "ambiguous": walk toward something but veer away before contact --
    press 'e' only (no collision happened)

Sessions save as sessions/session_<label>_<timestamp>.npz

pip install ultralytics opencv-python numpy --break-system-packages
"""

import time
import threading
import os
import numpy as np
import cv2
from ultralytics import YOLO

# ---- config -- MUST MATCH yolo_multiobject_tracking.py ----
SOURCE = "http://100.118.94.243:8080/video"
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
INFERENCE_IMGSZ = 320
FEATURE_CLASSES = ["none", "person", "pole", "wall", "vehicle", "curb"]
AGENT_SPEED_ASSUMED = 1.2

EMA_ALPHA = 0.25
SPEED_WINDOW = 8
SPEED_DEADBAND = 0.05
DIST_C = 0.3364
DIST_EXPONENT = 0.7769

ROTATE = False
PROCESS_WIDTH = None

MAX_TRACKS = 3
MATCH_MAX_DIST_FRAC = 0.25
TRACK_TIMEOUT_S = 1.0
TTC_SAFE_VALUE = 999.0

SESSIONS_DIR = "sessions"
RISK_FORMULA_SCALE = 1.5   # must match synthetic_dataset_generator.py's formula
RISK_FORMULA_OFFSET = 0.3


class LatestFrameReader:
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
        with self.lock:
            if self.latest_frame is None:
                return False, None
            return True, self.latest_frame.copy()

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


def bbox_area_to_distance(box_area_frac):
    box_area_frac = max(box_area_frac, 1e-4)
    est = DIST_C * (box_area_frac ** (-DIST_EXPONENT))
    return float(np.clip(est, 0.2, MAX_RANGE))


class Track:
    _next_id = 1

    def __init__(self, cls_name, centroid, dist, now):
        self.id = Track._next_id
        Track._next_id += 1
        self.cls_name = cls_name
        self.centroid = centroid
        self.box = None
        self.smoothed_dist = dist
        self.history = [(now, dist)]
        self.last_seen = now

    def update(self, cls_name, centroid, box, raw_dist, now):
        self.cls_name = cls_name
        self.centroid = centroid
        self.box = box
        self.smoothed_dist = EMA_ALPHA * raw_dist + (1 - EMA_ALPHA) * self.smoothed_dist
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
        speed = (d0 - d1) / dt
        if abs(speed) < SPEED_DEADBAND:
            return 0.0
        return speed

    def ttc(self):
        speed = self.closing_speed()
        if speed <= 0:
            return TTC_SAFE_VALUE
        return self.smoothed_dist / speed


def match_detections_to_tracks(detections, tracks, frame_width, now):
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

    for tr in unmatched_tracks:
        if now - tr.last_seen <= TRACK_TIMEOUT_S:
            updated.append(tr)

    updated.sort(key=lambda t: t.smoothed_dist)
    return updated[:MAX_TRACKS]


def instantaneous_risk(dist, closing_speed):
    """Same-shape fallback formula as synthetic labels, applied to real
    (noisy) measured distance/closing speed instead of simulated values."""
    if closing_speed <= 0:
        ttc = TTC_SAFE_VALUE
    else:
        ttc = dist / closing_speed
    risk = RISK_FORMULA_SCALE / (ttc + RISK_FORMULA_OFFSET)
    return float(np.clip(risk, 0.0, 1.0))


def compute_session_labels(frame_records, collision_times):
    """frame_records: list of dicts with keys 'timestamp', 'feature_vec',
    'dist', 'closing_speed'. collision_times: list of wall-clock times
    when 'c' was pressed. Returns array of per-frame ground-truth risk."""
    labels = np.zeros(len(frame_records), dtype=np.float32)
    collision_times_sorted = sorted(collision_times)

    for i, rec in enumerate(frame_records):
        t = rec["timestamp"]
        future_collisions = [c for c in collision_times_sorted if c >= t]
        if future_collisions:
            real_ttc = min(future_collisions) - t
            risk = RISK_FORMULA_SCALE / (real_ttc + RISK_FORMULA_OFFSET)
            labels[i] = float(np.clip(risk, 0.0, 1.0))
        else:
            labels[i] = instantaneous_risk(rec["dist"], rec["closing_speed"])

    return labels


def save_session(label, frame_records, collision_times):
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    y = compute_session_labels(frame_records, collision_times)
    X = np.array([r["feature_vec"] for r in frame_records], dtype=np.float32)

    fname = f"session_{label}_{int(time.time())}.npz"
    path = os.path.join(SESSIONS_DIR, fname)
    np.savez(path, X=X, y=y, label=label, n_collision_markers=len(collision_times))
    print(f"\n[SAVED] {path}  ({len(frame_records)} frames, "
          f"{len(collision_times)} collision marker(s), "
          f"mean risk={y.mean():.2f})\n")


def main():
    model = YOLO(MODEL)
    reader = LatestFrameReader(SOURCE)

    tracks = []
    session_active = False
    session_label = None
    frame_records = []
    collision_times = []

    print("=" * 60)
    print("VALIDATION SESSION RECORDER")
    print("  s = start session   c = mark collision   e = end session")
    print("  q = quit")
    print("=" * 60)

    while True:
        ok, frame = reader.read()
        if not ok:
            time.sleep(0.01)
            continue

        h, w = frame.shape[:2]
        now = time.time()

        results = model.predict(frame, verbose=False, imgsz=INFERENCE_IMGSZ)[0]

        detections = []
        for box in results.boxes:
            cls_name_raw = model.names[int(box.cls[0])]
            mapped = CLASS_MAP.get(cls_name_raw)
            if mapped is None:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area_frac = ((x2 - x1) * (y2 - y1)) / (w * h)
            raw_dist = bbox_area_to_distance(area_frac)
            centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
            detections.append((mapped, centroid, (int(x1), int(y1), int(x2), int(y2)), raw_dist))

        tracks = match_detections_to_tracks(detections, tracks, w, now)
        selected = min(tracks, key=lambda t: t.ttc()) if tracks else None

        if selected is not None:
            dist = selected.smoothed_dist
            closing_speed = selected.closing_speed()
            cls_name = selected.cls_name
            box = selected.box
        else:
            dist, closing_speed, cls_name, box = MAX_RANGE, 0.0, "none", None

        cls_idx = FEATURE_CLASSES.index(cls_name)
        feature_vec = [
            dist / MAX_RANGE,
            closing_speed,
            cls_idx / len(FEATURE_CLASSES),
            AGENT_SPEED_ASSUMED,
        ]

        if session_active:
            frame_records.append({
                "timestamp": now,
                "feature_vec": feature_vec,
                "dist": dist,
                "closing_speed": closing_speed,
            })

        # ---- display ----
        display = frame.copy()
        if box is not None:
            cv2.rectangle(display, box[:2], box[2:], (0, 0, 255), 3)

        status = f"REC [{session_label}] frames={len(frame_records)} markers={len(collision_times)}" \
            if session_active else "IDLE -- press 's' to start a session"
        color = (0, 0, 255) if session_active else (200, 200, 200)
        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(display, f"{cls_name} dist~{dist:.2f}m closing={closing_speed:+.2f}m/s",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        cv2.imshow("Validation Recorder", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s") and not session_active:
            session_label = input("\nSession label (e.g. dangerous1, safe1): ").strip() or "session"
            session_active = True
            frame_records = []
            collision_times = []
            print(f"[RECORDING] session '{session_label}' started.")

        elif key == ord("c") and session_active:
            collision_times.append(now)
            print(f"[MARKER] collision marked at t={now:.2f} "
                  f"({len(collision_times)} marker(s) so far)")

        elif key == ord("e") and session_active:
            save_session(session_label, frame_records, collision_times)
            session_active = False

        elif key == ord("q"):
            if session_active:
                print("\n[AUTO-SAVE] ending open session before quitting...")
                save_session(session_label, frame_records, collision_times)
            break

    reader.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
