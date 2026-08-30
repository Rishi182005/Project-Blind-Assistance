"""
CALIBRATION_K Multi-Point Calibration Tool (v2 -- fixed frame lag)
=====================================================================

FIXES FROM PREVIOUS VERSION:
1. SOURCE was pointing at a stale IP (192.168.29.115) that didn't match
   the phone stream actually in use (192.168.29.167) -- update if yours
   differs.
2. The previous version read frames directly via cv2.VideoCapture, which
   buffers frames over a WiFi stream -- by the time you pressed SPACE,
   the frame actually captured could be SECONDS old (same lag issue
   fixed in yolo_multiobject_tracking.py). If you moved or the delay
   was large, the area_frac captured didn't match the true distance you
   entered, which silently corrupts the whole calibration -- almost
   certainly why distances were coming out as 0.3-0.4m regardless of
   how far you actually stood.

This version uses a background thread that always keeps only the
FRESHEST frame, so SPACE always captures what's in front of the camera
RIGHT NOW, not a delayed frame from moments ago.

HOW TO USE: same as before --
1. Run this script: python calibrate_distance.py
2. Stand a person at a KNOWN distance (e.g. exactly 100cm from camera)
3. Wait a beat after moving/settling before pressing SPACE (even with
   the fix, give it half a second to catch up) -- watch the on-screen
   area_frac number stabilize before capturing
4. Press SPACE, type the true distance in cm when prompted
5. Repeat at 3-4 different distances (e.g. 50cm, 100cm, 200cm, 300cm)
6. Press 'q' when done -- prints the best-fit CALIBRATION_K

pip install ultralytics opencv-python numpy --break-system-packages
"""

import threading
import cv2
import numpy as np
from ultralytics import YOLO

# ---------- Config -- MUST MATCH yolo_multiobject_tracking.py ----------
SOURCE = "http://192.168.29.115:8080/video"   # sync this with your main script's SOURCE
MODEL = "yolov8n.pt"
TARGET_CLASS = "person"


class LatestFrameReader:
    """Always keeps only the most recent frame -- prevents SPACE from
    capturing a stale, buffered frame over a laggy network stream."""

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


def get_largest_person_box_area_frac(frame, model):
    """Returns the box_area_frac of the largest 'person' detection in the
    frame, or None if no person detected."""
    results = model.predict(frame, verbose=False)[0]
    h, w = frame.shape[:2]

    best_area_frac = None
    best_box = None
    for box in results.boxes:
        cls_name = model.names[int(box.cls[0])]
        if cls_name != TARGET_CLASS:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        area_frac = ((x2 - x1) * (y2 - y1)) / (w * h)
        if best_area_frac is None or area_frac > best_area_frac:
            best_area_frac = area_frac
            best_box = (int(x1), int(y1), int(x2), int(y2))

    return best_area_frac, best_box


def main():
    model = YOLO(MODEL)
    reader = LatestFrameReader(SOURCE)

    calibration_points = []  # list of (true_distance_m, area_frac)

    print("=" * 60)
    print("CALIBRATION MODE (v2 -- lag-fixed)")
    print("Stand a person at a KNOWN distance from the camera.")
    print("Wait for area_frac to stabilize on screen, THEN press SPACE.")
    print("Press 'q' when you've captured 3-4 points at different distances.")
    print("=" * 60)

    while True:
        ok, frame = reader.read()
        if not ok:
            continue

        area_frac, box = get_largest_person_box_area_frac(frame, model)

        display = frame.copy()
        if box is not None:
            cv2.rectangle(display, box[:2], box[2:], (0, 255, 0), 2)
            cv2.putText(display, f"area_frac={area_frac:.4f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, "No person detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(display, f"Points captured: {len(calibration_points)}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(display, "SPACE=capture  q=finish", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Calibration", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" ") and area_frac is not None:
            true_dist_cm = input("\nEnter the TRUE distance in cm right now: ")
            try:
                true_dist_m = float(true_dist_cm) / 100.0
                calibration_points.append((true_dist_m, area_frac))
                print(f"Captured: {true_dist_m:.2f}m -> area_frac={area_frac:.4f}")
            except ValueError:
                print("Invalid number, skipped.")

        elif key == ord("q"):
            break

    reader.release()
    cv2.destroyAllWindows()

    if len(calibration_points) < 2:
        print("\nNeed at least 2 points to calibrate. Run again and capture more.")
        return

    # Fit dist = C * area_frac ** (-p) via log-log linear regression.
    # We do NOT assume p=0.5 (the naive "area shrinks with distance^2"
    # geometric model) -- lens distortion, YOLO box tightness changing
    # with distance, and clipping at close range can all pull the real
    # exponent away from 0.5. Fitting both C and p from the data is far
    # more accurate than forcing p=0.5 and averaging implied K, which
    # is what caused close-range readings to be badly wrong before.
    dists = np.array([d for d, _ in calibration_points])
    areas = np.array([a for _, a in calibration_points])
    log_d = np.log(dists)
    log_a = np.log(areas)

    # slope, intercept of log_d = intercept + slope * log_a
    slope, intercept = np.polyfit(log_a, log_d, 1)
    exponent = -slope       # dist = C * area_frac ** (-exponent)
    C = float(np.exp(intercept))

    print(f"\n{'='*60}")
    print(f"CALIBRATION RESULTS ({len(calibration_points)} points)")
    print(f"{'='*60}")
    print(f"{'True dist (m)':>15}{'area_frac':>15}{'Est. (power fit)':>20}")
    for true_dist, area_frac in calibration_points:
        est = C * (area_frac ** (-exponent))
        err_pct = 100 * (est - true_dist) / true_dist
        print(f"{true_dist:>15.2f}{area_frac:>15.4f}{est:>17.2f}  ({err_pct:+.0f}%)")

    print(f"\nBest-fit model: dist = {C:.4f} * area_frac ** (-{exponent:.4f})")
    print(f"\nIf any point's error is still >15-20%, capture more points in that")
    print("range and re-run, or check for a measurement/entry mistake.")
    print(f"\nUpdate yolo_multiobject_tracking.py's bbox_area_to_distance():")
    print(f"  replace the CALIBRATION_K / sqrt(area_frac) formula with")
    print(f"  DIST_C = {C:.4f}")
    print(f"  DIST_EXPONENT = {exponent:.4f}")
    print(f"  est = DIST_C * (area_frac ** -DIST_EXPONENT)")


if __name__ == "__main__":
    main()