"""
Remove Collision Markings from .npz Session Files
===================================================

WHAT THIS DOES:
Some of your ambiguous sessions have collision markings ('c' at closest
approach), which causes the ground truth to use a collision-formula-driven
peak in risk leading up to that timestamp. This script removes that by
recomputing the ground truth using ONLY the instantaneous formula
(same as how "safe" sessions were labeled) for all timesteps.

Result: all ambiguous sessions will have consistent, instantaneous-formula
ground truth regardless of whether they had c-markers originally.

BEFORE: session might have risk=[0.1, 0.2, 0.5, 0.8, 0.9, 0.1] (sharp peak from c-marker)
AFTER: session will have risk=[0.1, 0.15, 0.25, 0.35, 0.32, 0.1] (smooth curve from instantaneous)

pip install numpy --break-system-packages
"""

import os
import glob
import numpy as np

# ---- config -- must match record_validation_sessions.py ----
SESSIONS_DIR = "og_sessions"
MAX_RANGE = 4.0
RISK_FORMULA_SCALE = 1.5
RISK_FORMULA_OFFSET = 0.3
TTC_SAFE_VALUE = 999.0


def instantaneous_risk(dist, closing_speed):
    """Compute risk from current distance and closing speed only.
    No knowledge of future collision markers."""
    if closing_speed <= 0:
        ttc = TTC_SAFE_VALUE
    else:
        ttc = dist / closing_speed
    risk = RISK_FORMULA_SCALE / (ttc + RISK_FORMULA_OFFSET)
    return float(np.clip(risk, 0.0, 1.0))


def main():
    paths = sorted(glob.glob(os.path.join(SESSIONS_DIR, "session_*.npz")))
    if not paths:
        print(f"No sessions found in '{SESSIONS_DIR}/'")
        return

    cleaned_count = 0

    for path in paths:
        data = np.load(path, allow_pickle=True)
        X = data["X"]
        y = data["y"]
        label = str(data["label"])
        n_collision_markers = int(data["n_collision_markers"])

        if n_collision_markers == 0:
            print(f"[SKIP] {os.path.basename(path)}  (no collision markers)")
            continue

        # Recompute y using only instantaneous formula
        # X[:,0] is normalized distance, X[:,1] is closing_speed
        y_new = np.zeros_like(y)
        for t in range(len(X)):
            dist_norm = X[t, 0]
            dist = dist_norm * MAX_RANGE  # denormalize
            closing_speed = X[t, 1]
            y_new[t] = instantaneous_risk(dist, closing_speed)

        # Compare before/after for logging
        old_max = y.max()
        new_max = y_new.max()
        old_mean = y.mean()
        new_mean = y_new.mean()

        # Save cleaned version back to disk
        np.savez(
            path,
            X=X,
            y=y_new,
            label=label,
            n_collision_markers=0,  # mark as "no markers" now
        )

        print(f"[CLEANED] {os.path.basename(path)}")
        print(f"  was: mean={old_mean:.3f}, max={old_max:.3f}  "
              f"(had {n_collision_markers} marker(s))")
        print(f"  now: mean={new_mean:.3f}, max={new_max:.3f}  "
              f"(instantaneous-only)")
        cleaned_count += 1

    print(f"\n{'='*60}")
    print(f"Cleaned {cleaned_count} session(s) -- collision markers removed.")
    print("Re-run train_on_recorded_sessions.py to retrain with the new labels.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()