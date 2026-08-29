"""
Final Evaluation of the GRU on the LOCKED REAL TEST SESSIONS
=============================================================

This script is intentionally separate from training.

It loads:
    1. risk_gru_model_final.keras
    2. session_split.json

and evaluates ONLY the sessions recorded in the manifest's "test" list.

The test sessions were never used for:
    - gradient updates
    - checkpoint selection
    - early stopping
    - hyperparameter selection

Therefore this is the number that should be used as the final
held-out real-world result.

IMPORTANT:
Do not edit session_split.json after training and before final evaluation.
If you record additional sessions later, create a NEW experiment/split
rather than silently changing the test set.

pip install tensorflow numpy --break-system-packages
"""

import json
import os
import numpy as np
from tensorflow import keras

MODEL_FILE = "risk_gru_model_final.keras"
SPLIT_FILE = "session_split.json"

SEQ_LEN = 30
STRIDE = 3

LOW_MED_BOUNDARY = 0.33
MED_HIGH_BOUNDARY = 0.66


def load_manifest(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found. Run train_on_recorded_sessions.py first."
        )

    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    required = {"train", "validation", "test"}
    missing = required - set(manifest)
    if missing:
        raise ValueError(
            f"Split manifest is missing keys: {sorted(missing)}"
        )

    return manifest


def load_test_sessions(test_paths):
    sessions = []

    for p in test_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"TEST session listed in manifest no longer exists:\n{p}"
            )

        data = np.load(p, allow_pickle=True)

        sessions.append({
            "path": p,
            "X": data["X"],
            "y": data["y"],
            "label": str(data["label"]),
        })

    return sessions


def build_windows(sessions, seq_len, stride):
    X_windows, y_windows, source_paths = [], [], []

    for sess in sessions:
        n = sess["X"].shape[0]

        if n < seq_len:
            print(
                f"[SKIP] {sess['path']} has only {n} frames "
                f"(< {seq_len}), so it cannot form a test window."
            )
            continue

        for start in range(0, n - seq_len + 1, stride):
            end = start + seq_len
            X_windows.append(sess["X"][start:end])
            y_windows.append(sess["y"][start:end])
            source_paths.append(sess["path"])

    if not X_windows:
        return None, None, []

    return (
        np.array(X_windows, dtype=np.float32),
        np.array(y_windows, dtype=np.float32),
        source_paths,
    )


def to_bucket(risk_array):
    buckets = np.zeros_like(risk_array, dtype=int)
    buckets[risk_array >= LOW_MED_BOUNDARY] = 1
    buckets[risk_array >= MED_HIGH_BOUNDARY] = 2
    return buckets


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test, verbose=0).squeeze(-1)

    mse = float(np.mean((y_pred - y_test) ** 2))

    true_buckets = to_bucket(y_test)
    pred_buckets = to_bucket(y_pred)

    overall_accuracy = float(
        (true_buckets == pred_buckets).mean() * 100
    )

    bucket_names = ["low", "medium", "high"]
    per_bucket = {}

    for b, name in enumerate(bucket_names):
        mask = true_buckets == b
        if mask.sum() == 0:
            per_bucket[name] = None
        else:
            per_bucket[name] = float(
                (pred_buckets[mask] == b).mean() * 100
            )

    valid = [v for v in per_bucket.values() if v is not None]
    macro_accuracy = float(np.mean(valid)) if valid else 0.0

    print("\n" + "=" * 65)
    print("FINAL HELD-OUT REAL TEST RESULTS")
    print("=" * 65)
    print(f"MSE: {mse:.4f}")
    print(f"Overall bucketed accuracy: {overall_accuracy:.2f}%")

    for name in bucket_names:
        value = per_bucket[name]
        if value is None:
            print(f"  {name:8s}: no instances")
        else:
            b = bucket_names.index(name)
            n = int((true_buckets == b).sum())
            print(f"  {name:8s}: {value:.2f}% (n={n})")

    print(f"Macro accuracy: {macro_accuracy:.2f}%")

    print(
        f"Prediction spread: min={y_pred.min():.3f} "
        f"max={y_pred.max():.3f} std={y_pred.std():.4f}"
    )

    print("\nConfusion matrix (rows=true, columns=predicted):")
    print(f"{'':>12s}{'low':>10s}{'medium':>10s}{'high':>10s}")

    for i, row_name in enumerate(bucket_names):
        row = [
            int(((true_buckets == i) & (pred_buckets == j)).sum())
            for j in range(3)
        ]
        print(
            f"{row_name:>12s}"
            f"{row[0]:>10d}"
            f"{row[1]:>10d}"
            f"{row[2]:>10d}"
        )

    return overall_accuracy, mse, macro_accuracy


def print_session_diagnostics(sessions):
    print("\nLOCKED TEST SESSIONS:")
    for sess in sessions:
        y = sess["y"]

        low = (y < LOW_MED_BOUNDARY).mean() * 100
        med = (
            (y >= LOW_MED_BOUNDARY) &
            (y < MED_HIGH_BOUNDARY)
        ).mean() * 100
        high = (y >= MED_HIGH_BOUNDARY).mean() * 100

        print(
            f"  {os.path.basename(sess['path']):42s} "
            f"label={sess['label']:12s} "
            f"frames={len(y):5d} "
            f"low={low:5.1f}% "
            f"med={med:5.1f}% "
            f"high={high:5.1f}%"
        )


def main():
    manifest = load_manifest(SPLIT_FILE)

    test_paths = manifest["test"]

    print(
        f"Loaded split manifest: {SPLIT_FILE}\n"
        f"Train sessions in manifest: {len(manifest['train'])}\n"
        f"Validation sessions in manifest: {len(manifest['validation'])}\n"
        f"FINAL TEST sessions in manifest: {len(test_paths)}"
    )

    if not test_paths:
        raise ValueError("The manifest contains no test sessions.")

    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"'{MODEL_FILE}' not found. Run train_on_recorded_sessions.py first."
        )

    test_sessions = load_test_sessions(test_paths)
    print_session_diagnostics(test_sessions)

    X_test, y_test, source_paths = build_windows(
        test_sessions,
        SEQ_LEN,
        STRIDE,
    )

    if X_test is None:
        print(
            f"\nNo test windows could be built. "
            f"Each test session needs at least {SEQ_LEN} frames."
        )
        return

    print(
        f"\nBuilt {X_test.shape[0]} final test windows "
        f"covering {X_test.shape[0] * SEQ_LEN} window-timesteps."
    )

    # Sanity check: every evaluation window must come from the manifest test set.
    manifest_test_set = {os.path.abspath(p) for p in test_paths}
    used_set = {os.path.abspath(p) for p in source_paths}

    unexpected = used_set - manifest_test_set
    if unexpected:
        raise RuntimeError(
            "SAFETY CHECK FAILED: evaluation included a session that is not "
            f"in the locked test set: {sorted(unexpected)}"
        )

    model = keras.models.load_model(MODEL_FILE)

    evaluate(model, X_test, y_test)

    print("\n" + "=" * 65)
    print("WHAT TO REPORT")
    print("=" * 65)
    print(
        f"Trained/fine-tuned using {len(manifest['train'])} real sessions, "
        f"selected the model checkpoint using {len(manifest['validation'])} "
        f"separate validation sessions, and evaluated once on "
        f"{len(manifest['test'])} completely held-out real test sessions."
    )
    print(
        "\nDo NOT describe the validation score as the final test score."
    )
    print(
        "The numbers above are the FINAL HELD-OUT TEST RESULTS."
    )


if __name__ == "__main__":
    main()
