"""
Train/Fine-tune GRU on REAL Recorded Sessions
================================================

IMPORTANT METHODOLOGY
---------------------
This version uses a proper three-way SPLIT BY SESSION:

    TRAIN      -> used to update GRU weights
    VALIDATION -> used to select the best epoch/checkpoint
    TEST       -> NEVER used during training or checkpoint selection

Windows are created only AFTER the session split, so overlapping windows
from the same recording can never appear on different sides of the split.

For the current 35 sessions, the default stratified split is approximately:
    24 train sessions
     5 validation sessions
     6 final test sessions

The exact session filenames are saved to SPLIT_FILE so the final test set
cannot silently change between runs.

Recommended workflow:
    1. Run this script to fine-tune the synthetic-pretrained GRU.
    2. This script selects the best epoch using VALIDATION macro accuracy.
    3. Run evaluate_on_recorded_sessions.py.
       That script evaluates ONLY the untouched TEST sessions.

The final test result should be reported as the unbiased real-world result.

pip install tensorflow numpy --break-system-packages
"""

import glob
import json
import os
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------- Config ----------
SESSIONS_DIR = "sessions ()"
PRETRAINED_MODEL_FILE = "risk_gru_model.keras"
OUTPUT_MODEL_FILE = "risk_gru_model_final.keras"
SPLIT_FILE = "session_split.json"

TRAIN_MODE = "finetune"       # "finetune" or "scratch"
FINETUNE_LR = 5e-4
SCRATCH_LR = 1e-3

SEQ_LEN = 30
N_FEATURES = 4
STRIDE = 3

# Session-level proportions. With the current 10/15/10 category counts,
# this produces 24 train / 5 validation / 6 test sessions.
TRAIN_FRACTION = 25 / 35
VAL_FRACTION = 5 / 35
TEST_FRACTION = 5 / 35

RANDOM_SEED = 42
EPOCHS = 80
CHECK_EVERY = 5
BATCH_SIZE = 16

LOW_MED_BOUNDARY = 0.33
MED_HIGH_BOUNDARY = 0.66

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def load_sessions(sessions_dir):
    paths = sorted(glob.glob(os.path.join(sessions_dir, "session_*.npz")))
    sessions = []

    for p in paths:
        data = np.load(p, allow_pickle=True)
        sessions.append({
            "path": p,
            "X": data["X"],
            "y": data["y"],
            "label": str(data["label"]),
        })

    return sessions


def session_category(label):
    """ambiguous2 -> ambiguous, dangerous25 -> dangerous, safe10 -> safe."""
    stripped = re.sub(r"\d+$", "", label)
    return stripped if stripped else label


def _allocate_counts(n, category, train_fraction, val_fraction, test_fraction):
    """
    Allocate train/validation/test sessions within one risk category.

    For the CURRENT 35-session collection:
        safe       (10) -> 7 train / 1 validation / 2 test
        ambiguous  (10) -> 7 train / 1 validation / 2 test
        dangerous  (15) -> 11 train / 3 validation / 1 test

    This gives an exact overall:
        25 train / 5 validation / 5 test

    We intentionally keep at least one session from every category in the
    final test set. The larger dangerous pool supplies more validation
    sessions because dangerous behavior is the most safety-critical category.
    """
    if n < 3:
        raise ValueError(
            f"Category '{category}' has only {n} session(s). "
            "Need at least 3 sessions per category."
        )

    # Exact allocation for the current dataset.
    # This avoids rounding accidentally producing only 4 final test sessions.
    if n == 10:
        return 7, 1, 2

    if n == 15:
        return 11, 3, 1

    # Generic fallback for future datasets.
    raw = np.array(
        [n * train_fraction, n * val_fraction, n * test_fraction],
        dtype=float,
    )
    counts = np.floor(raw).astype(int)

    counts[1] = max(1, counts[1])
    counts[2] = max(1, counts[2])

    while counts.sum() < n:
        remainders = raw - counts
        idx = int(np.argmax(remainders))
        counts[idx] += 1

    while counts.sum() > n:
        candidates = [
            i for i in range(3)
            if counts[i] > 1
        ]
        if not candidates:
            raise ValueError("Could not construct a valid 3-way split.")
        idx = max(candidates, key=lambda i: counts[i] - raw[i])
        counts[idx] -= 1

    return tuple(int(x) for x in counts)


def split_sessions_three_way(
    sessions,
    train_fraction=TRAIN_FRACTION,
    val_fraction=VAL_FRACTION,
    test_fraction=TEST_FRACTION,
    seed=RANDOM_SEED,
):
    """
    Stratified SESSION-level train/validation/test split.

    No window is created until AFTER this split.
    """
    if not np.isclose(train_fraction + val_fraction + test_fraction, 1.0):
        raise ValueError("TRAIN_FRACTION + VAL_FRACTION + TEST_FRACTION must equal 1.")

    rng = np.random.RandomState(seed)

    by_category = {}
    for sess in sessions:
        cat = session_category(sess["label"])
        by_category.setdefault(cat, []).append(sess)

    train_sessions, val_sessions, test_sessions = [], [], []

    for cat in sorted(by_category):
        cat_sessions = by_category[cat].copy()
        rng.shuffle(cat_sessions)

        n_train, n_val, n_test = _allocate_counts(
            len(cat_sessions),
            cat,
            train_fraction,
            val_fraction,
            test_fraction,
        )

        train_part = cat_sessions[:n_train]
        val_part = cat_sessions[n_train:n_train + n_val]
        test_part = cat_sessions[n_train + n_val:n_train + n_val + n_test]

        train_sessions.extend(train_part)
        val_sessions.extend(val_part)
        test_sessions.extend(test_part)

        print(
            f"  category '{cat}': {len(cat_sessions)} session(s) -> "
            f"{len(train_part)} train, {len(val_part)} validation, "
            f"{len(test_part)} test"
        )

    # Sort for reproducible display/output.
    train_sessions.sort(key=lambda s: s["path"])
    val_sessions.sort(key=lambda s: s["path"])
    test_sessions.sort(key=lambda s: s["path"])

    return train_sessions, val_sessions, test_sessions


def save_split_manifest(train_sessions, val_sessions, test_sessions):
    manifest = {
        "seed": RANDOM_SEED,
        "train_fraction": TRAIN_FRACTION,
        "validation_fraction": VAL_FRACTION,
        "test_fraction": TEST_FRACTION,
        "train": [s["path"] for s in train_sessions],
        "validation": [s["path"] for s in val_sessions],
        "test": [s["path"] for s in test_sessions],
    }

    with open(SPLIT_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSaved immutable session split manifest to: {SPLIT_FILE}")


def build_windows(sessions, seq_len, stride):
    X_windows, y_windows = [], []
    skipped = 0

    for sess in sessions:
        n = sess["X"].shape[0]

        if n < seq_len:
            skipped += 1
            print(
                f"[SKIP] {sess['path']} has only {n} frames "
                f"(< {seq_len}), not enough for one window."
            )
            continue

        for start in range(0, n - seq_len + 1, stride):
            end = start + seq_len
            X_windows.append(sess["X"][start:end])
            y_windows.append(sess["y"][start:end])

    if skipped:
        print(
            f"[INFO] {skipped} session(s) had fewer than {seq_len} frames "
            "and produced no windows."
        )

    if not X_windows:
        return None, None

    return (
        np.array(X_windows, dtype=np.float32),
        np.array(y_windows, dtype=np.float32),
    )


def to_bucket(risk_array, low_med=LOW_MED_BOUNDARY, med_high=MED_HIGH_BOUNDARY):
    buckets = np.zeros_like(risk_array, dtype=int)
    buckets[risk_array >= low_med] = 1
    buckets[risk_array >= med_high] = 2
    return buckets


def compute_timestep_sample_weights(
    y_windows,
    low_med=LOW_MED_BOUNDARY,
    med_high=MED_HIGH_BOUNDARY,
):
    """
    Compute a weight for EVERY timestep, rather than one weight per 30-frame
    window.

    This is better suited to temporal risk prediction because a sequence can
    contain a transition such as LOW -> MEDIUM -> HIGH. The dangerous
    timesteps should receive their own higher loss weight.

    We deliberately derive these weights ONLY from y_train.
    """
    y = np.asarray(y_windows, dtype=np.float32)

    buckets = np.zeros_like(y, dtype=np.int32)
    buckets[y >= low_med] = 1
    buckets[y >= med_high] = 2

    counts = np.array(
        [(buckets == b).sum() for b in range(3)],
        dtype=np.float64,
    )

    counts = np.clip(counts, 1.0, None)

    # Inverse-frequency weights, normalized so the average training
    # timestep has approximately weight 1.
    raw = counts.sum() / (3.0 * counts)
    raw = raw / np.average(raw, weights=counts)

    # Keep the weighting moderate. We want to improve representation of
    # medium/high risk without letting a few rare timesteps dominate.
    raw = np.clip(raw, 0.75, 2.5)

    weights = raw[buckets].astype(np.float32)

    print(
        "\nTimestep class distribution in TRAIN windows:"
        f"\n  low    : {int(counts[0])}"
        f"\n  medium : {int(counts[1])}"
        f"\n  high   : {int(counts[2])}"
    )
    print(
        "Timestep loss weights:"
        f"\n  low    : {raw[0]:.3f}"
        f"\n  medium : {raw[1]:.3f}"
        f"\n  high   : {raw[2]:.3f}"
    )

    return weights


def build_model(seq_len, n_features):
    model = keras.Sequential([
        layers.Input(shape=(seq_len, n_features)),
        layers.GRU(32, return_sequences=True),
        layers.GRU(16, return_sequences=True),
        layers.Dense(1, activation="sigmoid"),
    ])
    return model


def evaluate(model, X_data, y_data, tag):
    """
    Evaluation helper for TRAINING diagnostics and VALIDATION selection.

    IMPORTANT:
    This function must NEVER receive X_test/y_test during training.
    The final test set is handled only by evaluate_on_recorded_sessions.py.
    """
    y_pred = model.predict(X_data, verbose=0).squeeze(-1)

    mse = float(np.mean((y_pred - y_data) ** 2))

    true_buckets = to_bucket(y_data)
    pred_buckets = to_bucket(y_pred)

    overall_accuracy = float(
        (true_buckets == pred_buckets).mean() * 100
    )

    bucket_names = ["low", "medium", "high"]
    per_bucket_acc = {}

    for b, name in enumerate(bucket_names):
        mask = true_buckets == b
        if mask.sum() == 0:
            per_bucket_acc[name] = None
            continue

        per_bucket_acc[name] = float(
            (pred_buckets[mask] == b).mean() * 100
        )

    valid_accs = [
        v for v in per_bucket_acc.values()
        if v is not None
    ]
    macro_accuracy = float(np.mean(valid_accs)) if valid_accs else 0.0

    print(f"\n{'=' * 60}")
    print(tag)
    print(f"{'=' * 60}")
    print(f"MSE: {mse:.4f}")
    print(f"Bucketed accuracy: {overall_accuracy:.2f}%")

    for name in bucket_names:
        value = per_bucket_acc[name]
        if value is None:
            print(f"  {name:8s}: no instances")
        else:
            n = int((true_buckets == bucket_names.index(name)).sum())
            print(f"  {name:8s}: {value:.2f}% (n={n})")

    print(f"Macro accuracy: {macro_accuracy:.2f}%")
    print(
        f"Prediction spread: min={y_pred.min():.3f} "
        f"max={y_pred.max():.3f} std={y_pred.std():.4f}"
    )

    return overall_accuracy, mse, macro_accuracy


def print_session_list(title, sessions):
    print(f"\n{title} ({len(sessions)} sessions):")
    for s in sessions:
        print(f"  {os.path.basename(s['path'])}")


def main():
    sessions = load_sessions(SESSIONS_DIR)

    if len(sessions) < 9:
        print(
            f"Only found {len(sessions)} session(s). "
            "A meaningful 3-way split needs substantially more data."
        )
        return

    print(f"Loaded {len(sessions)} session(s).")

    # Diagnostic label distribution.
    print("\nPer-session label distribution:")
    for sess in sessions:
        y = sess["y"]
        low = (y < LOW_MED_BOUNDARY).mean() * 100
        med = ((y >= LOW_MED_BOUNDARY) & (y < MED_HIGH_BOUNDARY)).mean() * 100
        high = (y >= MED_HIGH_BOUNDARY).mean() * 100

        try:
            n_markers = int(
                np.load(sess["path"], allow_pickle=True)["n_collision_markers"]
            )
        except Exception:
            n_markers = 0

        print(
            f"  {os.path.basename(sess['path']):42s} "
            f"label={sess['label']:12s} markers={n_markers} "
            f"low={low:5.1f}% med={med:5.1f}% high={high:5.1f}% "
            f"mean={y.mean():.3f}"
        )

    # --------------------------------------------------------------
    # THE IMPORTANT FIX:
    # split WHOLE SESSIONS into train/validation/test FIRST.
    # --------------------------------------------------------------
    train_sessions, val_sessions, test_sessions = split_sessions_three_way(
        sessions
    )

    print_session_list("TRAIN SESSION SET", train_sessions)
    print_session_list("VALIDATION SESSION SET", val_sessions)
    print_session_list("FINAL TEST SESSION SET (LOCKED)", test_sessions)

    save_split_manifest(train_sessions, val_sessions, test_sessions)

    # Only now create sliding windows.
    X_train, y_train = build_windows(train_sessions, SEQ_LEN, STRIDE)
    X_val, y_val = build_windows(val_sessions, SEQ_LEN, STRIDE)

    if X_train is None or X_val is None:
        print(
            "\nNot enough frames to build train/validation windows. "
            f"Each session needs at least {SEQ_LEN} frames."
        )
        return

    print(
        f"\nTrain windows: {X_train.shape[0]} "
        f"| Validation windows: {X_val.shape[0]}"
    )
    print(
        "FINAL TEST WINDOWS ARE NOT BUILT HERE. "
        "They remain untouched until final evaluation."
    )

    # Load or build model.
    if TRAIN_MODE == "finetune":
        if not os.path.exists(PRETRAINED_MODEL_FILE):
            print(
                f"\n'{PRETRAINED_MODEL_FILE}' not found. "
                "Run train_risk_gru.py first or use TRAIN_MODE='scratch'."
            )
            return

        print(
            f"\nLoading synthetic-pretrained model from "
            f"{PRETRAINED_MODEL_FILE}..."
        )
        model = keras.models.load_model(PRETRAINED_MODEL_FILE)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=FINETUNE_LR),
            loss="mse",
        )
        tag = "FINE-TUNED"
    else:
        print("\nTraining GRU from scratch on real training sessions...")
        model = build_model(SEQ_LEN, N_FEATURES)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=SCRATCH_LR),
            loss="mse",
        )
        tag = "FROM-SCRATCH (real only)"

    # Baseline on VALIDATION only.
    if TRAIN_MODE == "finetune":
        print("\nEvaluating synthetic-pretrained model on VALIDATION baseline...")
        evaluate(
            model,
            X_val,
            y_val,
            "BEFORE FINE-TUNING -- VALIDATION SET",
        )

    # IMPORTANT:
    # We now weight EVERY TIMESTEP independently.
    # The weights are derived ONLY from TRAINING labels.
    train_weights_per_timestep = compute_timestep_sample_weights(y_train)

    print(
        f"\nTimestep sample-weight range: "
        f"min={train_weights_per_timestep.min():.2f} "
        f"max={train_weights_per_timestep.max():.2f}"
    )

    # --------------------------------------------------------------
    # Model selection:
    # VALIDATION is checked periodically.
    #
    # TEST IS NEVER TOUCHED HERE.
    # --------------------------------------------------------------
    best_score = -1.0
    best_macro = -1.0
    best_val_mse = None
    best_val_medium = -1.0
    best_val_high = -1.0
    best_weights = None
    best_epoch = 0

    epochs_done = 0

    while epochs_done < EPOCHS:
        chunk = min(CHECK_EVERY, EPOCHS - epochs_done)

        model.fit(
            X_train,
            y_train[..., np.newaxis],
            sample_weight=train_weights_per_timestep,
            epochs=chunk,
            batch_size=min(BATCH_SIZE, max(1, X_train.shape[0])),
            verbose=1,
        )

        epochs_done += chunk

        _, val_mse, val_macro = evaluate(
            model,
            X_val,
            y_val,
            f"EPOCH {epochs_done} -- VALIDATION SET",
        )

        # Recalculate validation bucket accuracies for checkpoint selection.
        # We prioritize medium/high-risk detection because these are the
        # safety-critical cases. Macro accuracy remains the secondary signal.
        val_pred = model.predict(X_val, verbose=0).squeeze(-1)
        val_true_b = to_bucket(y_val)
        val_pred_b = to_bucket(val_pred)

        medium_mask = val_true_b == 1
        high_mask = val_true_b == 2

        val_medium = (
            float((val_pred_b[medium_mask] == 1).mean() * 100)
            if medium_mask.any() else 0.0
        )
        val_high = (
            float((val_pred_b[high_mask] == 2).mean() * 100)
            if high_mask.any() else 0.0
        )

        # Safety score:
        #   50% medium + 50% high
        #
        # Macro accuracy is printed and used as a tie-breaker. This means
        # a checkpoint cannot win merely because it is excellent on the
        # abundant low-risk bucket.
        safety_score = 0.5 * val_medium + 0.5 * val_high

        print(
            f"  Validation safety score (medium/high average): "
            f"{safety_score:.2f}%"
        )

        better = (
            safety_score > best_score + 1e-6
            or (
                abs(safety_score - best_score) <= 1e-6
                and val_macro > best_macro
            )
            or (
                abs(safety_score - best_score) <= 1e-6
                and abs(val_macro - best_macro) <= 1e-6
                and val_mse < best_val_mse
            )
        )

        if better:
            best_score = safety_score
            best_macro = val_macro
            best_val_mse = val_mse
            best_val_medium = val_medium
            best_val_high = val_high
            best_weights = [w.copy() for w in model.get_weights()]
            best_epoch = epochs_done

            print(
                f"  ^ NEW BEST VALIDATION CHECKPOINT: "
                f"safety={best_score:.2f}% "
                f"(medium={best_val_medium:.2f}%, "
                f"high={best_val_high:.2f}%) "
                f"macro={best_macro:.2f}% MSE={best_val_mse:.4f}"
            )

    if best_weights is None:
        print("\nNo validation checkpoint was created.")
        return

    # Restore best VALIDATION-selected weights.
    model.set_weights(best_weights)

    print(
        f"\nRestored BEST VALIDATION checkpoint: epoch {best_epoch} "
        f"(validation macro={best_macro:.2f}%, "
        f"validation MSE={best_val_mse:.4f})"
    )

    # Final validation report.
    evaluate(
        model,
        X_val,
        y_val,
        f"FINAL SELECTED MODEL -- VALIDATION SET (epoch {best_epoch})",
    )

    # Save model + split manifest.
    model.save(OUTPUT_MODEL_FILE)

    print(f"\nModel saved to: {OUTPUT_MODEL_FILE}")
    print(f"Split manifest saved to: {SPLIT_FILE}")

    print(
        "\n" + "=" * 70 +
        "\nFINAL METHODOLOGY\n" +
        "=" * 70
    )
    print(
        f"  Train sessions      : {len(train_sessions)}\n"
        f"  Validation sessions : {len(val_sessions)}\n"
        f"  Test sessions       : {len(test_sessions)}\n"
        f"  Best epoch           : {best_epoch}\n"
        f"  Validation macro    : {best_macro:.2f}%\n"
        f"  Loss weighting       : timestep-level risk weighting\n"
    )
    print(
        "  IMPORTANT: The final TEST sessions were NOT used for training,\n"
        "  checkpoint selection, early stopping, or hyperparameter selection.\n"
    )
    print(
        "NEXT STEP:\n"
        f"  python evaluate_on_recorded_sessions_final_test.py\n"
        "\n"
        "That script loads session_split.json and evaluates ONLY the locked\n"
        "FINAL TEST session set."
    )


if __name__ == "__main__":
    main()
