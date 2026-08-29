"""
GRU Risk Model — Training + Bucketed Accuracy Evaluation
============================================================

WHAT THIS DOES:
1. Loads a synthetic risk dataset from disk (.npz file containing X, y)
2. Splits it into train (80%) and held-out test (20%) sets
3. Trains a GRU to predict per-timestep risk score
4. Evaluates on the held-out test set using bucketed accuracy
   (low/medium/high risk classification), not just MSE loss

WHY BUCKETED ACCURACY, NOT JUST MSE:
Your guide asked for ">92% accuracy" — that implies a classification-style
number ("correct vs incorrect"), not a raw regression loss. MSE tells you
how close the predicted number is on average, but doesn't translate
directly into the kind of percentage a guide/panel expects. This script
converts risk scores into low/medium/high buckets and measures how often
the predicted bucket matches the true bucket.

WHY THE TRAIN/TEST SPLIT MATTERS:
If you evaluate on the same data the model was trained on, the accuracy
number is inflated/meaningless -- the model may just be recalling data
it already memorized, not generalizing. This script always evaluates on
a held-out slice the model never saw during training.

USAGE:
  1. First run rebalanced_dataset_generator.py to create the .npz file
     (or point INPUT_FILE below at your own dataset with the same shape)
  2. Run this script: python train_evaluate_risk_gru.py

pip install tensorflow numpy --break-system-packages
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------- Config ----------
INPUT_FILE = "rebalanced_synthetic_risk_dataset.npz"
MODEL_SAVE_FILE = "risk_gru_model.keras"

SEQ_LEN = 30
N_FEATURES = 4
TEST_FRACTION = 0.2
RANDOM_SEED = 42
EPOCHS = 25
BATCH_SIZE = 32

# Risk bucket boundaries -- must match how you think about risk levels.
# Adjust if your project defines "low/medium/high" differently.
LOW_MED_BOUNDARY = 0.33
MED_HIGH_BOUNDARY = 0.66

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def load_dataset(path):
    data = np.load(path)
    return data["X"], data["y"]


def train_test_split(X, y, test_fraction, seed):
    n = X.shape[0]
    n_test = int(n * test_fraction)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    test_idx, train_idx = indices[:n_test], indices[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def build_model(seq_len, n_features):
    model = keras.Sequential([
        layers.Input(shape=(seq_len, n_features)),
        layers.GRU(32, return_sequences=True),
        layers.GRU(16, return_sequences=True),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def to_bucket(risk_array, low_med=LOW_MED_BOUNDARY, med_high=MED_HIGH_BOUNDARY):
    buckets = np.zeros_like(risk_array, dtype=int)
    buckets[risk_array >= low_med] = 1
    buckets[risk_array >= med_high] = 2
    return buckets


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test, verbose=0).squeeze(-1)
    test_mse = np.mean((y_pred - y_test) ** 2)

    true_buckets = to_bucket(y_test)
    pred_buckets = to_bucket(y_pred)
    overall_accuracy = (true_buckets == pred_buckets).mean() * 100

    print(f"\nHeld-out test MSE: {test_mse:.4f}")
    print(f"\n{'='*50}")
    print(f"BUCKETED ACCURACY (held-out test set): {overall_accuracy:.2f}%")
    print(f"{'='*50}")

    bucket_names = ["low", "medium", "high"]
    print("\nPer-bucket accuracy:")
    for b, name in enumerate(bucket_names):
        mask = true_buckets == b
        if mask.sum() == 0:
            print(f"  {name}: no test instances")
            continue
        bucket_acc = (pred_buckets[mask] == b).mean() * 100
        print(f"  {name:8s}: {bucket_acc:.2f}%  (n={mask.sum()})")

    print("\nConfusion matrix (rows=true, cols=predicted):")
    print(f"{'':>10s}{'low':>8s}{'medium':>8s}{'high':>8s}")
    for i, row_name in enumerate(bucket_names):
        row = [((true_buckets == i) & (pred_buckets == j)).sum() for j in range(3)]
        print(f"{row_name:>10s}{row[0]:>8d}{row[1]:>8d}{row[2]:>8d}")

    n_test = X_test.shape[0]
    print(f"\n{'='*50}")
    print("WHAT TO REPORT TO YOUR GUIDE:")
    print(f"  '>{overall_accuracy:.1f}% risk-bucket classification accuracy on a held-out")
    print(f"   20% test split (N={n_test} sequences, {n_test * SEQ_LEN} timesteps).'")
    print("  Follow with the honest caveat: this is synthetic-data accuracy;")
    print("  real-world sensor accuracy will be measured separately once")
    print("  hardware integration and real-world validation recordings are done.")
    print(f"{'='*50}")

    return overall_accuracy


if __name__ == "__main__":
    print(f"Loading dataset from {INPUT_FILE}...")
    X, y = load_dataset(INPUT_FILE)
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    X_train, y_train, X_test, y_test = train_test_split(X, y, TEST_FRACTION, RANDOM_SEED)
    print(f"Train: {X_train.shape[0]} sequences | Test (held out): {X_test.shape[0]} sequences")

    model = build_model(SEQ_LEN, N_FEATURES)

    print("\nTraining GRU...")
    model.fit(
        X_train, y_train[..., np.newaxis],
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    evaluate(model, X_test, y_test)

    model.save(MODEL_SAVE_FILE)
    print(f"\nModel saved to {MODEL_SAVE_FILE}")
    print(f"Load it later with: model = keras.models.load_model('{MODEL_SAVE_FILE}')")