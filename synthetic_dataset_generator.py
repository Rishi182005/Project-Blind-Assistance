"""
Rebalanced Synthetic Dataset Generator
========================================

WHY THIS EXISTS:
The original generator randomly sampled walking speed / obstacle speed /
starting distance uniformly. That's a reasonable first pass, but it meant
most randomly-generated scenarios drifted toward "obstacle closing in"
territory over a 30-step sequence, so ~92% of timesteps ended up labeled
"high risk" and only ~1.5% ended up "low risk." A model trained on that
looks great on paper (99%+ overall accuracy) partly because it can get
away with being weak on the rare "low risk" class without hurting the
aggregate number much.

THE FIX: instead of sampling parameters uniformly and letting risk fall
where it may, this generator explicitly decides WHICH SCENARIO TYPE each
sequence should be (safe / ambiguous / dangerous) FIRST, in equal thirds,
then samples parameters that are likely to produce that scenario. This is
called "stratified scenario generation" -- you're directly controlling
for the thing you want balanced, rather than hoping random sampling gets
you there.

Three scenario types, roughly equal counts:
  SAFE       -- obstacle far away and/or moving away (agent walks past it,
                or it's stationary and never gets close)
  AMBIGUOUS  -- obstacle at moderate distance, slow or uncertain closing
                speed (borderline cases -- genuinely useful training signal,
                since these are the hardest real-world calls too)
  DANGEROUS  -- obstacle close and closing in fast (genuine collision course)

OUTPUT: rebalanced_synthetic_risk_dataset.npz containing:
  X -- shape (4000, 30, 4): [distance, closing_speed, class_encoded, agent_speed]
  y -- shape (4000, 30): per-timestep risk score (0 to 1)

pip install numpy --break-system-packages
"""

import numpy as np

# ---------- Config ----------
SEQ_LEN = 30
N_FEATURES = 4
N_SEQUENCES = 4000
RANDOM_SEED = 42

# Risk bucket boundaries -- used only for the balance-check printout below,
# not for generation itself (generation works on scenario type, not buckets)
LOW_MED_BOUNDARY = 0.33
MED_HIGH_BOUNDARY = 0.66

OUTPUT_FILE = "rebalanced_synthetic_risk_dataset.npz"

np.random.seed(RANDOM_SEED)


def generate_one_sequence(scenario_type, seq_len=SEQ_LEN):
    """Generates one sequence's worth of parameters, biased toward the
    given scenario_type so the resulting risk distribution comes out
    roughly balanced across low/medium/high over the whole dataset."""

    if scenario_type == "safe":
        # Starts far, and either moving away or only slowly approaching --
        # never gets genuinely close during the sequence.
        start_dist = np.random.uniform(2.5, 4.0)
        agent_speed = np.random.uniform(0.8, 1.6)
        obstacle_speed = np.random.uniform(-0.8, 0.1)  # mostly moving away/neutral

    elif scenario_type == "ambiguous":
        # Moderate distance, moderate closing speed -- borderline cases
        # that could plausibly resolve either way, which is exactly the
        # useful "medium risk" training signal.
        start_dist = np.random.uniform(1.2, 2.5)
        agent_speed = np.random.uniform(0.8, 1.6)
        obstacle_speed = np.random.uniform(0.0, 0.6)

    else:  # "dangerous"
        # Close start and/or fast closing speed -- genuine collision course.
        start_dist = np.random.uniform(0.5, 1.5)
        agent_speed = np.random.uniform(0.8, 1.6)
        obstacle_speed = np.random.uniform(0.5, 1.5)

    cls = np.random.randint(0, 5)

    dist = start_dist
    X_seq = np.zeros((seq_len, N_FEATURES), dtype=np.float32)
    y_seq = np.zeros(seq_len, dtype=np.float32)

    for t in range(seq_len):
        closing_speed = agent_speed + obstacle_speed + np.random.normal(0, 0.05)
        dist = max(dist - closing_speed * 0.1, 0.05)

        ttc = dist / max(closing_speed, 1e-3) if closing_speed > 0 else 999.0
        risk = 1.5 / (ttc + 0.3)
        risk = float(np.clip(risk, 0.0, 1.0))

        X_seq[t] = [dist / 4.0, closing_speed, cls / 5.0, agent_speed]
        y_seq[t] = risk

    return X_seq, y_seq


def generate_balanced_dataset(n_sequences=N_SEQUENCES, seq_len=SEQ_LEN):
    X = np.zeros((n_sequences, seq_len, N_FEATURES), dtype=np.float32)
    y = np.zeros((n_sequences, seq_len), dtype=np.float32)

    scenario_types = np.random.choice(
        ["safe", "ambiguous", "dangerous"], size=n_sequences,
        p=[1/3, 1/3, 1/3],
    )

    for i, scenario in enumerate(scenario_types):
        X[i], y[i] = generate_one_sequence(scenario, seq_len)

    return X, y


def print_bucket_balance(risk_array):
    flat = risk_array.flatten()
    low = (flat < LOW_MED_BOUNDARY).sum()
    med = ((flat >= LOW_MED_BOUNDARY) & (flat < MED_HIGH_BOUNDARY)).sum()
    high = (flat >= MED_HIGH_BOUNDARY).sum()
    total = flat.size
    print(f"\nRisk bucket balance (all timesteps):")
    print(f"  low:    {low:6d}  ({100*low/total:.1f}%)")
    print(f"  medium: {med:6d}  ({100*med/total:.1f}%)")
    print(f"  high:   {high:6d}  ({100*high/total:.1f}%)")


if __name__ == "__main__":
    print("Generating REBALANCED synthetic dataset (equal safe/ambiguous/dangerous scenarios)...")
    X, y = generate_balanced_dataset()
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    print_bucket_balance(y)

    np.savez(OUTPUT_FILE, X=X, y=y)
    print(f"\nSaved to {OUTPUT_FILE}")
    print("Load it later with: data = np.load('rebalanced_synthetic_risk_dataset.npz'); X, y = data['X'], data['y']")