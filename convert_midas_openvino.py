import sys
from pathlib import Path

import torch
import openvino as ov

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
MIDAS_DIR = PROJECT_DIR / "MiDaS"
WEIGHTS = MIDAS_DIR / "weights" / "midas_v21_small_256.pt"

OUTPUT_DIR = PROJECT_DIR / "midas_openvino"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_XML = OUTPUT_DIR / "midas_v21_small_256.xml"

# Make the official MiDaS repository importable
sys.path.insert(0, str(MIDAS_DIR))

from midas.dpt_depth import DPTDepthModel
from midas.midas_net_custom import MidasNet_small


# ------------------------------------------------------------
# Load MiDaS Small
# ------------------------------------------------------------
print("Loading MiDaS Small...")

if not WEIGHTS.exists():
    raise FileNotFoundError(
        f"Checkpoint not found:\n{WEIGHTS}"
    )

# This is the architecture corresponding to the official
# midas_v21_small_256 checkpoint.
model = MidasNet_small(
    None,
    features=64,
    backbone="efficientnet_lite3",
    exportable=True,
    non_negative=True,
    blocks={"expand": True},
)

checkpoint = torch.load(
    WEIGHTS,
    map_location="cpu",
)

model.load_state_dict(checkpoint)
model.eval()

print("MiDaS Small loaded.")


# ------------------------------------------------------------
# Convert PyTorch -> OpenVINO
# ------------------------------------------------------------
print("Converting MiDaS Small to OpenVINO...")

# The checkpoint is designed for 256px inference.
example_input = torch.randn(
    1,
    3,
    256,
    256,
    dtype=torch.float32,
)

ov_model = ov.convert_model(
    model,
    example_input=example_input,
)

print("Conversion successful.")


# ------------------------------------------------------------
# Save OpenVINO model
# ------------------------------------------------------------
ov.save_model(
    ov_model,
    OUTPUT_XML,
)

print(f"Saved OpenVINO model to:")
print(OUTPUT_XML)

print("\nDONE.")