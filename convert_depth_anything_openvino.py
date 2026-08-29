import sys
from pathlib import Path

import torch
import openvino as ov

# Make the cloned Depth Anything V2 repository importable
PROJECT_DIR = Path(__file__).resolve().parent
DEPTH_REPO = PROJECT_DIR / "Depth-Anything-V2"
sys.path.insert(0, str(DEPTH_REPO))

from depth_anything_v2.dpt import DepthAnythingV2


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
CHECKPOINT = (
    DEPTH_REPO
    / "checkpoints"
    / "depth_anything_v2_vits.pth"
)

OUTPUT_DIR = PROJECT_DIR / "depth_openvino"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_MODEL = OUTPUT_DIR / "depth_anything_v2_vits_fp16.xml"


# ------------------------------------------------------------
# Load PyTorch Depth Anything V2 Small
# ------------------------------------------------------------
print("Loading Depth Anything V2 Small...")

model = DepthAnythingV2(
    encoder="vits",
    features=64,
    out_channels=[48, 96, 192, 384],
)

state_dict = torch.load(
    CHECKPOINT,
    map_location="cpu",
)

model.load_state_dict(state_dict)
model.eval()

print("PyTorch model loaded.")


# ------------------------------------------------------------
# Convert to OpenVINO
# ------------------------------------------------------------
print("Converting model to OpenVINO...")

# IMPORTANT:
# This follows the OpenVINO Depth Anything V2 conversion approach.
# Static 518x518 input is used during conversion.
example_input = torch.randn(
    1, 3, 518, 518,
    dtype=torch.float32,
)

ov_model = ov.convert_model(
    model,
    example_input=example_input,
    input=[1, 3, 518, 518],
)

print("OpenVINO conversion complete.")


# ------------------------------------------------------------
# Save FP32 OpenVINO model
# ------------------------------------------------------------
fp32_path = OUTPUT_DIR / "depth_anything_v2_vits_fp32.xml"

ov.save_model(
    ov_model,
    fp32_path,
)

print(f"Saved FP32 model: {fp32_path}")


# ------------------------------------------------------------
# Convert weights to FP16
# ------------------------------------------------------------
print("Converting OpenVINO model to FP16...")

compressed = ov.compress_quantize_weights(
    ov_model,
    mode="float16",
)

ov.save_model(
    compressed,
    OUTPUT_MODEL,
)

print(f"Saved FP16 model: {OUTPUT_MODEL}")

print("\nDONE.")