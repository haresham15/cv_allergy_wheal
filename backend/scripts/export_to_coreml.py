"""Export the trained Late-Fusion RGB-D U-Net to CoreML (.mlpackage).

Uses the predict() method (includes sigmoid) so the on-device model
outputs probabilities directly without post-processing.

Usage:
  # Export from best checkpoint:
  python -m backend.scripts.export_to_coreml

  # Export from specific checkpoint:
  python -m backend.scripts.export_to_coreml --checkpoint backend/models/checkpoints/epoch_050.pth

  # Export untrained (random weights, for pipeline testing):
  python -m backend.scripts.export_to_coreml --untrained
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

try:
    import coremltools as ct
except ImportError:
    print("ERROR: coremltools is required. Install with: pip install coremltools")
    sys.exit(1)

from backend.models.unet_rgbd import LateFusionUNet


# ── Normalization constants (must match Metal shader + training pipeline) ──
RGB_SCALE = 1.0 / 255.0      # uint8 → [0,1]
DEPTH_MIN = 0.1               # meters
DEPTH_MAX = 0.5               # meters
DEPTH_RANGE = DEPTH_MAX - DEPTH_MIN


class InferenceWrapper(torch.nn.Module):
    """Wraps the U-Net to call predict() (with sigmoid) for tracing."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, rgb, depth):
        return self.model.predict(rgb, depth)


def export_coreml(args):
    print("[CoreML] Initializing PyTorch model...")
    model = LateFusionUNet(n_classes=1, bilinear=False)

    # Load trained weights
    if not args.untrained:
        ckpt_path = args.checkpoint
        if not os.path.exists(ckpt_path):
            print(f"[CoreML] WARNING: Checkpoint not found at {ckpt_path}")
            print("[CoreML]          Exporting with random weights (use --untrained to silence)")
        else:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                epoch = checkpoint.get("epoch", "?")
                iou = checkpoint.get("val_iou", "?")
                print(f"[CoreML] Loaded checkpoint: epoch={epoch}, IoU={iou}")
            else:
                model.load_state_dict(checkpoint)
                print(f"[CoreML] Loaded raw state dict from {ckpt_path}")

    model.eval()

    # Wrap with sigmoid for inference
    wrapper = InferenceWrapper(model)
    wrapper.eval()

    HEIGHT, WIDTH = args.img_size, args.img_size
    dummy_rgb = torch.rand(1, 3, HEIGHT, WIDTH)
    dummy_depth = torch.rand(1, 1, HEIGHT, WIDTH)

    print(f"[CoreML] Tracing model at {HEIGHT}×{WIDTH}...")
    traced = torch.jit.trace(wrapper, (dummy_rgb, dummy_depth))

    print("[CoreML] Converting to CoreML mlprogram...")
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                name="rgbImage",
                shape=dummy_rgb.shape,
                dtype=float,
            ),
            ct.TensorType(
                name="depthMap",
                shape=dummy_depth.shape,
                dtype=float,
            ),
        ],
        outputs=[
            ct.TensorType(name="segmentationMask", dtype=float),
        ],
        minimum_deployment_target=ct.target.iOS16,
    )

    # ── Add metadata for on-device preprocessing reference ──
    mlmodel.author = "WhealTracker"
    mlmodel.short_description = (
        "Late-Fusion RGB-D U-Net for allergy wheal segmentation. "
        "Accepts normalised RGB [0,1] and depth [0,1] tensors. "
        "Outputs binary segmentation probabilities."
    )
    mlmodel.input_description["rgbImage"] = (
        f"RGB image normalised to [0,1] (scale={RGB_SCALE}). Shape: [1, 3, {HEIGHT}, {WIDTH}]"
    )
    mlmodel.input_description["depthMap"] = (
        f"Depth normalised from [{DEPTH_MIN},{DEPTH_MAX}]m to [0,1]. "
        f"Shape: [1, 1, {HEIGHT}, {WIDTH}]"
    )
    mlmodel.output_description["segmentationMask"] = (
        f"Wheal probability mask [0,1]. Shape: [1, 1, {HEIGHT}, {WIDTH}]"
    )

    # Save
    output_path = args.output
    mlmodel.save(output_path)
    print(f"[CoreML] ✓ Exported to: {output_path}")
    print(f"[CoreML]   Model size: {os.path.getsize(output_path) / 1e6:.1f} MB (approx)")


def main():
    parser = argparse.ArgumentParser(description="Export RGB-D U-Net to CoreML")

    parser.add_argument(
        "--checkpoint", type=str,
        default="backend/models/checkpoints/best_rgbd_unet.pth",
        help="Path to trained .pth checkpoint",
    )
    parser.add_argument(
        "--untrained", action="store_true",
        help="Export with random weights (for pipeline testing only)",
    )
    parser.add_argument(
        "--img-size", type=int, default=256,
        help="Input image dimensions (must match training)",
    )
    parser.add_argument(
        "--output", type=str,
        default="backend/models/WhealTrackerRGBD.mlpackage",
        help="Output path for the .mlpackage",
    )

    args = parser.parse_args()
    export_coreml(args)


if __name__ == "__main__":
    main()
