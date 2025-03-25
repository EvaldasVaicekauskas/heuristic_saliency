import pysaliency
import numpy as np
import os
import cv2
import argparse


def load_ground_truth_saliency(gt_dir):
    """Load ground truth saliency maps as normalized grayscale input_images."""
    gt_maps = {}

    for filename in sorted(os.listdir(gt_dir)):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            gt_map = cv2.imread(os.path.join(gt_dir, filename), cv2.IMREAD_GRAYSCALE)
            if gt_map is None:
                print(f"⚠️ Warning: Could not read ground truth map {filename}")
                continue

            # Normalize to range [0, 1]
            gt_map = gt_map.astype(np.float32) / 255.0
            gt_maps[filename] = gt_map

    return gt_maps


class SaliencyMapModel(pysaliency.Model):
    """Custom Pysaliency model to wrap predicted saliency maps for evaluation."""

    def __init__(self, saliency_maps):
        super().__init__()
        self.saliency_maps = saliency_maps  # Dictionary of saliency maps

    def _log_density(self, stimulus):
        """Compute log density of the saliency map for the given stimulus."""
        if stimulus not in self.saliency_maps:
            raise ValueError(f"Stimulus {stimulus} not found in saliency maps!")
        saliency_map = self.saliency_maps[stimulus]

        # Ensure saliency map is properly normalized
        saliency_map = saliency_map.astype(np.float32) + 1e-9  # Avoid log(0)
        saliency_map /= saliency_map.sum()  # Normalize to sum=1

        return np.log(saliency_map)


def evaluate_saliency_dataset(saliency_dir, gt_dir):
    """Evaluate predicted saliency maps against ground truth saliency maps."""
    print("📂 Loading ground truth saliency maps...")
    gt_maps = load_ground_truth_saliency(gt_dir)

    if not gt_maps:
        print("❌ No valid ground truth data found!")
        return

    print("📊 Evaluating saliency maps...")
    saliency_maps = {}

    for filename in sorted(os.listdir(saliency_dir)):

        if filename.startswith("saliency_"):
            gt_filename = filename.replace("saliency_", "", 1)  # Remove prefix
        else:
            gt_filename = filename  # Use filename as is



        if gt_filename not in gt_maps:
            print(f"⚠️ No matching ground truth for {filename} → Looking for {gt_filename}, skipping...")
            continue

        print(f"🔍 Processing: {filename} (Matching GT: {gt_filename})")

        pred_map = cv2.imread(os.path.join(saliency_dir, filename), cv2.IMREAD_GRAYSCALE)
        if pred_map is None:
            print(f"❌ Could not read predicted saliency map {filename}")
            continue

        # Normalize maps to [0,1]
        pred_map = pred_map.astype(np.float32) / 255.0
        saliency_maps[gt_filename] = pred_map  # Store with corrected filename

    if not saliency_maps:
        print("❌ No valid predicted saliency maps found!")
        return

    # Convert saliency maps to pysaliency Model
    model = SaliencyMapModel(saliency_maps)

    for filename in saliency_maps:
        gt_map = gt_maps[filename]

        # Compute evaluation metrics
        cc = pysaliency.metrics.CC(gt_map, saliency_maps[filename])
        sim = pysaliency.metrics.SIM(gt_map, saliency_maps[filename])
        kl_div = pysaliency.metrics.MIT_KLDiv(gt_map, saliency_maps[filename])

        print(f"✅ Results for {filename}:")
        print(f"   📌 Correlation Coefficient (CC): {cc:.4f}")
        print(f"   📌 Similarity Score (SIM): {sim:.4f}")
        print(f"   📌 KL-Divergence: {kl_div:.4f}")

def evaluate_saliency_image(pred_map, gt_map, normalize=True):
    """
    Evaluate a predicted saliency map against a ground truth saliency map.

    Parameters:
        pred_map (np.ndarray): Predicted saliency (grayscale).
        gt_map (np.ndarray): Ground truth saliency (grayscale).
        normalize (bool): Whether to normalize inputs to [0, 1].

    Returns:
        tuple: (CC, SIM, KL-Div)
    """
    if pred_map is None or gt_map is None:
        raise ValueError("Prediction or ground truth saliency map is None.")

    # Ensure same shape
    if pred_map.shape != gt_map.shape:
        raise ValueError(f"Shape mismatch: pred {pred_map.shape} vs. gt {gt_map.shape}")

    # Normalize to [0, 1] if needed
    if normalize:
        pred_map = pred_map.astype(np.float32) / 255.0
        gt_map = gt_map.astype(np.float32) / 255.0

    # Clamp edge cases
    pred_map = np.clip(pred_map, 0, 1)
    gt_map = np.clip(gt_map, 0, 1)

    # Pysaliency requires a model for some metrics
    model = SaliencyMapModel(pred_map)

    # Compute metrics
    cc = pysaliency.metrics.CC(gt_map, pred_map)
    sim = pysaliency.metrics.SIM(gt_map, pred_map)
    kl_div = pysaliency.metrics.MIT_KLDiv(pred_map,gt_map)

    return cc, sim, kl_div


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate saliency maps.")
    parser.add_argument("--saliency_dir", required=True, help="Path to the directory with predicted saliency maps")
    parser.add_argument("--gt_dir", required=True, help="Path to the directory with ground truth saliency maps")

    args = parser.parse_args()
    evaluate_saliency(args.saliency_dir, args.gt_dir)
