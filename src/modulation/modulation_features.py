
import cv2
import numpy as np
import os
import json
from skimage.filters import gabor
from skimage.color import rgb2gray
from collections import defaultdict

from modulation_utils import build_corrected_modulation_config

def modulation_colorfulness(image):
    """Measures the colorfulness of an image based on Hasler and Süsstrunk's method."""
    (B, G, R) = cv2.split(image.astype("float"))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    std_rg, std_yb = np.std(rg), np.std(yb)
    mean_rg, mean_yb = np.mean(rg), np.mean(yb)
    return np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)

def modulation_texture_energy(image):
    """Measures texture energy via Gabor filters."""
    gray = rgb2gray(image)
    filt_real, _ = gabor(gray, frequency=0.6)
    return float(np.mean(np.abs(filt_real)))

def modulation_intensity_variation(image):
    """Calculates the standard deviation of grayscale intensities."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray))


def modulation_hue_entropy(image: np.ndarray) -> float:
    """
    Calculates the entropy of the hue distribution in an image.

    Args:
        image (np.ndarray): Input image in BGR format (as read by OpenCV).

    Returns:
        float: Entropy value in range [0, ~5.5] (theoretical max ~ln(360) for full spectrum).
    """
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]  # Hue channel (0–179 in OpenCV)

    # Flatten and remove near-zero saturation pixels (grayscale-like regions)
    saturation = hsv[:, :, 1]
    mask = saturation > 20  # optional: remove desaturated pixels
    hue_values = hue[mask]

    if len(hue_values) == 0:
        return 0.0

    # Histogram and normalization
    hist, _ = np.histogram(hue_values, bins=36, range=(0, 180), density=True)
    hist += 1e-8  # avoid log(0)

    # Entropy
    entropy = -np.sum(hist * np.log(hist))
    return float(entropy)

def modulation_edge_density(image: np.ndarray) -> float:

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute gradients using Sobel
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Normalize and threshold edges
    edge_map = (magnitude > 50).astype(np.uint8)  # adjust threshold as needed

    # Compute edge density
    density = np.sum(edge_map) / edge_map.size
    return float(density)

def modulation_spatial_sparsity(image: np.ndarray) -> float:
    """
    Estimate how much of the image is visually sparse, using HSV-based activity.
    Returns a value in [0, 1], where 1.0 = completely sparse (empty).
    """

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]  # Value (intensity)
    s = hsv[:, :, 1]  # Saturation

    # Define active pixels: not too dark/bright and not desaturated
    active_mask = (v > 30) & (v < 230) & (s > 30)

    # Compute spatial activity ratio
    active_ratio = np.sum(active_mask) / active_mask.size
    sparsity = 1.0 - active_ratio
    return float(sparsity)

def modulation_hue_clustering_index(image: np.ndarray, grid_size: int = 8) -> float:
    """
    Measures hue clustering by computing hue variance across spatial grid regions.
    Higher score means more localized color clusters (i.e., grouping).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]  # Hue channel (0–179)

    h, w = hue.shape
    step_h = h // grid_size
    step_w = w // grid_size

    grid_hues = []

    for i in range(grid_size):
        for j in range(grid_size):
            region = hue[i * step_h:(i + 1) * step_h, j * step_w:(j + 1) * step_w]
            if region.size > 0:
                mean_hue = np.mean(region)
                grid_hues.append(mean_hue)

    # Compute variance of grid region hues
    if len(grid_hues) < 2:
        return 0.0

    variance = np.var(grid_hues)
    clustering_score = min(variance / 1000.0, 1.0)  # Normalize to [0, 1]
    return float(clustering_score)

def compute_image_features(image):
    colorfulness = modulation_colorfulness(image)
    texture = modulation_texture_energy(image)
    intensity = modulation_intensity_variation(image)
    hue = modulation_hue_entropy(image)
    edge = modulation_edge_density(image)
    sparsity = modulation_spatial_sparsity(image)
    hue_clustering = modulation_hue_clustering_index(image)

    return {
        "colorfulness": min(colorfulness / 120.0, 1.0),
        "texture_energy": min(texture / 0.1, 1.0),
        "intensity_variation": intensity / 255.0,
        "hue_entropy": min(hue / 2.2, 1.0),
        "edge_density": min(edge / 0.85, 1.0),
        "spatial_sparsity": sparsity,
        "hue_clustering_index": hue_clustering

    }




def precompute_modulation_features(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for filename in sorted(image_files):
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"❌ Skipping unreadable image: {filename}")
            continue

        # Extract feature vector as a dictionary
        feature_dict = compute_image_features(image)

        # Save as JSON using the same filename (e.g., cat_119.json)
        out_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".json")
        with open(out_path, "w") as f:
            json.dump(feature_dict, f, indent=2)

        print(f"✅ Saved features for {filename} → {out_path}")



if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "work_dataset", "input_images")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "src", "modulation", "data", "modulation_features_3")

    precompute_modulation_features(INPUT_DIR, OUTPUT_DIR)
    build_corrected_modulation_config(
        ga_config_path=os.path.join(PROJECT_ROOT, "src", "ga", "config.json"),
        modulation_config_path="modulation_config.json",
        output_path="modulation_config_cor.json"
    )

