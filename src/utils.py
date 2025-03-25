import cv2
import numpy as np
import os

def load_image(image_path):
    """Loads an image from a given path."""
    return cv2.imread(image_path)

def save_image(image, path):
    """Saves an image to a specified path."""
    cv2.imwrite(path, image)

def load_dataset(input_dir, gt_dir):
    """
    Returns list of tuples: (image_path, ground_truth_path)
    """
    image_names = sorted(os.listdir(input_dir))
    gt_names = sorted(os.listdir(gt_dir))
    matched = [(os.path.join(input_dir, img), os.path.join(gt_dir, f"fixation_{img}")) for img in image_names]
    return matched


def load_dataset_in_gt(input_dir, gt_dir):
    """
    Load image and ground truth saliency maps into memory.

    Returns:
        List of tuples: [(image, ground_truth), ...]
    """
    image_gt_data = []

    for filename in sorted(os.listdir(input_dir)):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(input_dir, filename)
            gt_path = os.path.join(gt_dir, filename)

            if not os.path.exists(gt_path):
                print(f"⚠️ Missing GT for: {filename}")
                continue

            image = cv2.imread(image_path)
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

            if image is not None and gt is not None:
                image_gt_data.append((image, gt))

    return image_gt_data

def weights_to_heuristic_config(weights):
    keys = ["h1", "h2", "h3", "h4", "h5"]
    return {
        key: {"enabled": True, "weight": float(w)}
        for key, w in zip(keys, weights)
    }

def create_hue_mask(hue_channel, target_hue, hue_range):
    """Generates a mask considering the circular nature of the hue values."""
    lower = int((target_hue - hue_range) % 180)
    upper = int((target_hue + hue_range) % 180)

    if lower < upper:
        # Regular case
        mask = cv2.inRange(hue_channel, lower, upper)
    else:
        # Hue wrap-around case (e.g., red hues spanning from 170° to 10°)
        mask1 = cv2.inRange(hue_channel, lower, 180)
        mask2 = cv2.inRange(hue_channel, 0, upper)
        mask = cv2.bitwise_or(mask1, mask2)

    return mask