import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


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

def precompute_heuristic_maps(model, dataset, heuristic_names):
    """
    Precompute heuristic maps for each image in the dataset.

    Returns:
        Dict mapping image_id -> dict of {heuristic_name: saliency_map}
    """
    precomputed = {}

    for idx, (image, _) in enumerate(dataset):
        heuristic_maps = model.generate_saliency_maps_for_image(image)
        precomputed[idx] = heuristic_maps

    return precomputed

def combine_maps_with_weights(heuristic_maps, config):
    combined = np.zeros_like(next(iter(heuristic_maps.values())), dtype=np.float32)

    for name, map_ in heuristic_maps.items():
        weight = config.get(name, {}).get("weight", 0.0)
        if weight >= 0.01:
            combined += map_ * weight

    return cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def load_precomputed_maps_from_folders(base_path, heuristic_names, INPUT_DIR, GT_DIR):
    """
    Load precomputed saliency maps into a nested dictionary:
    { image_name: { heuristic_name: map } }
    """
    image_filenames = [
        filename for filename in sorted(os.listdir(INPUT_DIR))
        if filename.lower().endswith((".jpg", ".jpeg", ".png"))
           and os.path.exists(os.path.join(INPUT_DIR, filename))
           and os.path.exists(os.path.join(GT_DIR, filename))
    ]

    precomputed = {}

    for image_name in image_filenames:
        image_maps = {}

        for h_name in heuristic_names:
            h_folder = os.path.join(base_path, h_name)
            map_path = os.path.join(h_folder, image_name)

            if not os.path.exists(map_path):
                print(f"⚠️ Warning: Missing map for {image_name} in {h_name}")
                continue

            saliency_map = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
            if saliency_map is None:
                print(f"⚠️ Failed to load {map_path}")
                continue

            image_maps[h_name] = saliency_map

        precomputed[image_name] = image_maps

    return precomputed


def precompute_and_save_heuristic_maps(model, input_dir, output_base_dir=None):
    """
    Run heuristic saliency generation for each image in `input_dir`
    and return all heuristic maps in memory.
    Optionally saves each heuristic map if `output_base_dir` is provided.

    Returns:
        Dict mapping image_name -> {heuristic_name: saliency_map}
    """
    if output_base_dir:
        os.makedirs(output_base_dir, exist_ok=True)

    precomputed = {}

    for filename in sorted(os.listdir(input_dir)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"⚠️ Could not read image: {filename}")
            continue

        heuristic_maps = model.generate_saliency_maps_for_image(image)
        precomputed[filename] = heuristic_maps  # ✅ Store by filename

        if output_base_dir:
            for h_name, h_map in heuristic_maps.items():
                h_dir = os.path.join(output_base_dir, h_name)
                os.makedirs(h_dir, exist_ok=True)

                save_path = os.path.join(h_dir, filename)
                success = cv2.imwrite(save_path, h_map)
                if not success:
                    print(f"❌ Failed to save {h_name} map for {filename}")
                else:
                    print(f"💾 Saved {h_name} → {save_path}")

    return precomputed  # ✅ Return the full dictionary!

def test_heuristic_visualization(image_path, heuristic_fn, color=(0, 1, 0)):
    """
    Test a single heuristic on an image:
    - Loads image
    - Applies heuristic function
    - Plots original image, saliency map, and contour overlay

    Args:
        image_path: Path to image file
        heuristic_fn: A function that takes an image and returns a saliency map
        color: Tuple with normalized RGB values (0–1) for overlay (default: green)
    """
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    saliency_map = heuristic_fn(image)

    # --- Generate overlay using your previous method (simplified for one map)
    # overlay_image = image_rgb.copy()
    # contours, _ = cv2.findContours(saliency_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contour_color = tuple(int(c * 255) for c in color)
    # cv2.drawContours(overlay_image, contours, -1, contour_color, thickness=2)

    overlay_image = image_rgb.copy()
    mask = saliency_map > 30  # Threshold can be adjusted
    alpha = 0.5

    # Create a solid color image with the same shape
    color_rgb = np.array(color) * 255
    color_layer = np.zeros_like(image, dtype=np.uint8)
    color_layer[:, :] = color_rgb.astype(np.uint8)

    # Apply mask
    overlay_image[mask] = cv2.addWeighted(image[mask], 1 - alpha, color_layer[mask], alpha, 0)

    # --- Plot all three views
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(image_rgb)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(saliency_map, cmap='gray')
    axs[1].set_title("Saliency Map")
    axs[1].axis("off")

    axs[2].imshow(overlay_image)
    axs[2].set_title("Overlay with Contours")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()

def save_heuristic_comparison_visualization(
        image_path,
        gt_path,
        heuristic_fn,
        output_path,
        cmap_saliency='gray',
        cmap_gt='hot'
):
    """
    Save a side-by-side visualization of:
    - Original image
    - Heuristic-generated saliency map
    - Ground truth fixation map

    Args:
        image_path (str): Path to the input image
        gt_path (str): Path to the ground truth saliency/fixation map (grayscale)
        heuristic_fn (function): Function that takes image and returns saliency map
        output_path (str): Where to save the result (.png, .jpg, etc.)
        cmap_saliency (str): Colormap for the saliency map
        cmap_gt (str): Colormap for ground truth
    """
    # Load input image and GT
    image = cv2.imread(image_path)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image is None or gt is None:
        print(f"❌ Failed to load image or ground truth for {image_path}")
        return

    # Generate heuristic saliency
    saliency = heuristic_fn(image)

    # Plot and save
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(image_rgb)
    #axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(saliency, cmap=cmap_saliency)
    #axs[1].set_title("Heuristic Saliency")
    axs[1].axis("off")

    axs[2].imshow(gt, cmap=cmap_gt)
    #axs[2].set_title("Ground Truth Fixation")
    axs[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"💾 Saved comparison to: {output_path}")