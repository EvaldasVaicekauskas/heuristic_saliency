from os import statvfs_result

import numpy as np

import matplotlib.pyplot as plt
import os

import cv2

"""
Stage	Purpose
1. Soft thresholding (e.g. 80%)	Keep general structure, remove noise
2. Iterative fixation proposal	Use circular kernel to find peak regions
3. Overlap suppression	Ensure spatially distinct fixation points
4. Amplify peak regions	Apply modulation based on peak-shift theory
5. Global blur (optional)	Smooth saliency for human-like appearance
6. Optional center-bias proportional to the uniformity of saliency map
"""



def visualize_saliency_debug(
    image,
    saliency_map,
    gt_map=None,
    initial_saliency_map=None,
    stage_name="Modelio rezultatas",
    save_dir=None,
    filename=None
):
    """
    Visualizes postprocessing steps alongside original image, GT, and original saliency map.

    Args:
        image (np.ndarray): Original image (BGR or RGB)
        saliency_map (np.ndarray): Current saliency map (postprocessed)
        gt_map (np.ndarray or None): Ground truth saliency map
        initial_saliency_map (np.ndarray or None): Original saliency map before postprocessing
        stage_name (str): Label for the current postprocessing step
        save_dir (str or None): If set, saves to disk
        filename (str or None): Used for file naming
    """
    panels = 2 + int(gt_map is not None) + int(initial_saliency_map is not None)
    plt.figure(figsize=(4 * panels, 4))

    i = 1
    if image is not None:
        plt.subplot(1, panels, i)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Paveikslas")
        plt.axis("off")
        i += 1

    if initial_saliency_map is not None:
        plt.subplot(1, panels, i)
        plt.imshow(initial_saliency_map, cmap='gray')
        plt.title("Modelio rezultatas")
        plt.axis("off")
        i += 1

    plt.subplot(1, panels, i)
    plt.imshow(saliency_map, cmap='gray')
    plt.title(stage_name)
    plt.axis("off")
    i += 1

    if gt_map is not None:
        plt.subplot(1, panels, i)
        plt.imshow(gt_map, cmap='gray')
        plt.title("Stebėtojų dėmesio pasiskirstymas")
        plt.axis("off")
        i += 1

    #plt.tight_layout()

    if save_dir and filename:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{os.path.splitext(filename)[0]}.png")
        plt.savefig(out_path)
        plt.close()
    else:
        print(f"[INFO] Visualization for '{stage_name}' skipped (non-interactive backend).")

def visualize_saliency_map(
        saliency_map,
        save_dir,
        filename
):
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{os.path.splitext(filename)[0]}.png")

    plt.imshow(saliency_map, cmap='gray')
    plt.axis('off')  # no axes
    plt.tight_layout(pad=0)  # remove padding
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # full extent

    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"💾 Saved saliency map to: {out_path}")



def visualize_fixation_overlay(saliency_map, fixation_coords, kernel_radius=None, kernel_rad_perc = 0.05, image=None, title="Fixations",
                               save_path=None):
    """
    Visualizes fixation points and their Gaussian kernel footprints over a saliency map.

    Args:
        saliency_map (np.ndarray): The saliency map (2D array).
        fixation_coords (list): List of (x, y) fixation center coordinates.
        kernel_radius (int): Radius of the Gaussian kernel used in detection.
        image (np.ndarray or None): Optional image background to show under overlay.
        title (str): Title for the figure.
        save_path (str or None): If provided, saves figure to this path.
    """


    h, w = saliency_map.shape
    overlay = np.stack([saliency_map] * 3, axis=-1).astype(np.uint8)

    if kernel_radius is None:
        kernel_radius = int(kernel_rad_perc * min(h, w))

    fig, ax = plt.subplots(figsize=(8, 6))

    if image is not None:
        background = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(background, alpha=0.35)

    ax.imshow(saliency_map, cmap='hot', alpha=0.65)

    for (x, y) in fixation_coords:
        circ = plt.Circle((x, y), radius=kernel_radius, color='cyan', fill=False, linewidth=2)
        ax.add_patch(circ)
        ax.plot(x, y, 'bo')  # fixation center

    ax.set_title(title)
    ax.axis("off")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def visualize_fixation_overlay_dual(
        saliency_map,
        gt_map,
        fixation_coords,
        kernel_radius=None,
        kernel_rad_perc=0.05,
        image=None,
        title="Fixations (saliency vs GT)",
        save_path=None
):
    """
    Visualizes fixation points and their kernel footprints on both the predicted saliency map
    and the ground truth saliency map.

    Args:
        saliency_map (np.ndarray): The saliency map (2D array).
        gt_map (np.ndarray): Ground truth saliency map (2D array).
        fixation_coords (list): List of (x, y) fixation center coordinates.
        kernel_radius (int or None): Radius of Gaussian kernel (optional).
        kernel_rad_perc (float): Kernel size as percentage of shortest image dimension.
        image (np.ndarray or None): Background image (for saliency overlay).
        title (str): Overall figure title.
        save_path (str or None): If set, saves the figure to this location.
    """
    h, w = saliency_map.shape

    if kernel_radius is None:
        kernel_radius = int(kernel_rad_perc * min(h, w))

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Saliency + Fixations Overlay
    if image is not None:
        background = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axs[0].imshow(background, alpha=0.00)

    axs[0].imshow(saliency_map, cmap='gray', alpha=1.0)
    axs[0].set_title("Modelio rezultatas")
    axs[0].axis("off")

    if fixation_coords is not None:
        for (x, y) in fixation_coords:
            circ = plt.Circle((x, y), radius=kernel_radius, color='cyan', fill=False, linewidth=2)
            axs[0].add_patch(circ)
            axs[0].plot(x, y, 'bo')

    # 2. GT + Fixations Overlay
    axs[1].imshow(gt_map, cmap='gray')
    axs[1].set_title("Stebėtojų dėmesio pasiskirstymas")
    axs[1].axis("off")

    if fixation_coords is not None:
        for (x, y) in fixation_coords:
            circ = plt.Circle((x, y), radius=kernel_radius, color='cyan', fill=False, linewidth=2)
            axs[1].add_patch(circ)
            axs[1].plot(x, y, 'bo')

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def thresholding_perc(saliency_map: np.ndarray, threshold_percentile: float = 80) -> np.ndarray:
    """
    Zeros out low saliency pixels based on a percentile threshold (soft thresholding).

    Args:
        saliency_map (np.ndarray): Float32 input saliency map
        threshold_percentile (float): Percentile above which values are retained

    Returns:
        np.ndarray: Thresholded saliency map (same dtype as input)
    """
    if saliency_map.dtype != np.float32:
        saliency_map = saliency_map.astype(np.float32)

    threshold_value = np.percentile(saliency_map, threshold_percentile)
    return np.where(saliency_map >= threshold_value, saliency_map, 0.0)

def create_gaussian_kernel(radius: int, kernel_depth: float = None) -> np.ndarray:
    """Generates a normalized 2D Gaussian kernel with a given radius."""
    #if sigma is None:
    sigma = radius / kernel_depth

    size = 2 * radius + 1
    x = np.arange(0, size, 1) - radius
    y = np.arange(0, size, 1) - radius
    xx, yy = np.meshgrid(x, y)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.max()

def detect_fixations_from_saliency_map_oo(
    saliency_map: np.ndarray,
    kernel_radius: int = None,
    kernel_rad_perc: float = 0.05,
    kernel_depth: float = 2,
    max_fixations: int = 7,
    min_response_frac: float = 1.45,
    overlap_thresh: float = 0.5
):
    """
    Detects fixation centers from a saliency map using iterative Gaussian kernel scanning.
    """
    map_copy = saliency_map.astype(np.float32).copy()
    h, w = map_copy.shape

    if kernel_radius is None:
        kernel_radius = int(kernel_rad_perc * min(h, w))
    offset = kernel_radius
    kernel = create_gaussian_kernel(kernel_radius, kernel_depth)

    claim_mask = np.zeros_like(map_copy, dtype=np.float32)
    fixation_coords = []
    kernel_sums = []

    while len(fixation_coords) < max_fixations:
        response = cv2.filter2D(map_copy, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        max_val = np.max(response)

        if max_val < min_response_frac * np.average(response):
            break  # no more strong peaks

        y, x = np.unravel_index(np.argmax(response), response.shape)

        # Boundaries in map
        y1, y2 = max(0, y - offset), min(h, y + offset + 1)
        x1, x2 = max(0, x - offset), min(w, x + offset + 1)

        # Boundaries in kernel
        ky1 = offset - (y - y1)
        ky2 = ky1 + (y2 - y1)
        kx1 = offset - (x - x1)
        kx2 = kx1 + (x2 - x1)

        kernel_crop = kernel[ky1:ky2, kx1:kx2]
        existing = claim_mask[y1:y2, x1:x2]
        overlap = np.sum(np.minimum(existing, kernel_crop))

        if overlap > overlap_thresh:
            # Reduce influence & retry (don't skip, just suppress a bit)
            print(overlap, overlap_thresh)
            map_copy[y1:y2, x1:x2] *= overlap_thresh
            continue

        # Accept fixation
        fixation_coords.append((x, y))
        kernel_sums.append(max_val)

        claim_mask[y1:y2, x1:x2] += kernel_crop
        map_copy[y1:y2, x1:x2] *= (1 - kernel_crop)

    max_score = max(kernel_sums) if kernel_sums else 1.0
    norm_scores = [s / max_score for s in kernel_sums]

    return fixation_coords, norm_scores, kernel_radius

def detect_fixations_from_saliency_map_o(
    saliency_map: np.ndarray,
    kernel_radius: int = None,
    kernel_rad_perc: float = 0.05,
    kernel_depth: float = 2,
    max_fixations: int = 7,
    min_response_frac: float = 1.45,
    overlap_val: float = 1.2
):
    """
    Detects fixation centers from a saliency map using iterative Gaussian kernel scanning.
    Uses overlap-aware suppression in the response map to prevent infinite loops.

    Returns:
        fixation_coords: List of (x, y) points
        norm_scores: Normalized fixation strengths (max = 1.0)
        kernel_radius: Final radius used
    """
    map_copy = saliency_map.astype(np.float32).copy()
    h, w = map_copy.shape

    if kernel_radius is None:
        kernel_radius = int(kernel_rad_perc * min(h, w))
    offset = kernel_radius
    kernel = create_gaussian_kernel(kernel_radius, kernel_depth)

    # Claim map tracks accepted fixation coverage
    claim_mask = np.zeros_like(map_copy, dtype=np.float32)

    # Initial response map (will be suppressed independently)
    response = cv2.filter2D(map_copy, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    response_copy = response.copy()

    fixation_coords = []
    kernel_sums = []

    while len(fixation_coords) < max_fixations:
        max_val = np.max(response_copy)
        if max_val < min_response_frac * np.average(response_copy):
            break  # No more strong peaks

        y, x = np.unravel_index(np.argmax(response_copy), response_copy.shape)

        # Bounding boxes
        y1, y2 = max(0, y - offset), min(h, y + offset + 1)
        x1, x2 = max(0, x - offset), min(w, x + offset + 1)

        ky1 = offset - (y - y1)
        ky2 = ky1 + (y2 - y1)
        kx1 = offset - (x - x1)
        kx2 = kx1 + (x2 - x1)

        kernel_crop = kernel[ky1:ky2, kx1:kx2]
        existing = claim_mask[y1:y2, x1:x2]

        overlap = np.sum(np.minimum(existing, kernel_crop))
        max_possible = np.sum(kernel_crop)
        overlap_frac = overlap / max_possible

        if overlap_frac > 0.2:
            # Suppress only in response priority map
            response_copy[y1:y2, x1:x2] *= (1.0 - np.clip(overlap_frac*overlap_val, 0.0, 1.0))
            continue

        # Accept fixation
        fixation_coords.append((x, y))
        kernel_sums.append(max_val)

        claim_mask[y1:y2, x1:x2] += kernel_crop
        map_copy[y1:y2, x1:x2] *= (1 - kernel_crop)
        response_copy[y1:y2, x1:x2] *= (1 - kernel_crop)

    # Normalize fixation scores
    max_score = max(kernel_sums) if kernel_sums else 1.0
    norm_scores = [s / max_score for s in kernel_sums]

    return fixation_coords, norm_scores, kernel_radius

def detect_fixations_from_saliency_map(
    saliency_map: np.ndarray,
    kernel_radius: int = None,
    kernel_rad_perc: float = 0.05,
    kernel_depth: float = 2,
    max_fixations: int = 7,
    min_response_frac: float = 1.45,
    overlap_val: float = 0.5
):
    """
    Detects fixation centers from a saliency map using iterative Gaussian kernel scanning.
    Uses overlap-aware suppression in the response map to prevent infinite loops.

    Returns:
        fixation_coords: List of (x, y) points
        norm_scores: Normalized fixation strengths (max = 1.0)
        kernel_radius: Final radius used
    """
    map_copy = saliency_map.astype(np.float32).copy()
    h, w = map_copy.shape

    if kernel_radius is None:
        kernel_radius = int(kernel_rad_perc * min(h, w))
    offset = kernel_radius
    kernel = create_gaussian_kernel(kernel_radius, kernel_depth)

    # Claim map tracks accepted fixation coverage
    claim_mask = np.zeros_like(map_copy, dtype=np.float32)

    # Initial response map (will be suppressed independently)
    response = cv2.filter2D(map_copy, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    response_copy = response.copy()

    fixation_coords = []
    kernel_sums = []

    while len(fixation_coords) < max_fixations:
        max_val = np.max(response_copy)
        if max_val < min_response_frac * np.average(response_copy):
            break  # No more strong peaks

        y, x = np.unravel_index(np.argmax(response_copy), response_copy.shape)

        # Bounding boxes
        y1, y2 = max(0, y - offset), min(h, y + offset + 1)
        x1, x2 = max(0, x - offset), min(w, x + offset + 1)

        ky1 = offset - (y - y1)
        ky2 = ky1 + (y2 - y1)
        kx1 = offset - (x - x1)
        kx2 = kx1 + (x2 - x1)

        kernel_crop = kernel[ky1:ky2, kx1:kx2]
        existing = claim_mask[y1:y2, x1:x2]

        overlap = np.sum(np.minimum(existing, kernel_crop))
        max_possible = np.sum(kernel_crop)
        overlap_frac = overlap / max_possible

        if overlap_frac > overlap_val:
            # Suppress only in response priority map
            response_copy[y1:y2, x1:x2] *= (1.0 - np.clip(overlap_frac, 0.0, 1.0))
            continue

        # Accept fixation
        fixation_coords.append((x, y))
        kernel_sums.append(max_val)

        claim_mask[y1:y2, x1:x2] += kernel_crop
        map_copy[y1:y2, x1:x2] *= (1 - kernel_crop)
        response_copy[y1:y2, x1:x2] *= (1 - kernel_crop)

    # Normalize fixation scores
    max_score = max(kernel_sums) if kernel_sums else 1.0
    norm_scores = [s / max_score for s in kernel_sums]

    return fixation_coords, norm_scores, kernel_radius


def draw_fixations_into_map(
        shape,
        fixation_coords,
        fixation_scores,
        kernel_radius=30,
        kernel_depth = 2,
        amplify_strength=1.0,
        normalize_output=True
):
    """
    Generates a peak-shift-enhanced saliency map by placing Gaussian activations at fixation points.

    Args:
        shape (tuple): Shape of the output saliency map (H, W)
        fixation_coords (list): List of (x, y) fixation points
        fixation_scores (list): Corresponding importance values for fixations
        kernel_radius (int): Radius of the Gaussian kernel
        sigma (float): Sigma for Gaussian kernel (defaults to radius / 2)
        amplify_strength (float): Global multiplier for kernel amplitudes
        normalize_output (bool): Whether to normalize result to [0, 255]

    Returns:
        np.ndarray: Amplified saliency map (float32 or uint8)
    """
    h, w = shape
    heatmap = np.zeros((h, w), dtype=np.float32)

    kernel = create_gaussian_kernel(kernel_radius, kernel_depth)
    k_h, k_w = kernel.shape
    k_off = kernel_radius

    for (x, y), strength in zip(fixation_coords, fixation_scores):
        y1, y2 = max(0, y - k_off), min(h, y + k_off + 1)
        x1, x2 = max(0, x - k_off), min(w, x + k_off + 1)

        ky1 = k_off - (y - y1)
        ky2 = ky1 + (y2 - y1)
        kx1 = k_off - (x - x1)
        kx2 = kx1 + (x2 - x1)

        kernel_crop = kernel[ky1:ky2, kx1:kx2]
        heatmap[y1:y2, x1:x2] += amplify_strength * strength * kernel_crop

    if normalize_output:
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        return heatmap.astype(np.uint8)

    return heatmap

def amplify_fixations_into_map(
        original_map: np.ndarray,
        fixation_coords,
        fixation_scores,
        kernel_radius=30,
        kernel_depth=2,
        amplify_strength=1.0,
        normalize_output=True
):
    """
    Multiplies Gaussian-shaped weights centered at fixations into the original saliency map.

    Args:
        original_map (np.ndarray): Input saliency map (grayscale).
        fixation_coords (list): List of (x, y) fixation centers.
        fixation_scores (list): Strength per fixation.
        kernel_radius (int): Radius of Gaussian kernel.
        kernel_depth (float): Sharpness of Gaussian (larger = flatter).
        amplify_strength (float): Global intensity multiplier for all fixations.
        normalize_output (bool): Normalize output to [0, 255].

    Returns:
        np.ndarray: Saliency map with peak-shift enhancement.
    """
    h, w = original_map.shape
    saliency = original_map.astype(np.float32).copy()

    weight_map = np.ones_like(saliency)

    kernel_radius = int(kernel_radius)
    kernel = create_gaussian_kernel(kernel_radius, kernel_depth)
    k_off = kernel_radius

    for (x, y), strength in zip(fixation_coords, fixation_scores):
        y1, y2 = max(0, y - k_off), min(h, y + k_off + 1)
        x1, x2 = max(0, x - k_off), min(w, x + k_off + 1)

        ky1 = k_off - (y - y1)
        ky2 = ky1 + (y2 - y1)
        kx1 = k_off - (x - x1)
        kx2 = kx1 + (x2 - x1)

        kernel_crop = kernel[ky1:ky2, kx1:kx2]
        multiplier = 1.0 + amplify_strength * strength * kernel_crop
        weight_map[y1:y2, x1:x2] *= multiplier

    # Multiply saliency by the boost map
    saliency *= weight_map/2
    saliency += weight_map/2 * 64

    if normalize_output:
        saliency = cv2.normalize(saliency, None, 0, 255, cv2.NORM_MINMAX)
        return saliency.astype(np.uint8)

    return saliency

def apply_blur(saliency_map: np.ndarray, sigma_perc: float = 0.03) -> np.ndarray:
    """
    Applies Gaussian blur with sigma proportional to image size.

    Args:
        saliency_map (np.ndarray): Input saliency map (uint8 or float32).
        sigma_perc (float): Sigma as fraction of image's shorter dimension (e.g., 0.03)

    Returns:
        np.ndarray: Blurred saliency map (uint8).
    """

    h, w = saliency_map.shape
    sigma = sigma_perc * min(h, w)

    return cv2.GaussianBlur(saliency_map.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)



def postprocess_saliency_map(saliency_map: np.ndarray, image=None, gt_map=None, filename=""):

    """
    Applies peak-shift inspired postprocessing to enhance salient regions.

    Args:
        saliency_map (np.ndarray): Raw saliency map

    Returns:
        np.ndarray: Refined saliency map
    """

    # === Configurable parameters ===
    #thresholding_percentile = 0.98
    kernel_radius = None                # For fixation candidate detection
    kernel_rad_perc_overal = 0.35
    kernel_rad_perc_mid = 0.25
    kernel_rad_perc = 0.12
    #kernel_rad_perc_fine = 0.00
    max_fixation_count_overal = 1
    max_fixation_count_mid = 2
    max_fixation_count = 5            # Maximum number of peak centers
    #max_fixation_count_fine = 0
    min_response_frac = 1.5
    overlap_penalty = 0.7             # Suppress proportionaly to overlap
    kernel_depth = 1.7                # Kernel shape
    amplify_strength = 0.7            # For peak boosting
    sigma_perc = 0.02                 # Final Gaussian smoothing
    #use_center_bias = False           # Optional

    # # === Config tunning ===
    #
    # kernel_radius = None  # For fixation candidate detection
    # kernel_rad_perc_overal = 0.45
    # kernel_rad_perc_mid = 0.30
    # kernel_rad_perc = 0.17
    # kernel_rad_perc_fine = 0.00
    # max_fixation_count_overal = 1
    # max_fixation_count_mid = 2
    # max_fixation_count = 5  # Maximum number of peak centers
    # max_fixation_count_fine = 0
    # min_response_frac = 1.5
    # overlap_penalty = 0.6  # Suppress proportionaly to overlap
    # kernel_depth = 1.5  # Kernel shape
    # amplify_strength = 0.8  # For peak boosting
    # sigma_perc = 0.03  # Final Gaussian smoothing

    # === Dirs ===
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    save_dir = os.path.join(PROJECT_ROOT, "src", "generate_maps", "data", "post_proc_test")

    initial_saliency_map = saliency_map.copy()

    # === Postprocessing steps (modular) ===
    # mask = thresholding_perc(saliency_map, thresholding_percentile)

    visualize_saliency_debug(
        image=image,
        saliency_map=saliency_map,
        gt_map=gt_map,
        initial_saliency_map=None,
        stage_name="Modelio rezultatas",
        save_dir=save_dir,
        filename=f"{filename}_preproc_sal_gt.png"
    )

    fixations_overal, scores_overal, calc_radius_overal = detect_fixations_from_saliency_map_o(
        saliency_map,
        kernel_radius=kernel_radius,
        kernel_rad_perc=kernel_rad_perc_overal,
        kernel_depth = kernel_depth,
        max_fixations=max_fixation_count_overal,
        min_response_frac=min_response_frac*0.0,
        overlap_val=overlap_penalty
    )


    visualize_fixation_overlay_dual(
        saliency_map=saliency_map,
        gt_map=gt_map,
        fixation_coords=fixations_overal,
        kernel_radius=kernel_radius,
        kernel_rad_perc=kernel_rad_perc_overal,
        image=image,
        title=" ",#"Fixations (Overal fixation)",
        save_path=f"{save_dir}/{filename}_ov_fixations_overlay.png"
    )

    fixations_mid, scores_mid, calc_radius_mid = detect_fixations_from_saliency_map_o(
        saliency_map,
        kernel_radius=kernel_radius,
        kernel_rad_perc=kernel_rad_perc_mid,
        kernel_depth = kernel_depth,
        max_fixations=max_fixation_count_mid,
        min_response_frac=min_response_frac*0.0,
        overlap_val=overlap_penalty
    )


    visualize_fixation_overlay_dual(
        saliency_map=saliency_map,
        gt_map=gt_map,
        fixation_coords=fixations_mid,
        kernel_radius=kernel_radius,
        kernel_rad_perc=kernel_rad_perc_mid,
        image=image,
        title=" ",#"Fixations (Secondary fixation)",
        save_path=f"{save_dir}/{filename}_mid_fixations_overlay.png"
    )

    fixations, scores, calc_radius = detect_fixations_from_saliency_map_o(
        saliency_map,
        kernel_radius=kernel_radius,
        kernel_rad_perc=kernel_rad_perc,
        kernel_depth=kernel_depth,
        max_fixations=max_fixation_count,
        min_response_frac=min_response_frac,
        overlap_val=overlap_penalty
    )


    visualize_fixation_overlay_dual(
        saliency_map=saliency_map,
        gt_map=gt_map,
        fixation_coords=fixations,
        kernel_radius=kernel_radius,
        kernel_rad_perc=kernel_rad_perc,
        image=image,
        title=" ",#"Fixations (fixations)",
        save_path=f"{save_dir}/{filename}_fixations_overlay.png"
    )

    peak_shifted_ov = amplify_fixations_into_map(
        original_map=saliency_map,
        fixation_coords=fixations_overal,
        fixation_scores=scores_overal,
        kernel_radius=calc_radius_overal,
        kernel_depth=kernel_depth,
        amplify_strength=amplify_strength*4,
        normalize_output=True
    )

    peak_shifted_mid = amplify_fixations_into_map(
        original_map=peak_shifted_ov,
        fixation_coords=fixations_mid,
        fixation_scores=scores_mid,
        kernel_radius=calc_radius_mid,
        kernel_depth=kernel_depth,
        amplify_strength=amplify_strength*2,
        normalize_output=True
    )

    peak_shifted = amplify_fixations_into_map(
        original_map=peak_shifted_mid,
        fixation_coords=fixations,
        fixation_scores=scores,
        kernel_radius=calc_radius,
        kernel_depth=kernel_depth,
        amplify_strength=amplify_strength*1,
        normalize_output=False
    )



    visualize_fixation_overlay_dual(
        saliency_map=peak_shifted,
        gt_map=gt_map,
        fixation_coords=None,
        kernel_radius=kernel_radius,
        kernel_rad_perc=kernel_rad_perc,
        image=image,
        title=" ",#"Saliency (fully peakshifted)",
        save_path=f"{save_dir}/{filename}_peakshift_overlay.png"
    )

    # Apply Gaussian blur in float32
    mask = apply_blur(peak_shifted, sigma_perc)  # float32


    visualize_fixation_overlay_dual(
        saliency_map=mask,
        gt_map=gt_map,
        fixation_coords=None,
        kernel_radius=None,
        kernel_rad_perc=kernel_rad_perc,
        image=image,
        title=" ",#"Saliency postprocessed",
        save_path=f"{save_dir}/{filename}_postproc_overlay.png"
    )


    return mask