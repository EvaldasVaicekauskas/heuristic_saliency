
import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_gradient_magnitude

from utils import create_hue_mask


from skimage.color import rgb2lab

from skimage import graph
from skimage.feature import local_binary_pattern
from skimage.filters.rank import entropy


from skimage.segmentation import slic
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KDTree

from skimage.segmentation import flood

from skimage.filters import gaussian
from skimage.segmentation import felzenszwalb
from skimage.util import img_as_float


from skimage.morphology import disk
from skimage.measure import label
from skimage.filters import sobel
from skimage.metrics import structural_similarity as ssim

from skimage.segmentation import watershed

from skimage.feature import peak_local_max

from skimage.color import lab2rgb


from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

def detect_red_saliency(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    return cv2.normalize(red_mask, None, 0, 255, cv2.NORM_MINMAX)

### Grouping and binding

## Grouping

# Colour matching

def extract_colored_blobs_by_histogram_cc_bin(image, num_bins=2, min_blob_size=25,max_blob_size=50,sigma_percent=0.0032):
    """
    Segment image into color blobs using LAB color histogram binning (no superpixels).
    Features are assigned based on histogram bin center rather than per-blob mean.

    Returns:
        features: LAB bin center of each blob
        positions: XY centroid of each blob (normalized)
        masks: Binary masks of blobs
    """
    # 0. Blur the image
    sigma = sigma_percent*np.sqrt(image.shape[0]**2 +  image.shape[1]**2)
    image_blur = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)


    # 1. Convert to LAB space
    lab = cv2.cvtColor(image_blur, cv2.COLOR_BGR2LAB)
    lab_float = lab.astype(np.float32)

    # 2. Quantize LAB into histogram bins
    bins = np.linspace(0, 256, num_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])  # Center of each bin
    bin_indices = np.digitize(lab_float.reshape(-1, 3), bins) - 1  # Ensure valid indices
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    height, width = image.shape[:2]
    features = []
    positions = []
    masks = []

    unique_bins = np.unique(bin_indices, axis=0)
    print(f"Used color bins: {len(unique_bins)} / {num_bins ** 3} total")

    # Iterate over unique bin triplets (L*, a*, b* bin index)
    for bin_id in np.unique(bin_indices, axis=0):
        # Create mask for pixels matching this bin
        mask = np.all(bin_indices == bin_id, axis=1).reshape(height, width).astype(np.uint8)

        # Connected component analysis
        num_labels, labels = cv2.connectedComponents(mask)

        for label in range(1, num_labels):  # Skip background
            blob_mask = (labels == label).astype(np.uint8)

            if cv2.countNonZero(blob_mask) < min_blob_size:
                continue
            if cv2.countNonZero(blob_mask) > max_blob_size:
                continue

            # LAB feature is now the bin center, not mean of pixels
            lab_bin = [bin_centers[i] for i in bin_id]

            # Compute centroid position
            ys, xs = np.where(blob_mask > 0)
            cy, cx = ys.mean() / height, xs.mean() / width

            features.append(lab_bin)
            positions.append([cx, cy])
            masks.append(blob_mask)

    plot_colored_blob_masks_corrected(image, features, positions, masks, bin_indices, lab)

    return np.array(features, dtype=np.float32), np.array(positions, dtype=np.float32), masks

def grouping_color_histogram_bins(image, num_bins=6, min_blob_size_percent=0.00026,max_blob_size_percent=0.09,max_group_area_ratio=0.05):
    """
    Group blobs by LAB histogram bin and score perceptual grouping using position proximity.

    Args:
        image: Input BGR image.
        num_bins: Number of histogram bins per LAB channel.
        min_blob_size: Minimum blob size in pixels.

    Returns:
        Saliency map (uint8, 0–255).
    """


    image_area = image.shape[0] * image.shape[1]

    min_blob_size = int(min_blob_size_percent*image_area)
    max_blob_size = int(max_blob_size_percent * image_area)

    # Step 1: Extract color-bin blobs
    features, positions, masks = extract_colored_blobs_by_histogram_cc_bin(
        image, num_bins=num_bins, min_blob_size=min_blob_size,max_blob_size=max_blob_size
    )

    if len(features) == 0:
        return np.zeros(image.shape[:2], dtype=np.uint8)

    print(f"Total blobs: {len(masks)}")
    print(f"Unique color bins: {len(np.unique(features, axis=0))}")

    # Step 2: Group blobs by LAB bin value
    grouped_blobs = defaultdict(list)
    for idx, feat in enumerate(features):
        grouped_blobs[tuple(feat)].append(idx)


    # Step 3 (debugging groups)
    clusters = np.full(len(masks), -1)
    for group_id, (lab_bin, indices) in enumerate(grouped_blobs.items()):
        for idx in indices:
            clusters[idx] = group_id

    debug_cluster_overlay(image, clusters, masks)

    # Step 3: Compute saliency per group
    saliency_map = np.zeros(image.shape[:2], dtype=np.float32)

    for lab_bin, member_indices in grouped_blobs.items():
        if len(member_indices) < 3:
            continue  # Skip singleton groups
        if len(member_indices) > 45:
            continue  # Skip large groups

        # Check total area of group blobs
        group_area = sum(cv2.countNonZero(masks[idx]) for idx in member_indices)
        if group_area > max_group_area_ratio * image_area:
            continue  # Skip overly large groups (likely background)


        group_strength = compute_group_saliency(
            member_indices, features, positions,w_m=0.5, w_f=0, w_p=0.0,  mode='combined_colour'  # or 'combined'
        )

        for idx in member_indices:
            saliency_map[masks[idx] > 0] = np.maximum(saliency_map[masks[idx] > 0], group_strength)

    return cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def debug_cluster_overlay(image, clusters, masks):
    """
    Visualizes clustered blobs with distinct colors.
    """
    overlay = image.copy()

    for idx, mask in enumerate(masks):
        if clusters[idx] == -1:
            continue  # Skip noise

        color = np.array(cluster_colors(clusters[idx])[:3]) * 255
        color = color.astype(np.uint8)

        # Find where to apply overlay
        mask_indices = np.where(mask > 0)
        overlay_region = overlay[mask_indices]

        # Expand color to match shape
        color_overlay = np.tile(color, (overlay_region.shape[0], 1))

        # Blend
        blended = cv2.addWeighted(overlay_region, 0.5, color_overlay, 0.5, 0)
        overlay[mask_indices] = blended

    # Plot
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Grouped Blobs")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def cluster_colors(cluster_id):
    """
    Get a distinct color for a cluster using matplotlib's color cycle.
    """
    cmap = plt.get_cmap("tab20")  # or 'hsv', 'nipy_spectral' for more variety
    return cmap(cluster_id % 20)  # Returns RGBA (0–1 range)

def compute_group_saliency(member_indices, features, positions, w_m=1.0, w_f=1, w_p=1.0, mode='combined_colour'):
    """
    Compute saliency of a group based on its size, spatial proximity, and feature consistency.

    Args:
        member_indices: indices of blobs in the group
        features: full feature array (e.g., LAB)
        positions: full normalized (x, y) centroid positions
        mode: 'size', 'spatial', 'feature', or 'combined'

    Returns:
        Float saliency score (can be scaled later)
    """
    if len(member_indices) == 0:
        return 0.0
    if len(member_indices) == 1:
        return 1.0  # Default score for single blob

    features_only = features[member_indices]
    positions_only = positions[member_indices]

    score_size = 1*len(member_indices)
    score_spatial = 1 / (np.mean(pairwise_distances(positions_only)) + 1e-5)
    score_feature = 1 / (np.mean(pairwise_distances(features_only)) + 1e-5)

    if mode == 'size':
        return score_size
    elif mode == 'spatial':
        return score_spatial
    elif mode == 'feature':
        return score_feature
    elif mode == 'combined_colour':
        print(f"scores: {score_size,score_spatial,score_size * score_spatial}")
        return score_size * score_spatial

    elif mode == 'combined_line':
        w_size, w_spatial, w_feature = w_m, w_p, w_f

        log_score = (
            w_size    * np.log(score_size + 1e-5) +
            w_spatial * np.log(score_spatial + 1e-5) +
            w_feature * np.log(score_feature + 1e-5)
        )
        score = np.exp(log_score)  # Convert back to normal scale
        print(f"log-scaled: {score_size=}, {score_spatial=}, {score_feature=}, final={score}")
        return score
    else:
        raise ValueError(f"Unknown scoring mode: {mode}")

def plot_colored_blob_masks_corrected(image, features_lab, positions, masks, bin_indices, lab_image):
    """
    Visualizes:
    1. Raw bin-wise color masks from feature LAB values.
    2. Final blob masks colored by LAB.

    Args:
        image: Original input image (BGR).
        features_lab: LAB mean of each blob (OpenCV style).
        positions: Not used here.
        masks: List of binary masks per blob.
        bin_indices: Original bin index array (H*W, 3).
        lab_image: LAB image used to derive bin_indices.
    """
    height, width = image.shape[:2]
    lab_raw = np.zeros((height, width, 3), dtype=np.float32)
    lab_blob = np.zeros((height, width, 3), dtype=np.float32)

    # --- RAW MASKS: Go through all pixels and assign average color of bin
    flat_lab = lab_image.reshape(-1, 3)
    flat_lab_raw = lab_raw.reshape(-1, 3)

    # Unique bin ids and their mean colors
    unique_bins = np.unique(bin_indices, axis=0)
    bin_colors = {}

    for bin_id in unique_bins:
        mask = np.all(bin_indices == bin_id, axis=1)
        mean_lab = flat_lab[mask].mean(axis=0)
        lab_sk = np.array([
            mean_lab[0] * (100.0 / 255.0),
            mean_lab[1] - 128.0,
            mean_lab[2] - 128.0
        ], dtype=np.float32)
        bin_colors[tuple(bin_id)] = lab_sk
        flat_lab_raw[mask] = lab_sk

    lab_raw = flat_lab_raw.reshape(height, width, 3)

    # --- BLOB MASKS
    for lab_color, mask in zip(features_lab, masks):
        lab_sk = np.array([
            lab_color[0] * (100.0 / 255.0),
            lab_color[1] - 128.0,
            lab_color[2] - 128.0
        ], dtype=np.float32)
        lab_blob[mask.astype(bool)] = lab_sk

    # Convert and clip to avoid visual artifacts
    rgb_raw = np.clip(lab2rgb(lab_raw), 0, 1)
    rgb_blob = np.clip(lab2rgb(lab_blob), 0, 1)

    rgb_raw_vis = (rgb_raw * 255).astype(np.uint8)
    rgb_blob_vis = (rgb_blob * 255).astype(np.uint8)

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(14, 6))
    axs[0].imshow(rgb_raw_vis)
    axs[0].set_title("Raw Color Bin Masks")
    axs[0].axis("off")

    axs[1].imshow(rgb_blob_vis)
    axs[1].set_title("Connected Color Blobs")
    axs[1].axis("off")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    axs[2].imshow(image_rgb)
    axs[2].set_title("Original image")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()

# Line grouping

def line_extraction(image, sigma_percent = 0.0016*0.5, min_length_ratio=0.01, max_length_ratio=0.5):

    # 0. Blur the image
    sigma = sigma_percent * np.sqrt(image.shape[0] ** 2 + image.shape[1] ** 2)
    image_blur = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)

    lsd = cv2.createLineSegmentDetector(refine=2)
    lines = lsd.detect(gray)[0]

    minLineLength = 0.047*np.sqrt(image.shape[0] ** 2 + image.shape[1] ** 2)
    maxLineGap = 0.011*np.sqrt(image.shape[0] ** 2 + image.shape[1] ** 2)

    edges = cv2.Canny(gray, 60, 200)
    lines = cv2.HoughLinesP(edges, rho=3, theta=np.pi / 180, threshold=40,
                            minLineLength=minLineLength, maxLineGap=maxLineGap)


    features = []
    positions = []
    masks = []

    min_length = min_length_ratio * max(width, height)
    max_length = max_length_ratio * max(width, height)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.hypot(x2 - x1, y2 - y1)
            if length < min_length or length > max_length:
                continue

            orientation = np.arctan2(y2 - y1, x2 - x1)
            normalized_length = length / max(width, height)
            mx, my = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height

            # Create binary mask for this line using OpenCV line with thickness
            mask = np.zeros((height, width), dtype=np.uint8)
            thickness = int(np.clip(normalized_length * max(width, height) * 0.01, 3, 20))
            cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), color=255, thickness=thickness)

            features.append([orientation, normalized_length])
            positions.append([mx, my])
            masks.append(mask)

        return np.array(features, dtype=np.float32), np.array(positions, dtype=np.float32), masks

def group_lines_by_orientation(features, masks, positions, image_shape, num_bins=12):
    orientation_bins = np.linspace(-np.pi, np.pi, num_bins + 1)
    grouped = defaultdict(list)

    for i, (orientation, length) in enumerate(features):
        bin_idx = np.digitize(orientation, orientation_bins) - 1
        bin_idx = np.clip(bin_idx, 0, num_bins - 1)
        grouped[bin_idx].append(i)

    clusters = np.full(len(masks), -1)
    for group_id, (bin_idx, indices) in enumerate(grouped.items()):
        for idx in indices:
            clusters[idx] = group_id

    # Visualization
    overlay = np.zeros((*image_shape[:2], 3), dtype=np.uint8)
    colors = plt.cm.get_cmap("tab20", num_bins)
    for i, mask in enumerate(masks):
        if clusters[i] == -1:
            continue
        color = np.array(colors(clusters[i])[:3]) * 255
        overlay[mask > 0] = color.astype(np.uint8)

    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Line Orientation Groups")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return clusters

def grouping_line_hough(image, num_bins=36, max_lines = 15):

    features, positions, masks = line_extraction(image)

    if len(features) == 0:
        return np.zeros(image.shape[:2], dtype=np.uint8)

    clusters = group_lines_by_orientation(features, masks, positions, image.shape, num_bins=num_bins)

    saliency_map = np.zeros(image.shape[:2], dtype=np.float32)
    for group_id in np.unique(clusters):
        if group_id == -1:
            continue

        member_indices = np.where(clusters == group_id)[0]
        if len(member_indices) < 2:
            continue
        if len(member_indices) > max_lines:
            continue

        group_strength = compute_group_saliency(member_indices, features, positions,w_m=0.4, w_f=1, w_p=0.3,  mode='combined_line')

        for idx in member_indices:
            saliency_map[masks[idx] > 0] = np.maximum(saliency_map[masks[idx] > 0], group_strength)

    return cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)



### Isolation

## Proximity

# Proximity grouping (colour space) + global outliers

def proximity_colour_loc_glob(image, n_segments=200, compactness=10, local_radius=0.2):
    """
    Compute saliency based on perceptual grouping and contrast between superpixels,
    using both global distinctiveness and local contrast based on spatial proximity.

    Args:
        image: Input RGB image (BGR if OpenCV-loaded)
        n_segments: Number of superpixels for SLIC
        compactness: Compactness parameter for SLIC
        local_radius: Normalized radius in [0, 1] for local contrast (Euclidean distance in image space)

    Returns:
        saliency_map: Normalized uint8 saliency map
    """
    height, width = image.shape[:2]

    # Convert to LAB for perceptual color distance
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    segments = slic(image, n_segments=n_segments, compactness=compactness, start_label=0)

    features = []
    positions = []
    segment_ids = np.unique(segments)

    for seg_id in segment_ids:
        mask = (segments == seg_id)
        color = lab_image[mask].mean(axis=0)  # L, A, B
        y, x = np.argwhere(mask).mean(axis=0)  # Position
        features.append(color)
        positions.append([x / width, y / height])  # Normalized positions

    features = np.array(features)
    positions = np.array(positions)

    # Normalize feature vectors
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # --- Global Contrast: Euclidean distance to global mean
    global_scores = np.linalg.norm(features_scaled, axis=1)

    # --- Local Contrast: distance-weighted feature difference from nearby regions
    tree = KDTree(positions)
    neighbors = tree.query_radius(positions, r=local_radius)

    local_scores = []
    for i, neighbor_ids in enumerate(neighbors):
        diffs = []
        for j in neighbor_ids:
            if i == j:
                continue
            dist = np.linalg.norm(positions[i] - positions[j]) + 1e-6  # avoid div by 0
            feature_dist = np.linalg.norm(features_scaled[i] - features_scaled[j])
            diffs.append(feature_dist / dist)
        if diffs:
            local_scores.append(np.mean(diffs))
        else:
            local_scores.append(0.0)

    local_scores = np.array(local_scores)

    # --- Combine scores
    final_scores = 0.6 * global_scores + 0.4 * local_scores

    # --- Map scores back to saliency map
    saliency_map = np.zeros((height, width), dtype=np.float32)
    for seg_id, score in zip(segment_ids, final_scores):
        saliency_map[segments == seg_id] = score

    saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX)
    return saliency_map.astype(np.uint8)

def proximity_colour_global(image, n_segments=200, compactness=10):
    """
    Compute saliency based on global contrast of LAB color features across superpixels.
    Returns a saliency map where regions distinct from the global mean are more salient.
    """
    height, width = image.shape[:2]
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    segments = slic(image, n_segments=n_segments, compactness=compactness, start_label=0)

    features = []
    segment_ids = np.unique(segments)

    for seg_id in segment_ids:
        mask = (segments == seg_id)
        color = lab_image[mask].mean(axis=0)  # L, A, B
        features.append(color)

    features = np.array(features)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    global_scores = np.linalg.norm(features_scaled, axis=1)

    saliency_map = np.zeros((height, width), dtype=np.float32)
    for seg_id, score in zip(segment_ids, global_scores):
        saliency_map[segments == seg_id] = score

    saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX)
    return saliency_map.astype(np.uint8)

def proximity_colour_local(image, n_segments=200, compactness=10, local_radius=0.2):
    """
    Compute saliency based on local contrast between superpixels,
    using spatial proximity (not just adjacency).
    """
    height, width = image.shape[:2]
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    segments = slic(image, n_segments=n_segments, compactness=compactness, start_label=0)

    features = []
    positions = []
    segment_ids = np.unique(segments)

    for seg_id in segment_ids:
        mask = (segments == seg_id)
        color = lab_image[mask].mean(axis=0)
        y, x = np.argwhere(mask).mean(axis=0)
        features.append(color)
        positions.append([x / width, y / height])

    features = np.array(features)
    positions = np.array(positions)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    tree = KDTree(positions)
    neighbors = tree.query_radius(positions, r=local_radius)

    local_scores = []
    for i, neighbor_ids in enumerate(neighbors):
        diffs = []
        for j in neighbor_ids:
            if i == j:
                continue
            dist = np.linalg.norm(positions[i] - positions[j]) + 1e-6
            feature_dist = np.linalg.norm(features_scaled[i] - features_scaled[j])
            diffs.append(feature_dist / dist)
        if diffs:
            local_scores.append(np.mean(diffs))
        else:
            local_scores.append(0.0)

    saliency_map = np.zeros((height, width), dtype=np.float32)
    for seg_id, score in zip(segment_ids, local_scores):
        saliency_map[segments == seg_id] = score

    saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX)
    return saliency_map.astype(np.uint8)

# Feature

def color_lab_deviation(image):
    image_float = img_as_float(image)
    lab = rgb2lab(image_float)
    mean_color = np.mean(lab.reshape(-1, 3), axis=0)
    diff = np.linalg.norm(lab - mean_color, axis=2)
    return diff.astype(np.float32)

def intensity_map(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32)

def texture_entropy(image, neighborhood_radius=10):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ent = entropy(gray, disk(neighborhood_radius))
    return ent.astype(np.float32)

def texture_lbp(image, radius=35, n_points=3, method='nri_uniform'):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius, method)
    return lbp.astype(np.float32)

def texture_gabor_energy(image, orientations=8, scales=[4, 8, 16], ksize=15, sigma=4.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    rows, cols = gray.shape
    energy_map = np.zeros((rows, cols), dtype=np.float32)

    for theta in np.linspace(0, np.pi, orientations, endpoint=False):
        for lambd in scales:
            kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma=0.5, psi=0)
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            energy_map += filtered ** 2

    return energy_map

# Context

def global_mean_context(feature_map):
    return np.full_like(feature_map, np.mean(feature_map))

def local_blur_context(feature_map, sigma_ratio=0.005):
    h, w = feature_map.shape[:2]
    sigma = sigma_ratio * (h + w) / 2
    return cv2.GaussianBlur(feature_map, (0, 0), sigma)

def local_multi_scale_context(feature_map, sigma_ratios=[0.005, 0.015, 0.030], weights=None):
    """
    Build a context map from multi-scale blurred versions.
    Args:
        feature_map: 2D float32 array.
        sigmas: list of gaussian blur sigmas (larger = more global).
        weights: optional list of weights per sigma (defaults to 1/sigma).
    """
    h, w = feature_map.shape[:2]
    sigmas = [r * (h + w) / 2 for r in sigma_ratios]

    if weights is None:
        weights = [1.0 / s for s in sigmas]  # smaller sigma = stronger weight

    context_sum = np.zeros_like(feature_map, dtype=np.float32)
    total_weight = 0.0

    for sigma, weight in zip(sigmas, weights):
        blurred = cv2.GaussianBlur(feature_map, (0, 0), sigma)
        difference = np.abs(feature_map - blurred)
        context_sum += weight * difference
        total_weight += weight

    return context_sum / total_weight

# Scoring

def score_absolute_difference(feature_map, context_map):
    return np.abs(feature_map - context_map)

def score_zscore(feature_map, context_map):
    std = np.std(feature_map)
    return (feature_map - context_map) / std if std > 0 else np.zeros_like(feature_map)

def score_threshold_top_percent(feature_map, context_map, percentile=95):
    difference = score_absolute_difference(feature_map, context_map)
    threshold = np.percentile(difference, percentile)
    return (difference >= threshold).astype(np.uint8) * 255

# Mapping

def normalize_to_uint8(saliency):
    return cv2.normalize(saliency, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Full

# Texture_gabor_energy
def isolation_tex_gab_global_abs(image):
    feature = texture_gabor_energy(image)
    context = global_mean_context(feature)
    score = score_absolute_difference(feature, context)
    return normalize_to_uint8(score)


def isolation_tex_gab_local_abs(image):
    feature = texture_gabor_energy(image)
    context = local_blur_context(feature)
    score = score_absolute_difference(feature, context)
    return normalize_to_uint8(score)


def isolation_tex_gab_local_multi(image):
    feature = texture_gabor_energy(image)
    score = local_multi_scale_context(feature, weights=[0.5, 0.3, 0.2])
    return normalize_to_uint8(score)

# Texture_LBP
def isolation_tex_lbp_global_abs(image):
    feature = texture_lbp(image)
    context = global_mean_context(feature)
    score = score_absolute_difference(feature, context)
    return normalize_to_uint8(score)

def isolation_tex_lbp_local_abs(image):
    feature = texture_lbp(image)
    context = local_blur_context(feature)
    score = score_absolute_difference(feature, context)
    return normalize_to_uint8(score)


def isolation_tex_lbp_local_multi(image):
    feature = texture_lbp(image)
    score = local_multi_scale_context(feature, weights=[0.5, 0.3, 0.2])
    return normalize_to_uint8(score)

# Texture_entropy
def isolation_tex_entropy_global_abs(image):
    feature = texture_entropy(image)
    context = global_mean_context(feature)
    score = score_absolute_difference(feature, context)
    return normalize_to_uint8(score)

def isolation_tex_entropy_local_abs(image):
    feature = texture_entropy(image)
    context = local_blur_context(feature)
    score = score_absolute_difference(feature, context)
    return normalize_to_uint8(score)

def isolation_tex_entropy_local_multi(image):
    feature = texture_entropy(image)
    score = local_multi_scale_context(feature, weights=[0.5, 0.3, 0.2])
    return normalize_to_uint8(score)

# Intensity
def isolation_intensity_global_abs(image):
    feature = intensity_map(image)
    context = global_mean_context(feature)
    score = score_absolute_difference(feature, context)
    return normalize_to_uint8(score)

def isolation_intensity_local_abs(image):
    feature = intensity_map(image)
    context = local_blur_context(feature)
    score = score_absolute_difference(feature, context)
    return normalize_to_uint8(score)

def isolation_intensity_local_multi(image):
    feature = intensity_map(image)
    score = local_multi_scale_context(feature, weights=[0.5, 0.3, 0.2])
    return normalize_to_uint8(score)

# LAB
def isolation_color_global_abs(image):
    feature = color_lab_deviation(image)
    context = global_mean_context(feature)
    score = score_absolute_difference(feature, context)
    return normalize_to_uint8(score)

def isolation_color_local_abs(image):
    feature = color_lab_deviation(image)
    context = local_blur_context(feature)
    score = score_absolute_difference(feature, context)
    return normalize_to_uint8(score)

def isolation_color_local_multi(image):
    feature = color_lab_deviation(image)
    score = local_multi_scale_context(feature, weights=[0.5, 0.3, 0.2])
    return normalize_to_uint8(score)


### Contrast

## Edges

# Luminance = Intensity?

def contrast_edge_luminance_Laplacian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast_map = cv2.Laplacian(gray, cv2.CV_64F)
    contrast_map = np.abs(contrast_map)
    return cv2.normalize(contrast_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def contrast_edge_luminance_sobel(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=25)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=25)
    sobel_mag = cv2.magnitude(sobelx, sobely)
    sobel_norm = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX)
    return sobel_norm.astype(np.uint8)

## Colour contrast (chromatic edges) - LAB

# sobel - pixel value

def contrast_edge_colour_sobel(image):
    """
    Detect color-based edges by applying Sobel operator to A and B channels in LAB space.
    Returns a normalized grayscale map of color edge strength.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    _, a, b = cv2.split(lab)

    # Apply Sobel to A and B channels
    sobel_a = cv2.Sobel(a, cv2.CV_64F, 1, 1, ksize=25)
    sobel_b = cv2.Sobel(b, cv2.CV_64F, 1, 1, ksize=25)

    # Combine both gradient magnitudes
    color_edge = cv2.magnitude(sobel_a, sobel_b)

    # Normalize to 8-bit image
    norm_edge = cv2.normalize(color_edge, None, 0, 255, cv2.NORM_MINMAX)
    return norm_edge.astype(np.uint8)

# Local color gradient

def contrast_edge_color_gradient(image, sigma=1.5):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    _, a, b = cv2.split(lab)

    # Compute gradient magnitude in A and B channels
    grad_a = gaussian_gradient_magnitude(a.astype(np.float32), sigma=sigma)
    grad_b = gaussian_gradient_magnitude(b.astype(np.float32), sigma=sigma)

    # Combine gradients
    grad_color = np.sqrt(grad_a**2 + grad_b**2)

    norm_grad = cv2.normalize(grad_color, None, 0, 255, cv2.NORM_MINMAX)
    return norm_grad.astype(np.uint8)


# Super-pixel region

def contrast_region_color_slic(image, num_segments=240, base_threshold=90, colorfulness=0):
    """Detect color contrast using superpixel segmentation (SLIC) and ΔE distance."""
    # Convert image to LAB color space and float format for SLIC
    image_float = img_as_float(image)
    lab_image = rgb2lab(image_float)

    # Scaled contrast_threshold
    contrast_threshold = base_threshold * (1 + (colorfulness / 30))  # Example scaling
    #print(f"contrast_threshold: {contrast_threshold}, for colourfulness: {colorfulness}")

    # Apply SLIC superpixel segmentation
    segments = slic(image_float, n_segments=num_segments, compactness=50, sigma=5, start_label=1)

    # Compute mean LAB color for each superpixel
    mean_colors = {}
    for segment_id in np.unique(segments):
        mask = segments == segment_id
        mean_color = np.mean(lab_image[mask], axis=0)
        mean_colors[segment_id] = mean_color

    # Create a Region Adjacency Graph (RAG) for neighboring superpixels
    rag = graph.rag_mean_color(lab_image, segments)

    # Initialize contrast map
    contrast_map = np.zeros(segments.shape, dtype=np.uint8)

    # Calculate color contrast between neighboring superpixels
    for edge in rag.edges:
        region1, region2 = edge
        color1 = mean_colors[region1]
        color2 = mean_colors[region2]
        # Compute Euclidean distance (Delta E)
        color_distance = np.linalg.norm(color1 - color2)

        if color_distance > contrast_threshold:
            # Highlight high contrast areas
            dominant_region = region1 if np.linalg.norm(mean_colors[region1]) > np.linalg.norm(mean_colors[region2]) else region2
            contrast_map[segments == dominant_region] = 255

    return contrast_map

# Hue opposition - complementary color mapping



# Some extra colour based methods:


# Hue histogram separation (e.g. between center-surround patches)
# Colorfulness boost via metrics (Hasler & Süsstrunk, 2003)
# Opponent color channels (RG/BY) as in early biological models (Itti et al.)

## Texture

# Texture contrast (Parkhurst)

def contrast_texture_gabor(image, orientations=8, scales=3, ksize=15, sigma=4.0):
    """
    Compute texture contrast using a bank of Gabor filters and local energy difference.
    Inspired by Parkhurst et al. (2004)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32) / 255.0  # Normalize

    rows, cols = gray.shape
    texture_energy = np.zeros((rows, cols), dtype=np.float32)

    # Define orientations and frequencies
    thetas = np.linspace(0, np.pi, orientations, endpoint=False)
    lambdas = [4, 8, 16]  # wavelengths (scales)

    for theta in thetas:
        for lambd in lambdas:
            kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma=0.5, psi=0)
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            energy = np.square(filtered)
            texture_energy += energy

    # Local contrast: subtract blurred version
    local_avg = cv2.GaussianBlur(texture_energy, (15, 15), 5)
    contrast_map = np.abs(texture_energy - local_avg)

    # Normalize to [0,255] for visualization
    contrast_map = cv2.normalize(contrast_map, None, 0, 255, cv2.NORM_MINMAX)
    return contrast_map.astype(np.uint8)

# Local Binary Patterns

def contrast_texture_lbp(image, radius=35, n_points=3, method='nri_uniform', post_threshold=180):
    """
    Local Binary Pattern-based saliency (micro-texture).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius, method)
    # Normalize
    lbp_norm = (lbp - lbp.min()) / (lbp.max() - lbp.min()) * 255
    lbp_norm = lbp_norm.astype(np.uint8)

    # Apply post-threshold to suppress low-variation areas
    lbp_thresh = np.where(lbp_norm >= post_threshold, lbp_norm, 0).astype(np.uint8)

    return lbp_thresh

# Entropy based (texture complexity)
# (Chaos is salient) "neuroaesthetics"

def contrast_texture_entropy(image, neighborhood_radius=5,entropy_threshold=5.0):
    """
    Entropy-based saliency map (texture complexity).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ent = entropy(gray, disk(neighborhood_radius))

    # Zero-out values below entropy threshold
    ent_min = ent.min()
    ent_max = ent.max()
    ent[ent < entropy_threshold] = 0

    # Normalization
    ent = (ent - ent_min) / (ent_max - ent_min) * 255
    return ent.astype(np.uint8)






# Complementary colour mapping

def detect_complementary_color_saliency(image, top_n=3, hue_range=10, proximity_thickness=25):
        """Detect complementary color regions and highlight the complementary regions near dominant areas."""
        # Convert to HSV and extract hue channel
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue_channel = hsv_image[:, :, 0]
        saturation_channel = hsv_image[:, :, 1]
        value_channel = hsv_image[:, :, 2]

        # Define thresholds for filtering non-vivid colors
        saturation_threshold = 80  # Minimum saturation to be considered "vivid"
        value_threshold = 90  # Avoid near-black areas

        # Create a mask for pixels that are sufficiently colorful
        vivid_mask = cv2.bitwise_and(
            cv2.inRange(saturation_channel, saturation_threshold, 255),
            cv2.inRange(value_channel, value_threshold, 255)
        )

        # Apply vivid color mask to hue channel before processing
        filtered_hue_channel = cv2.bitwise_and(hue_channel, hue_channel, mask=vivid_mask)

        # Create hue histogram
        hist_resolution = 1
        num_bins = 180 // hist_resolution
        hist = cv2.calcHist([filtered_hue_channel], [0], None, [num_bins], [0, 180])

        # Select top N dominant hues, avoiding repeated complementary pairs
        dominant_hues = []
        used_complementaries = set()

        dominant_hue_bins = np.argsort(hist.flatten())[::-1]
        for bin_index in dominant_hue_bins:
            candidate_hue = bin_index * hist_resolution
            complementary_hue = (candidate_hue + 90) % 180  # 180° shift for complementary color

            if complementary_hue not in used_complementaries:
                dominant_hues.append(candidate_hue)
                used_complementaries.add(candidate_hue)
                used_complementaries.add(complementary_hue)

            if len(dominant_hues) >= top_n:
                break

        # Initialize saliency map
        complementary_saliency_map = np.zeros_like(filtered_hue_channel, dtype=np.uint8)

        for dominant_hue in dominant_hues:
            complementary_hue = (dominant_hue + 90) % 180

            # Create binary masks for dominant and complementary hues
            dominant_mask = create_hue_mask(filtered_hue_channel, dominant_hue, hue_range)
            complementary_mask = create_hue_mask(filtered_hue_channel, complementary_hue, hue_range)

            # Find contours for dominant regions
            contours, _ = cv2.findContours(dominant_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Create a proximity mask around dominant regions
            proximity_mask = np.zeros_like(complementary_mask)
            for cnt in contours:
                cv2.drawContours(proximity_mask, [cnt], -1, 255, thickness=proximity_thickness)

            # Highlight complementary regions that are near dominant regions
            nearby_complementary = cv2.bitwise_and(complementary_mask, proximity_mask)
            complementary_saliency_map = cv2.bitwise_or(complementary_saliency_map, nearby_complementary)

        return complementary_saliency_map

### Symmetry

# objects_vertical

symmetry_axis = 'vertical'

def symmetry_superpixel_reflection(image, num_segments=40, symmetry_axis='vertical', min_region_size=30,threshold=0.55):
    image_float = img_as_float(image)
    segments = slic(image_float, n_segments=num_segments, compactness=10, sigma=4, start_label=1)

    output_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for seg_val in np.unique(segments):
        mask = (segments == seg_val).astype(np.uint8) * 255

        # Skip small regions
        if cv2.countNonZero(mask) < min_region_size:
            continue

        x, y, w, h = cv2.boundingRect(mask)
        region = mask[y:y+h, x:x+w]

        # Mirror along axis
        if symmetry_axis == 'vertical':
            mirrored = cv2.flip(region, 1)
        elif symmetry_axis == 'horizontal':
            mirrored = cv2.flip(region, 0)
        else:
            raise ValueError("symmetry_axis must be 'vertical' or 'horizontal'")

        # Crop to the smallest overlapping shape
        min_h, min_w = min(region.shape[0], mirrored.shape[0]), min(region.shape[1], mirrored.shape[1])
        region = region[:min_h, :min_w]
        mirrored = mirrored[:min_h, :min_w]

        # Compare region and its reflection
        score = compute_symmetry_score(region, symmetry_axis, threshold=0.0)

        #score=compute_symmetry_score(mask,symmetry_axis,threshold)
        if score > threshold:
            output_mask[y:y+h, x:x+w][region > 0] = 255*score

    return output_mask

def symmetry_superpixel_reflection_preproc(image, num_segments=40, symmetry_axis='vertical', min_region_size=30,threshold=0.55):
    image_float = img_as_float(image)
    #segments = slic(image_float, n_segments=num_segments, compactness=10, sigma=4, start_label=1)
    segments = preprocess_and_segment_superpixels(image, num_segments=60, compactness=10, sigma=1)
    output_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for seg_val in np.unique(segments):
        mask = (segments == seg_val).astype(np.uint8) * 255

        # Skip small regions
        if cv2.countNonZero(mask) < min_region_size:
            continue

        x, y, w, h = cv2.boundingRect(mask)
        region = mask[y:y+h, x:x+w]

        # Mirror along axis
        if symmetry_axis == 'vertical':
            mirrored = cv2.flip(region, 1)
        elif symmetry_axis == 'horizontal':
            mirrored = cv2.flip(region, 0)
        else:
            raise ValueError("symmetry_axis must be 'vertical' or 'horizontal'")

        # Crop to the smallest overlapping shape
        min_h, min_w = min(region.shape[0], mirrored.shape[0]), min(region.shape[1], mirrored.shape[1])
        region = region[:min_h, :min_w]
        mirrored = mirrored[:min_h, :min_w]

        # Compare region and its reflection
        #score = ssim(region, mirrored)
        score = compute_symmetry_score(region, symmetry_axis, threshold=0.0)
        if score > threshold:
            output_mask[y:y+h, x:x+w][region > 0] = 255*score

    return output_mask

def symmetry_seeded_region_reflection(image, symmetry_axis='vertical', min_region_size=30, sigma=5,threshold=0.55):
    """
    Region symmetry detection using seeded region growing from intensity maxima.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blurred = gaussian(gray, sigma=sigma)

    # Use local maxima as seeds (ensure float64 image for peak_local_max)
    coordinates = peak_local_max(gray_blurred, min_distance=20, threshold_abs=0.3)

    output_mask = np.zeros_like(gray, dtype=np.uint8)

    for y, x in coordinates:
        # Perform flood fill from the seed point
        region = flood(gray, seed_point=(y, x), tolerance=25)

        # Skip if region is invalid or empty
        if region is None or not np.any(region):
            continue

        # Create binary mask
        region_mask = np.zeros_like(gray, dtype=np.uint8)
        region_mask[region] = 255

        # Skip small regions
        if cv2.countNonZero(region_mask) < min_region_size:
            continue

        # Bounding box for the region
        x0, y0, w, h = cv2.boundingRect(region_mask)
        cropped = region_mask[y0:y0+h, x0:x0+w]

        # Reflect the cropped region
        if symmetry_axis == 'vertical':
            mirrored = cv2.flip(cropped, 1)
        elif symmetry_axis == 'horizontal':
            mirrored = cv2.flip(cropped, 0)
        else:
            raise ValueError("symmetry_axis must be 'vertical' or 'horizontal'")

        # Ensure same shape for comparison
        min_h, min_w = min(cropped.shape[0], mirrored.shape[0]), min(cropped.shape[1], mirrored.shape[1])
        cropped = cropped[:min_h, :min_w]
        mirrored = mirrored[:min_h, :min_w]

        # Skip tiny regions that SSIM can't process
        if min_h < 7 or min_w < 7:
            continue

        # Compare using SSIM
        #score = ssim(cropped, mirrored, data_range=255)
        score = compute_symmetry_score(cropped, symmetry_axis, threshold=0.0)
        if score > threshold:
            region_out = output_mask[y0:y0+min_h, x0:x0+min_w]
            region_out[cropped > 0] = np.maximum(region_out[cropped > 0], int(255 * score))
        # if score > 0.55:
        #     output_mask[y:y+h, x:x+w][region > 0] = 255*score

    return output_mask

def symmetry_felzenszwalb_reflection(image, scale=100, sigma=3, min_size=100, symmetry_axis='vertical',
                                     min_region_size=100,threshold=0.55):
    """
    Detect symmetric regions using Felzenszwalb segmentation.

    Args:
        image: Input BGR image
        scale: Float. Higher means larger clusters.
        sigma: Smoothing prior to segmentation
        min_size: Minimum component size
        symmetry_axis: 'vertical' or 'horizontal'
        min_region_size: Ignore regions smaller than this

    Returns:
        output_mask: Symmetry saliency map
    """
    image_float = img_as_float(image)
    segments = felzenszwalb(image_float, scale=scale, sigma=sigma, min_size=min_size)

    output_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for seg_val in np.unique(segments):
        mask = (segments == seg_val).astype(np.uint8) * 255

        if cv2.countNonZero(mask) < min_region_size:
            continue

        x, y, w, h = cv2.boundingRect(mask)
        region = mask[y:y + h, x:x + w]

        # Mirror the region
        if symmetry_axis == 'vertical':
            mirrored = cv2.flip(region, 1)
        elif symmetry_axis == 'horizontal':
            mirrored = cv2.flip(region, 0)
        else:
            raise ValueError("symmetry_axis must be 'vertical' or 'horizontal'")

        # Ensure compatible shapes
        min_h, min_w = min(region.shape[0], mirrored.shape[0]), min(region.shape[1], mirrored.shape[1])
        region = region[:min_h, :min_w]
        mirrored = mirrored[:min_h, :min_w]

        if min_h < 7 or min_w < 7:
            continue  # too small for SSIM

        #score = ssim(region, mirrored, data_range=255)
        score = compute_symmetry_score(region, symmetry_axis, threshold=0.0)
        if score > threshold:
            region_out = output_mask[y:y + min_h, x:x + min_w]
            region_out[region > 0] = np.maximum(region_out[region > 0], (255 * score).astype(np.uint8))

    return output_mask

def symmetry_watershed_reflection(image, symmetry_axis='vertical', min_region_size=100, sigma_blur=4,threshold=0.55):
    """
    Detect symmetric regions using Watershed segmentation.

    Args:
        image: Input BGR image.
        symmetry_axis: 'vertical' or 'horizontal'
        min_region_size: Minimum region size to keep.
        sigma_blur: Gaussian blur sigma before edge detection

    Returns:
        output_mask: Symmetry saliency map
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: Edge map via Sobel
    gradient = sobel(gray)

    # Step 2: Markers from local maxima in distance transform
    distance = cv2.GaussianBlur(gray, (0, 0), sigma_blur)
    distance = cv2.distanceTransform((gray > 40).astype(np.uint8), cv2.DIST_L2, 5)
    coordinates = peak_local_max(distance, footprint=np.ones((65, 65)), labels=gray)
    local_maxi = np.zeros_like(gray, dtype=bool)
    local_maxi[tuple(coordinates.T)] = True
    markers = label(local_maxi)

    # Step 3: Apply watershed
    segments = watershed(gradient, markers, mask=gray > 50)

    output_mask = np.zeros_like(gray, dtype=np.uint8)

    for seg_val in np.unique(segments):
        if seg_val == 0:
            continue

        mask = (segments == seg_val).astype(np.uint8) * 255
        if cv2.countNonZero(mask) < min_region_size:
            continue

        x, y, w, h = cv2.boundingRect(mask)
        region = mask[y:y+h, x:x+w]

        if symmetry_axis == 'vertical':
            mirrored = cv2.flip(region, 1)
        elif symmetry_axis == 'horizontal':
            mirrored = cv2.flip(region, 0)
        else:
            raise ValueError("symmetry_axis must be 'vertical' or 'horizontal'")

        min_h, min_w = min(region.shape[0], mirrored.shape[0]), min(region.shape[1], mirrored.shape[1])
        region = region[:min_h, :min_w]
        mirrored = mirrored[:min_h, :min_w]

        if min_h < 7 or min_w < 7:
            continue

        #scoreo = ssim(region, mirrored, data_range=255)
        score = compute_symmetry_score(region,symmetry_axis,threshold=0.0)

        if score > threshold:
            region_out = output_mask[y:y+min_h, x:x+min_w]
            region_out[region > 0] = np.maximum(region_out[region > 0], (255 * score).astype(np.uint8))

    return output_mask

# objects_horizontal

symmetry_axis = 'horizontal'

def symmetry_superpixel_reflection_h(image, num_segments=40, symmetry_axis='horizontal', min_region_size=30,threshold=0.55):
    image_float = img_as_float(image)
    segments = slic(image_float, n_segments=num_segments, compactness=10, sigma=4, start_label=1)

    output_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for seg_val in np.unique(segments):
        mask = (segments == seg_val).astype(np.uint8) * 255

        # Skip small regions
        if cv2.countNonZero(mask) < min_region_size:
            continue

        x, y, w, h = cv2.boundingRect(mask)
        region = mask[y:y+h, x:x+w]

        # Mirror along axis
        if symmetry_axis == 'vertical':
            mirrored = cv2.flip(region, 1)
        elif symmetry_axis == 'horizontal':
            mirrored = cv2.flip(region, 0)
        else:
            raise ValueError("symmetry_axis must be 'vertical' or 'horizontal'")

        # Crop to the smallest overlapping shape
        min_h, min_w = min(region.shape[0], mirrored.shape[0]), min(region.shape[1], mirrored.shape[1])
        region = region[:min_h, :min_w]
        mirrored = mirrored[:min_h, :min_w]

        # Compare region and its reflection
        score = compute_symmetry_score(region, symmetry_axis, threshold=0.0)

        #score=compute_symmetry_score(mask,symmetry_axis,threshold)
        if score > threshold:
            output_mask[y:y+h, x:x+w][region > 0] = 255*score

    return output_mask


def symmetry_superpixel_reflection_preproc_h(image, num_segments=40, symmetry_axis='horizontal', min_region_size=30,threshold=0.55):
    image_float = img_as_float(image)
    #segments = slic(image_float, n_segments=num_segments, compactness=10, sigma=4, start_label=1)
    segments = preprocess_and_segment_superpixels(image, num_segments=60, compactness=10, sigma=1)
    output_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for seg_val in np.unique(segments):
        mask = (segments == seg_val).astype(np.uint8) * 255

        # Skip small regions
        if cv2.countNonZero(mask) < min_region_size:
            continue

        x, y, w, h = cv2.boundingRect(mask)
        region = mask[y:y+h, x:x+w]

        # Mirror along axis
        if symmetry_axis == 'vertical':
            mirrored = cv2.flip(region, 1)
        elif symmetry_axis == 'horizontal':
            mirrored = cv2.flip(region, 0)
        else:
            raise ValueError("symmetry_axis must be 'vertical' or 'horizontal'")

        # Crop to the smallest overlapping shape
        min_h, min_w = min(region.shape[0], mirrored.shape[0]), min(region.shape[1], mirrored.shape[1])
        region = region[:min_h, :min_w]
        mirrored = mirrored[:min_h, :min_w]

        # Compare region and its reflection
        #score = ssim(region, mirrored)
        score = compute_symmetry_score(region, symmetry_axis, threshold=0.0)
        if score > threshold:
            output_mask[y:y+h, x:x+w][region > 0] = 255*score

    return output_mask

def symmetry_seeded_region_reflection_h(image, symmetry_axis='horizontal', min_region_size=30, sigma=5,threshold=0.55):
    """
    Region symmetry detection using seeded region growing from intensity maxima.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blurred = gaussian(gray, sigma=sigma)

    # Use local maxima as seeds (ensure float64 image for peak_local_max)
    coordinates = peak_local_max(gray_blurred, min_distance=20, threshold_abs=0.3)

    output_mask = np.zeros_like(gray, dtype=np.uint8)

    for y, x in coordinates:
        # Perform flood fill from the seed point
        region = flood(gray, seed_point=(y, x), tolerance=25)

        # Skip if region is invalid or empty
        if region is None or not np.any(region):
            continue

        # Create binary mask
        region_mask = np.zeros_like(gray, dtype=np.uint8)
        region_mask[region] = 255

        # Skip small regions
        if cv2.countNonZero(region_mask) < min_region_size:
            continue

        # Bounding box for the region
        x0, y0, w, h = cv2.boundingRect(region_mask)
        cropped = region_mask[y0:y0+h, x0:x0+w]

        # Reflect the cropped region
        if symmetry_axis == 'vertical':
            mirrored = cv2.flip(cropped, 1)
        elif symmetry_axis == 'horizontal':
            mirrored = cv2.flip(cropped, 0)
        else:
            raise ValueError("symmetry_axis must be 'vertical' or 'horizontal'")

        # Ensure same shape for comparison
        min_h, min_w = min(cropped.shape[0], mirrored.shape[0]), min(cropped.shape[1], mirrored.shape[1])
        cropped = cropped[:min_h, :min_w]
        mirrored = mirrored[:min_h, :min_w]

        # Skip tiny regions that SSIM can't process
        if min_h < 7 or min_w < 7:
            continue

        # Compare using SSIM
        #score = ssim(cropped, mirrored, data_range=255)
        score = compute_symmetry_score(cropped, symmetry_axis, threshold=0.0)
        if score > threshold:
            region_out = output_mask[y0:y0+min_h, x0:x0+min_w]
            region_out[cropped > 0] = np.maximum(region_out[cropped > 0], int(255 * score))
        # if score > 0.55:
        #     output_mask[y:y+h, x:x+w][region > 0] = 255*score

    return output_mask

def symmetry_felzenszwalb_reflection_h(image, scale=100, sigma=3, min_size=100, symmetry_axis='horizontal',
                                     min_region_size=100,threshold=0.55):
    """
    Detect symmetric regions using Felzenszwalb segmentation.

    Args:
        image: Input BGR image
        scale: Float. Higher means larger clusters.
        sigma: Smoothing prior to segmentation
        min_size: Minimum component size
        symmetry_axis: 'vertical' or 'horizontal'
        min_region_size: Ignore regions smaller than this

    Returns:
        output_mask: Symmetry saliency map
    """
    image_float = img_as_float(image)
    segments = felzenszwalb(image_float, scale=scale, sigma=sigma, min_size=min_size)

    output_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for seg_val in np.unique(segments):
        mask = (segments == seg_val).astype(np.uint8) * 255

        if cv2.countNonZero(mask) < min_region_size:
            continue

        x, y, w, h = cv2.boundingRect(mask)
        region = mask[y:y + h, x:x + w]

        # Mirror the region
        if symmetry_axis == 'vertical':
            mirrored = cv2.flip(region, 1)
        elif symmetry_axis == 'horizontal':
            mirrored = cv2.flip(region, 0)
        else:
            raise ValueError("symmetry_axis must be 'vertical' or 'horizontal'")

        # Ensure compatible shapes
        min_h, min_w = min(region.shape[0], mirrored.shape[0]), min(region.shape[1], mirrored.shape[1])
        region = region[:min_h, :min_w]
        mirrored = mirrored[:min_h, :min_w]

        if min_h < 7 or min_w < 7:
            continue  # too small for SSIM

        #score = ssim(region, mirrored, data_range=255)
        score = compute_symmetry_score(region, symmetry_axis, threshold=0.0)
        if score > threshold:
            region_out = output_mask[y:y + min_h, x:x + min_w]
            region_out[region > 0] = np.maximum(region_out[region > 0], (255 * score).astype(np.uint8))

    return output_mask

def symmetry_watershed_reflection_h(image, symmetry_axis='horizontal', min_region_size=100, sigma_blur=4,threshold=0.55):
    """
    Detect symmetric regions using Watershed segmentation.

    Args:
        image: Input BGR image.
        symmetry_axis: 'vertical' or 'horizontal'
        min_region_size: Minimum region size to keep.
        sigma_blur: Gaussian blur sigma before edge detection

    Returns:
        output_mask: Symmetry saliency map
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: Edge map via Sobel
    gradient = sobel(gray)

    # Step 2: Markers from local maxima in distance transform
    distance = cv2.GaussianBlur(gray, (0, 0), sigma_blur)
    distance = cv2.distanceTransform((gray > 40).astype(np.uint8), cv2.DIST_L2, 5)
    coordinates = peak_local_max(distance, footprint=np.ones((65, 65)), labels=gray)
    local_maxi = np.zeros_like(gray, dtype=bool)
    local_maxi[tuple(coordinates.T)] = True
    markers = label(local_maxi)

    # Step 3: Apply watershed
    segments = watershed(gradient, markers, mask=gray > 50)

    output_mask = np.zeros_like(gray, dtype=np.uint8)

    for seg_val in np.unique(segments):
        if seg_val == 0:
            continue

        mask = (segments == seg_val).astype(np.uint8) * 255
        if cv2.countNonZero(mask) < min_region_size:
            continue

        x, y, w, h = cv2.boundingRect(mask)
        region = mask[y:y+h, x:x+w]

        if symmetry_axis == 'vertical':
            mirrored = cv2.flip(region, 1)
        elif symmetry_axis == 'horizontal':
            mirrored = cv2.flip(region, 0)
        else:
            raise ValueError("symmetry_axis must be 'vertical' or 'horizontal'")

        min_h, min_w = min(region.shape[0], mirrored.shape[0]), min(region.shape[1], mirrored.shape[1])
        region = region[:min_h, :min_w]
        mirrored = mirrored[:min_h, :min_w]

        if min_h < 7 or min_w < 7:
            continue

        #scoreo = ssim(region, mirrored, data_range=255)
        score = compute_symmetry_score(region,symmetry_axis,threshold=0.0)

        if score > threshold:
            region_out = output_mask[y:y+min_h, x:x+min_w]
            region_out[region > 0] = np.maximum(region_out[region > 0], (255 * score).astype(np.uint8))

    return output_mask

# utils for symmetry functions:

def preprocess_and_segment_superpixels(image, num_segments=60, compactness=10, sigma=1):
    """
    Applies edge-preserving preprocessing and SLIC segmentation.

    Args:
        image (np.ndarray): Input BGR image (uint8).
        num_segments (int): Number of superpixels.
        compactness (float): Balance between color and space proximity.
        sigma (float): Sigma for bilateral filter.

    Returns:
        segments (np.ndarray): Superpixel labels (2D array).
        preprocessed (np.ndarray): Preprocessed image used for segmentation.
    """
    # Convert to LAB for color-aware processing
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # Apply bilateral filter to each channel separately
    filtered_channels = []
    for i in range(3):
        channel = lab_image[:, :, i]
        filtered = cv2.bilateralFilter(channel, d=9, sigmaColor=sigma * 20, sigmaSpace=sigma * 20)
        filtered_channels.append(filtered)

    # Merge channels and convert to float
    filtered_lab = cv2.merge(filtered_channels)
    filtered_rgb = cv2.cvtColor(filtered_lab, cv2.COLOR_Lab2BGR)
    filtered_rgb_float = img_as_float(filtered_rgb)

    # Apply SLIC on filtered image
    segments = slic(
        filtered_rgb_float,
        n_segments=num_segments,
        compactness=compactness,
        sigma=0,
        start_label=1
    )



    return segments

def compute_symmetry_score(region_mask, symmetry_axis='vertical', threshold=0.05):
    """
    Computes symmetry score between a region and its reflection.

    Args:
        region_mask (np.ndarray): Binary mask of the region (single-channel).
        symmetry_axis (str): 'vertical' or 'horizontal'.
        threshold (float): Minimum SSIM score to be considered symmetrical.

    Returns:
        score (float): SSIM score if above threshold, otherwise 0.
    """
    # Bounding box for the region
    x0, y0, w, h = cv2.boundingRect(region_mask)
    cropped = region_mask[y0:y0 + h, x0:x0 + w]

    # Reflect the cropped region
    if symmetry_axis == 'vertical':
        mirrored = cv2.flip(cropped, 1)
    elif symmetry_axis == 'horizontal':
        mirrored = cv2.flip(cropped, 0)
    else:
        raise ValueError("symmetry_axis must be 'vertical' or 'horizontal'")

    # Match shape
    min_h = min(region_mask.shape[0], mirrored.shape[0])
    min_w = min(region_mask.shape[1], mirrored.shape[1])
    region = region_mask[:min_h, :min_w]
    mirrored = mirrored[:min_h, :min_w]

    # Compute SSIM
    score = ssim(region, mirrored, data_range=255)
    return score if score > threshold else 0


