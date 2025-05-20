import cv2
import numpy as np
import os

from src.utils import create_hue_mask

from skimage.segmentation import slic
from skimage.color import rgb2lab
from skimage.util import img_as_float
from skimage import graph

from .heuristics.heuristics import HEURISTIC_FUNCTIONS

class SaliencyModel:
    """Dynamic heuristic-based saliency model with configurable weights and activation."""

    def __init__(self, input_dir=None, output_dir=None, vis_dir=None, heuristic_config=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.vis_dir = vis_dir
        self.heuristic_config = heuristic_config or {}

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        if vis_dir:
            os.makedirs(vis_dir, exist_ok=True)

    # ========================== Image evaluation ===============================

    def calculate_overall_saturation(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation_channel = hsv[:, :, 1]
        return np.mean(saturation_channel) / 255

    def measure_colorfulness(self, image):
        """Measure the colorfulness of an image using Hasler and Süsstrunk's metric."""
        # Convert to RGB
        (B, G, R) = cv2.split(image.astype("float"))

        # Calculate Red-Green and Yellow-Blue differences
        rg = np.abs(R - G)
        yb = np.abs(0.5 * (R + G) - B)

        # Compute mean and standard deviation
        std_rg, mean_rg = np.std(rg), np.mean(rg)
        std_yb, mean_yb = np.std(yb), np.mean(yb)

        # Calculate colorfulness score
        colorfulness = np.sqrt(std_rg ** 2 + std_yb ** 2) + (0.3 * np.sqrt(mean_rg ** 2 + mean_yb ** 2))
        return colorfulness

    # =========================== Heuristic Functions ===========================

    def detect_red_saliency(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2
        return cv2.normalize(red_mask, None, 0, 255, cv2.NORM_MINMAX)

    def detect_contrast_saliency(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast_map = cv2.Laplacian(gray, cv2.CV_64F)
        contrast_map = np.abs(contrast_map)
        return cv2.normalize(contrast_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def detect_saturation_saliency(self, image, overall_saturation):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation_channel = hsv[:, :, 1]
        adjusted_saturation = saturation_channel - (overall_saturation * 255)
        return np.clip(adjusted_saturation, 0, 255).astype(np.uint8)

    def detect_color_contrast_blobs(self, image):
        """Detects significant blobs by refining contrast detection using Otsu's thresholding and area filtering."""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Compute Scharr gradients
        grad_lx = cv2.Scharr(lab[:, :, 0], cv2.CV_64F, 1, 0)
        grad_ly = cv2.Scharr(lab[:, :, 0], cv2.CV_64F, 0, 1)
        grad_l = cv2.magnitude(grad_lx, grad_ly)

        grad_ax = cv2.Scharr(lab[:, :, 1], cv2.CV_64F, 1, 0)
        grad_ay = cv2.Scharr(lab[:, :, 1], cv2.CV_64F, 0, 1)
        grad_a = cv2.magnitude(grad_ax, grad_ay)

        grad_bx = cv2.Scharr(lab[:, :, 2], cv2.CV_64F, 1, 0)
        grad_by = cv2.Scharr(lab[:, :, 2], cv2.CV_64F, 0, 1)
        grad_b = cv2.magnitude(grad_bx, grad_by)

        # Combine gradients
        combined_gradient = np.sqrt(grad_l ** 2 + grad_a ** 2 + grad_b ** 2)

        # Normalize and smooth the gradient map
        gradient_map = cv2.normalize(combined_gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        smoothed_map = cv2.GaussianBlur(gradient_map, (5, 5), 0)

        # Apply Otsu's thresholding to reduce oversaturation
        _, threshold_map = cv2.threshold(smoothed_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological closing to enhance blobs
        kernel = np.ones((10, 10), np.uint8)

        blob_map = cv2.morphologyEx(threshold_map, cv2.MORPH_CLOSE, kernel)

        # Fill detected contours to emphasize inner blob regions
        contours, _ = cv2.findContours(blob_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_blob_map = np.zeros_like(blob_map)
        cv2.drawContours(filled_blob_map, contours, -1, (255), thickness=cv2.FILLED)

        return filled_blob_map

    def detect_superpixel_color_contrast(self, image, num_segments=240, base_threshold=90, colorfulness=0):
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

    def detect_complementary_color_saliency(self, image, top_n=3, hue_range=10, proximity_thickness=25):
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

    # =========================== Dynamic Saliency Functions ===========================

    def combine_saliency_maps(self, heuristic_maps):
        """Dynamically combine enabled saliency maps using their defined weights."""
        combined_map = np.zeros_like(heuristic_maps[0][0], dtype=np.float32)

        for saliency_map, weight in heuristic_maps:
            combined_map += saliency_map * weight

        # Normalize the result
        combined_map = cv2.normalize(combined_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return combined_map

    def visualize_heuristics(self, maps_with_colors):
        """Visualizes heuristics dynamically with distinct colors."""
        height, width = maps_with_colors[0][0].shape
        visual_image = np.zeros((height, width, 3), dtype=np.uint8)

        for saliency_map, color in maps_with_colors:
            for channel, intensity in enumerate(color):
                visual_image[:, :, channel] += (saliency_map * intensity).astype(np.uint8)

        return visual_image

    def overlay_heuristic_contours(self, image, heuristic_maps_with_colors):
        """Draws contours of heuristic maps on top of the original image using distinct colors."""
        # Copy the original image for visualization
        overlay_image = image.copy()

        for saliency_map, color in heuristic_maps_with_colors:
            # Detect contours from each heuristic map
            contours, _ = cv2.findContours(saliency_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw the contours on the overlay image
            contour_color = tuple(int(c * 255) for c in color)  # Convert (0-1) color values to (0-255)
            cv2.drawContours(overlay_image, contours, -1, contour_color, thickness=2)

        return overlay_image


    # =========================== Main Processing Function ===========================

    def process_images(self):
        for filename in os.listdir(self.input_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(self.input_dir, filename)
                image = cv2.imread(image_path)

                if image is None:
                    print(f"Warning: Could not read {filename}")
                    continue

                heuristic_maps = []
                visual_maps = []

                # Image metrics
                overall_saturation = self.calculate_overall_saturation(image)
                overall_colorfulness = self.measure_colorfulness(image)

                # Dynamically apply enabled heuristics based on the config
                if self.heuristic_config["red"]["enabled"]:
                    red_map = self.detect_red_saliency(image)
                    heuristic_maps.append((red_map, self.heuristic_config["red"]["weight"]))
                    visual_maps.append((red_map, (0, 0, 1)))  # Red visualization

                if self.heuristic_config["contrast"]["enabled"]:
                    contrast_map = self.detect_contrast_saliency(image)
                    heuristic_maps.append((contrast_map, self.heuristic_config["contrast"]["weight"]))
                    visual_maps.append((contrast_map, (1, 0, 0)))  # Blue visualization

                if self.heuristic_config["saturation"]["enabled"]:
                    saturation_map = self.detect_saturation_saliency(image, overall_saturation)
                    heuristic_maps.append((saturation_map, self.heuristic_config["saturation"]["weight"]))
                    visual_maps.append((saturation_map, (0, 1, 0)))  # Green visualization

                if self.heuristic_config["superpixel_contrast"]["enabled"]:
                    superpixel_contrast_map = self.detect_superpixel_color_contrast(image, 350, 20, overall_colorfulness)
                    heuristic_maps.append(
                        (superpixel_contrast_map, self.heuristic_config["superpixel_contrast"]["weight"]))
                    visual_maps.append((superpixel_contrast_map, (1, 1, 0)))  # Yellow visualization

                if self.heuristic_config["complementary_colors"]["enabled"]:
                    complementary_saliency_map = self.detect_complementary_color_saliency(image)
                    heuristic_maps.append(
                        (complementary_saliency_map, self.heuristic_config["complementary_colors"]["weight"]))
                    visual_maps.append((complementary_saliency_map, (1, 0, 1)))  # Magenta visualization

                # Combine maps and visualize
                combined_saliency = self.combine_saliency_maps(heuristic_maps)
                visualization = self.visualize_heuristics(visual_maps)
                contour_overlay = self.overlay_heuristic_contours(image,
                                                                  visual_maps)  # New visualization on top of the original image

                # Save results
                output_path = os.path.join(self.output_dir, f"saliency_{filename}")
                vis_path = os.path.join(self.vis_dir, f"visualization_{filename}")
                contour_vis_path = os.path.join(self.vis_dir,
                                                f"contour_overlay_{filename}")  # New path for contour visualization

                cv2.imwrite(output_path, combined_saliency)
                cv2.imwrite(vis_path, visualization)
                cv2.imwrite(contour_vis_path, contour_overlay)  # Save the contour overlay

                # Print status update
                print(f"Processed: {filename} → Saliency Map: {output_path} "
                      f"| Visualization: {vis_path} | Contour Overlay: {contour_vis_path}")

    def generate_saliency_for_image(self, image):
        heuristic_maps = []

        # Global image metrics
        overall_saturation = self.calculate_overall_saturation(image)
        overall_colorfulness = self.measure_colorfulness(image)

        if self.heuristic_config.get("red", {}).get("enabled", False):
            red_map = self.detect_red_saliency(image)
            heuristic_maps.append((red_map, self.heuristic_config["red"]["weight"]))

        if self.heuristic_config.get("contrast", {}).get("enabled", False):
            contrast_map = self.detect_contrast_saliency(image)
            heuristic_maps.append((contrast_map, self.heuristic_config["contrast"]["weight"]))

        if self.heuristic_config.get("saturation", {}).get("enabled", False):
            sat_map = self.detect_saturation_saliency(image, overall_saturation)
            heuristic_maps.append((sat_map, self.heuristic_config["saturation"]["weight"]))

        if self.heuristic_config.get("superpixel_contrast", {}).get("enabled", False):
            sp_map = self.detect_superpixel_color_contrast(image, 350, 20, overall_colorfulness)
            heuristic_maps.append((sp_map, self.heuristic_config["superpixel_contrast"]["weight"]))

        if self.heuristic_config.get("complementary_colors", {}).get("enabled", False):
            cc_map = self.detect_complementary_color_saliency(image)
            heuristic_maps.append((cc_map, self.heuristic_config["complementary_colors"]["weight"]))

        combined_saliency = self.combine_saliency_maps(heuristic_maps)
        return combined_saliency

    def generate_saliency_maps_for_image_o(self, image):
        heuristic_maps = {}

        overall_saturation = self.calculate_overall_saturation(image)
        overall_colorfulness = self.measure_colorfulness(image)

        if "red" in self.heuristic_config:
            heuristic_maps["red"] = self.detect_red_saliency(image)

        if "contrast" in self.heuristic_config:
            heuristic_maps["contrast"] = self.detect_contrast_saliency(image)

        if "saturation" in self.heuristic_config:
            heuristic_maps["saturation"] = self.detect_saturation_saliency(image, overall_saturation)

        if "superpixel_contrast" in self.heuristic_config:
            heuristic_maps["superpixel_contrast"] = self.detect_superpixel_color_contrast(
                image, 350, 20, overall_colorfulness
            )

        if "complementary_colors" in self.heuristic_config:
            heuristic_maps["complementary_colors"] = self.detect_complementary_color_saliency(image)

        if "symmetry" in self.heuristic_config:
            sym_map = symmetry_saliency(image)
            heuristic_maps["symmetry"] = sym_map

        return heuristic_maps



    def generate_saliency_maps_for_image(self, image):
        heuristic_maps = {}

        for name, weight in self.heuristic_config.items():
            weight = weight.get("weight", 0)
            if weight < 1e-6:
                continue
            if name not in HEURISTIC_FUNCTIONS:
                print(f"[Warning] Heuristic not found: {name}")
                continue
            heuristic_fn = HEURISTIC_FUNCTIONS[name]
            saliency_map = heuristic_fn(image)
            heuristic_maps[name] = saliency_map

        return heuristic_maps
