import cv2
import numpy as np
import os


def crop_padding(image, padding_color=(128, 128, 128)):
    """
    Crops uniform padding from all sides of the image based on the given padding_color.

    Args:
        image (np.ndarray): BGR image (uint8).
        padding_color (tuple): Padding color to remove (B, G, R).

    Returns:
        Cropped image.
    """
    mask = np.any(image != padding_color, axis=-1)

    if not np.any(mask):
        print("⚠️ Entire image is padding.")
        return image  # Return unchanged

    y_min, y_max = np.where(mask)[0][[0, -1]]
    x_min, x_max = np.where(mask)[1][[0, -1]]

    return image[y_min:y_max + 1, x_min:x_max + 1]

def get_crop_bounds(image, padding_color=(126, 126, 126)):
    """
    Returns the cropping bounds (y_min, y_max, x_min, x_max) that exclude padding.
    """
    mask = np.any(image != padding_color, axis=-1)

    if not np.any(mask):
        raise ValueError("The entire image is padding.")

    y_indices, x_indices = np.where(mask)
    y_min, y_max = y_indices[0], y_indices[-1]
    x_min, x_max = x_indices[0], x_indices[-1]

    return y_min, y_max + 1, x_min, x_max + 1  # +1 for slicing compatibility

def crop_using_bounds(image, bounds):
    y_min, y_max, x_min, x_max = bounds
    return image[y_min:y_max, x_min:x_max]


## Directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "cat", "stimuli")
GT_DIR = os.path.join(PROJECT_ROOT, "data", "cat", "fixation_maps")
OUTPUT_IM_DIR = os.path.join(PROJECT_ROOT, "data", "cat_cropped", "stimuli")
OUTPUT_GT_DIR = os.path.join(PROJECT_ROOT, "data", "cat_cropped", "fixation_maps")

IMAGE_NAME = "/"+"cat_197.jpg"

original = cv2.imread(INPUT_DIR + IMAGE_NAME)
gt = cv2.imread(GT_DIR+IMAGE_NAME, cv2.IMREAD_GRAYSCALE)



# Step 1: Find cropping bounds from the original padded image
bounds = get_crop_bounds(original, padding_color=(126, 126, 126))
print(bounds)

# Step 2: Crop both images using the same bounds
cropped_image = crop_using_bounds(original, bounds)
cropped_gt = crop_using_bounds(gt, bounds)

# Save or return cropped outputs
cv2.imwrite(OUTPUT_IM_DIR+IMAGE_NAME, cropped_image)
cv2.imwrite(OUTPUT_GT_DIR+IMAGE_NAME, cropped_gt)
