import cv2
import numpy as np

def load_image(image_path):
    """Loads an image from a given path."""
    return cv2.imread(image_path)

def save_image(image, path):
    """Saves an image to a specified path."""
    cv2.imwrite(path, image)

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