import cv2
import numpy as np
import os


def resize_saliency_map(saliency_path, stimulus_path, dataset_name, save_path=None):
    """
    Resize saliency maps based on dataset-specific rules.

    - dataset_paint: Resize while preserving aspect ratio and apply padding to match stimulus size.
    - dataset_pand: Resize to exactly match the stimulus image dimensions.

    :param saliency_path: Path to the original saliency map
    :param stimulus_path: Path to the corresponding stimulus image
    :param dataset_name: "paint" or "pand" (determines resizing method)
    :param save_path: Where to save the processed saliency map (if None, overwrite)
    """
    # Load input_images
    saliency = cv2.imread(saliency_path, cv2.IMREAD_GRAYSCALE)
    stimulus = cv2.imread(stimulus_path)

    if saliency is None or stimulus is None:
        print(f"Error: Unable to load {saliency_path} or {stimulus_path}")
        return

    stim_h, stim_w = stimulus.shape[:2]
    sal_h, sal_w = saliency.shape[:2]

    if dataset_name == "paint":
        # Compute aspect ratios
        stim_aspect = stim_w / stim_h
        sal_aspect = sal_w / sal_h

        # Determine whether to crop horizontally or vertically
        if sal_aspect > stim_aspect:  # Too wide, crop left and right
            new_w = int(sal_h * stim_aspect)
            start_x = (sal_w - new_w) // 2
            cropped_saliency = saliency[:, start_x:start_x + new_w]
        else:  # Too tall, crop top and bottom
            new_h = int(sal_w / stim_aspect)
            start_y = (sal_h - new_h) // 2
            cropped_saliency = saliency[start_y:start_y + new_h, :]

        # Resize cropped saliency map to match stimulus dimensions
        processed_saliency = cv2.resize(cropped_saliency, (stim_w, stim_h), interpolation=cv2.INTER_CUBIC)


    elif dataset_name == "pand":
        # Resize saliency map to exactly match the stimulus image dimensions
        processed_saliency = cv2.resize(saliency, (stim_w, stim_h), interpolation=cv2.INTER_CUBIC)

    else:
        print(f"Error: Unsupported dataset name {dataset_name}")
        return

    # Save the processed saliency map
    if save_path is None:
        save_path = saliency_path  # Overwrite original if not specified
    cv2.imwrite(save_path, processed_saliency)
    print(f"Processed {saliency_path}: Resized to ({stim_w}, {stim_h}) and saved to {save_path}")
