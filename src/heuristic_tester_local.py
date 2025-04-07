import os
import cv2

#import src.heuristics

from heuristics import *  # Pull in updated functions
from utils import test_heuristic_visualization
from utils import save_heuristic_comparison_visualization
from src.heuristics import contrast_edge_luminance_Laplacian

PROJECT_ROOT = "/home/evaldas/PycharmProjects/heuristic_saliency"
INPUT_DIR = PROJECT_ROOT+"/data/full_dataset/input_images/"
GT_DIR = PROJECT_ROOT + "/data/full_dataset/fixation_maps/"
OUT_DIR = PROJECT_ROOT + "/data/heuristic_visualizations/"
IMAGE_NAME = "paint_88.png"
FUNC_NAME = "texture_entropy_"
IMAGE_PATH = INPUT_DIR+IMAGE_NAME
GT_PATH = GT_DIR + IMAGE_NAME



save_heuristic_comparison_visualization(
    image_path=IMAGE_PATH,
    gt_path=GT_PATH,
    heuristic_fn=contrast_texture_entropy,
    output_path=OUT_DIR+FUNC_NAME+IMAGE_NAME
)