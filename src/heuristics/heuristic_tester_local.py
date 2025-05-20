#import src.heuristics

from heuristics import *  # Pull in updated functions
#from src.heuristics.heuristics import isolation_tex_gab_local_abs
#from utils import test_heuristic_visualization
from src.utils import save_heuristic_comparison_visualization,save_isolation_comparison_visualization,save_symmetry_comparison_visualization

PROJECT_ROOT = "/home/evaldas/PycharmProjects/heuristic_saliency"
INPUT_DIR = PROJECT_ROOT+"/data/full_dataset/input_images/"
GT_DIR = PROJECT_ROOT + "/data/full_dataset/fixation_maps/"
OUT_DIR = PROJECT_ROOT + "/data/heuristic_visualizations/"
IMAGE_NAME = "pand_nature-morte-aux-fleurs-jaunes-1956 .jpg"
FUNC_NAME = "symmetry_superpixel_preproc_"
IMAGE_PATH = INPUT_DIR+IMAGE_NAME
GT_PATH = GT_DIR + IMAGE_NAME



# save_heuristic_comparison_visualization(
#     image_path=IMAGE_PATH,
#     gt_path=GT_PATH,
#     heuristic_fn=grouping_line_hough,
#     output_path=OUT_DIR+FUNC_NAME+IMAGE_NAME
# )

# save_isolation_comparison_visualization(
#     image_path=IMAGE_PATH,
#     gt_path=GT_PATH,
#     sal1_fn=isolation_tex_entropy_global_abs,
#     sal2_fn=isolation_tex_entropy_local_abs,
#     sal3_fn=isolation_tex_entropy_local_multi,
#     output_path=OUT_DIR + FUNC_NAME + IMAGE_NAME
#
# )

save_symmetry_comparison_visualization(
    image_path=IMAGE_PATH,
    gt_path=GT_PATH,
    sal1_fn=symmetry_superpixel_preproc,
    sal2_fn=symmetry_superpixel_preproc_h,
    output_path=OUT_DIR + FUNC_NAME + IMAGE_NAME

)