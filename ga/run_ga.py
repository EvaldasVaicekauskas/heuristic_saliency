import cv2
import os
import json

from src.saliency import SaliencyModel
from src.utils import load_dataset_in_gt
from ga.genetic import compute_fitness
from ga.genetic import weights_to_config
from ga.genetic import initialize_population
from ga.genetic import evaluate_population

## Directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = PROJECT_ROOT+"/data/input_images/"
GT_DIR = PROJECT_ROOT+"/data/fixation_maps"

## Population initiation
# Load config
with open(os.path.join(PROJECT_ROOT, "ga", "config.json"), "r") as f:
    config_data = json.load(f)

ENABLED_HEURISTICS = list(config_data["heuristics"].keys())
INITIAL_WEIGHTS = list(config_data["heuristics"].values())
WEIGHT_THRESHOLD = config_data.get("weight_threshold", 0.01)
POP_SIZE = config_data.get("population_size",10)
NUM_GENES = len(ENABLED_HEURISTICS)

population = initialize_population(POP_SIZE,NUM_GENES)

## Load data
dataset = load_dataset_in_gt(INPUT_DIR, GT_DIR)

## Run population
evaluated_population = evaluate_population(population, dataset, ENABLED_HEURISTICS, WEIGHT_THRESHOLD)

# Example: print top result
best_individual, best_fitness = max(evaluated_population, key=lambda x: x[1])
print(f"🥇 Best fitness in population: {best_fitness:.4f}")
print(f"🔧 Weights: {best_individual}")


# heuristic_config = weights_to_config(INITIAL_WEIGHTS, ENABLED_HEURISTICS,WEIGHT_THRESHOLD)
#
# # Load one pair of image and GT
# pairs = load_image_gt_pairs(PROJECT_ROOT+"/data/input_images/", PROJECT_ROOT+"/data/fixation_maps/")
# img_path, gt_path = pairs[0]
#
# # Load the image
# # image = cv2.imread(img_path)
# # gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)  # ground truth saliency
#
# # Run the model
# # model = SaliencyModel(heuristic_config=heuristic_config)
# # pred_map = model.generate_saliency_for_image(image)
#
# # Optional: Save output for visual check
# #cv2.imwrite("predicted_saliency.png", pred_map)
#
#
# active = [name for name, cfg in heuristic_config.items() if cfg["enabled"]]
# print(f"Active heuristics this round: {active}")
# fitness = compute_fitness(heuristic_config, INPUT_DIR, GT_DIR)
# print(fitness)