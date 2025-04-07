
import os
import json
import random

from src.saliency import SaliencyModel
from src.utils import load_dataset_in_gt, precompute_heuristic_maps, load_precomputed_maps_from_folders, precompute_and_save_heuristic_maps

from ga.genetic import initialize_population, evaluate_population, weights_to_config



## Directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = PROJECT_ROOT+"/data/full_dataset/input_images/"
GT_DIR = PROJECT_ROOT+"/data/full_dataset/fixation_maps"
PRECOMP_BASE = PROJECT_ROOT+"/data/precomputed_maps/"

## Population initiation
# Load config
with open(os.path.join(PROJECT_ROOT, "ga", "config.json"), "r") as f:
    config_data = json.load(f)

ENABLED_HEURISTICS = list(config_data["heuristics"].keys())
INITIAL_WEIGHTS = list(config_data["heuristics"].values())
WEIGHT_THRESHOLD = config_data.get("weight_threshold", 0.01)
POP_SIZE = config_data.get("population_size",10)
NUM_GENES = len(ENABLED_HEURISTICS)
NUM_GENERATIONS = config_data.get("generations")
MUTATION_RATE = config_data.get("mutation_rate")
CROSSOVER_RATE = config_data.get("crossover_rate")

# Load Dataset
image_filenames = sorted([
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])
dataset = load_dataset_in_gt(INPUT_DIR, GT_DIR)

# Initial heuristic maps
# Create model once (heuristics enabled for all)
intial_config = weights_to_config(INITIAL_WEIGHTS, ENABLED_HEURISTICS, threshold=WEIGHT_THRESHOLD)
model = SaliencyModel(None, None, None, intial_config)

# Precompute all saliency maps for each image
print(f"Initializing heuristic_maps: {ENABLED_HEURISTICS}")
PRECOMPUTED_MAPS = precompute_and_save_heuristic_maps(model, INPUT_DIR, PRECOMP_BASE)
#PRECOMPUTED_MAPS = load_precomputed_maps_from_folders(PRECOMP_BASE, ENABLED_HEURISTICS, INPUT_DIR, GT_DIR)

# === Initial Population (Evaluate Once) ===
print(f"Initializing population of size: {POP_SIZE}, genes: {NUM_GENES}")
population = initialize_population(POP_SIZE, NUM_GENES)
evaluated_parents = evaluate_population(
    population, PRECOMPUTED_MAPS, dataset, image_filenames, ENABLED_HEURISTICS, WEIGHT_THRESHOLD
)

# === Evolution Loop ===
for generation in range(NUM_GENERATIONS):
    print(f"\n🌱 Generation {generation + 1}/{NUM_GENERATIONS}")


    # 1. Generate offspirng (crossover + mutation)
    children = []
    for parent, _ in evaluated_parents:

        if random.random() < CROSSOVER_RATE:
            parent1, _ = random.choice(evaluated_parents)
            parent2, _ = random.choice(evaluated_parents)

            child = [
                random.choice([g1, g2])  # Uniform crossover
                for g1, g2 in zip(parent1, parent2)
            ]
        else:
            # Just clone a random parent
            parent, _ = random.choice(evaluated_parents)
            child = parent[:]

        #child = parent[:]  # Copy
        for i in range(NUM_GENES):
            if random.random() < MUTATION_RATE:
                child[i] += random.uniform(-0.1, 0.1)
                child[i] = min(max(child[i], 0.0), 1.0)

        children.append(child)

    # 2. Evaluate children only
    evaluated_children = evaluate_population(
        population, PRECOMPUTED_MAPS, dataset, image_filenames, ENABLED_HEURISTICS, WEIGHT_THRESHOLD
    )


    # 3. Combine parents + children
    combined_population = evaluated_parents + evaluated_children
    combined_population.sort(key=lambda x: x[1], reverse=True)

    # 4. Print best individual
    best_individual, best_fitness = combined_population[0]
    print(f"🥇 Best Fitness: {best_fitness:.4f}")
    for name, weight in zip(ENABLED_HEURISTICS, best_individual):
        print(f"   🔹 {name}: {weight:.4f}")

    # 5. Select next generation
    evaluated_parents = combined_population[:POP_SIZE]

# ## Run population
# evaluated_population = evaluate_population(population, dataset, ENABLED_HEURISTICS, WEIGHT_THRESHOLD)
#
# # Example: print top result
# best_individual, best_fitness = max(evaluated_population, key=lambda x: x[1])
# print(f"🥇 Best fitness in population: {best_fitness:.4f}")
# print(f"🔧 Weights: {best_individual}")


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