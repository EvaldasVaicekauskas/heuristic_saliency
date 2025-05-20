
import os
import json
import random
import sys

from src.saliency import SaliencyModel
from src.utils import *

from src.ga.genetic import initialize_population, evaluate_population, weights_to_config



## Directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "full_dataset", "input_images")
GT_DIR = os.path.join(PROJECT_ROOT, "data", "full_dataset", "fixation_maps")
PRECOMP_BASE = os.path.join(PROJECT_ROOT, "data", "precomputed_maps")

RUN_FOLDER = create_run_output_folder(os.path.join(PROJECT_ROOT, "ga_log"))

# INPUT_DIR = PROJECT_ROOT+"/data/full_dataset/input_images/"
# GT_DIR = PROJECT_ROOT+"/data/full_dataset/fixation_maps"
# PRECOMP_BASE = PROJECT_ROOT+"/data/precomputed_maps/"

## Population initiation
# Load config
with open(os.path.join(PROJECT_ROOT,"src", "ga", "config.json"), "r") as f:
    config_data = json.load(f)

ENABLED_HEURISTICS = list(config_data["heuristics"].keys())
INITIAL_WEIGHTS = list(config_data["heuristics"].values())
WEIGHT_THRESHOLD = config_data.get("weight_threshold", 0.01)
POP_SIZE = config_data.get("population_size",10)
NUM_GENES = len(ENABLED_HEURISTICS)
NUM_GENERATIONS = config_data.get("generations")
MUTATION_RATE = config_data.get("mutation_rate")
MUTATION_RANGE = config_data.get("mutation_range")
CROSSOVER_RATE = config_data.get("crossover_rate")

# Load Dataset
image_filenames = sorted([
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])
dataset = load_dataset_in_gt(INPUT_DIR, GT_DIR)
sample_image_name = image_filenames[30]
sample_image = cv2.imread(INPUT_DIR+"/"+sample_image_name)

# Initial heuristic maps
# Create model once (heuristics enabled for all)
intial_config = weights_to_config(INITIAL_WEIGHTS, ENABLED_HEURISTICS, threshold=WEIGHT_THRESHOLD)
model = SaliencyModel(None, None, None, intial_config)

# Precompute all saliency maps for each image
print(f"Initializing heuristic_maps: {ENABLED_HEURISTICS}")
#PRECOMPUTED_MAPS = precompute_and_save_heuristic_maps(model, INPUT_DIR, PRECOMP_BASE)
PRECOMPUTED_MAPS = load_precomputed_maps_from_folders(PRECOMP_BASE, ENABLED_HEURISTICS, INPUT_DIR, GT_DIR)

# === Initial Population (Evaluate Once) ===
print(f"Initializing population of size: {POP_SIZE}, genes: {NUM_GENES}")
copy_config_file(src_config_path=PROJECT_ROOT+"/src/ga/config.json", dst_folder=RUN_FOLDER+"/")
population = initialize_population(POP_SIZE, NUM_GENES)
evaluated_parents = evaluate_population(
    population, PRECOMPUTED_MAPS, dataset, image_filenames, ENABLED_HEURISTICS, WEIGHT_THRESHOLD
)

#sys.exit("Stopping program after initial population.")

SAVE_EVERY_N = 5

# === Evolution Loop ===
for generation in range(NUM_GENERATIONS):
    print(f"\n🌱 Generation {generation + 1}/{NUM_GENERATIONS}")


    # 1. Generate offspirng (crossover + mutation)
    individuals = [ind for ind, *_ in evaluated_parents]
    children = []
    for parent in individuals:
        if random.random() < CROSSOVER_RATE:
            parent1 = random.choice(individuals)
            parent2 = random.choice(individuals)

            child = [
                random.choice([g1, g2])
                for g1, g2 in zip(parent1, parent2)
            ]
        else:
            parent = random.choice(individuals)
            child = parent[:]

            # Mutate at least one gene
            mutate_index = random.randint(0, NUM_GENES - 1)
            child[mutate_index] += random.uniform(-MUTATION_RANGE, MUTATION_RANGE)
            child[mutate_index] = min(max(child[mutate_index], 0.0), 1.0)

        # Additional mutation
        for i in range(NUM_GENES):
            if random.random() < MUTATION_RATE:
                child[i] += random.uniform(-MUTATION_RANGE, MUTATION_RANGE)
                child[i] = min(max(child[i], 0.0), 1.0)

        children.append(child)

    # 2. Evaluate children only
    evaluated_children = evaluate_population(
        children, PRECOMPUTED_MAPS, dataset, image_filenames, ENABLED_HEURISTICS, WEIGHT_THRESHOLD
    )


    # 3. Combine parents + children
    combined_population = evaluated_parents + evaluated_children
    combined_population.sort(key=lambda x: x[1], reverse=True)

    top_k = combined_population[:POP_SIZE]
    avg_fitness = sum(x[1] for x in top_k) / POP_SIZE
    avg_sim = sum(x[2] for x in top_k) / POP_SIZE
    avg_cc = sum(x[3] for x in top_k) / POP_SIZE
    avg_kl = sum(x[4] for x in top_k) / POP_SIZE

    # 4. Print best individual
    #best_individual, best_fitness = combined_population[0]
    best_individual, best_fitness, best_sim, best_cc, best_kl = combined_population[0]
    print(f"🥇 Best Fitness: {best_fitness:.4f}")
    for name, weight in zip(ENABLED_HEURISTICS, best_individual):
        print(f"   🔹 {name}: {weight:.4f}")

    # 5. Select next generation
    evaluated_parents = combined_population[:POP_SIZE]

    # 6. Logging
    if generation % SAVE_EVERY_N == 0:
        avg_fitness = sum(fitness for _, fitness, _, _, _ in combined_population[:POP_SIZE]) / POP_SIZE
        # log_fitness(generation, best_fitness, avg_fitness,best_id,RUN_FOLDER+"/fitness.csv")
        log_fitness(
            generation,
            best_fitness,
            avg_fitness=avg_fitness,
            best_id=None,  # combined_population.index((best_individual, best_fitness, best_sim, best_cc, best_kl)),
            best_sim=best_sim,
            best_cc=best_cc,
            best_kl=best_kl,
            avg_sim=avg_sim,
            avg_cc=avg_cc,
            avg_kl=avg_kl,
            csv_path=RUN_FOLDER + "/fitness.csv"
        )
        save_best_individual(
            generation, best_individual, best_fitness, best_sim, best_cc, best_kl,
            ENABLED_HEURISTICS, os.path.join(RUN_FOLDER, "best_individual.json")
        )
        save_generation_state(generation, evaluated_parents, ENABLED_HEURISTICS, RUN_FOLDER + "/generations")
        heuristic_maps = PRECOMPUTED_MAPS[sample_image_name]  # or regenerate if needed
        weights_dict = {name: weight for name, weight in zip(ENABLED_HEURISTICS, best_individual)}
        combined = combine_saliency_maps(heuristic_maps, weights_dict)
        save_combined_map(sample_image, combined, generation, RUN_FOLDER)

