import os
import json
import random
import sys
import cv2

from src.saliency import SaliencyModel
from src.utils import *
from src.modulation.modulation_utils import (
    load_modulation_config,
    initialize_modulated_individual,
    split_genome,
    compute_modulated_weights
)
from src.ga.genetic_mod import evaluate_modulated_population, compute_population_std
from src.modulation.modulation_utils import load_feature_vectors, save_best_individual_mod, save_generation_state_mod


# === Directories ===
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "work_dataset", "input_images")
GT_DIR = os.path.join(PROJECT_ROOT, "data", "work_dataset", "fixation_maps")
PRECOMP_BASE = os.path.join(PROJECT_ROOT, "data", "precomputed_maps")
MOD_FEATURES_DIR = os.path.join(PROJECT_ROOT, "src", "modulation", "data", "modulation_features")
RUN_FOLDER = create_run_output_folder(os.path.join(PROJECT_ROOT, "ga_log"))

# === Configs ===
with open(os.path.join(PROJECT_ROOT, "src", "ga", "config.json"), "r") as f:
    config_data = json.load(f)
with open(os.path.join(PROJECT_ROOT, "src", "modulation", "modulation_config.json"), "r") as f:
    mod_config = json.load(f)
MOD_GROUPS = mod_config["groups"]
MOD_FEATURES = mod_config["features"]
MODULATION_FEATURE_DIR = os.path.join(PROJECT_ROOT, "src", "modulation", "data", "modulation_features")

# Create heuristic → group mapping
HEURISTIC_TO_GROUP = {
    heuristic: group
    for group, heuristics in MOD_GROUPS.items()
    for heuristic in heuristics
}

ENABLED_HEURISTICS = list(config_data["heuristics"].keys())
INITIAL_WEIGHTS = list(config_data["heuristics"].values())
WEIGHT_THRESHOLD = config_data.get("weight_threshold", 0.01)
POP_SIZE = config_data.get("population_size", 10)
NUM_GENERATIONS = config_data.get("generations")
MUTATION_RATE = config_data.get("mutation_rate")
MUTATION_RANGE = config_data.get("mutation_range")
CROSSOVER_RATE = config_data.get("crossover_rate")

MOD_FEATURES, MOD_GROUPS = mod_config["features"], mod_config["groups"]
NUM_GENES = len(ENABLED_HEURISTICS) + len(MOD_GROUPS) * len(MOD_FEATURES)

# === Dataset & Features ===
image_filenames = sorted(f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))
dataset = load_dataset_in_gt(INPUT_DIR, GT_DIR)
mod_feature_vectors = load_feature_vectors(MOD_FEATURES_DIR)

# === Precomputed saliency maps ===
print(f"Initializing heuristic_maps: {ENABLED_HEURISTICS}")
PRECOMPUTED_MAPS = load_precomputed_maps_from_folders(PRECOMP_BASE, ENABLED_HEURISTICS, INPUT_DIR, GT_DIR)

# === Initialize Population ===
print(f"Initializing modulation-aware population of size {POP_SIZE}, genome size: {NUM_GENES}")
copy_config_file(src_config_path=os.path.join(PROJECT_ROOT, "src", "ga", "config.json"), dst_folder=RUN_FOLDER + "/",file_name="used_config.json")
copy_config_file(src_config_path=os.path.join(PROJECT_ROOT, "src", "modulation", "modulation_config.json"), dst_folder=RUN_FOLDER + "/", file_name="used_modulation_config.json")
population = [initialize_modulated_individual(ENABLED_HEURISTICS, MOD_GROUPS, MOD_FEATURES) for _ in range(POP_SIZE)]

evaluated_parents = evaluate_modulated_population(
    population,
    PRECOMPUTED_MAPS,
    dataset,
    image_filenames,
    ENABLED_HEURISTICS,
    MOD_GROUPS,
    MOD_FEATURES,
    HEURISTIC_TO_GROUP,
    MOD_FEATURES_DIR,
    WEIGHT_THRESHOLD
)

SAVE_EVERY_N = 5
sample_image_name = image_filenames[32]
sample_image = cv2.imread(os.path.join(INPUT_DIR, sample_image_name))

initial_mutation_rate = MUTATION_RATE
initial_mutation_range = MUTATION_RANGE

last_best_fitness = 0

# === Evolution Loop ===
for generation in range(NUM_GENERATIONS):
    print(f"\n🌱 Generation {generation + 1}/{NUM_GENERATIONS}")

    # Linear decay based on generation
    MUTATION_RATE = initial_mutation_rate * (1 - generation / NUM_GENERATIONS)
    MUTATION_RANGE = initial_mutation_range * (1 - generation / NUM_GENERATIONS)

    # Diversity-based boost
    std_dev = compute_population_std(evaluated_parents)
    diversity_scale = min(std_dev / 0.1, 1.0)  # Normalize

    # Encourage mutation when diversity is low
    MUTATION_RATE = MUTATION_RATE * (2.0 - diversity_scale)
    MUTATION_RANGE = MUTATION_RANGE * (2.0 - diversity_scale)

    # Mutation cropping
    MUTATION_RATE = min(max(MUTATION_RATE, 0.01), 0.5)
    MUTATION_RANGE = min(max(MUTATION_RANGE, 0.01), 0.5)
    if generation % 5 == 0:
        print(f"  ↪️ Adaptive Mutation Rate: {MUTATION_RATE:.4f}, Range: {MUTATION_RANGE:.4f}")


    individuals = [ind for ind, *_ in evaluated_parents]
    children = []
    for parent in individuals:
        if random.random() < CROSSOVER_RATE:
            parent1 = random.choice(individuals)
            parent2 = random.choice(individuals)
            child = [random.choice([g1, g2]) for g1, g2 in zip(parent1, parent2)]
        else:
            parent = random.choice(individuals)
            child = parent[:]
            mutate_index = random.randint(0, NUM_GENES - 1)
            child[mutate_index] += random.uniform(-MUTATION_RANGE, MUTATION_RANGE)
            child[mutate_index] = min(max(child[mutate_index], 0.0), 1.0)

        for i in range(NUM_GENES):
            if random.random() < MUTATION_RATE:
                child[i] += random.uniform(-MUTATION_RANGE, MUTATION_RANGE)
                child[i] = min(max(child[i], 0.0), 1.0)

        children.append(child)

    evaluated_children = evaluate_modulated_population(
        children,
        PRECOMPUTED_MAPS,
        dataset,
        image_filenames,
        ENABLED_HEURISTICS,
        MOD_GROUPS,
        MOD_FEATURES,
        HEURISTIC_TO_GROUP,
        MOD_FEATURES_DIR,
        WEIGHT_THRESHOLD
    )

    combined_population = evaluated_parents + evaluated_children
    combined_population.sort(key=lambda x: x[1], reverse=True)
    evaluated_parents = combined_population[:POP_SIZE]

    best_individual, best_fitness, best_sim, best_cc, best_kl = evaluated_parents[0]
    avg_fitness = sum(x[1] for x in evaluated_parents) / POP_SIZE
    avg_sim = sum(x[2] for x in evaluated_parents) / POP_SIZE
    avg_cc = sum(x[3] for x in evaluated_parents) / POP_SIZE
    avg_kl = sum(x[4] for x in evaluated_parents) / POP_SIZE

    print(f"🥇 Best Fitness: {best_fitness:.4f}, last best: {last_best_fitness:.4f}")
    base_weights, mod_weights = split_genome(best_individual, ENABLED_HEURISTICS, MOD_GROUPS, MOD_FEATURES)
    # for name, weight in zip(ENABLED_HEURISTICS, base_weights):
    #     print(f"   🔹 {name}: {weight:.4f}")

    last_best_fitness = best_fitness

    # Save logs and saliency maps
    if generation % SAVE_EVERY_N == 0:
        log_fitness(
            generation, best_fitness, avg_fitness, best_id=None,
            best_sim=best_sim, best_cc=best_cc, best_kl=best_kl,
            avg_sim=avg_sim, avg_cc=avg_cc, avg_kl=avg_kl,
            csv_path=os.path.join(RUN_FOLDER, "fitness.csv")
        )
        save_best_individual_mod(
            generation, best_individual, best_fitness, best_sim, best_cc, best_kl,
            ENABLED_HEURISTICS, MOD_GROUPS, MOD_FEATURES, os.path.join(RUN_FOLDER, "best_individual.json")
        )

        save_generation_state_mod(generation, evaluated_parents, ENABLED_HEURISTICS, MOD_GROUPS, MOD_FEATURES, os.path.join(RUN_FOLDER, "generations"))

        sample_features = mod_feature_vectors[os.path.splitext(sample_image_name)[0]]
        mod_weights_vec = compute_modulated_weights(base_weights, mod_weights, sample_features, ENABLED_HEURISTICS, {
            h: group for group, hs in MOD_GROUPS.items() for h in hs
        })
        combined_map = combine_saliency_maps(PRECOMPUTED_MAPS[sample_image_name], {
            h: w for h, w in zip(ENABLED_HEURISTICS, mod_weights_vec)
        })
        save_combined_map(sample_image, combined_map, generation, RUN_FOLDER)
