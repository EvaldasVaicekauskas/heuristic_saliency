import cv2
import numpy as np
import random
import os
import json

from src.saliency import SaliencyModel
from src.evaluation.evaluation import evaluate_saliency_image
from src.utils import combine_maps_with_weights

from src.modulation.modulation_utils import (
    initialize_modulated_individual,
    split_genome,
    compute_modulated_weights
)

# === Fitness Logic ===
def fitness_function(cc, sim, kld):
    return (cc / 1 + sim / 1) - kld / 1

# === GA Core Functions ===
def initialize_modulated_population(pop_size, heuristics, groups, features):
    return [
        initialize_modulated_individual(heuristics, groups, features)
        for _ in range(pop_size)
    ]

def evaluate_modulated_population(population, precomputed_maps, dataset, image_filenames,
                                   heuristics, groups, features, heuristic_to_group,
                                   modulation_feature_dir, weight_threshold):

    evaluated = []
    for idx, genome in enumerate(population):
        print(f"Individual {idx}/{len(population)}")
        fitness, avg_sim, avg_cc, avg_kld = compute_fitness_mod(
            genome, precomputed_maps, dataset, image_filenames,
            heuristics, groups, features, heuristic_to_group,
            modulation_feature_dir, weight_threshold
        )
        evaluated.append((genome, fitness, avg_sim, avg_cc, avg_kld))
    return evaluated

def compute_fitness_mod(genome, precomputed_maps, dataset, image_filenames,
                        heuristics, groups, features, heuristic_to_group,
                        modulation_feature_dir, weight_threshold):
    cc_list, sim_list, kld_list = [], [], []

    base_weights, modulation_weights = split_genome(genome, heuristics, groups, features)

    for idx, filename in enumerate(image_filenames):
        heuristic_maps = precomputed_maps[filename]
        gt_map = dataset[idx][1]

        # Load modulation features
        feat_path = os.path.join(modulation_feature_dir, os.path.splitext(filename)[0] + ".json")
        with open(feat_path, "r") as f:
            feature_dict = json.load(f)
        image_feature_vec = np.array([feature_dict[f] for f in features], dtype=np.float32)

        # Apply modulation
        modulated_weights = compute_modulated_weights(
            base_weights, modulation_weights, image_feature_vec,
            heuristics, heuristic_to_group
        )

        config = mod_weights_to_config(modulated_weights, heuristics, threshold=weight_threshold)
        combined = combine_maps_with_weights(heuristic_maps, config)

        # Resize GT if needed
        if combined.shape != gt_map.shape:
            gt_map = cv2.resize(gt_map, (combined.shape[1], combined.shape[0]))

        cc, sim, kld = evaluate_saliency_image(combined, gt_map, b_cc=True, b_sim=True, b_kl=True)

        if np.isnan(cc): cc = -2.0
        if np.isnan(cc): sim = -2.0
        if np.isnan(kld): kld = 2.0

        cc_list.append(cc)
        sim_list.append(sim)
        kld_list.append(kld)

    avg_cc = np.mean(cc_list)
    avg_sim = np.mean(sim_list)
    avg_kld = np.mean(kld_list)
    fitness = fitness_function(avg_cc, avg_sim, avg_kld)

    #print(f"Fitness: {fitness:.4f}, Avg CC: {avg_cc:.4f}, SIM: {avg_sim:.4f}, KL: {avg_kld:.4f}, over {len(cc_list)} samples")
    return fitness, avg_sim, avg_cc, avg_kld

def compute_population_std(individuals):
    genes = [ind for ind, *_ in individuals]
    arr = np.array(genes)
    return np.mean(np.std(arr, axis=0))

# === Helper ===
def mod_weights_to_config(weights, heuristic_names, threshold=0.01):
    config = {}
    for name, weight in zip(heuristic_names, weights):
        config[name] = {
            "enabled": weight >= threshold,
            "weight": float(weight)
        }
    return config
