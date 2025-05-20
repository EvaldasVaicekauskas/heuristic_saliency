
import cv2
import random
import numpy as np


from src.saliency import SaliencyModel
from src.evaluation.evaluation import evaluate_saliency_image  # assuming you have this
from src.utils import combine_maps_with_weights




def weights_to_config(weights, heuristic_names, threshold=0.01):
    """
    Maps a list of weights to a heuristic_config dict using descriptive heuristic names.
    Only includes heuristics with weight >= threshold.
    """
    config = {}
    for name, weight in zip(heuristic_names, weights):
        config[name] = {
            "enabled": weight >= threshold,
            "weight": float(weight)
        }
    return config

def fitness_function(cc, sim, kld):
    fitness = (cc/0.06 +sim/0.3 )-kld/2.5
    fitness = (cc / 1 + sim / 1) - kld / 1

    return fitness

def compute_fitness(heuristic_config, dataset):
    """
    Evaluates fitness by averaging CC + SIM - KL over all image pairs.
    Lower KL = better → so we subtract it.
    """

    model = SaliencyModel(heuristic_config=heuristic_config)

    cc_list, sim_list, kld_list = [], [], []

    for image, gt_map in dataset:
        #pred_map = model.generate_saliency_for_image(image)
        pred_map = model.combine_saliency_maps(heuristic_config)

        if pred_map.shape != gt_map.shape:
            gt_map = cv2.resize(gt_map, (pred_map.shape[1], pred_map.shape[0]))

        cc, sim, kld = evaluate_saliency_image(pred_map, gt_map)

        cc_list.append(cc)
        sim_list.append(sim)
        kld_list.append(kld)

    avg_cc = np.mean(cc_list)
    avg_sim = np.mean(sim_list)
    avg_kld = np.mean(kld_list)

    fitness = fitness_function(avg_cc, avg_sim, avg_kld)
    print(f"Fitness: {fitness}, Avg cc: {avg_cc}, Avg sim: {avg_sim}, Avg KLd: {avg_kld}, over {len(cc_list)} samples")

    return fitness



def initialize_population(pop_size, num_genes):
    """
    Creates a population of random individuals.
    Each individual is a list of floats (weights for heuristics).
    """
    return [
        [random.uniform(0.0, 1) for _ in range(num_genes)]
        for _ in range(pop_size)
    ]

def evaluate_population(population, precomputed_maps, dataset, image_filenames, heuristic_names, weight_threshold):
    """
    Evaluate each individual in the population using the fitness function.

    Returns:
        List of tuples: [(individual, fitness_score), ...]
    """

    evaluated = []
    metrics = []
    for individual in population:
        # print("--------------------------------------")
        print(f"Individual {len(evaluated)}")
        config = weights_to_config(individual, heuristic_names, threshold=weight_threshold)
        # for name, cfg in config.items():
        #     print(f"{name}: {cfg['weight']:.4f}")

        fitness, avg_sim, avg_cc, avg_kl = compute_fitness_from_config(config, precomputed_maps, dataset, image_filenames)
        evaluated.append((individual, fitness,avg_sim,avg_cc,avg_kl))

    return evaluated

def compute_fitness_from_config(config, precomputed_maps, dataset, image_filenames):
    cc_list, sim_list, kld_list = [], [], []

    for idx in range(len(dataset)):
        filename = image_filenames[idx]
        heuristic_maps = precomputed_maps[filename]
        gt_map = dataset[idx][1]

        combined = combine_maps_with_weights(heuristic_maps, config)

        # Resize GT map if needed
        if combined.shape != gt_map.shape:
            gt_map = cv2.resize(gt_map, (combined.shape[1], combined.shape[0]))

        cc, sim, kld = evaluate_saliency_image(combined, gt_map,b_cc=True,b_sim=True,b_kl=True)
        cc_list.append(cc)
        sim_list.append(sim)
        kld_list.append(kld)

    avg_cc = np.mean(cc_list)
    avg_sim = np.mean(sim_list)
    avg_kld = np.mean(kld_list)

    fitness = fitness_function(avg_cc, avg_sim, avg_kld)
    print(f"Fitness: {fitness}, Avg cc: {avg_cc}, Avg sim: {avg_sim}, Avg KLd: {avg_kld}, over {len(cc_list)} samples")

    return fitness, avg_cc, avg_sim, avg_kld

