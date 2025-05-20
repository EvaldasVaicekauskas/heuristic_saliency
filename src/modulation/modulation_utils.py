# Re-run code after kernel reset to define the modulation genome handling functions

import os
import json
import numpy as np



def build_corrected_modulation_config(ga_config_path, modulation_config_path, output_path):
    with open(ga_config_path, "r") as f:
        ga_config = json.load(f)

    with open(modulation_config_path, "r") as f:
        mod_config = json.load(f)

    ga_heuristics = set(ga_config["heuristics"].keys())

    # Flatten all modulation-config heuristics
    mod_heuristics = set()
    for group in mod_config["groups"].values():
        mod_heuristics.update(group)

    # Find missing
    missing = sorted(ga_heuristics - mod_heuristics)

    if not missing:
        print("✅ All heuristics are already included in modulation config.")
        return

    print("⚠️ Missing heuristics not assigned to modulation groups:")
    for h in missing:
        print(f"  - {h}")

    # Create corrected config with "misc" added
    corrected_groups = mod_config["groups"].copy()
    corrected_groups["misc"] = missing

    corrected_config = {
        "features": mod_config["features"],
        "groups": corrected_groups
    }

    with open(output_path, "w") as f:
        json.dump(corrected_config, f, indent=2)

    print(f"\n✅ Corrected modulation config written to: {output_path}")

def initialize_modulated_individual(heuristics, groups, features):
    """
    Initialize an individual with random base weights and modulation weights.

    Returns:
        genome: np.array of size len(heuristics) + len(groups) * len(features)
    """
    num_base_weights = len(heuristics)
    num_modulation_weights = len(groups) * len(features)

    base_weights = np.random.uniform(0, 1, size=num_base_weights)
    modulation_weights = np.random.uniform(0, 1, size=num_modulation_weights)

    return np.concatenate([base_weights, modulation_weights])

def split_genome(genome, heuristics, groups, features):
    """
    Splits genome vector into base and modulation weights.

    Returns:
        base_weights: np.array
        modulation_weights_dict: { group_name: np.array of feature weights }
    """
    num_base = len(heuristics)
    num_features = len(features)

    base_weights = genome[:num_base]
    mod_raw = genome[num_base:]

    modulation_weights = {
        group: mod_raw[i * num_features:(i + 1) * num_features]
        for i, group in enumerate(groups)
    }

    return base_weights, modulation_weights

def compute_modulated_weights(base_weights, modulation_weights, image_features, heuristics, heuristic_to_group):
    """
    Applies modulation to base weights using image features and group modulation.

    Returns:
        modulated_weights: list of final heuristic weights after modulation
    """
    modulated_weights = []
    for i, h in enumerate(heuristics):
        group = heuristic_to_group[h]
        mod_vec = modulation_weights[group]
        # dot_p = np.dot(mod_vec, image_features)
        #
        # # Scaled sigmoid → [-1, 1]
        # activation = 2 * (1 / (1 + np.exp(-dot_p)) - 0.5)
        #
        # modulated_weights.append(base_weights[i] * activation)

        mod_vec_centered = 2 * (np.array(mod_vec) - 0.5)  # [-1, 1]
        dot_p = np.dot(mod_vec_centered, image_features)
        activation = max(0.0, dot_p)  # cancel if negative
        modulated = base_weights[i] * activation
        modulated_weights.append(modulated)

    return modulated_weights

def load_modulation_config(path):
    with open(path, "r") as f:
        config = json.load(f)
    return config["features"], config["groups"]

def build_heuristic_to_group(group_to_heuristics):
    """
    Builds a dictionary mapping each heuristic to its group name.

    Args:
        group_to_heuristics (dict): group name -> list of heuristics

    Returns:
        heuristic_to_group (dict): heuristic name -> group name
    """
    heuristic_to_group = {}
    for group, heuristics in group_to_heuristics.items():
        for h in heuristics:
            heuristic_to_group[h] = group
    return heuristic_to_group

def load_feature_vectors(feature_dir):
    """
    Loads modulation feature vectors from JSON files using base filename (no extension) as key.

    Returns:
        Dictionary: { "image_basename": feature_vector }
    """
    features = {}
    for filename in os.listdir(feature_dir):
        if not filename.lower().endswith(".json"):
            continue

        base_name = os.path.splitext(filename)[0]
        with open(os.path.join(feature_dir, filename), "r") as f:
            feature_dict = json.load(f)
            features[base_name] = list(feature_dict.values())

    return features

def save_best_individual_mod(generation, individual, fitness, avg_sim, avg_cc, avg_kl,
                             heuristic_names, groups, features, out_path="results/best_individual.json"):
    base_weights, modulation_weights = split_genome(individual, heuristic_names, groups, features)

    data = {
        "generation": int(generation),
        "fitness": float(round(fitness, 4)),
        "avg_sim": float(round(avg_sim, 4)),
        "avg_cc": float(round(avg_cc, 4)),
        "avg_kl": float(round(avg_kl, 4)),
        "base_weights": {
            name: float(round(weight, 4))
            for name, weight in zip(heuristic_names, base_weights)
        },
        "modulation_weights": {
            group: [float(round(w, 4)) for w in mod_vec]
            for group, mod_vec in modulation_weights.items()
        }
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

def save_generation_state_mod(generation, evaluated_population, heuristic_names, groups, features, folder="results/generations"):
    os.makedirs(folder, exist_ok=True)
    generation_data = []

    for individual, fitness, avg_sim, avg_cc, avg_kl in evaluated_population:
        base_weights, mod_weights = split_genome(individual, heuristic_names, groups, features)
        weights = {
            "fitness": float(round(fitness, 4)),
            "avg_sim": float(round(avg_sim, 4)),
            "avg_cc": float(round(avg_cc, 4)),
            "avg_kl": float(round(avg_kl, 4)),
            "base_weights": {
                name: float(round(w, 4)) for name, w in zip(heuristic_names, base_weights)
            },
            "modulation_weights": {
                group: [float(round(w, 4)) for w in mod_weights[group]]
                for group in groups
            }
        }
        generation_data.append(weights)

    with open(os.path.join(folder, f"gen_{generation:02d}.json"), "w") as f:
        json.dump(generation_data, f, indent=2)
