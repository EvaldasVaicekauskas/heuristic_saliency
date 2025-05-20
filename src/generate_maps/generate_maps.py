import os
import json
import numpy as np
import cv2

from src.evaluation.evaluation import evaluate_saliency_image
from src.utils import combine_maps_with_weights, load_precomputed_maps_from_folders, load_dataset_in_gt
from src.modulation.modulation_utils import compute_modulated_weights, load_feature_vectors
from src.generate_maps.postprocessing import postprocess_saliency_map
from src.generate_maps.postprocessing import visualize_saliency_debug,visualize_saliency_map

# === Configuration ===
USE_MODULATION = True
EVALUATE = True
computed = False
WEIGHT_THRESHOLD = 0.01

# === Paths ===
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "work_dataset", "input_images")
GT_DIR = os.path.join(PROJECT_ROOT, "data", "work_dataset", "fixation_maps")
PRECOMP_BASE = os.path.join(PROJECT_ROOT, "data", "precomputed_maps")
MOD_FEATURES_DIR = os.path.join(PROJECT_ROOT, "src", "modulation", "data", "modulation_features")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "src", "generate_maps", "data", "output_maps_heuristic")
BEST_INDIVIDUAL_PATH = os.path.join(PROJECT_ROOT, "src", "generate_maps", "data", "configs", "best_individual.json")
#BEST_INDIVIDUAL_PATH = os.path.join(PROJECT_ROOT, "src", "generate_maps", "data", "configs", "ones_individual.json")

if computed:
    INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "work_dataset", "input_images")
    GT_DIR = os.path.join(PROJECT_ROOT, "data", "benchmark_data", "fixation_maps")
    SAL_DIR = os.path.join(PROJECT_ROOT, "data", "benchmark_data", "IKN")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "src", "generate_maps", "data", "output_maps_IKN")
    saliency_maps = load_dataset_in_gt(INPUT_DIR, SAL_DIR)


# === Load best individual & extract configs ===
with open(BEST_INDIVIDUAL_PATH, "r") as f:
    best_individual = json.load(f)

# Base or modulated weights
if "weights" in best_individual:
    base_weights = best_individual["weights"]
    ENABLED_HEURISTICS = list(base_weights.keys())
    base_weights_list = [base_weights[h] for h in ENABLED_HEURISTICS]
    modulation_weights = None
    use_modulation = False
else:
    base_weights = best_individual["base_weights"]
    modulation_weights = best_individual["modulation_weights"]
    ENABLED_HEURISTICS = list(base_weights.keys())
    base_weights_list = [base_weights[h] for h in ENABLED_HEURISTICS]
    use_modulation = USE_MODULATION  # use switch here


# If modulated, extract group info and features
if modulation_weights:
    MOD_GROUPS = list(modulation_weights.keys())
    # Reconstruct heuristic → group map
    HEURISTIC_TO_GROUP = {
        h: group for group in MOD_GROUPS for h in ENABLED_HEURISTICS if h.startswith(group)
    }
    # Load modulation features from external directory
    MOD_FEATURES = best_individual.get("modulation_config", {}).get("features", [])
else:
    HEURISTIC_TO_GROUP = {}
    MOD_FEATURES = []

# === Utilities ===
def save_map(output_dir, filename, map_data):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, map_data)

def generate_saliency_map(heuristic_maps, base_weights, modulation_weights,
                          heuristics, heuristic_to_group, image_features,
                          use_modulation, weight_threshold):
    if use_modulation:
        mod_weights = compute_modulated_weights(base_weights, modulation_weights, image_features,
                                                heuristics, heuristic_to_group)
        config = {
            h: {"weight": w}
            for h, w in zip(heuristics, mod_weights) if w >= weight_threshold
        }
    else:
        config = {
            h: {"weight": w}
            for h, w in zip(heuristics, list(base_weights.values())) if w >= weight_threshold
        }
    return combine_maps_with_weights(heuristic_maps, config)

# === Main Process ===
image_filenames = sorted(f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))
dataset = load_dataset_in_gt(INPUT_DIR, GT_DIR)
precomputed_maps = load_precomputed_maps_from_folders(PRECOMP_BASE, ENABLED_HEURISTICS, INPUT_DIR, GT_DIR)
mod_feature_vectors = load_feature_vectors(MOD_FEATURES_DIR) if use_modulation else {}
results = []
reresults = []

for i, filename in enumerate(image_filenames):
    print(f"🔹 Processing: {filename}")
    heuristic_maps = precomputed_maps[filename]

    image_features = None
    if use_modulation:
        image_features = mod_feature_vectors[os.path.splitext(filename)[0]]

    if computed:

        saliency_map = saliency_maps[i][1]

    else:
        saliency_map = generate_saliency_map(
            heuristic_maps, base_weights_list, modulation_weights,
            ENABLED_HEURISTICS, HEURISTIC_TO_GROUP, image_features,
            use_modulation, WEIGHT_THRESHOLD
        )

    if EVALUATE:
        gt_map = dataset[i][1]
        # Resize GT map if needed
        if saliency_map.shape != gt_map.shape:
            gt_map = cv2.resize(gt_map, (saliency_map.shape[1], saliency_map.shape[0]))

        cc, sim, kld = evaluate_saliency_image(saliency_map, gt_map, b_cc=True, b_sim=True, b_kl=True)

        results.append({
            "filename": filename,
            "SIM": float(sim),
            "CC": float(cc),
            "KL": float(kld)
        })

        print(f"Evaluated: {filename} -- Fitness: {sim+cc-kld}, SIM: {sim}, CC: {cc}, KLD: {kld}")

    # Optional postprocessing step
    if True:
        saliency_map = postprocess_saliency_map(saliency_map, dataset[i][0], dataset[i][1], filename)

    save_dir = OUTPUT_DIR#os.path.join(PROJECT_ROOT, "src", "generate_maps", "data", "summed_isolation")
    visualize_saliency_debug(
        image=dataset[i][0],
        saliency_map=saliency_map,
        gt_map=dataset[i][1],
        initial_saliency_map=None,
        stage_name="Modelio rezultatas",
        save_dir=save_dir,
        filename=f"{filename}_image_sal_gt.png"
    )

    visualize_saliency_map(saliency_map,save_dir,filename=f"{filename}_sal.png")

    REEVALUATE = True
    if REEVALUATE:
        gt_map = dataset[i][1]
        # Resize GT map if needed
        if saliency_map.shape != gt_map.shape:
            gt_map = cv2.resize(gt_map, (saliency_map.shape[1], saliency_map.shape[0]))

        cc, sim, kld = evaluate_saliency_image(saliency_map, gt_map, b_cc=True, b_sim=True, b_kl=True)

        reresults.append({
            "filename": filename,
            "SIM": float(sim),
            "CC": float(cc),
            "KL": float(kld)
        })

        print(f"Reevaluated: {filename}-- Fitness: {sim+cc-kld}, SIM: {sim}, CC: {cc}, KLD: {kld}")

    #save_map(OUTPUT_DIR, filename, saliency_map)

# === Save per-image metrics ===

sim_vals = [r["SIM"] for r in results]
cc_vals  = [r["CC"] for r in results]
kl_vals  = [r["KL"] for r in results]

sim_avg, sim_std = np.mean(sim_vals), np.std(sim_vals)
cc_avg,  cc_std  = np.mean(cc_vals),  np.std(cc_vals)
kl_avg,  kl_std  = np.mean(kl_vals),  np.std(kl_vals)

# Compute fitness per sample
fitness_vals = [s + c - k for s, c, k in zip(sim_vals, cc_vals, kl_vals)]
fitness_avg  = np.mean(fitness_vals)
fitness_std  = np.std(fitness_vals)

print("\n📊 Evaluation Summary:")
print(f"  SIM: {sim_avg:.4f} ± {sim_std:.4f}")
print(f"  CC:  {cc_avg:.4f} ± {cc_std:.4f}")
print(f"  KL:  {kl_avg:.4f} ± {kl_std:.4f}")
print(f"  fitness:  {fitness_avg:.4f} ± {fitness_std:.4f}")
print(f"  fitness:  {sim_avg+cc_avg-kl_avg} ± {sim_std+cc_std+kl_std}")

sim_vals = [r["SIM"] for r in reresults]
cc_vals  = [r["CC"] for r in reresults]
kl_vals  = [r["KL"] for r in reresults]

sim_avg, sim_std = np.mean(sim_vals), np.std(sim_vals)
cc_avg,  cc_std  = np.mean(cc_vals),  np.std(cc_vals)
kl_avg,  kl_std  = np.mean(kl_vals),  np.std(kl_vals)

# Compute fitness per sample
fitness_vals = [s + c - k for s, c, k in zip(sim_vals, cc_vals, kl_vals)]
fitness_avg  = np.mean(fitness_vals)
fitness_std  = np.std(fitness_vals)

print("\n📊 Postproc evaluation Summary:")
print(f"  SIM: {sim_avg:.4f} ± {sim_std:.4f}")
print(f"  CC:  {cc_avg:.4f} ± {cc_std:.4f}")
print(f"  KL:  {kl_avg:.4f} ± {kl_std:.4f}")
print(f"  fitness:  {fitness_avg:.4f} ± {fitness_std:.4f}")
print(f"  fitness:  {sim_avg+cc_avg-kl_avg} ± {sim_std+cc_std+kl_std}")

os.makedirs("results", exist_ok=True)
with open("results/evaluation_scores.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ {len(results)} saliency maps generated and evaluated.")