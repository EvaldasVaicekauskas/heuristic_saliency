import os
from utils import resize_saliency_map

# Define dataset paths
datasets = {
    "paint": {
        "stimuli_folder": "/home/evaldas/PycharmProjects/datasets/paint/stimuli/",
        "saliency_folder": "/home/evaldas/PycharmProjects/datasets/paint/fixation_maps/",
        "save_folder": "/home/evaldas/PycharmProjects/datasets/paint/fixation_maps_match_size"
    },
    "pand": {
        "stimuli_folder": "/home/evaldas/PycharmProjects/datasets/pand/stimuli/",
        "saliency_folder": "/home/evaldas/PycharmProjects/datasets/pand/fixation_maps/",
        "save_folder": "/home/evaldas/PycharmProjects/datasets/pand/fixation_maps_match_size"
    }
}

# Process each dataset separately
for dataset_name, paths in datasets.items():
    stimuli_folder = paths["stimuli_folder"]
    saliency_folder = paths["saliency_folder"]
    save_folder = paths["save_folder"]

    os.makedirs(save_folder, exist_ok=True)

    print(f"\n🔄 Processing dataset: {dataset_name.upper()}")

    for filename in os.listdir(stimuli_folder):
        stim_path = os.path.join(stimuli_folder, filename)
        sal_path = os.path.join(saliency_folder, filename)  # Assuming saliency maps have a "saliency_" prefix

        if not os.path.exists(sal_path):
            print(f"⚠️ Skipping {filename}: No matching saliency map found.")
            continue

        save_path = os.path.join(save_folder, filename)
        resize_saliency_map(sal_path, stim_path, dataset_name, save_path)

print("\n✅ All saliency maps resized successfully!")
