import cv2
from src.utils import load_image_gt_pairs  # or dataset.py if you moved it

def test_dataset_loader():
    input_dir = "/home/evaldas/PycharmProjects/heuristic_saliency/data/input_images/"
    gt_dir = "/home/evaldas/PycharmProjects/heuristic_saliency/data/fixation_maps/"  # or ground_truth_saliency

    pairs = load_image_gt_pairs(input_dir, gt_dir)
    print(f"✅ Loaded {len(pairs)} image-ground-truth pairs.\n")

    for i, (img_path, gt_path) in enumerate(pairs[:5]):
        print(f"[{i}] Image: {img_path} | GT: {gt_path}")
        img = cv2.imread(img_path)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"❌ Failed to load image: {img_path}")
        elif gt is None:
            print(f"❌ Failed to load ground truth: {gt_path}")
        elif img.shape[:2] != gt.shape[:2]:
            print(f"⚠️ Size mismatch: {img.shape[:2]} vs. {gt.shape[:2]}")
        else:
            print(f"✅ Pair is OK. Size: {img.shape[:2]}")

if __name__ == "__main__":
    test_dataset_loader()
