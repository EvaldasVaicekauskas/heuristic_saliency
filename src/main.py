
import argparse
from saliency import SaliencyModel

def main():
    parser = argparse.ArgumentParser(description="Run heuristic saliency model with visualization.")
    parser.add_argument("--input", type=str, required=True, help="Path to input image folder")
    parser.add_argument("--output", type=str, required=True, help="Path to save output saliency maps")
    parser.add_argument("--visualization", type=str, required=True, help="Path to save heuristic visualizations")
    args = parser.parse_args()

    # Define heuristic settings (Enable/Disable & Weight)
    heuristic_config = {
        "red": {"enabled": True, "weight": 0.2},
        "contrast": {"enabled": True, "weight": 0.2},
        "saturation": {"enabled": True, "weight": 0.2},
        "superpixel_contrast": {"enabled": True, "weight": 0.2},
        "complementary_colors": {"enabled": True, "weight": 0.2}
    }

    # Initialize and run the model
    model = SaliencyModel(args.input, args.output, args.visualization, heuristic_config)
    model.process_images()

if __name__ == "__main__":
    main()