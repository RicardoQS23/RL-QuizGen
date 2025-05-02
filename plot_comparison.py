import argparse
from utils.utilities import load_data
from utils.plotting import plot_comparison

def parse_args():
    parser = argparse.ArgumentParser(description='Plot comparison of RL agent and baseline')
    parser.add_argument('--test_number', type=str, default="test1", help='Test number for saving results')
    parser.add_argument('--alfa_values', type=float, nargs='+', default=[0, 0.25, 0.5, 0.75, 1],
                       help='List of alfa values to test')
    return parser.parse_args()

def main():
    args = parse_args()
    # Load data
    universe, targets = load_data(args.test_number)
    
    # Create comparison plots
    plot_comparison(
        universe=universe,
        targets=targets,
        alfa_values=args.alfa_values,
        test_number=args.test_number,
        output_name=f"{args.test_number}/comparison",
        save=True,
        show=False
    )

if __name__ == "__main__":
    main() 