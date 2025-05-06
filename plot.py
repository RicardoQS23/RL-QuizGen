import argparse
from utils.plotting import plot_all_results_a3c

def parse_args():
    parser = argparse.ArgumentParser(description='Plot A3C results')
    parser.add_argument('--test_num', type=str, required=True, help='Test number to plot')
    parser.add_argument('--num_workers', type=int, required=True, help='Number of workers')
    parser.add_argument('--alfa_values', type=float, nargs='+', default=[0, 0.25, 0.5, 0.75, 1],
                       help='List of alfa values to plot')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Plot results for each worker
    for worker_id in range(args.num_workers):
        print(f"Plotting results for worker {worker_id}...")
        plot_all_results_a3c(args.test_num, args.alfa_values, worker_id)
    
    print("Plotting complete!")

if __name__ == "__main__":
    main()
