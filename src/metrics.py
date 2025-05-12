import json
import numpy as np
import os
import time
import argparse

def load_inference_data(test_num, alfa_values):
    inference_data = {}
    for alfa in alfa_values:
        file_path = os.path.join('jsons', test_num, 'agent_inference', f'inference_states_alfa_{alfa}.json')
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                inference_data[alfa] = json.load(f)
        else:
            print(f"Warning: Inference data not found for alfa = {alfa}.")
    return inference_data

def calculate_metrics(inference_data, alfa_values):
    metrics = {
        "data_metrics": {},
        "system_metrics": {}
    }
    
    total_inference_time = 0
    total_iterations = 0

    for alfa in alfa_values:
        if alfa in inference_data:
            all_states = inference_data[alfa]
            if len(all_states) == 0:
                continue
            num_episodes = len(all_states)
            rewards = [state['reward'] for state in all_states]
            episode_durations = [len(state['states']) for state in all_states]
            
            avg_reward = np.mean(rewards)
            avg_duration = np.mean(episode_durations)
            
            action_history = np.zeros(4)  # Adjust if number of actions is different
            for state in all_states:
                for action in state['actions']:
                    action_history[action] += 1
            action_distribution = action_history / np.sum(action_history) if np.sum(action_history) > 0 else action_history
            
            inference_start_time = time.time()  # Placeholder, not real inference time
            total_iterations += num_episodes
            inference_end_time = time.time()
            total_inference_time += inference_end_time - inference_start_time

            metrics["data_metrics"][alfa] = {
                'avg_reward': avg_reward,
                'avg_duration': avg_duration,
                'action_distribution': action_distribution.tolist(),
            }

    metrics["system_metrics"]["total_inference_time"] = total_inference_time
    metrics["system_metrics"]["total_iterations"] = total_iterations

    return metrics

def save_metrics(metrics, test_num):
    dir_path = os.path.join('jsons', test_num)
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, 'metrics.json')
    with open(file_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {file_path}")

def main(test_num):
    alfa_values = [0, 0.25, 0.5, 0.75, 1]

    inference_data = load_inference_data(test_num, alfa_values)
    metrics = calculate_metrics(inference_data, alfa_values)
    save_metrics(metrics, test_num)

    print("\nData Metrics:")
    for alfa, data in metrics["data_metrics"].items():
        print(f"Alfa {alfa}:")
        print(f"  Avg Reward: {data['avg_reward']}")
        print(f"  Avg Duration: {data['avg_duration']}")
        print(f"  Action Distribution: {data['action_distribution']}")
    
    print("\nSystem Metrics:")
    print(f"Total Inference Time: {metrics['system_metrics']['total_inference_time']:.2f} seconds")
    print(f"Total Iterations: {metrics['system_metrics']['total_iterations']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate metrics for a given test set.")
    parser.add_argument('--test_num', type=str, required=True, help='Test set identifier (e.g., test11)')
    args = parser.parse_args()
    main(args.test_num)
