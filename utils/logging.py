import json
import torch
import pickle
import numpy as np
import os

def save_to_log(data, output_name, mode='a', flag=True):   
    with open(f"{output_name}.log", mode) as f:
        f.write(data + '\n')
    if flag:
        print(data)

def create_worker_dir(log_path):
    # Ensure the directory exists
    os.makedirs(log_path, exist_ok=True)


def save_to_json(data, output_name):
    """Save the data to a JSON file."""
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        else:
            return obj
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_name), exist_ok=True)
    
    with open(f"{output_name}.json", "w") as f:
        json.dump(convert(data), f, indent=4)

def save_data(training_data, alfa, test_num):
    """Save all training data to JSON files."""
    # Create base directories
    base_dirs = [
        f'../jsons/{test_num}/success',
        f'../jsons/{test_num}/reward',
        f'../jsons/{test_num}/reward_dim1',
        f'../jsons/{test_num}/reward_dim2',
        f'../jsons/{test_num}/qvalues',
        f'../jsons/{test_num}/actions',
        f'../jsons/{test_num}/loss',
        f'../jsons/{test_num}/exploration_ratio'
    ]
    
    for directory in base_dirs:
        os.makedirs(directory, exist_ok=True)
    
    # Save data to JSON files
    save_to_json(training_data['success_episodes'], f'../jsons/{test_num}/success/all_success_alfa_{alfa}')
    save_to_json(training_data['episode_rewards'], f'../jsons/{test_num}/reward/all_rewards_alfa_{alfa}')
    save_to_json(training_data['episode_rewards_dim1'], f'../jsons/{test_num}/reward_dim1/all_rewards_dim1_alfa_{alfa}')
    save_to_json(training_data['episode_rewards_dim2'], f'../jsons/{test_num}/reward_dim2/all_rewards_dim2_alfa_{alfa}')
    save_to_json(training_data['episode_avg_qvalues'], f'../jsons/{test_num}/qvalues/all_qvalues_alfa_{alfa}')
    save_to_json(training_data['episode_actions'], f'../jsons/{test_num}/actions/all_actions_alfa_{alfa}')
    save_to_json(training_data['episode_losses'], f'../jsons/{test_num}/loss/all_losses_alfa_{alfa}')
    
    explore_exploit_ratio = np.array(training_data['exploration_counts']) / (
        np.array(training_data['exploration_counts']) + np.array(training_data['exploitation_counts'])
    )
    save_to_json(explore_exploit_ratio, f'../jsons/{test_num}/exploration_ratio/all_exploration_ratio_alfa_{alfa}')

def save_agent(save_path, agent):
    """Save the agent's model."""
    try:
        # Create directory for saving agent if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        if hasattr(agent, 'model') and isinstance(agent.model, torch.nn.Module):
            # For DQN agent
            torch.save(agent.model.state_dict(), save_path)
            print(f"Model saved at {save_path}")
        elif hasattr(agent, 'actor_critic'):
            # For A3C agent
            agent.save(save_path)
            print(f"A3C model saved at {save_path}")
            # Save training data separately
            training_data = agent.get_training_data()
            save_data(training_data, agent.training_data.get('alfa', 0), agent.training_data.get('test_num', 'default'))
        else:
            # For other agents
            with open(save_path.replace(".pth", ".pkl"), 'wb') as f:
                pickle.dump(agent, f)
            print(f"Agent object saved at {save_path.replace('.pth', '.pkl')}")
    except Exception as e:
        print(f"Failed to save agent model: {e}") 