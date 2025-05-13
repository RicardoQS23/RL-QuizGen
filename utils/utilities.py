import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils.logging import save_to_log
def setup_directories(test_num):
    """Create necessary directories for saving results"""
    directories = [
        f'../data/{test_num}',
        f'../logs/{test_num}',
        f'../saved_agents/{test_num}',
        f'../images/{test_num}',
        f'../jsons/{test_num}/success',
        f'../jsons/{test_num}/reward',
        f'../jsons/{test_num}/reward_dim1',
        f'../jsons/{test_num}/reward_dim2',
        f'../jsons/{test_num}/qvalues',
        f'../jsons/{test_num}/actions',
        f'../jsons/{test_num}/loss',
        f'../jsons/{test_num}/exploration_ratio',
        f'../jsons/{test_num}/universes',
        f'../jsons/{test_num}/agent_inference',
        f'../jsons/{test_num}/baseline_inference'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def create_worker_dir(log_path):
    # Ensure the directory exists
    os.makedirs(log_path, exist_ok=True)

def compute_reward(alfa, state, targets):
    """Compute the reward for a given state and targets."""
    target_dim1 = targets[0]
    target_dim2 = targets[1]
    vec1, vec2 = state[:target_dim1.shape[0]], state[target_dim1.shape[0]:]
    first_dim_metric = cosine_similarity(vec1.reshape(1, -1), target_dim1.reshape(1, -1))[0][0]
    second_dim_metric = cosine_similarity(vec2.reshape(1, -1), target_dim2.reshape(1, -1))[0][0]
    
    reward = alfa * first_dim_metric + (1 - alfa) * second_dim_metric
    return reward

def get_best_state_inference(visited_states, targets, alfa):
    """Get the best state from the inference data."""
    rewards = []
    target_dim1 = targets[0]
    target_dim2 = targets[1]
    for state in visited_states:
        vec1, vec2 = state[:target_dim1.shape[0]], state[target_dim1.shape[0]:] 
        
        first_dim_metric = cosine_similarity(vec1.reshape(1, -1), target_dim1.reshape(1, -1))[0][0]
        second_dim_metric = cosine_similarity(vec2.reshape(1, -1), target_dim2.reshape(1, -1))[0][0]
        
        reward = alfa * first_dim_metric + (1 - alfa) * second_dim_metric
        rewards.append(reward)
    
    best_idx = np.argmax(rewards)
    return rewards[best_idx]

def get_inference_data(test_number, alfa, universe, targets, alfa_values=[0, 0.25, 0.5, 0.75, 1]):
    # Load agent inference data
    visited_states = load_agent_data(test_number, alfa)
    # Load baseline data
    baseline_states = load_baseline_data(test_number)
    
    best_inference_reward = []
    for inference_states in visited_states:
        best_inference_reward.append(get_best_state_inference(universe[inference_states], targets, alfa))

    #round np.mean(best_inference_reward) to 4 decimal places
    mean_reward = np.mean(best_inference_reward)
    best_reward = compute_reward(alfa, universe[baseline_states[alfa_values.index(alfa)]], targets) 
    std_reward = np.std(best_inference_reward)
    save_to_log(f"{len(visited_states)} inference runs for alfa {alfa}: Average reward -> {mean_reward:.3f}, BRUTEFORCE Reward -> {best_reward:.3f}, Std -> {std_reward:.3f}", f'../logs/{test_number}/inference')

    return visited_states[np.argmax(best_inference_reward)], baseline_states

def get_best_state(universe, targets, alfa, num_topics):
    """Find the best state in the universe based on the reward value"""
    rewards = []
    target_dim1 = targets[0]
    target_dim2 = targets[1]
    
    for state in universe:
        vec1, vec2 = state[:num_topics], state[num_topics:] 
        
        first_dim_metric = cosine_similarity(vec1.reshape(1, -1), target_dim1.reshape(1, -1))[0][0]
        second_dim_metric = cosine_similarity(vec2.reshape(1, -1), target_dim2.reshape(1, -1))[0][0]
        
        reward = alfa * first_dim_metric + (1 - alfa) * second_dim_metric
        rewards.append(reward)
    
    best_idx = np.argmax(rewards)
    return best_idx, rewards[best_idx]


def load_data(test_number):
    """Load universe and targets data from JSON files."""
    with open(f"../jsons/{test_number}/universes/targets.json", "r") as f:
        targets = json.load(f)
        targets = [np.array(t, dtype=np.float32) for t in targets]
    
    with open(f"../jsons/{test_number}/universes/universe.json", "r") as f:
        universe = json.load(f)
        universe = np.array(universe, dtype=np.float32)
    
    return universe, targets

def load_agent_data(test_number, alfa):
    """Load agent inference data for a specific alfa value."""
    input_name = f"../jsons/{test_number}/agent_inference/inference_states_alfa_{alfa}"
    with open(f'{input_name}.json', 'r') as f:
        agent_data = json.load(f)
        agent_data = [np.array(d, dtype=np.int32) for d in agent_data]
        # agent_data = np.array(agent_data, dtype=np.int32)
    return agent_data

def load_baseline_data(test_number):
    """Load baseline inference data."""
    input_name = f"../jsons/{test_number}/baseline_inference/baseline_states"
    with open(f'{input_name}.json', 'r') as f:
        best_state = json.load(f)
        best_state = np.array(best_state, dtype=np.int32)
    return best_state