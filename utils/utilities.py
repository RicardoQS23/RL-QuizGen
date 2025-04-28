import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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
        f'../jsons/{test_num}/universes'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def compute_reward(alfa, state, targets, universe_shape):
    """Compute the reward for a given state and targets."""
    target_dim1 = targets[0]
    target_dim2 = targets[1]
    vec1, vec2 = state[:universe_shape[1]-5], state[universe_shape[1]-5:]
    first_dim_metric = cosine_similarity(vec1.reshape(1, -1), target_dim1.reshape(1, -1))[0][0]
    second_dim_metric = cosine_similarity(vec2.reshape(1, -1), target_dim2.reshape(1, -1))[0][0]
    
    reward = alfa * first_dim_metric + (1 - alfa) * second_dim_metric
    return reward

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
        agent_data = np.array(agent_data, dtype=np.int32)
    return agent_data

def load_baseline_data(test_number):
    """Load baseline inference data."""
    input_name = f"../jsons/{test_number}/baseline_inference/baseline_states"
    with open(f'{input_name}.json', 'r') as f:
        best_state = json.load(f)
        best_state = np.array(best_state, dtype=np.int32)
    return best_state