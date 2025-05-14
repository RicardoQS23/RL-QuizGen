import time
import torch
import argparse
import numpy as np
from agents.dqn_agent import DQNAgent
from agents.a2c_agent import A2CAgent
from agents.a3c_agent import A3CAgent
from agents.sarsa_agent import SARSAAgent
from environments.custom_env import CustomEnv
from utils.logging import save_to_log, save_to_json
from utils.utilities import load_data, get_best_state

def parse_args():
    parser = argparse.ArgumentParser(description='Inference of RL agent with different configurations')
    # Environment parameters
    parser.add_argument('--agent_type', type=str, default='dqn', choices=['dqn', 'a2c', 'a3c', 'sarsa'])
    parser.add_argument('--test_num', type=str, default="test1", help='Test number for saving results')
    parser.add_argument('--reward_threshold', type=float, default=0.85, help='Reward threshold for episode completion')
    parser.add_argument('--max_iterations', type=int, default=100, help='Maximum iterations per episode')
    
    # Training parameters
    parser.add_argument('--max_episodes', type=int, default=5000, help='Maximum number of episodes')
    parser.add_argument('--gamma', type=float, default=0.95, help='Discount factor')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--eps', type=float, default=1.0, help='Initial epsilon for exploration')
    parser.add_argument('--eps_decay', type=float, default=0.997, help='Epsilon decay rate')
    parser.add_argument('--eps_min', type=float, default=0.05, help='Minimum epsilon value')
    parser.add_argument('--target_sync_freq', type=int, default=1000, help='Frequency of target network updates')
    
    # Data generation parameters
    parser.add_argument('--data_source', type=int, default=1, choices=[0, 1, 2],
                       help='Data source: 0=load existing, 1=generate from real data, 2=generate synthetic')
    parser.add_argument('--num_topics', type=int, default=10, help='Number of topics for universe generation')
    parser.add_argument('--universe_size', type=int, default=10000, help='Size of the universe')
    parser.add_argument('--quiz_size', type=int, default=10, help='Number of MCQs per quiz')
    parser.add_argument('--generator_type', type=str, default='uniform', 
                        choices=['uniform', 'topic_focused', 'difficulty_focused', 'topic_diverse', 'difficulty_diverse', 'correlated'])
    
    # Experiment parameters
    parser.add_argument('--alfa_values', type=float, nargs='+', default=[0, 0.25, 0.5, 0.75, 1],
                       help='List of alfa values to test')
    parser.add_argument('--seed', type=int, default=23, help='Random seed for reproducibility')
    
    return parser.parse_args()

def agent_inference(env, agent, test_num, start_state):
    steps = 0
    done = False
    env.state = start_state
    all_states = [env.state]
    action_history = np.zeros(4, dtype=int)
    save_to_log(f"Starting inference from state {env.state}...", f'../logs/{test_num}/inference')
    while not done:
        steps += 1
        action, _, _ = agent.get_action(env.universe[env.state], epsilon=0.0)  # Use greedy policy
        next_state, reward, done, re, _, _ = env.step(action, steps)
        env.state = next_state
        save_to_log(f"Step {steps}: State = {env.state} Action={action}, Reward={re}",
                        f'../logs/{test_num}/inference')
        all_states.append(env.state)
        action_history[action] += 1
    save_to_log("Inference complete!", f'../logs/{test_num}/inference')
    return all_states, action_history

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Load data
    universe, targets = load_data(args.test_num)
    #universe = np.append(universe, np.zeros((universe.shape[0], 1)), axis=1)
    
    # Train agents for each alfa value
    save_to_log(f"Starting inference for {args.test_num}...", f'../logs/{args.test_num}/inference', mode='w')
    #for reward_threshold in [0.80, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1.0]:
    #    save_to_log(f"Reward threshold: {reward_threshold}", f'../logs/{args.test_num}/inference')
    all_best_states = []
    num_iterations = 0
    total_time = 0
    # Run inference for each alfa value
    for alfa in args.alfa_values:
        save_to_log(f"Inference for alfa = {alfa}...", 
                    f'../logs/{args.test_num}/inference')
        # Find baseline solution
        best_state, best_reward = get_best_state(universe, targets, alfa, num_topics=args.num_topics)
        all_best_states.append(best_state)
        
        save_to_log(f"Baseline for alfa = {alfa} -> State: {best_state}, Reward: {best_reward}",
                f'../logs/{args.test_num}/inference')
        # Run inference 10 times
        agent_inference_states = []
        agent_action_history = []
        for _ in range(10):
            start_state = np.random.choice(args.universe_size, 1)[0]
            save_to_log(f"Starting inference for {args.test_num}...", 
                        f'../logs/{args.test_num}/inference')
            # Create environment and agent
            env = CustomEnv(universe=universe, target_dim1=targets[0], target_dim2=targets[1], max_iterations=args.max_iterations,
                        num_topics=args.num_topics, alfa=alfa, reward_threshold=args.reward_threshold, state=start_state)
            
            # Create agent based on specified type
            if args.agent_type == 'dqn':
                agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_space.n, device=device,
                                lr=args.lr, gamma=args.gamma, target_sync_freq=args.target_sync_freq,
                                batch_size=args.batch_size)
            elif args.agent_type == 'a2c':
                agent = A2CAgent(state_dim=env.state_dim, action_dim=env.action_space.n, device=device,
                            lr=args.lr, gamma=args.gamma)
            elif args.agent_type == 'a3c':
                agent = A3CAgent(state_dim=env.state_dim, action_dim=env.action_space.n, device=device,
                            lr=args.lr, gamma=args.gamma)
            elif args.agent_type == 'sarsa':
                agent = SARSAAgent(state_dim=env.state_dim, action_dim=env.action_space.n, device=device,
                                lr=args.lr, gamma=args.gamma, eps=args.eps, eps_decay=args.eps_decay,
                                eps_min=args.eps_min)
            else:
                raise ValueError(f"Unknown agent type: {args.agent_type}")
            
            if args.agent_type == 'a2c':
                agent.load(f"../saved_agents/{args.test_num}/agent_alfa_{alfa}_bias.pkl")
            else:
                agent.load(f"../saved_agents/{args.test_num}/agent_alfa_{alfa}_bias.pth")
            
            if args.agent_type == 'a3c':
                agent.actor_critic.eval()
            else:
                agent.model.eval()
            # Train agent
            start_time = time.time()
            inference_states, action_history = agent_inference(env, agent, args.test_num, start_state)
            end_time = time.time()
            total_time += end_time - start_time
            num_iterations += len(inference_states)
            agent_inference_states.append(inference_states)
            agent_action_history.append(action_history)
        agent_action_dist=np.sum(agent_action_history, axis=0)/np.sum(np.sum(agent_action_history, axis=0))
        save_to_log(f"Agent action distribution for alfa = {alfa}: {agent_action_dist}", f'../logs/{args.test_num}/inference')
        save_to_json(agent_inference_states, f'../jsons/{args.test_num}/agent_inference/inference_states_alfa_{alfa}')
    save_to_json(all_best_states, f'../jsons/{args.test_num}/baseline_inference/baseline_states')
    #save_to_log(f"Avg number of iterations for reward_th={reward_threshold}: {round(num_iterations/(len(args.alfa_values)*10))}", f'../logs/{args.test_num}/inference')
    save_to_log(f"Avg number of iterations: {round(num_iterations/(len(args.alfa_values)*10))}", f'../logs/{args.test_num}/inference')
    save_to_log(f"Avg time per inference: {total_time/(len(args.alfa_values)*10):.2f}", f'../logs/{args.test_num}/inference')
    save_to_log(f"Inference complete!", f'../logs/{args.test_num}/inference')
if __name__ == "__main__":
    main() 