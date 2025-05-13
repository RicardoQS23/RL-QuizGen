import os
import json
import time
import torch
import random
import argparse
import numpy as np
import pandas as pd
from agents.dqn_agent import DQNAgent
from agents.a2c_agent import A2CAgent
from agents.a3c_agent import A3CAgent
from agents.sarsa_agent import SARSAAgent
from utils.plotting import plot_all_results, plot_all_results_a3c
from environments.custom_env import CustomEnv
from utils.logging import save_to_log, save_data, save_agent
from utils.utilities import setup_directories, get_best_state
from universe_generator import generate_universe_from_real_data, generate_synthetic_universe, load_universe, save_universe

def parse_args():
    parser = argparse.ArgumentParser(description='Train RL agent with different configurations')
    
    # Agent type parameter
    parser.add_argument('--agent_type', type=str, default='dqn', choices=['dqn', 'a2c', 'a3c', 'sarsa'],
                       help='Type of agent to use: dqn, a3c, or sarsa')
    
    # Replay buffer type parameter
    parser.add_argument('--replay_buffer', type=str, default='prioritized', choices=['normal', 'prioritized'],
                       help='Type of replay buffer to use: normal or prioritized')
    
    # Target distribution parameter
    parser.add_argument('--target_distribution', type=str, default='uniform',
                       choices=['uniform', 'sparse_topic', 'sparse_difficulty', 'sparse_topic_difficulty'], 
                       help='Type of target distribution to use: uniform (both uniform), sparse_topic (sparse topics,\
                          uniform difficulty), sparse_difficulty (uniform topics, sparse difficulty), or sparse_topic_difficulty (sparse topics, sparse difficulty)')
    
    # Environment parameters
    parser.add_argument('--test_num', type=str, default="test1", help='Test number for saving results')
    parser.add_argument('--reward_threshold', type=float, default=0.85, help='Reward threshold for episode completion')
    parser.add_argument('--max_iterations', type=int, default=100, help='Maximum iterations per episode')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of worker threads')
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
    parser.add_argument('--dataset', type=str, default='medical', choices=['math', 'medical'],
                       help='Dataset to use when generating from real data: math or medical')
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


def train_agent(env, agent, max_episodes, test_num, args):
    """Train the agent for specified number of episodes"""
    all_visited_states = set()
    
    # For A3C, start the workers and let them handle training
    if isinstance(agent, A3CAgent):
        #save_to_log(f"Starting A3C training with {agent.num_workers} workers...", f'../logs/{test_num}/training')
        print(f"Starting A3C training with {agent.num_workers} workers...")
        agent.training_data['max_episodes'] = max_episodes
        agent.start_workers(env)
        
        # Wait for workers to complete their episodes
        while any(worker.is_alive() for worker in agent.workers):
            time.sleep(1)  # Check every second
            
        # Stop workers and clean up
        agent.stop_workers()
        #save_to_log(f'A3C training complete! Total Visited States: {len(agent.all_visited_states)}', f'../logs/{test_num}/training')
        return
    
    # For other agents (DQN, SARSA), use single-agent training
    for ep in range(max_episodes):
        visited_states = set()
        done = False
        total_reward = total_reward_dim1 = total_reward_dim2 = 0
        exploration_count = exploitation_count = 0
        num_iterations = 0
        episode_actions = []
        episode_avg_qvalues = []
        
        state = env.reset()
        save_to_log(f"Episode {ep + 1} started on state {state}...", f'../logs/{test_num}/training')
        
        while not done:
            if state not in all_visited_states:
                all_visited_states.add(state)
            if state not in visited_states:
                visited_states.add(state)
                
            action, mean_q_value, explore = agent.get_action(env.universe[state], agent.training_data['epsilon'])
            next_state, reward, done, success, reward_dim1, reward_dim2 = env.step(action, num_iterations)
            
            agent.train_step(env.universe[state], action, reward, env.universe[next_state], done)
            
            total_reward += reward
            total_reward_dim1 += reward_dim1
            total_reward_dim2 += reward_dim2
            state = next_state
            
            if explore:
                exploration_count += 1
            else:
                exploitation_count += 1
            
            episode_actions.append(action)
            episode_avg_qvalues.append(mean_q_value)
            save_to_log(f'Iteration {num_iterations + 1} State={state} Action={action} Reward={reward}', 
                       f'../logs/{test_num}/training', flag=False)
            num_iterations += 1
            
        agent.training_data['episode_count'] += 1
        agent.training_data['epsilon'] = max(args.eps_min, agent.training_data['epsilon'] * args.eps_decay)
        
        save_to_log(f'EP{ep + 1} Number of Visited States={len(visited_states)} EpisodeReward={total_reward/num_iterations}', 
                   f'../logs/{test_num}/training', flag=False)
        
        agent.update_episode_data(total_reward, total_reward_dim1, total_reward_dim2, 
                                exploration_count, exploitation_count, success,
                                episode_actions, episode_avg_qvalues, num_iterations)
    
    save_to_log(f'Train complete! Total Visited States: {len(all_visited_states)}', f'../logs/{test_num}/training')

def main():
    program_start_time = time.time()
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # Setup directories
    setup_directories(args.test_num)
    
    # Handle data source
    if args.data_source == 0:  # Load existing data
        try:
            universe, targets = load_universe(args.test_num)
            num_topics = args.num_topics
            save_to_log("Loaded existing data", f'../logs/{args.test_num}/training', mode='w')
        except (FileNotFoundError, json.JSONDecodeError) as e:
            save_to_log(f"Error loading existing data: {e}", f'../logs/{args.test_num}/training', mode='w')
            save_to_log("Falling back to generating from real data", f'../logs/{args.test_num}/training')
            universe, targets, num_topics = generate_universe_from_real_data(
                test_num=args.test_num,
                num_topics=args.num_topics,
                universe_size=args.universe_size,
                quiz_size=args.quiz_size,
                dataset=args.dataset
            )
    elif args.data_source == 1:  # Generate from real data
        save_to_log("Generating universe from real data", f'../logs/{args.test_num}/training', mode='w')
        universe, targets, num_topics = generate_universe_from_real_data(
            test_num=args.test_num,
            num_topics=args.num_topics,
            universe_size=args.universe_size,
            quiz_size=args.quiz_size,
            dataset=args.dataset,
            target_distribution=args.target_distribution
        )
    else:  # Generate synthetic data
        save_to_log("Generating synthetic universe", f'../logs/{args.test_num}/training', mode='w')
        universe, targets = generate_synthetic_universe(
            test_num=args.test_num,
            num_topics=args.num_topics,
            universe_size=args.universe_size,
            num_difficulties=5,
            universe_distribution=args.generator_type,
            target_distribution=args.target_distribution
        )
        num_topics = args.num_topics
    # Log initial information
    save_to_log(f"Starting {args.test_num} ... using device: {device}", f'../logs/{args.test_num}/training')
    save_to_log(f"Parameters used: {args}", f'../logs/{args.test_num}/training')
    save_to_log(f"Universe has shape: {universe.shape}", f'../logs/{args.test_num}/training')
    save_to_log(f"Training agent with target1 = {targets[0]} and target2 = {targets[1]}", 
                f'../logs/{args.test_num}/training')
    
    # Train agents for each alfa value
    for alfa in args.alfa_values:
        start_time = time.time()
        save_to_log(f"Computing the Baseline solution for alfa = {alfa} ...", f'../logs/{args.test_num}/training')
        
        # Find baseline solution
        best_state, best_reward = get_best_state(universe, targets, alfa, num_topics)
        save_to_log(f"Baseline for alfa = {alfa} -> State: {best_state}, Reward: {best_reward}\n"
                   f"Training agent with alfa = {alfa} and max episodes of {args.max_episodes}",
                   f'../logs/{args.test_num}/training')
        
        # Create environment and agent
        env = CustomEnv(universe=universe, target_dim1=targets[0], target_dim2=targets[1], 
                       num_topics=num_topics, alfa=alfa, reward_threshold=args.reward_threshold, max_iterations=args.max_iterations)
        
        # Create agent based on specified type
        if args.agent_type == 'dqn':
            agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_space.n, device=device,
                           lr=args.lr, gamma=args.gamma, target_sync_freq=args.target_sync_freq,
                           batch_size=args.batch_size, replay_buffer_type=args.replay_buffer)
        elif args.agent_type == 'a2c':
            agent = A2CAgent(state_dim=env.state_dim, action_dim=env.action_space.n, device=device,
                lr=args.lr, gamma=args.gamma, update_interval=5,
                eps=args.eps, eps_decay=args.eps_decay, eps_min=args.eps_min)
        elif args.agent_type == 'a3c':
            agent = A3CAgent(state_dim=env.state_dim, action_dim=env.action_space.n, device=device,
                           lr=args.lr, gamma=args.gamma, entropy_beta=0.01, num_workers=args.num_workers, test_num=args.test_num, alfa=alfa)
        elif args.agent_type == 'sarsa':
            agent = SARSAAgent(state_dim=env.state_dim, action_dim=env.action_space.n, device=device,
                             lr=args.lr, gamma=args.gamma, eps=args.eps, eps_decay=args.eps_decay,
                             eps_min=args.eps_min)
        else:
            raise ValueError(f"Unknown agent type: {args.agent_type}")
        
        # Train agent
        train_agent(env, agent, args.max_episodes, args.test_num, args)
        
        # Save results
        save_agent(os.path.join(f"../saved_agents/{args.test_num}", f"agent_alfa_{alfa}_bias.pth"), agent)
        if args.agent_type != 'a3c':
            save_data(agent.training_data, alfa, args.test_num)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        save_to_log(f"Number of Replay counts: {agent.training_data['replay_count']}\n"
                   f"Time elapsed for agent with alfa = {alfa}: {elapsed_time:.4f} seconds\n{50 * '-'}",
                   f'../logs/{args.test_num}/training')
    

    # Plot results after training
    if isinstance(agent, A3CAgent):
        print("\nPlotting results...")
        for worker_id in range(args.num_workers):
            print(f"Plotting results for worker {worker_id}...")
            plot_all_results_a3c(args.test_num, args.alfa_values, worker_id)
        print("Plotting complete!")
    else:
        plot_all_results(args.test_num, args.alfa_values)
    
    # Log final information
    program_end_time = time.time()
    program_elapsed_time = program_end_time - program_start_time
    save_to_log(f"Total elapsed time of the program: {program_elapsed_time:.4f} seconds", 
                f'../logs/{args.test_num}/training')

if __name__ == "__main__":
    main() 