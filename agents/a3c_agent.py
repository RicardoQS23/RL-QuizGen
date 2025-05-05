import argparse
import numpy as np
from threading import Thread, Lock
from multiprocessing import cpu_count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
import seaborn as sns
from .base_agent import BaseAgent

# Global vars
CUR_EPISODE = 0
REWARD_HISTORY = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, entropy_beta=0.01):
        super(ActorCritic, self).__init__()
        # Shared layers
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        # Actor head
        self.actor = nn.Linear(16, action_dim)
        
        # Critic head
        self.critic = nn.Linear(16, 1)
        
        self.entropy_beta = entropy_beta

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        # Actor output (action probabilities)
        action_probs = F.softmax(self.actor(x), dim=-1)
        
        # Critic output (state value)
        state_value = self.critic(x)
        
        return action_probs, state_value


class WorkerAgent(Thread):
    def __init__(self, env, global_actor_critic, device, max_episodes, gamma=0.95, update_interval=5, global_agent=None, worker_id=0):
        super(WorkerAgent, self).__init__()
        self.env = env
        self.global_actor_critic = global_actor_critic
        self.device = device
        self.max_episodes = max_episodes
        self.gamma = gamma
        self.update_interval = update_interval
        self.global_agent = global_agent
        self.worker_id = worker_id
        
        # Local network
        self.local_actor_critic = ActorCritic(
            env.state_dim,
            env.action_space.n,
            0.01  # Fixed small entropy coefficient
        ).to(device)
        self.local_actor_critic.load_state_dict(self.global_actor_critic.state_dict())
        
        self.optimizer = torch.optim.Adam(self.local_actor_critic.parameters(), lr=0.0005)
        self.lock = Lock()
        
        # Worker-specific metrics
        self.worker_metrics = {
            'rewards': [],
            'rewards_dim1': [],
            'rewards_dim2': [],
            'successes': [],
            'visited_states': set(),
            'action_probs': []  # Track action probabilities
        }

    def n_step_td_target(self, rewards, next_v_value, done):
        td_targets = np.zeros_like(rewards)
        cumulative = 0 if done else next_v_value
        for k in reversed(range(len(rewards))):
            cumulative = rewards[k] + self.gamma * cumulative
            td_targets[k] = cumulative
        return td_targets
        
    def run(self):
        for episode in range(self.max_episodes):
            state = self.env.reset()
            state_batch, action_batch, reward_batch = [], [], []
            episode_reward = total_reward_dim1 = total_reward_dim2 = 0
            episode_actions = []
            episode_avg_qvalues = []
            visited_states = set()
            done = False
            num_iterations = 0
            
            while not done:
                if state not in visited_states:
                    visited_states.add(state)
                
                state_tensor = torch.FloatTensor(self.env.universe[state]).unsqueeze(0).to(self.device)
                action_probs, _ = self.local_actor_critic(state_tensor)
                
                # Store action probabilities for analysis
                self.worker_metrics['action_probs'].append(action_probs.detach().cpu().numpy())
                
                # Always take the action with highest probability
                action = torch.argmax(action_probs).item()
                
                next_state, reward, done, success, reward_dim1, reward_dim2 = self.env.step(action, num_iterations)
                
                state_batch.append(self.env.universe[state])
                action_batch.append([action])
                reward_batch.append([reward])
                
                episode_reward += reward
                total_reward_dim1 += reward_dim1
                total_reward_dim2 += reward_dim2
                
                episode_actions.append(action)
                episode_avg_qvalues.append(action_probs.mean().item())
                
                if len(state_batch) >= self.update_interval or done:
                    states = torch.FloatTensor(np.array(state_batch)).to(self.device)
                    actions = torch.LongTensor(np.array(action_batch)).to(self.device)
                    rewards = np.array(reward_batch)
                    
                    next_state_tensor = torch.FloatTensor(self.env.universe[next_state]).unsqueeze(0).to(self.device)
                    _, next_value = self.local_actor_critic(next_state_tensor)
                    td_targets = self.n_step_td_target(rewards, next_value.item(), done)
                    td_targets = torch.FloatTensor(td_targets).to(self.device).view(-1, 1)
                    
                    # Get current values and action probabilities
                    action_probs, values = self.local_actor_critic(states)
                    
                    # Calculate advantages
                    advantages = (td_targets - values.detach()).view(-1)
                    
                    # Calculate losses
                    log_probs = torch.log(action_probs)
                    actor_loss = -(log_probs.gather(1, actions) * advantages.unsqueeze(1)).mean()
                    critic_loss = advantages.pow(2).mean()
                    entropy = -(action_probs * log_probs).sum(dim=1).mean()
                    
                    # Total loss with minimal entropy regularization
                    loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
                    
                    # Update global network
                    with self.lock:
                        self.optimizer.zero_grad()
                        loss.backward()
                        for local_param, global_param in zip(
                            self.local_actor_critic.parameters(),
                            self.global_actor_critic.parameters()
                        ):
                            global_param._grad = local_param.grad
                        self.optimizer.step()
                        
                        # Update local network
                        self.local_actor_critic.load_state_dict(self.global_actor_critic.state_dict())
                        
                        # Update global agent's training data
                        if self.global_agent is not None:
                            # Ensure we don't divide by zero
                            if num_iterations > 0:
                                reward = episode_reward/num_iterations
                                reward_dim1 = total_reward_dim1/num_iterations
                                reward_dim2 = total_reward_dim2/num_iterations
                            else:
                                reward = episode_reward
                                reward_dim1 = total_reward_dim1
                                reward_dim2 = total_reward_dim2
                                
                            # Update worker-specific metrics
                            self.worker_metrics['rewards'].append(reward)
                            self.worker_metrics['rewards_dim1'].append(reward_dim1)
                            self.worker_metrics['rewards_dim2'].append(reward_dim2)
                            self.worker_metrics['successes'].append(success)
                            self.worker_metrics['visited_states'].update(visited_states)
                            
                            # Update global metrics
                            self.global_agent.training_data['episode_rewards'].append(reward)
                            self.global_agent.training_data['episode_rewards_dim1'].append(reward_dim1)
                            self.global_agent.training_data['episode_rewards_dim2'].append(reward_dim2)
                            self.global_agent.training_data['episode_actions'].append(episode_actions)
                            self.global_agent.training_data['episode_avg_qvalues'].append(episode_avg_qvalues)
                            self.global_agent.training_data['episode_losses'].append(loss.item())
                            self.global_agent.training_data['success_episodes'].append(success)
                            self.global_agent.training_data['episode_count'] += 1
                            
                            # Update visited states
                            self.global_agent.all_visited_states.update(visited_states)
                            
                            # Print detailed worker metrics every 10 episodes
                            if episode % 10 == 0:
                                avg_reward = np.mean(self.worker_metrics['rewards'][-10:])
                                avg_success = np.mean(self.worker_metrics['successes'][-10:])
                                print(f"\nWorker {self.worker_id} Metrics (Last 10 episodes):")
                                print(f"Average Reward: {avg_reward:.4f}")
                                print(f"Success Rate: {avg_success:.2%}")
                                print(f"Total States Visited: {len(self.worker_metrics['visited_states'])}")
                                
                                # Print action distribution
                                if len(self.worker_metrics['action_probs']) > 0:
                                    recent_probs = np.mean(self.worker_metrics['action_probs'][-10:], axis=0)
                                    print(f"Action Distribution: {recent_probs}\n")
                    
                    state_batch, action_batch, reward_batch = [], [], []
                
                state = next_state
                num_iterations += 1
            
            print(f"[Worker {self.worker_id}] Episode {episode + 1} Reward: {episode_reward/num_iterations if num_iterations > 0 else episode_reward}")


class A3CAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, device, lr=0.0005, gamma=0.95, num_workers=None):
        super().__init__(state_dim, action_dim, device)
        self.lr = lr
        self.gamma = gamma
        self.num_workers = num_workers if num_workers is not None else cpu_count()
        
        # Initialize networks
        self._init_networks()
        
        # Training parameters
        self.update_interval = 5
        self.workers = []
        self.lock = Lock()
        self.all_visited_states = set()  # Track all visited states
        
        # Worker-specific data
        self.worker_data = {}
    
    def _init_networks(self):
        """Initialize the actor-critic network"""
        self.actor_critic = ActorCritic(self.state_dim, self.action_dim, 0.01).to(self.device)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=self.lr)
    
    def get_action(self, state, epsilon):
        """Get action from the agent given a state"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, _ = self.actor_critic(state)
        
        # Epsilon-greedy exploration
        if np.random.random() < epsilon:
            action = np.random.randint(self.action_dim)
            explore = True
        else:
            action = torch.multinomial(action_probs, 1).item()
            explore = False
            
        mean_q_value = action_probs.mean().item()
        return action, mean_q_value, explore
    
    def train_step(self, state, action, reward, next_state, done):
        """Perform a training step"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        action = torch.LongTensor([action]).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        done = torch.FloatTensor([done]).to(self.device)
        
        # Get current action probabilities and state value
        action_probs, value = self.actor_critic(state)
        
        # Get next state value
        with torch.no_grad():
            _, next_value = self.actor_critic(next_state)
        
        # Calculate advantage
        advantage = reward + (1 - done) * self.gamma * next_value - value
        
        # Calculate actor loss
        log_probs = torch.log(action_probs)
        actor_loss = -(log_probs.gather(1, action.unsqueeze(1)) * advantage.detach()).mean()
        
        # Calculate critic loss
        critic_loss = advantage.pow(2).mean()
        
        # Calculate entropy loss for exploration
        entropy = -(action_probs * log_probs).sum(dim=1).mean()
        
        # Total loss
        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Track loss in training data
        self.training_data['episode_losses'].append(loss.item())
    
    def start_workers(self, env):
        """Start worker threads for parallel training"""
        self.workers = [
            WorkerAgent(env, self.actor_critic, self.device, self.training_data['max_episodes'],
                       self.gamma, self.update_interval, self, worker_id=i)
            for i in range(self.num_workers)
        ]
        for worker in self.workers:
            worker.start()
    
    def stop_workers(self):
        """Stop all worker threads"""
        for worker in self.workers:
            worker.join()
        self.workers = []
    
    def save(self, path):
        """Save the agent's model"""
        # Create a dictionary of only the necessary components
        save_dict = {
            'model_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_data': {
                'episode_rewards': self.training_data['episode_rewards'],
                'episode_rewards_dim1': self.training_data['episode_rewards_dim1'],
                'episode_rewards_dim2': self.training_data['episode_rewards_dim2'],
                'episode_actions': self.training_data['episode_actions'],
                'episode_avg_qvalues': self.training_data['episode_avg_qvalues'],
                'episode_losses': self.training_data['episode_losses'],
                'episode_count': self.training_data['episode_count'],
                'step_count': self.training_data['step_count'],
                'replay_count': self.training_data['replay_count']
            }
        }
        torch.save(save_dict, path)
    
    def load(self, path):
        """Load the agent's model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore training data if available
        if 'training_data' in checkpoint:
            for key, value in checkpoint['training_data'].items():
                self.training_data[key] = value
    
    def get_training_data(self):
        """Get the agent's training data"""
        return self.training_data

    def get_worker_metrics(self):
        """Get metrics for each worker"""
        return {worker.worker_id: worker.worker_metrics for worker in self.workers}
