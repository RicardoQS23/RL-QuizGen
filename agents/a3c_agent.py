import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from threading import Thread, Lock
from multiprocessing import cpu_count
from .base_agent import BaseAgent

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, entropy_beta=0.01):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        
        # Actor head - outputs action probabilities
        self.actor_fc = nn.Linear(16, action_dim)
        
        # Critic head - outputs state value
        self.critic_fc = nn.Linear(16, 1)
        
        # Entropy coefficient for exploration
        self.entropy_beta = entropy_beta

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        # Actor output - action probabilities
        action_probs = F.softmax(self.actor_fc(x), dim=-1)
        
        # Critic output - state value
        value = self.critic_fc(x)
        
        return action_probs, value

class A3CWorker(Thread):
    def __init__(self, worker_id, env, global_actor_critic, device, gamma=0.95, update_interval=5):
        Thread.__init__(self)
        self.worker_id = worker_id
        self.env = env
        self.device = device
        self.gamma = gamma
        self.update_interval = update_interval
        
        # Global network
        self.global_actor_critic = global_actor_critic
        
        # Local network
        self.local_actor_critic = ActorCritic(
            env.observation_space.shape[0],
            env.action_space.n
        ).to(device)
        
        # Synchronize with global network
        self.sync_networks()
        
        # Training data for this worker
        self.training_data = {
            "episode_rewards": [],
            "episode_rewards_dim1": [],
            "episode_rewards_dim2": [],
            "episode_actions": [],
            "episode_avg_qvalues": [],
            "episode_losses": [],
            "exploration_counts": [],
            "exploitation_counts": [],
            "success_episodes": [],
            "step_count": 0,
            "replay_count": 0,
            "total_losses": [],
            "policy_losses": [],
            "value_losses": [],
            "entropies": [],
            "steps_per_update": []
        }
        
        # Lock for thread synchronization
        self.lock = Lock()

    def sync_networks(self):
        """Synchronize local network with global network"""
        self.local_actor_critic.load_state_dict(self.global_actor_critic.state_dict())

    def compute_advantages(self, rewards, values, next_value, dones):
        """Compute advantages using n-step returns"""
        advantages = []
        returns = []
        R = next_value
        
        for r, v, done in zip(reversed(rewards), reversed(values), reversed(dones)):
            R = r + self.gamma * R * (1 - done)
            advantage = R - v
            returns.insert(0, R)
            advantages.insert(0, advantage)
            
        return torch.tensor(advantages, device=self.device), torch.tensor(returns, device=self.device)

    def train(self, states, actions, rewards, next_states, dones):
        """Train the local network and update global network"""
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Get predictions
        action_probs, values = self.local_actor_critic(states)
        _, next_values = self.local_actor_critic(next_states)
        next_values = next_values.detach()
        
        # Compute advantages and returns
        advantages, returns = self.compute_advantages(
            rewards, values, next_values[-1], dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute losses
        policy_loss = -(action_probs.gather(1, actions.unsqueeze(1)) * advantages.unsqueeze(1)).mean()
        value_loss = F.mse_loss(values, returns.unsqueeze(1))
        entropy = -(action_probs * torch.log(action_probs + 1e-10)).sum(1).mean()
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss - self.local_actor_critic.entropy_beta * entropy
        
        # Update local network
        self.local_actor_critic.zero_grad()
        total_loss.backward()
        
        # Update global network
        with self.lock:
            for local_param, global_param in zip(self.local_actor_critic.parameters(), 
                                               self.global_actor_critic.parameters()):
                if global_param.grad is not None:
                    global_param.grad = local_param.grad
        
        # Synchronize local network with global network
        self.sync_networks()
        
        # Log metrics
        self.training_data['total_losses'].append(total_loss.item())
        self.training_data['policy_losses'].append(policy_loss.item())
        self.training_data['value_losses'].append(value_loss.item())
        self.training_data['entropies'].append(entropy.item())
        self.training_data['steps_per_update'].append(len(states))
        
        return total_loss.item(), policy_loss.item(), value_loss.item(), entropy.item()

    def run(self):
        """Main training loop for the worker"""
        while True:
            state = self.env.reset()
            episode_reward = 0
            episode_reward_dim1 = 0
            episode_reward_dim2 = 0
            exploration_count = 0
            exploitation_count = 0
            states, actions, rewards, next_states, dones = [], [], [], [], []
            
            while True:
                # Get action
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action_probs, value = self.local_actor_critic(state_tensor)
                
                # Sample action from policy
                action = torch.multinomial(action_probs, 1).item()
                
                # Take action
                next_state, reward, done, success, reward_dim1, reward_dim2 = self.env.step(action)
                
                # Store transition
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)
                
                # Update episode statistics
                episode_reward += reward
                episode_reward_dim1 += reward_dim1
                episode_reward_dim2 += reward_dim2
                
                state = next_state
                
                # Update if enough transitions or episode is done
                if len(states) >= self.update_interval or done:
                    self.train(states, actions, rewards, next_states, dones)
                    states, actions, rewards, next_states, dones = [], [], [], [], []
                
                if done:
                    # Update training data
                    self.training_data['episode_rewards'].append(episode_reward)
                    self.training_data['episode_rewards_dim1'].append(episode_reward_dim1)
                    self.training_data['episode_rewards_dim2'].append(episode_reward_dim2)
                    self.training_data['exploration_counts'].append(exploration_count)
                    self.training_data['exploitation_counts'].append(exploitation_count)
                    self.training_data['success_episodes'].append(success)
                    break

class A3CAgent:
    def __init__(self, state_dim, action_dim, device, lr=0.0005, gamma=0.95, update_interval=5, num_workers=4):
        self.device = device
        self.gamma = gamma
        self.update_interval = update_interval
        self.num_workers = num_workers
        
        # Global network
        self.global_actor_critic = ActorCritic(state_dim, action_dim).to(device)
        self.global_actor_critic.share_memory()  # Enable sharing between processes
        
        # Create workers
        self.workers = []
        self.envs = []
        
        # Training data
        self.training_data = {
            "episode_rewards": [],
            "episode_rewards_dim1": [],
            "episode_rewards_dim2": [],
            "episode_actions": [],
            "episode_avg_qvalues": [],
            "episode_losses": [],
            "exploration_counts": [],
            "exploitation_counts": [],
            "success_episodes": [],
            "step_count": 0,
            "replay_count": 0,
            "total_losses": [],
            "policy_losses": [],
            "value_losses": [],
            "entropies": [],
            "steps_per_update": [],
            "epsilon": 1.0,  # Add epsilon to training data
            "episode_count": 0
        }

    def start_workers(self, env):
        """Start worker threads"""
        # Create environments for each worker
        for _ in range(self.num_workers):
            self.envs.append(env)
        
        # Create and start workers
        for i in range(self.num_workers):
            worker = A3CWorker(
                worker_id=i,
                env=self.envs[i],
                global_actor_critic=self.global_actor_critic,
                device=self.device,
                gamma=self.gamma,
                update_interval=self.update_interval
            )
            self.workers.append(worker)
            worker.start()

    def stop_workers(self):
        """Stop all worker threads"""
        for worker in self.workers:
            worker.join()

    def get_action(self, state, epsilon=0.1):
        """Get action from the global network"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, value = self.global_actor_critic(state_tensor)
        
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = np.random.randint(self.global_actor_critic.action_dim)
            explore = True
        else:
            action = torch.multinomial(action_probs, 1).item()
            explore = False
        
        return action, value.item(), explore

    def train_step(self, state, action, reward, next_state, done):
        """Single training step - not used in A3C as training is done by workers"""
        pass  # Training is handled by workers

    def save(self, path):
        """Save the global network"""
        torch.save(self.global_actor_critic.state_dict(), path)

    def load(self, path):
        """Load the global network"""
        self.global_actor_critic.load_state_dict(torch.load(path))

    def update_episode_data(self, total_reward, total_reward_dim1, total_reward_dim2, 
                           exploration_count, exploitation_count, success,
                           episode_actions, episode_avg_qvalues, num_iterations):
        """Update training data after each episode"""
        self.training_data['episode_rewards'].append(total_reward)
        self.training_data['episode_rewards_dim1'].append(total_reward_dim1)
        self.training_data['episode_rewards_dim2'].append(total_reward_dim2)
        self.training_data['exploration_counts'].append(exploration_count)
        self.training_data['exploitation_counts'].append(exploitation_count)
        self.training_data['success_episodes'].append(success)
        self.training_data['episode_actions'].append(episode_actions)
        self.training_data['episode_avg_qvalues'].append(episode_avg_qvalues)
        
        self.training_data['episode_count'] += 1 