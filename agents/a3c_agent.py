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
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # Shared layers
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        
        # Actor head
        self.actor = nn.Linear(16, action_dim)
        
        # Critic head
        self.critic = nn.Linear(16, 1)
        
        self.entropy_beta = 0.01

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Actor output (action probabilities)
        action_probs = F.softmax(self.actor(x), dim=-1)
        
        # Critic output (state value)
        state_value = self.critic(x)
        
        return action_probs, state_value


class WorkerAgent(Thread):
    def __init__(self, env, global_actor_critic, device, max_episodes, gamma=0.95, update_interval=5):
        super(WorkerAgent, self).__init__()
        self.env = env
        self.global_actor_critic = global_actor_critic
        self.device = device
        self.max_episodes = max_episodes
        self.gamma = gamma
        self.update_interval = update_interval
        
        # Local network
        self.local_actor_critic = ActorCritic(
            env.observation_space.shape[0],
            env.action_space.n
        ).to(device)
        self.local_actor_critic.load_state_dict(self.global_actor_critic.state_dict())
        
        self.optimizer = torch.optim.Adam(self.local_actor_critic.parameters(), lr=0.0005)
        self.lock = Lock()
        
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
            episode_reward = 0
            done = False
            
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action_probs, _ = self.local_actor_critic(state_tensor)
                action = torch.multinomial(action_probs, 1).item()
                
                next_state, reward, done, _ = self.env.step(action)
                
                state_batch.append(state)
                action_batch.append([action])
                reward_batch.append([reward])
                
                if len(state_batch) >= self.update_interval or done:
                    states = torch.FloatTensor(np.array(state_batch)).to(self.device)
                    actions = torch.LongTensor(np.array(action_batch)).to(self.device)
                    rewards = np.array(reward_batch)
                    
                    next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
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
                    
                    # Total loss
                    loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
                    
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
                    
                    state_batch, action_batch, reward_batch = [], [], []
                
                episode_reward += reward
                state = next_state
            
            print(f"[Worker {self.name}] Episode {episode + 1} Reward: {episode_reward}")


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
        
    def _init_networks(self):
        """Initialize the actor-critic network"""
        self.actor_critic = ActorCritic(self.state_dim, self.action_dim).to(self.device)
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
                       self.gamma, self.update_interval)
            for _ in range(self.num_workers)
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
        torch.save({
            'model_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load(self, path):
        """Load the agent's model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
