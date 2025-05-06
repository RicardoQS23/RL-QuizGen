import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base_agent import BaseAgent

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # Shared feature extraction layers
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        
        # Actor head - outputs action probabilities
        self.actor_fc = nn.Linear(32, action_dim)
        
        # Critic head - outputs state value
        self.critic_fc = nn.Linear(32, 1)
        
        # Entropy coefficient for exploration
        self.entropy_beta = 0.01

    def forward(self, x):
        # Shared feature extraction
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Actor output - action probabilities
        action_probs = F.softmax(self.actor_fc(x), dim=-1)
        
        # Critic output - state value
        value = self.critic_fc(x)
        
        return action_probs, value

    def compute_loss(self, states, actions, rewards, next_states, dones, gamma):
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Get current action probabilities and values
        action_probs, values = self(states)
        
        # Get next state values
        _, next_values = self(next_states)
        next_values = next_values.detach()
        
        # Compute TD targets
        td_targets = rewards + (1 - dones) * gamma * next_values
        
        # Compute advantages
        advantages = td_targets - values.detach()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute policy loss
        policy_loss = -(action_probs.gather(1, actions.unsqueeze(1)) * advantages.unsqueeze(1)).mean()
        
        # Compute entropy for exploration
        entropy = -(action_probs * torch.log(action_probs + 1e-10)).sum(1).mean()
        
        # Compute value loss
        value_loss = F.mse_loss(values, td_targets)
        
        # Total loss combines policy loss, value loss, and entropy regularization
        total_loss = policy_loss + 0.5 * value_loss - self.entropy_beta * entropy
        
        return total_loss, policy_loss.item(), value_loss.item(), entropy.item()

class A3CAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, device, lr=0.0005, gamma=0.95, entropy_start=0.01, entropy_end=0.001, entropy_decay=0.995):
        super().__init__(state_dim, action_dim, device)
        self.lr = lr
        self.gamma = gamma
        self.device = device
        
        # Entropy decay parameters
        self.entropy_start = entropy_start
        self.entropy_end = entropy_end
        self.entropy_decay = entropy_decay
        
        # Initialize actor-critic network
        self.actor_critic = ActorCritic(state_dim, action_dim).to(device)
        self.actor_critic.entropy_beta = entropy_start  # Set initial entropy coefficient
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        # Initialize training data
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
            "epsilon": 1.0,
            "episode_count": 0,
            "step_count": 0,
            "replay_count": 0,
            "beta_start": 0.4,
            "beta_increment_per_episode": 0.0
        }

    def get_action(self, state, epsilon):
        """Get action from the agent given a state"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, value = self.actor_critic(state)
            
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = np.random.randint(self.action_dim)
            explore = True
        else:
            action = torch.multinomial(action_probs, 1).item()
            explore = False
            
        return action, value.item(), explore

    def update_entropy(self):
        """Update the entropy coefficient using decay"""
        self.actor_critic.entropy_beta = max(
            self.entropy_end,
            self.actor_critic.entropy_beta * self.entropy_decay
        )

    def train_step(self, state, action, reward, next_state, done):
        """Perform a training step"""
        # Store experience
        self.training_data['step_count'] += 1
        
        # Update entropy coefficient
        self.update_entropy()
        
        # Convert to tensors
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        
        # Get current action probabilities and values
        action_probs, value = self.actor_critic(state)
        
        # Get next state value
        _, next_value = self.actor_critic(next_state)
        
        # Compute TD target
        td_target = reward + (1 - done) * self.gamma * next_value.detach()
        
        # Compute advantage
        advantage = td_target - value.detach()
        
        # Normalize advantage (handle single value case)
        if advantage.numel() > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        
        # Compute policy loss
        policy_loss = -(action_probs[0, action] * advantage).mean()
        
        # Compute entropy for exploration
        entropy = -(action_probs * torch.log(action_probs + 1e-10)).sum(1).mean()
        
        # Compute value loss
        value_loss = F.mse_loss(value, td_target)
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss - self.actor_critic.entropy_beta * entropy
        
        # Update network
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
        self.optimizer.step()
        
        # Track loss
        self.training_data['episode_losses'].append(total_loss.item())
        self.training_data['replay_count'] += 1

    def save(self, path):
        """Save the agent's model"""
        state_dict = {
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_data': self.training_data
        }
        torch.save(state_dict, path)

    def load(self, path):
        """Load the agent's model"""
        checkpoint = torch.load(path)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_data = checkpoint['training_data'] 