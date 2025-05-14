import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from threading import Thread, Lock
from .base_agent import BaseAgent

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, entropy_beta=0.01):
        super(ActorCritic, self).__init__()
        # Shared feature extraction layers
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
        # Shared feature extraction
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        # Actor output - action probabilities
        action_probs = F.softmax(self.actor_fc(x), dim=-1)
        
        # Critic output - state value
        value = self.critic_fc(x)
        
        return action_probs, value

class A2CWorker(Thread):
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
                
                # Epsilon-greedy action selection
                if np.random.random() < 0.1:  # 10% exploration
                    action = np.random.randint(self.env.action_space.n)
                    explore = True
                else:
                    action = torch.multinomial(action_probs, 1).item()
                    explore = False
                
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
                if explore:
                    exploration_count += 1
                else:
                    exploitation_count += 1
                
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

class A2CAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, device, lr=0.0005, gamma=0.95, update_interval=5,
                 eps=1.0, eps_decay=0.997, eps_min=0.05, entropy_beta=0.01, entropy_decay=0.995, entropy_min=0.001):
        super().__init__(state_dim, action_dim, device)
        self.lr = lr
        self.gamma = gamma
        self.update_interval = update_interval
        self.device = device
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.entropy_beta = entropy_beta
        self.entropy_decay = entropy_decay
        self.entropy_min = entropy_min
        
        # Initialize global actor-critic network
        self.global_actor_critic = ActorCritic(state_dim, action_dim, entropy_beta).to(device)
        self.optimizer = torch.optim.Adam(self.global_actor_critic.parameters(), lr=lr)
        
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
            "step_count": 0,
            "replay_count": 0,
            "total_losses": [],
            "policy_losses": [],
            "value_losses": [],
            "entropies": [],
            "steps_per_update": [],
            "epsilon": self.eps,
            "entropy_beta": self.entropy_beta,
            "episode_count": 0
        }

    def get_action(self, state, epsilon):
        """Get action from the agent given a state"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, value = self.global_actor_critic(state)
            
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = np.random.randint(self.action_dim)
            explore = True
        else:
            action = torch.multinomial(action_probs, 1).item()
            explore = False
            
        return action, value.item(), explore

    def train_step(self, state, action, reward, next_state, done):
        """Perform a training step"""
        # Store experience
        self.training_data['step_count'] += 1
        
        # Convert to tensors and move to device
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        action = torch.LongTensor([action]).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        done = torch.FloatTensor([done]).to(self.device)
        
        # Get current action probabilities and values
        action_probs, value = self.global_actor_critic(state)
        
        # Get next state value
        _, next_value = self.global_actor_critic(next_state)
        
        # Compute TD target
        td_target = reward + (1 - done) * self.gamma * next_value.detach()
        
        # Compute advantage
        advantage = td_target - value.detach()
        
        # Normalize advantage (handle single value case and constant advantages)
        if advantage.numel() > 1:
            std = advantage.std()
            if std > 0:
                advantage = (advantage - advantage.mean()) / (std + 1e-8)
        
        # Compute policy loss
        policy_loss = -(action_probs[0, action] * advantage).mean()
        
        # Compute entropy for exploration
        entropy = -(action_probs * torch.log(action_probs + 1e-10)).sum(1).mean()
        
        # Compute value loss
        value_loss = F.mse_loss(value, td_target)
        
        # Total loss with current entropy beta
        total_loss = policy_loss + 0.5 * value_loss - self.training_data['entropy_beta'] * entropy
        
        # Update network with gradient clipping
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.global_actor_critic.parameters(), 0.5)
        self.optimizer.step()
        
        # Track metrics
        self.training_data['episode_losses'].append(total_loss.item())
        self.training_data['total_losses'].append(total_loss.item())
        self.training_data['policy_losses'].append(policy_loss.item())
        self.training_data['value_losses'].append(value_loss.item())
        self.training_data['entropies'].append(entropy.item())
        self.training_data['replay_count'] += 1

    def save(self, path):
        """Save the agent's model"""
        state_dict = {
            'global_actor_critic_state_dict': self.global_actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_data': self.training_data
        }
        torch.save(state_dict, path)

    def load(self, path):
        """Load the agent's model"""
        try:
            # Load the checkpoint with map_location set to the device
            checkpoint = torch.load(path, map_location=torch.device('cpu'), weights_only=False,
                                     pickle_module=torch.serialization.pickle)
            
            # Process the loaded checkpoint
            if isinstance(checkpoint, dict):
                print("Loading from dictionary")
                if 'global_actor_critic_state_dict' in checkpoint:
                    self.global_actor_critic.load_state_dict(checkpoint['global_actor_critic_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'training_data' in checkpoint:
                    for key, value in checkpoint['training_data'].items():
                        if key in self.training_data:
                            self.training_data[key] = value
            else:
                raise ValueError(f"Unexpected checkpoint type: {type(checkpoint)}")
            
            print(f"Successfully loaded model from {path}")
            
        except Exception as e:
            print(f"Error loading model from {path}: {str(e)}")
            print("Please ensure the model file exists and is in the correct format.")
            raise

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
        
        # Update epsilon
        self.training_data['epsilon'] = max(self.eps_min, 
                                          self.training_data['epsilon'] * self.eps_decay)
        
        # Update entropy beta
        self.training_data['entropy_beta'] = max(self.entropy_min,
                                               self.training_data['entropy_beta'] * self.entropy_decay)
        self.global_actor_critic.entropy_beta = self.training_data['entropy_beta']
        
        self.training_data['episode_count'] += 1 