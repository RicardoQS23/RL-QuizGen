import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from agents.base_agent import BaseAgent
from replay_buffers.normal_buffer import NormalReplayBuffer
from replay_buffers.prioritized_buffer import PrioritizedReplayBuffer

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class DQNAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, device, lr=0.0005, gamma=0.95, 
                 target_sync_freq=1000, batch_size=128, replay_buffer_type='normal'):
        super().__init__(state_dim, action_dim, device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_sync_freq = target_sync_freq
        self.device = device
        self.model = DQN(state_dim, action_dim).to(device)
        self.target_model = DQN(state_dim, action_dim).to(device)
        self.target_update()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        
        # Initialize appropriate replay buffer
        if replay_buffer_type == 'normal':
            self.buffer = NormalReplayBuffer(capacity=100000)
        else:  # prioritized
            self.buffer = PrioritizedReplayBuffer(capacity=100000)
        self.replay_buffer_type = replay_buffer_type
        

    def target_update(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state, epsilon):
        if random.random() < epsilon:
            state = torch.tensor(state, dtype=torch.float32).view(1, -1).to(self.device)
            q_values = self.model(state)
            rand_action = random.randint(0, self.action_dim - 1)
            return rand_action, np.mean(q_values.cpu().detach().numpy()), True
        else:
            state = torch.tensor(state, dtype=torch.float32).view(1, -1).to(self.device)
            with torch.no_grad():
                q_values = self.model(state)
            action = q_values.argmax().item()
            return action, np.mean(q_values.cpu().detach().numpy()), False

    def train_step(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        if len(self.buffer) >= self.batch_size:
            self.replay()
        
        self.training_data['step_count'] += 1
        if self.training_data['step_count'] % self.target_sync_freq == 0:
            self.target_update()

    def replay(self):
        if self.replay_buffer_type == 'normal':
            states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
            weights = torch.ones(self.batch_size, device=self.device)
        else:
            states, actions, rewards, next_states, dones, indices, weights = self.buffer.sample(self.batch_size)
            weights = torch.FloatTensor(weights).to(self.device)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = (current_q_values - target_q_values.unsqueeze(1)).pow(2) * weights.unsqueeze(1)
        loss = loss.mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.replay_buffer_type == 'prioritized':
            # Update priorities in the buffer
            td_errors = (current_q_values - target_q_values.unsqueeze(1)).abs().detach().cpu().numpy()
            self.buffer.update_priorities(indices, td_errors)
        
        # Track loss in training data
        self.training_data['episode_losses'].append(loss.item())
        self.training_data['replay_count'] += 1

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=torch.device(self.device)))
        self.target_update() 