import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from agents.base_agent import BaseAgent

class SARSANetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(SARSANetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class SARSAAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, device, lr=0.0005, gamma=0.95, 
                 eps=1.0, eps_decay=0.997, eps_min=0.05):
        super().__init__(state_dim, action_dim, device)
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        
        # Initialize network
        self.model = SARSANetwork(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
    
    def get_action(self, state, epsilon):
        """Get action from the agent given a state"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        
        if np.random.random() < epsilon:
            action = np.random.randint(self.action_dim)
            explore = True
        else:
            action = q_values.argmax().item()
            explore = False
            
        mean_q_value = q_values.mean().item()
        return action, mean_q_value, explore
    
    def train_step(self, state, action, reward, next_state, done):
        """Perform a SARSA training step"""
        # Convert to tensors
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        action = torch.LongTensor([action]).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        done = torch.FloatTensor([float(done)]).to(self.device)
        
        # Get current Q-value
        current_q = self.model(state).gather(1, action.unsqueeze(1))
        
        # CHANGED: use ε-greedy policy to select next action (on-policy SARSA)
        next_action, _, _ = self.get_action(next_state.cpu().numpy()[0], self.eps)
        next_action = torch.LongTensor([next_action]).to(self.device)
        with torch.no_grad():
            next_q = self.model(next_state).gather(1, next_action.unsqueeze(1))
        
        # Compute target Q-value using SARSA
        target_q = reward + (1 - done) * self.gamma * next_q
        
        # Compute loss and update
        loss = F.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
        
        # Track training data
        self.training_data['episode_losses'].append(loss.item())
        self.training_data['step_count'] += 1
    
    def save(self, path):
        """Save the agent's model"""
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        """Load the agent's model"""
        self.model.load_state_dict(torch.load(path, map_location=torch.device(self.device)))