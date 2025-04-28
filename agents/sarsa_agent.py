import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base_agent import BaseAgent

class SARSAAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, device, lr=0.0005, gamma=0.95, 
                 eps=1.0, eps_decay=0.997, eps_min=0.05):
        super().__init__(state_dim, action_dim, device)
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        
        # Placeholder for Q-network
        self.q_network = None
        self.optimizer = None
        
        # Initialize network
        self._init_network()
    
    def _init_network(self):
        """Initialize the Q-network"""
        # TODO: Implement proper Q-network architecture
        pass
    
    def get_action(self, state, epsilon):
        """Get action from the agent given a state"""
        # TODO: Implement SARSA action selection
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state)
        
        if np.random.random() < epsilon:
            action = np.random.randint(self.action_dim)
            explore = True
        else:
            action = q_values.argmax().item()
            explore = False
            
        mean_q_value = q_values.mean().item()
        return action, mean_q_value, explore
    
    def train_step(self, state, action, reward, next_state, done):
        """Perform a training step"""
        # TODO: Implement SARSA training step
        pass
    
    def save(self, path):
        """Save the agent's model"""
        # TODO: Implement model saving
        pass
    
    def load(self, path):
        """Load the agent's model"""
        # TODO: Implement model loading
        pass 