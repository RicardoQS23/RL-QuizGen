import numpy as np
from .base_buffer import BaseReplayBuffer

class PrioritizedReplayBuffer(BaseReplayBuffer):
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        super().__init__(capacity)
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = beta_increment
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            self.priorities = np.append(self.priorities, 0)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return None, None, None, None, None, None
        
        # Calculate sampling probabilities
        probs = self.priorities[:len(self.buffer)] ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get samples
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])
        
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones), indices, weights)
    
    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
        self.max_priority = max(self.max_priority, max(priorities)) 