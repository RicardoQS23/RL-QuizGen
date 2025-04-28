import numpy as np
from abc import ABC, abstractmethod

class BaseReplayBuffer(ABC):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def __len__(self):
        return len(self.buffer)
    
    @abstractmethod
    def push(self, state, action, reward, next_state, done):
        """Add a transition to the buffer"""
        pass
    
    @abstractmethod
    def sample(self, batch_size):
        """Sample a batch of transitions"""
        pass
    
    @abstractmethod
    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions (only for prioritized replay)"""
        pass 