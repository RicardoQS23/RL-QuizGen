import numpy as np
from .base_buffer import BaseReplayBuffer

class NormalReplayBuffer(BaseReplayBuffer):
    def __init__(self, capacity):
        super().__init__(capacity)
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in batch])
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def update_priorities(self, indices, priorities):
        """No-op for normal replay buffer"""
        pass 