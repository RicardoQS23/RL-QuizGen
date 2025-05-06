import gym
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class CustomEnv(gym.Env):
    def __init__(self, universe, target_dim1, target_dim2, num_topics, alfa=0.5, reward_threshold=0.85, state=None):
        super(CustomEnv, self).__init__()
        self.alfa = alfa
        self.reward_threshold = reward_threshold
        self.target_dim1 = target_dim1
        self.target_dim2 = target_dim2
        
        self.num_topics = num_topics
        self.universe = universe
        self.universe_size = self.universe.shape[0]

        self.state = state   #idx of the current state
        self.state_dim = self.universe.shape[1]
        self.action_space = gym.spaces.Discrete(4)

        self.topic_similarities_matrix = self.compute_similarities_matrix(mode=0)
        self.difficulty_similarities_matrix = self.compute_similarities_matrix(mode=1)
    
    def reset(self):
        """Resets the environment by picking a random valid state."""
        self.state = np.random.choice(self.universe_size, 1)[0]  # Pick 1 row index
        return self.state

    def step(self, action, num_iterations, max_iterations=100):
        """Take a step in the environment given an action."""
        if action in [0, 1]:  # Similar actions
            self.state = self.choose_similar(mode=action%2)
        else:  # Different actions
            self.state = self.choose_different(mode=action%2)

        # Add the current state to memory
        state_value = self.universe[self.state]

        vec1, vec2 = state_value[:self.num_topics], state_value[self.num_topics:] 

        # Reward function
        first_dim_metric = cosine_similarity(vec1.reshape(1, -1), self.target_dim1.reshape(1, -1))[0][0]
        second_dim_metric = cosine_similarity(vec2.reshape(1, -1), self.target_dim2.reshape(1, -1))[0][0]
        reward = self.alfa * first_dim_metric + (1 - self.alfa) * second_dim_metric
        done = num_iterations >= max_iterations or reward > self.reward_threshold
        success = 1 if reward > self.reward_threshold else 0
        return self.state, reward, done, success, first_dim_metric, second_dim_metric

    def choose_similar(self, mode):
        if mode == 0:
            row = self.topic_similarities_matrix[self.state]
        elif mode == 1:
            row = self.difficulty_similarities_matrix[self.state]
        flag = True
        mask_threshold = 0.95
        # Mask out overly similar states (similarity > 0.95)
        while flag:
            mask = row <= mask_threshold
            # Apply mask and get top N most different (lowest similarity)
            filtered = row[mask]
            num_valid_states = len(filtered)
            num_candidates = min(num_valid_states, 25)
            flag = num_candidates == 0
            mask_threshold += 0.01
        candidate_indices = np.argsort(filtered)[-num_candidates:]
        return np.flatnonzero(mask)[candidate_indices[np.random.randint(num_candidates)]]

    def choose_different(self, mode):
        if mode == 0:
            row = self.topic_similarities_matrix[self.state]
        elif mode == 1:
            row = self.difficulty_similarities_matrix[self.state]
        flag = True
        mask_threshold = 0.95
        # Mask out overly similar states (similarity > 0.95)
        while flag:
            mask = row <= mask_threshold
            # Apply mask and get top N most different (lowest similarity)
            filtered = row[mask]
            num_valid_states = len(filtered)
            num_candidates = min(num_valid_states, 1000)
            flag = num_candidates == 0
            mask_threshold += 0.01
        candidate_indices = np.argsort(filtered)[:num_candidates]
        return np.flatnonzero(mask)[candidate_indices[np.random.randint(num_candidates)]]

    def compute_similarities_matrix(self, mode):
        """Compute the similarities matrix for the universe."""
        if mode == 0:
            # Compute the similarities matrix for the first vector
            similarities_matrix = cosine_similarity(self.universe[:, :self.num_topics])
        elif mode == 1:
            # Compute the cosine similarity matrix
            similarities_matrix = cosine_similarity(self.universe[:, self.num_topics:])

        return similarities_matrix 

    def clone(self):
        """Create a deep copy of the environment."""
        return CustomEnv(
            universe=self.universe.copy(),
            target_dim1=self.target_dim1.copy(),
            target_dim2=self.target_dim2.copy(),
            num_topics=int(self.num_topics),
            alfa=float(self.alfa),
            reward_threshold=float(self.reward_threshold),
            state=int(self.state)
        ) 