from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, state_dim, action_dim, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
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

    @abstractmethod
    def get_action(self, state, epsilon):
        """Get action from the agent given a state"""
        pass

    @abstractmethod
    def train_step(self, state, action, reward, next_state, done):
        """Perform a training step"""
        pass

    @abstractmethod
    def save(self, path):
        """Save the agent's model"""
        pass

    @abstractmethod
    def load(self, path):
        """Load the agent's model"""
        pass

    def update_episode_data(self, total_reward, total_reward_dim1, total_reward_dim2, 
                          exploration_count, exploitation_count, success, 
                          episode_actions, episode_avg_qvalues, num_iterations):
        """Update training data after an episode"""
        self.training_data['episode_rewards'].append(total_reward/num_iterations)
        self.training_data['episode_rewards_dim1'].append(total_reward_dim1/num_iterations)
        self.training_data['episode_rewards_dim2'].append(total_reward_dim2/num_iterations)
        self.training_data['exploration_counts'].append(exploration_count)
        self.training_data['exploitation_counts'].append(exploitation_count)
        self.training_data['success_episodes'].append(success)
        self.training_data['episode_actions'].append(episode_actions)
        self.training_data['episode_avg_qvalues'].append(episode_avg_qvalues) 