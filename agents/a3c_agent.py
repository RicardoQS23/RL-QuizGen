import numpy as np
from threading import Thread, Lock
from multiprocessing import cpu_count
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logging import save_to_log, save_to_json
from utils.utilities import create_worker_dir
import json

import matplotlib.pyplot as plt
import seaborn as sns
from .base_agent import BaseAgent

import os
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_data(training_data, alfa, test_num, worker_id):
    """Save all training data to JSON files."""
    # Create base directories
    base_dirs = [
        f'../jsons/{test_num}/{worker_id}/success',
        f'../jsons/{test_num}/{worker_id}/reward',
        f'../jsons/{test_num}/{worker_id}/reward_dim1',
        f'../jsons/{test_num}/{worker_id}/reward_dim2',
        f'../jsons/{test_num}/{worker_id}/qvalues',
        f'../jsons/{test_num}/{worker_id}/actions',
        f'../jsons/{test_num}/{worker_id}/loss',
        f'../jsons/{test_num}/{worker_id}/num_iterations'
    ]
    
    for directory in base_dirs:
        os.makedirs(directory, exist_ok=True)
    
    # Save data to JSON files
    save_to_json(training_data['successes'], f'../jsons/{test_num}/{worker_id}/success/all_success_alfa_{alfa}')
    save_to_json(training_data['episode_rewards'], f'../jsons/{test_num}/{worker_id}/reward/all_rewards_alfa_{alfa}')
    save_to_json(training_data['episode_rewards_dim1'], f'../jsons/{test_num}/{worker_id}/reward_dim1/all_rewards_dim1_alfa_{alfa}')
    save_to_json(training_data['episode_rewards_dim2'], f'../jsons/{test_num}/{worker_id}/reward_dim2/all_rewards_dim2_alfa_{alfa}')
    save_to_json(training_data['episode_avg_qvalues'], f'../jsons/{test_num}/{worker_id}/qvalues/all_qvalues_alfa_{alfa}')
    save_to_json(training_data['episode_actions'], f'../jsons/{test_num}/{worker_id}/actions/all_actions_alfa_{alfa}')
    save_to_json(training_data['episode_losses'], f'../jsons/{test_num}/{worker_id}/loss/all_losses_alfa_{alfa}')
    save_to_json(training_data['num_iterations'], f'../jsons/{test_num}/{worker_id}/num_iterations/all_num_iterations_alfa_{alfa}')
    

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, entropy_beta=0.01):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.actor = nn.Linear(16, action_dim)
        self.critic = nn.Linear(16, 1)
        self.entropy_beta = entropy_beta

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action_logits = self.actor(x)
        action_probs = F.softmax(action_logits, dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value


class WorkerAgent(Thread):
    def __init__(self, env, global_actor_critic, device, max_episodes, gamma, update_interval,
                 global_agent, global_optimizer, shared_lock, worker_id=0, test_num=0):
        super(WorkerAgent, self).__init__()
        self.env = env
        self.global_actor_critic = global_actor_critic
        self.device = device
        self.max_episodes = max_episodes
        self.gamma = gamma
        self.update_interval = update_interval
        self.global_agent = global_agent
        self.optimizer = global_optimizer
        self.lock = shared_lock
        self.worker_id = worker_id
        self.test_num = test_num

        self.local_actor_critic = ActorCritic(
            env.state_dim,
            env.action_space.n,
            self.global_agent.entropy_beta
        ).to(device)

        self.local_actor_critic.load_state_dict(self.global_actor_critic.state_dict())

        self.worker_training_data = {
            'episode_rewards': [],
            'episode_rewards_dim1': [],
            'episode_rewards_dim2': [],
            'episode_actions': [],
            'episode_avg_qvalues': [],
            'episode_losses': [],
            'episode_count': 0,
            'successes': [],
            'num_iterations': [],
            'action_probs': [],
        }

    def n_step_td_target(self, rewards, next_v_value, done):
        td_targets = np.zeros_like(rewards)
        cumulative = 0 if done else next_v_value
        for k in reversed(range(len(rewards))):
            cumulative = rewards[k] + self.gamma * cumulative
            td_targets[k] = cumulative
        return td_targets

    def run(self):
        create_worker_dir(f'../logs/{self.test_num}/{self.worker_id}')

        save_to_log(f"Starting worker {self.worker_id} ...", 
        f'../logs/{self.test_num}/{self.worker_id}/training')
        
        for episode in range(self.max_episodes):
            state = self.env.reset()
            save_to_log(f"Episode {episode + 1} started on state {state}...", f'../logs/{self.test_num}/{self.worker_id}/training')

            state_batch, action_batch, reward_batch = [], [], []
            episode_reward = total_reward_dim1 = total_reward_dim2 = 0
            episode_actions = []
            episode_avg_qvalues = []
            action_probs_list = []
            visited_states = set()
            done = False
            num_iterations = 0

            while not done:
                if state not in visited_states:
                    visited_states.add(state)

                state_tensor = torch.FloatTensor(self.env.universe[state]).unsqueeze(0).to(self.device)
                action_probs, _ = self.local_actor_critic(state_tensor)
                
                # Ensure action probabilities are valid
                action_probs = action_probs.clamp(min=1e-7, max=1-1e-7)
                action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
                
                action = torch.multinomial(action_probs, 1).item()

                next_state, reward, done, success, reward_dim1, reward_dim2 = self.env.step(action, num_iterations)

                state_batch.append(self.env.universe[state])
                action_batch.append([action])
                reward_batch.append([reward])

                episode_reward += reward
                total_reward_dim1 += reward_dim1
                total_reward_dim2 += reward_dim2

                episode_actions.append(action)
                episode_avg_qvalues.append(action_probs.max().item())
                action_probs_list.append(action_probs.detach().cpu().numpy().tolist()[0])

                if len(state_batch) >= self.update_interval or done:
                    states = torch.FloatTensor(np.array(state_batch)).to(self.device)
                    actions = torch.LongTensor(np.array(action_batch)).to(self.device)
                    rewards = np.array(reward_batch)

                    next_state_tensor = torch.FloatTensor(self.env.universe[next_state]).unsqueeze(0).to(self.device)
                    _, next_value = self.local_actor_critic(next_state_tensor)
                    td_targets = self.n_step_td_target(rewards, next_value.item(), done)
                    td_targets = torch.FloatTensor(td_targets).to(self.device).view(-1, 1)

                    action_probs, values = self.local_actor_critic(states)
                    advantages = (td_targets - values.detach()).view(-1)
                    
                    # Ensure action probabilities are valid for loss calculation
                    action_probs = action_probs.clamp(min=1e-7, max=1-1e-7)
                    action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
                    
                    log_probs = torch.log(action_probs)
                    actor_loss = -(log_probs.gather(1, actions) * advantages.unsqueeze(1)).mean()
                    critic_loss = advantages.pow(2).mean()
                    entropy = -(action_probs * log_probs).sum(dim=1).mean()
                    loss = actor_loss + critic_loss - self.global_agent.entropy_beta * entropy

                    with self.lock:
                        self.optimizer.zero_grad()
                        loss.backward()
                        # Apply gradients to global network with gradient clipping
                        for global_param, local_param in zip(
                                self.global_actor_critic.parameters(), self.local_actor_critic.parameters()):
                            if local_param.grad is not None:
                                global_param._grad = local_param.grad.clone()
                        #torch.nn.utils.clip_grad_norm_(self.global_actor_critic.parameters(), 40.0)
                        self.optimizer.step()
                        
                        self.local_actor_critic.load_state_dict(self.global_actor_critic.state_dict())

                        self.worker_training_data['episode_losses'].append(loss.item())

                    save_to_log(f'Iteration {num_iterations + 1} State={state} Action={action} Reward={reward}', 
                        f'../logs/{self.test_num}/{self.worker_id}/training', flag=False)
                    
                    state = next_state
                    num_iterations += 1

                    state_batch, action_batch, reward_batch = [], [], []
            
            self.worker_training_data['episode_count'] += 1

            save_to_log(f'EP{episode + 1} Number of Visited States={len(visited_states)} EpisodeReward={episode_reward/num_iterations}', 
                   f'../logs/{self.test_num}/{self.worker_id}/training', flag=False)

            self.worker_training_data['episode_rewards'].append(episode_reward/num_iterations)
            self.worker_training_data['episode_rewards_dim1'].append(total_reward_dim1/num_iterations)
            self.worker_training_data['episode_rewards_dim2'].append(total_reward_dim2/num_iterations)
            self.worker_training_data['num_iterations'].append(num_iterations)
            self.worker_training_data['successes'].append(success)
            self.worker_training_data['episode_actions'].append(episode_actions)
            self.worker_training_data['episode_avg_qvalues'].append(episode_avg_qvalues)
            self.worker_training_data['action_probs'].extend(action_probs_list)
            if episode % 10 == 0:
                avg_reward = np.mean(self.worker_training_data['episode_rewards'][-10:])
                avg_success = np.mean(self.worker_training_data['successes'][-10:])

                print(f"\nWorker {self.worker_id} Metrics (Last 10 episodes):")
                print(f"Average Reward: {avg_reward:.4f}")
                print(f"Success Rate: {avg_success:.2%}")
                
                if len(self.worker_training_data['action_probs']) > 0:
                    recent_probs = np.array(self.worker_training_data['action_probs'][-10:]).mean(axis=0)
                    print(f"Action Distribution: {recent_probs}\n")

            print(f"[Worker {self.worker_id}] Episode {episode + 1} Reward: {episode_reward / num_iterations if num_iterations > 0 else episode_reward}")

        ##SAVE HERE THE TRAINING DATA
        save_data(self.worker_training_data, self.global_agent.alfa, self.test_num, self.worker_id)
        save_to_log('Train complete!', f'../logs/{self.test_num}/{self.worker_id}/training')

class A3CAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, device, lr=0.0005, gamma=0.95, entropy_beta=0.01, num_workers=None, test_num=None, alfa=0.0):
        super().__init__(state_dim, action_dim, device)
        self.lr = lr
        self.gamma = gamma
        self.entropy_beta = entropy_beta
        self.test_num = test_num
        self.alfa = alfa
        self.num_workers = num_workers if num_workers is not None else cpu_count()

        self.update_interval = 1
        self.workers = []
        self.lock = Lock()
        self.worker_ids = []  # Store worker IDs

        self.actor_critic = ActorCritic(self.state_dim, self.action_dim, self.entropy_beta).to(self.device)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=self.lr)

        self.worker_data = {}
        self.training_data = {}

        
    
    def get_action(self, state, epsilon=None):
        """
        Selects an action given a state using the actor-critic policy.
        Epsilon is unused in A3C but kept for compatibility with other agent interfaces.
        Returns:
            action (int): chosen action
            mean_q_value (float): mean of action probabilities (not true Q-values)
            use_random (bool): always False for A3C since it uses policy sampling
        """
        self.actor_critic.eval()
        with torch.no_grad():
           state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
           action_probs, _ = self.actor_critic(state_tensor)
    
        #    Clamp and normalize to ensure valid distribution
           action_probs = action_probs.clamp(min=1e-7, max=1.0 - 1e-7)
           action_probs /= action_probs.sum(dim=-1, keepdim=True)
    
           action = torch.multinomial(action_probs, 1).item()
           mean_q_value = action_probs.mean().item()
    
        return action, mean_q_value, False
        #pass

    

    def train_step(self, state, action, reward, next_state, done):
        # Left as is for single-step training; multi-thread training handled in workers
        pass
    
    def save_global_data(self, global_data, alfa, test_num):
        base_path = f'../jsons/{test_num}/global'
        os.makedirs(base_path, exist_ok=True)

        def save_json(data, name):
            with open(f'{base_path}/{name}_alfa_{alfa}.json', 'w') as f:
                json.dump(data, f)

        save_json(global_data['successes'], 'success_all')
        save_json(global_data['rewards'], 'rewards_all')
        save_json(global_data['rewards_dim1'], 'rewards_dim1_all')
        save_json(global_data['rewards_dim2'], 'rewards_dim2_all')
        save_json(global_data['qvalues'], 'qvalues_all')
        save_json(global_data['actions'], 'actions_all')
        save_json(global_data['loss'], 'loss_all')
        save_json(global_data['iterations'], 'iterations_all')



    def start_workers(self, env):
        self.workers = []
        self.worker_ids = []  # Reset worker IDs
        for i in range(self.num_workers):
            worker = WorkerAgent(
                env.clone(), self.actor_critic, self.device,
                self.training_data['max_episodes'],
                self.gamma, self.update_interval,
                self, self.optimizer, self.lock, worker_id=i, test_num=self.test_num
            )
            self.workers.append(worker)
            self.worker_ids.append(i)  # Store worker ID
            worker.start()

    def stop_workers(self):
        for worker in self.workers:
            worker.join()

        # Aggregate global data after all workers finish
        global_data = {
            'successes': [],
            'rewards': [],
            'rewards_dim1': [],
            'rewards_dim2': [],
            'qvalues': [],
            'actions': [],
            'loss': [],
            'iterations': [],
        }

        for worker in self.workers:
            data = worker.worker_training_data
            global_data['successes'].extend(data['successes'])
            global_data['rewards'].extend(data['episode_rewards'])
            global_data['rewards_dim1'].extend(data['episode_rewards_dim1'])
            global_data['rewards_dim2'].extend(data['episode_rewards_dim2'])
            global_data['qvalues'].extend(data['episode_avg_qvalues'])
            global_data['actions'].extend(data['episode_actions'])
            global_data['loss'].extend(data['episode_losses'])
            global_data['iterations'].extend(data['num_iterations'])
            
        for key in global_data:
            try:
                global_data[key] = sorted(global_data[key], key=lambda x: x if isinstance(x, (int, float)) else len(x))
            except Exception:
                # Fallback if sorting fails (e.g., if contents are non-comparable)
                pass

        self.save_global_data(global_data, self.alfa, self.test_num)
        self.training_data = {
            'episode_rewards': global_data['rewards'],
            'episode_rewards_dim1': global_data['rewards_dim1'],
            'episode_rewards_dim2': global_data['rewards_dim2'],
            'episode_actions': global_data['actions'],
            'episode_avg_qvalues': global_data['qvalues'],
            'episode_losses': global_data['loss'],
            'episode_count': len(global_data['rewards']),
            'step_count': 0,
            'replay_count': 0
            }
        self.workers = []

    def save(self, path):
        save_dict = {
            'model_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_data': {
                'episode_rewards': self.training_data['episode_rewards'],
                'episode_rewards_dim1': self.training_data['episode_rewards_dim1'],
                'episode_rewards_dim2': self.training_data['episode_rewards_dim2'],
                'episode_actions': self.training_data['episode_actions'],
                'episode_avg_qvalues': self.training_data['episode_avg_qvalues'],
                'episode_losses': self.training_data['episode_losses'],
                'episode_count': self.training_data['episode_count'],
                'step_count': self.training_data['step_count'],
                'replay_count': self.training_data['replay_count']
            }
        }
        torch.save(save_dict, path)

    def load(self, path):
        """Load the agent's model"""
        try:
            # Try different loading methods
            checkpoint = None
            errors = []
            
            # Method 1: Standard torch.load
            try:
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                print("Successfully loaded using standard torch.load")
            except Exception as e:
                errors.append(f"Standard torch.load failed: {str(e)}")
            
            # Method 2: Try with pickle directly
            if checkpoint is None:
                try:
                    import pickle
                    with open(path, 'rb') as f:
                        checkpoint = pickle.load(f)
                    print("Successfully loaded using pickle")
                except Exception as e:
                    errors.append(f"Pickle load failed: {str(e)}")
            
            # Method 3: Try with different pickle protocol
            if checkpoint is None:
                try:
                    import pickle
                    with open(path, 'rb') as f:
                        checkpoint = pickle.load(f, encoding='latin1')
                    print("Successfully loaded using pickle with latin1 encoding")
                except Exception as e:
                    errors.append(f"Pickle load with latin1 encoding failed: {str(e)}")
            
            if checkpoint is None:
                raise RuntimeError(f"All loading methods failed. Errors:\n" + "\n".join(errors))
            
            # Process the loaded checkpoint
            if isinstance(checkpoint, A3CAgent):
                print("Loading from A3CAgent instance")
                # Check which attributes are available in the loaded agent
                if hasattr(checkpoint, 'actor_critic'):
                    self.actor_critic.load_state_dict(checkpoint.actor_critic.state_dict())
                elif hasattr(checkpoint, 'model'):
                    self.actor_critic.load_state_dict(checkpoint.model.state_dict())
                
                if hasattr(checkpoint, 'optimizer'):
                    self.optimizer.load_state_dict(checkpoint.optimizer.state_dict())
                
                if hasattr(checkpoint, 'training_data'):
                    for key, value in checkpoint.training_data.items():
                        if key in self.training_data:
                            self.training_data[key] = value
            elif isinstance(checkpoint, dict):
                print("Loading from dictionary")
                if 'model_state_dict' in checkpoint:
                    self.actor_critic.load_state_dict(checkpoint['model_state_dict'])
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

    def get_training_data(self):
        return self.training_data

    def get_worker_metrics(self):
        return {worker.worker_id: worker.worker_metrics for worker in self.workers}

    def get_worker_ids(self):
        """Return the list of worker IDs."""
        return self.worker_ids