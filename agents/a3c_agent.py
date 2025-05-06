import argparse
import numpy as np
from threading import Thread, Lock
from multiprocessing import cpu_count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
import seaborn as sns
from .base_agent import BaseAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
                 global_agent, global_optimizer, shared_lock, test_num, worker_id=0):
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
        }

    def n_step_td_target(self, rewards, next_v_value, done):
        td_targets = np.zeros_like(rewards)
        cumulative = 0 if done else next_v_value
        for k in reversed(range(len(rewards))):
            cumulative = rewards[k] + self.gamma * cumulative
            td_targets[k] = cumulative
        return td_targets

    def run(self):
        save_to_log(f"Starting worker {self.worker_id} ...", 
        f'../logs/{test_num}/{self.worker_id}/training')
        
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
                
                self.worker_metrics['action_probs'].append(action_probs.detach().cpu().numpy())
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
                action_probs_list.append(action_probs.cpu().numpy())

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
            self.worker_training_data['action_probs'].append(action_probs_list)
            if episode % 10 == 0:
                avg_reward = np.mean(self.worker_training_data['episode_rewards'][-10:])
                avg_success = np.mean(self.worker_training_data['successes'][-10:])

                print(f"\nWorker {self.worker_id} Metrics (Last 10 episodes):")
                print(f"Average Reward: {avg_reward:.4f}")
                print(f"Success Rate: {avg_success:.2%}")
                
                if len(self.worker_training_data['action_probs']) > 0:
                    recent_probs = np.mean(self.worker_training_data['action_probs'][-10:], axis=0)
                    print(f"Action Distribution: {recent_probs}\n")

            print(f"[Worker {self.worker_id}] Episode {episode + 1} Reward: {episode_reward / num_iterations if num_iterations > 0 else episode_reward}")

        save_to_log(f'Train complete! Total Visited States: {len(all_visited_states)}', f'../logs/{self.test_num}/{self.worker_id}/training')

class A3CAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, device, lr=0.0005, gamma=0.95, entropy_beta=0.01, num_workers=None, test_num=None):
        super().__init__(state_dim, action_dim, device)
        self.lr = lr
        self.gamma = gamma
        self.entropy_beta = entropy_beta
        self.test_num = test_num
        
        self.num_workers = num_workers if num_workers is not None else cpu_count()

        self.update_interval = 1
        self.workers = []
        self.lock = Lock()
        #self.all_visited_states = set()

        self.actor_critic = ActorCritic(self.state_dim, self.action_dim, self.entropy_beta).to(self.device)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=self.lr)

        self.worker_data = {}
        
    
    def get_action(self, state, epsilon=None):
        print('YEDKHOL')
        pass
        '''
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, _ = self.actor_critic(state)
        action = torch.multinomial(action_probs, 1).item()
        mean_q_value = action_probs.mean().item()
        return action, mean_q_value, False
        '''
    

    def train_step(self, state, action, reward, next_state, done):
        # Left as is for single-step training; multi-thread training handled in workers
        pass

    def start_workers(self, env):
        self.workers = [
            WorkerAgent(
                env.clone(), self.actor_critic, self.device,
                self.training_data['max_episodes'],
                self.gamma, self.update_interval,
                self, self.optimizer, self.lock, self.test_num, worker_id=i
            )
            for i in range(self.num_workers)
        ]
        for worker in self.workers:
            worker.start()

    def stop_workers(self):
        for worker in self.workers:
            worker.join()
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
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'training_data' in checkpoint:
            for key, value in checkpoint['training_data'].items():
                self.training_data[key] = value

    def get_training_data(self):
        return self.training_data

    def get_worker_metrics(self):
        return {worker.worker_id: worker.worker_metrics for worker in self.workers}
