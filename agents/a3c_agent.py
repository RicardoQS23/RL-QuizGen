# agents/a3c_models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from threading import Lock
import copy

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, action_dim)
        self.entropy_beta = 0.01
        self.opt = torch.optim.Adam(self.parameters(), lr=0.0005)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=-1)

    def train_step(self, states, actions, advantages):
        log_probs = self(states)
        loss = self.compute_loss(log_probs, actions, advantages)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.opt = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    def train_step(self, states, td_targets):
        values = self(states)
        loss = self.compute_loss(values, td_targets)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()
        

class A3CWorker(Thread):
    def __init__(self, env_template, agent, max_episodes, device, lock, worker_id):
        super().__init__()
        self.env = copy.deepcopy(env_template)
        self.agent = agent
        self.max_episodes = max_episodes
        self.device = device
        self.lock = lock
        self.worker_id = worker_id

        self.state_dim = self.env.state_dim
        self.action_dim = self.env.action_space.n

        self.actor = copy.deepcopy(agent.global_actor).to(device)
        self.critic = copy.deepcopy(agent.global_critic).to(device)

    def n_step_td_target(self, rewards, next_v_value, done):
        td_targets = np.zeros_like(rewards)
        cumulative = 0 if done else next_v_value
        for k in reversed(range(len(rewards))):
            cumulative = rewards[k] + self.agent.gamma * cumulative
            td_targets[k] = cumulative
        return td_targets

    def run(self):
        while True:
            with self.lock:
                if self.agent.training_data['episode_count'] >= self.max_episodes:
                    break
                self.agent.training_data['episode_count'] += 1
                current_episode = self.agent.training_data['episode_count']

            state = self.env.reset()
            state_batch, action_batch, reward_batch = [], [], []
            episode_reward, done = 0, False

            while not done:
                state_tensor = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(self.device)
                log_probs = self.actor(state_tensor)
                probs = torch.exp(log_probs).cpu().detach().numpy()[0]
                action = np.random.choice(self.action_dim, p=probs)

                next_state, reward, done, _, reward1, reward2 = self.env.step(action)

                state_batch.append(state)
                action_batch.append([action])
                reward_batch.append([reward])

                if len(state_batch) >= 5 or done:
                    states = torch.tensor(np.array(state_batch), dtype=torch.float32).to(self.device)
                    actions = torch.tensor(np.array(action_batch), dtype=torch.long).to(self.device)
                    rewards = np.array(reward_batch)

                    next_state_tensor = torch.tensor(np.array(next_state), dtype=torch.float32).unsqueeze(0).to(self.device)
                    next_v = self.critic(next_state_tensor).item()
                    td_targets = self.n_step_td_target(rewards, next_v, done)
                    td_targets = torch.tensor(td_targets, dtype=torch.float32).to(self.device).view(-1, 1)

                    values = self.critic(states)
                    advantages = (td_targets - values.detach()).view(-1)

                    with self.lock:
                        self.agent.global_actor.train_step(states, actions, advantages)
                        self.agent.global_critic.train_step(states, td_targets)
                        self.actor.load_state_dict(self.agent.global_actor.state_dict())
                        self.critic.load_state_dict(self.agent.global_critic.state_dict())

                    state_batch, action_batch, reward_batch = [], [], []

                episode_reward += reward
                state = next_state

            with self.lock:
                self.agent.training_data['total_rewards'].append(episode_reward)


class A3CAgent:
    def __init__(self, state_dim, action_dim, device, lr=0.0005, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma

        self.global_actor = Actor(state_dim, action_dim).to(device)
        self.global_critic = Critic(state_dim).to(device)

        self.training_data = {
            'episode_count': 0,
            'epsilon': 1.0,
            'total_rewards': [],
            'replay_count': 0,
            'episode_info': []
        }

        self.lock = Lock()

    def get_action(self, state, epsilon=None):
        state_tensor = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(self.device)
        log_probs = self.global_actor(state_tensor)
        probs = torch.exp(log_probs).cpu().detach().numpy()[0]
        action = np.random.choice(self.action_dim, p=probs)
        return action, np.mean(probs), False

    def train_step(self, *args, **kwargs):
        pass  # A3C uses internal async updates

    def update_episode_data(self, reward, r1, r2, explore_count, exploit_count, success,
                            episode_actions, episode_avg_qvalues, num_iter):
        self.training_data['total_rewards'].append(reward)
        self.training_data['episode_info'].append({
            'reward': reward,
            'actions': episode_actions,
            'avg_qvalues': episode_avg_qvalues,
            'explore': explore_count,
            'exploit': exploit_count,
            'success': success,
            'steps': num_iter
        })

    def train_async(self, env_template, max_episodes):
        self.CUR_EPISODE = 0
        self.REWARD_HISTORY = []

        workers = [
            A3CWorker(env_template, self, max_episodes, self.device, self.lock, i)
            for i in range(os.cpu_count())
        ]

        for w in workers:
            w.start()
        for w in workers:
            w.join()
