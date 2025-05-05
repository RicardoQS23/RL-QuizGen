import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from threading import Thread, Lock
from multiprocessing import cpu_count
from .base_agent import BaseAgent

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, lr=0.0005):
        print(f"Initializing Actor with state_dim: {state_dim}, action_dim: {action_dim}")
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.entropy_beta = 0.01
        self.max_grad_norm = 0.5

    def forward(self, x):
        print(f"Forward pass input shape: {x.shape}")
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=-1)

    def compute_loss(self, log_probs, actions, advantages):
        print(f"Computing loss with log_probs shape: {log_probs.shape}, actions shape: {actions.shape}, advantages shape: {advantages.shape}")
        if log_probs.dim() == 1:
            log_probs = log_probs.unsqueeze(0)
        actions = actions.view(-1)
        log_probs_taken = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
        advantages = advantages.view(-1)
        policy_loss = -(log_probs_taken * advantages).mean()
        entropy = -(log_probs.exp() * log_probs).sum(dim=1).mean()
        return policy_loss - self.entropy_beta * entropy

    def train_step(self, states, actions, advantages):
        print(f"Training step with states shape: {states.shape}, actions shape: {actions.shape}, advantages shape: {advantages.shape}")
        log_probs = self(states)
        loss = self.compute_loss(log_probs, actions, advantages)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.opt.step()
        return loss.item()

class Critic(nn.Module):
    def __init__(self, state_dim, lr=0.001):
        print(f"Initializing Critic with state_dim: {state_dim}")
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.max_grad_norm = 0.5

    def forward(self, x):
        print(f"Forward pass input shape: {x.shape}")
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def compute_loss(self, v_pred, td_targets):
        print(f"Computing loss with v_pred shape: {v_pred.shape}, td_targets shape: {td_targets.shape}")
        return F.mse_loss(v_pred, td_targets)

    def train_step(self, states, td_targets):
        print(f"Training step with states shape: {states.shape}, td_targets shape: {td_targets.shape}")
        values = self(states)
        loss = self.compute_loss(values, td_targets)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.opt.step()
        return loss.item()

class A3CWorker(Thread):
    def __init__(self, worker_id, env, global_actor, global_critic, agent, device, gamma=0.99, update_interval=5, max_episodes=1000):
        print(f"Initializing A3CWorker with worker_id: {worker_id}, env: {env}, global_actor: {global_actor}, global_critic: {global_critic}, agent: {agent}, device: {device}, gamma: {gamma}, update_interval: {update_interval}, max_episodes: {max_episodes}")
        Thread.__init__(self)
        self.worker_id = worker_id
        self.env = env
        self.device = device
        self.gamma = gamma
        self.update_interval = update_interval
        self.max_episodes = max_episodes
        self.global_actor = global_actor
        self.global_critic = global_critic
        self.agent = agent
        self.actor = Actor(env.state_dim, env.action_space.n).to(device)
        self.critic = Critic(env.state_dim).to(device)
        self.sync_networks()
        self.network_lock = Lock()
        self.episode_count = 0
        self.running = True
        self.error = None

    def get_epsilon(self):
        """Get epsilon value from agent"""
        print(f"Getting epsilon from agent")
        with self.agent.data_lock:
            return self.agent.training_data['epsilon']

    def update_epsilon(self):
        """Update epsilon value in agent"""
        print(f"Updating epsilon in agent") 
        with self.agent.data_lock:
            self.agent.training_data['epsilon'] = max(
                self.agent.eps_min,
                self.agent.training_data['epsilon'] * self.agent.eps_decay
            )

    def sync_networks(self):
        """Synchronize local networks with global networks"""
        print(f"Synchronizing networks")
        self.actor.load_state_dict(self.global_actor.state_dict())
        self.critic.load_state_dict(self.global_critic.state_dict())

    def n_step_td_target(self, rewards, next_v_value, done):
        """Compute n-step TD targets"""
        print(f"Computing n-step TD targets")
        td_targets = np.zeros_like(rewards)
        cumulative = 0 if done else next_v_value
        for k in reversed(range(len(rewards))):
            cumulative = rewards[k] + self.gamma * cumulative
            td_targets[k] = cumulative
        return td_targets

    def run(self):
        """Main training loop for the worker"""
        try:
            print(f"Starting training loop")
            while self.running and self.episode_count < self.max_episodes:
                try:
                    print(f"Resetting environment")
                    state = self.env.reset()
                    state_batch, action_batch, reward_batch = [], [], []
                    episode_reward = 0
                    episode_reward_dim1 = 0
                    episode_reward_dim2 = 0
                    exploration_count = 0
                    exploitation_count = 0
                    episode_actions = []
                    episode_avg_qvalues = []
                    episode_losses = []  # Track losses for this episode
                    done = False

                    while not done:
                        try:
                            # Get state features and convert to tensor
                            state_features = self.env.universe[state]
                            state_tensor = torch.FloatTensor(state_features).unsqueeze(0).to(self.device)
                            
                            # Get action probabilities
                            with torch.no_grad():
                                log_probs = self.actor(state_tensor)
                                probs = torch.exp(log_probs).cpu().numpy()[0]
                                value = self.critic(state_tensor).item()
                            
                            # Epsilon-greedy action selection
                            epsilon = self.get_epsilon()
                            if np.random.random() < epsilon:
                                action = np.random.randint(self.env.action_space.n)
                                explore = True
                            else:
                                action = np.random.choice(self.env.action_space.n, p=probs)
                                explore = False
                            
                            # Take action
                            next_state, reward, done, success, reward_dim1, reward_dim2 = self.env.step(action, 100)
                            
                            # Store transition
                            state_batch.append(state_features)
                            action_batch.append([action])
                            reward_batch.append([reward])
                            
                            # Update episode statistics
                            episode_reward += reward
                            episode_reward_dim1 += reward_dim1
                            episode_reward_dim2 += reward_dim2
                            if explore:
                                exploration_count += 1
                            else:
                                exploitation_count += 1
                            
                            episode_actions.append(action)
                            episode_avg_qvalues.append(value)
                            
                            # Update if enough transitions or episode is done
                            if len(state_batch) >= self.update_interval or done:
                                try:
                                    print(f"Updating global networks")
                                    states = torch.FloatTensor(np.array(state_batch)).to(self.device)
                                    actions = torch.LongTensor(np.array(action_batch)).to(self.device)
                                    rewards = np.array(reward_batch)
                                    
                                    # Get next state value
                                    next_state_features = self.env.universe[next_state]
                                    next_state_tensor = torch.FloatTensor(next_state_features).unsqueeze(0).to(self.device)
                                    next_v = self.critic(next_state_tensor).item()
                                    
                                    # Compute TD targets and advantages
                                    td_targets = self.n_step_td_target(rewards, next_v, done)
                                    td_targets = torch.FloatTensor(td_targets).to(self.device).view(-1, 1)
                                    
                                    values = self.critic(states)
                                    advantages = (td_targets - values.detach()).view(-1)
                                    
                                    # Update global networks with network lock
                                    with self.network_lock:
                                        # Compute gradients
                                        actor_loss = self.actor.compute_loss(self.actor(states), actions, advantages)
                                        critic_loss = self.critic.compute_loss(self.critic(states), td_targets)
                                        
                                        # Track losses
                                        episode_losses.append((actor_loss.item() + critic_loss.item()) / 2)
                                        
                                        # Zero gradients
                                        self.actor.opt.zero_grad()
                                        self.critic.opt.zero_grad()
                                        self.global_actor.opt.zero_grad()
                                        self.global_critic.opt.zero_grad()
                                        
                                        # Backward pass
                                        actor_loss.backward()
                                        critic_loss.backward()
                                        
                                        # Clip gradients
                                        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor.max_grad_norm)
                                        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic.max_grad_norm)
                                        
                                        # Update global networks
                                        for param, global_param in zip(self.actor.parameters(), self.global_actor.parameters()):
                                            if param.grad is not None:
                                                global_param.grad = param.grad.clone()
                                        for param, global_param in zip(self.critic.parameters(), self.global_critic.parameters()):
                                            if param.grad is not None:
                                                global_param.grad = param.grad.clone()
                                        
                                        # Step optimizers
                                        self.global_actor.opt.step()
                                        self.global_critic.opt.step()
                                        
                                        # Sync networks
                                        self.sync_networks()
                                    
                                    state_batch, action_batch, reward_batch = [], [], []
                                except Exception as e:
                                    print(f"Warning: Worker {self.worker_id} failed to update networks: {str(e)}")
                                    continue
                            
                            state = next_state
                        except Exception as e:
                            print(f"Warning: Worker {self.worker_id} encountered error in step: {str(e)}")
                            done = True
                            break
                    
                    # Update training data through agent
                    try:
                        print(f"Updating training data")
                        # Calculate average loss for the episode
                        avg_episode_loss = np.mean(episode_losses) if episode_losses else 0.0
                        
                        self.agent.update_episode_data(
                            episode_reward, episode_reward_dim1, episode_reward_dim2,
                            exploration_count, exploitation_count, success,
                            episode_actions, episode_avg_qvalues, self.episode_count
                        )
                        
                        # Update episode losses
                        with self.agent.data_lock:
                            if len(self.agent.training_data['episode_losses']) < self.episode_count + 1:
                                self.agent.training_data['episode_losses'].append(avg_episode_loss)
                            else:
                                self.agent.training_data['episode_losses'][self.episode_count] = avg_episode_loss
                        
                        self.episode_count += 1
                        self.update_epsilon()
                    except Exception as e:
                        print(f"Warning: Worker {self.worker_id} failed to update training data: {str(e)}")
                except Exception as e:
                    print(f"Warning: Worker {self.worker_id} encountered error in episode: {str(e)}")
                    continue
        except Exception as e:
            self.error = e
            print(f"Error in worker {self.worker_id}: {str(e)}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        try:
            print(f"Cleaning up resources")
            # Clear any remaining data
            state_batch, action_batch, reward_batch = [], [], []
            # Ensure networks are synced one last time
            with self.network_lock:
                self.sync_networks()
        except Exception as e:
            print(f"Warning: Worker {self.worker_id} failed to cleanup: {str(e)}")

    def stop(self):
        """Stop the worker thread"""
        print(f"Stopping worker {self.worker_id}")
        self.running = False
        try:
            self.join(timeout=1.0)
        except Exception as e:
            print(f"Warning: Failed to stop worker {self.worker_id}: {str(e)}")

class A3CAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, device, lr=0.0005, gamma=0.99, update_interval=5, num_workers=8, max_episodes=1000):
        print(f"Initializing A3CAgent with state_dim: {state_dim}, action_dim: {action_dim}, device: {device}, lr: {lr}, gamma: {gamma}, update_interval: {update_interval}, num_workers: {num_workers}, max_episodes: {max_episodes}")
        super().__init__(state_dim, action_dim, device)
        self.lr = lr
        self.gamma = gamma
        self.update_interval = update_interval
        self.num_workers = num_workers
        self.max_episodes = max_episodes
        self.eps_decay = 0.999
        self.eps_min = 0.01
        
        # Initialize global networks
        self.global_actor = Actor(state_dim, action_dim, lr).to(device)
        self.global_critic = Critic(state_dim, lr).to(device)
        
        # Initialize training data with thread-safe dictionary
        self.training_data = {
            "episode_rewards": [],
            "episode_rewards_dim1": [],
            "episode_rewards_dim2": [],
            "episode_actions": [],
            "episode_avg_qvalues": [],
            "exploration_counts": [],
            "exploitation_counts": [],
            "success_episodes": [],
            "episode_losses": [],  # Added episode_losses
            "epsilon": 1.0,
            "episode_count": 0
        }
        
        # Initialize workers
        self.workers = []
        self.network_lock = Lock()  # Lock for network updates
        self.data_lock = Lock()     # Lock for data updates

    def start_workers(self, env):
        """Initialize and start worker threads"""
        print(f"Starting worker threads")
        self.workers = []
        for worker_id in range(self.num_workers):
            worker = A3CWorker(
                worker_id=worker_id,
                env=env,
                global_actor=self.global_actor,
                global_critic=self.global_critic,
                agent=self,
                device=self.device,
                gamma=self.gamma,
                update_interval=self.update_interval,
                max_episodes=self.max_episodes
            )
            self.workers.append(worker)
            worker.start()

    def stop_workers(self):
        """Stop all worker threads"""
        print(f"Stopping worker threads")
        for worker in self.workers:
            try:
                worker.stop()
                if worker.is_alive():
                    worker.join(timeout=1.0)
                if worker.error:
                    print(f"Worker {worker.worker_id} stopped with error: {worker.error}")
            except Exception as e:
                print(f"Warning: Failed to stop worker: {str(e)}")
        self.workers = []

    def get_action(self, state, epsilon):
        """Get action from the agent given a state"""
        print(f"Getting action from agent")
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            log_probs = self.global_actor(state_tensor)
            probs = torch.exp(log_probs).cpu().numpy()[0]
            value = self.global_critic(state_tensor).item()
        
        if np.random.random() < epsilon:
            action = np.random.randint(self.action_dim)
            explore = True
        else:
            action = np.random.choice(self.action_dim, p=probs)
            explore = False
        
        return action, value, explore

    def train_step(self, state, action, reward, next_state, done):
        """Perform a training step"""
        print(f"Performing training step")
        # This is handled by the workers
        pass

    def save(self, path):
        """Save the agent's model"""
        try:
            print(f"Saving model")
            # Create a copy of training data without locks
            training_data_copy = {
                k: v for k, v in self.training_data.items() 
                if k not in ['network_lock', 'data_lock']
            }
            
            state_dict = {
                'global_actor_state_dict': self.global_actor.state_dict(),
                'global_critic_state_dict': self.global_critic.state_dict(),
                'training_data': training_data_copy,
                'gamma': self.gamma,
                'lr': self.lr,
                'update_interval': self.update_interval
            }
            torch.save(state_dict, path)
        except Exception as e:
            print(f"Warning: Failed to save model: {str(e)}")

    def load(self, path):
        """Load the agent's model"""
        try:
            print(f"Loading model")
            checkpoint = torch.load(path)
            with self.network_lock:  # Lock when updating networks
                self.global_actor.load_state_dict(checkpoint['global_actor_state_dict'])
                self.global_critic.load_state_dict(checkpoint['global_critic_state_dict'])
            with self.data_lock:  # Lock when updating training data
                self.training_data = checkpoint['training_data']
            self.gamma = checkpoint['gamma']
            self.lr = checkpoint['lr']
            self.update_interval = checkpoint['update_interval']
        except Exception as e:
            print(f"Warning: Failed to load model: {str(e)}")

    def update_episode_data(self, total_reward, total_reward_dim1, total_reward_dim2, 
                          exploration_count, exploitation_count, success,
                          episode_actions, episode_avg_qvalues, num_iterations):
        """Update training data after each episode"""
        try:
            print(f"Updating episode data")
            with self.data_lock:  # Lock when updating training data
                self.training_data['episode_rewards'].append(float(total_reward))
                self.training_data['episode_rewards_dim1'].append(float(total_reward_dim1))
                self.training_data['episode_rewards_dim2'].append(float(total_reward_dim2))
                self.training_data['exploration_counts'].append(int(exploration_count))
                self.training_data['exploitation_counts'].append(int(exploitation_count))
                self.training_data['success_episodes'].append(bool(success))
                self.training_data['episode_actions'].append(episode_actions)
                self.training_data['episode_avg_qvalues'].append(episode_avg_qvalues)
                self.training_data['episode_count'] += 1
                # Add a placeholder for episode_losses if not present
                if len(self.training_data['episode_losses']) < self.training_data['episode_count']:
                    self.training_data['episode_losses'].append(0.0)  # Default loss value
        except Exception as e:
            print(f"Warning: Failed to update episode data: {str(e)}") 