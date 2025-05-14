import os
import torch
from agents.a3c_agent import A3CAgent
from environments.custom_env import CustomEnv
from utils.utilities import load_data, get_best_state

device = torch.device("cuda")
folder = "../saved_agents/A2CMedicalNewReward/"
test_num = "A2CMedicalNewReward"
alfa_values = [0, 0.25, 0.5, 0.75, 1]

universe, targets = load_data(test_num)
num_topics = 10
reward_threshold = 0.85

for alfa in alfa_values:
    filename = f"agent_alfa_{alfa}_bias.pkl"
    path = os.path.join(folder, filename)
    print(f"🔁 Attempting to load: {path}")

    try:
        # Create dummy env to get correct dimensions
        best_state, _ = get_best_state(universe, targets, alfa, num_topics)
        env = CustomEnv(
            universe=universe,
            target_dim1=targets[0],
            target_dim2=targets[1],
            num_topics=num_topics,
            alfa=alfa,
            reward_threshold=reward_threshold,
            state=best_state
        )
        state_dim = env.state_dim
        action_dim = env.action_space.n

        # Create dummy agent
        agent = A3CAgent(state_dim, action_dim, device)

        # Load corrupt-style full object using pickle
        import pickle
        with open(path, 'rb') as f:
            old_agent = pickle.load(f)

        # Copy weights and data
        agent.global_actor_critic.load_state_dict(old_agent.global_actor_critic.state_dict())
        agent.optimizer.load_state_dict(old_agent.optimizer.state_dict())
        agent.training_data = old_agent.training_data

        # Save proper checkpoint
        clean_path = path.replace(".pkl", ".pth")
        torch.save({
            'global_actor_critic_state_dict': agent.global_actor_critic.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'training_data': agent.training_data
        }, clean_path)
        print(f"✅ Converted and saved: {clean_path}")

    except Exception as e:
        print(f"❌ Failed for {alfa}: {e}")
