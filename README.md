# Train
The script to train the Reinforcement Learning model has the options to control a variety of parameters:
- Agent type: DQN (default), SARSA, A2C, A3C -> --agent_type
- Experience replay buffer: PER - Prioritized Experience Replay Buffer (default), Normal -> --replay_buffer
- Target Distribution: Uniform (default), Sparse_Topic, Sparse_Difficulty, Sparse -> --target_distribution
- Test Name -> --test_num
- Reward_threshold: Reward threshold that defines episode's stop condition, 0.85 (default) -> --reward_threshold
- Max iterations: Max iterations that defines episode's stop condition
- Number of workers: Number of worker threads for A3C, 2 (default) -> num_workers
- Max episodes: Number of max episodes for training session, 5000 (default) -> --max_episodes
- Gamma: Discount factor hyperparameter, 0.95 (default) -> --gamma
- Learning rate: Learning rate hyperparameter, 5e-4 (default) -> --lr
- Batch size: Batch size hyperparameter, 128 (default) -> --batch_size
- Epsilon: Initial epsilon for exploration, 1 (default) -> --eps
- Epsilon decay: Epsilon decay rate, 0.997 (default) -> --eps_decay
- Minimum Epsilon: Minimum Epsilon value, 0.05 (default) -> --eps_min
- Target Sync Frequency: Frequency of target network updates, 1000 (default) -> --target_sync_freq
- Data Source: Data source for the dataset, Load Existing(0), Generate from Real data(1, default), Generate from Synthetic data(2) -> --data_source
- Dataset: Dataset to use when generating from real data, Medical (default) and Math -> --dataset
- Number of topics: Number of topics for universe generation, 10 (default) -> --num_topics
- Universe size: Size of the universe, 10 000 (default) -> --num_topics
- Quiz size: Size of the universe, 10 (default) -> --quiz_size
- Generator type: Universe bias for the generation of the synthetic dataset, uniform (default), topic_focused, difficulty_focused, topic_diverse, difficulty_diverse, correlated -> --generator_type
- Alfa values: List of alfa values hyperparameters, \[0, 0.25, 0.5, 0.75, 1\](default) -> --alfa_values
- Random Seed: Ransom seed for reproducibility, 23 (default) -> --seed

Example of a program call: `python3 train.py --agent_type sarsa --max_episodes 1000 --dataset math --universe_size 30000`

When the program runs, it creates 5 folders on the outside directory: `data`, `jsons`, `logs`, `saved_agents`, `images`

In order for the program to run correctly, the `medical.csv` and `math.csv` files should be inserted inside the `data` folder
