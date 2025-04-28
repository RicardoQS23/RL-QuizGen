import os
import json
import random
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from utils.logging import save_to_log

def flatten_inhomogeneous_array(inhomogeneous_array):
    """Flatten an inhomogeneous array into a 1D numpy array."""
    inhomogeneous_array = np.array([np.array(x, dtype=object) for x in inhomogeneous_array], dtype=object)
    homogeneous_array = [x.reshape(1) if x.ndim == 0 else x for x in inhomogeneous_array]
    flattened_array = np.concatenate(homogeneous_array)
    return np.array(flattened_array, dtype=np.float32)

def generate_targets(quizzes_df):
    """Generate target vectors for topic and difficulty coverage."""
    dummy_topic_coverage = quizzes_df['topic_coverage'].values[0]
    dummy_diff_coverage = quizzes_df['difficulty_coverage'].values[0]
    
    target1 = np.random.uniform(0, 1, size=dummy_topic_coverage.shape)
    target1 = target1 / np.sum(target1)  # Normalize to sum to 1
    target1 = np.clip(target1, 0, 1)  # Ensure values are between 0 and 1    
    target2 = np.random.uniform(0, 1, size=dummy_diff_coverage.shape) 
    target2 = target2 / np.sum(target2)  # Normalize to sum to 1
    target2 = np.clip(target2, 0, 1)  # Ensure values are between 0 and 1    
    
    return target1, target2

def generate_quizzes(mcqs_df, output_name, universe_size=10000, quiz_size=10):
    """Generate quizzes from MCQs with topic and difficulty coverage."""
    # Compute the difficulty distribution for each set
    def compute_difficulty_distribution(quiz_mcqs):
        difficulties = mcqs_df.loc[mcqs_df['id'].isin(quiz_mcqs), 'difficulty']
        difficulty_vector = np.zeros(5)  # Initialize a zero vector of size 5
        for diff in difficulties:
            difficulty_vector[np.argmax(diff)] += 1  # Increment the count for the corresponding difficulty
        return difficulty_vector / len(difficulties)  # Normalize to get the distribution
    
    # Function to calculate topic distribution for each set
    def compute_topic_coverage(quiz_mcqs):
        topic_vector = np.zeros(len(unique_topics))  # Initialize zero vector
        topics = mcqs_df.loc[mcqs_df['id'].isin(quiz_mcqs), 'topic']
        topic_counts = topics.value_counts(normalize=True).to_dict()  # Normalize to percentage
        
        # Fill the vector
        for topic, freq in topic_counts.items():
            topic_vector[topic_index[topic]] = freq

        return topic_vector
    
    N = universe_size
    # Total MCQs
    mcqs = mcqs_df['id'].tolist()
    random.shuffle(mcqs)

    # Step 1: Ensure each MCQ appears at least once
    sets = set()  # Use a set to ensure uniqueness
    for i in range(0, len(mcqs), quiz_size):
        subset = tuple(mcqs[i:i+quiz_size])
        if len(subset) == quiz_size:  # Ensure we always have full sets
            sets.add(subset)
    # Step 2: Generate more unique sets until reaching N
    while len(sets) < N:
        new_set = tuple(sorted(random.sample(mcqs, quiz_size)))  # Sorting helps prevent ordering issues
        sets.add(new_set)  # Add only if unique

    # Convert back to list of sets
    sets = list(sets)
    # Debugging: Check for NA values
    for s in sets:
        if any(v is None for v in s):  # Detect NA values
            print("Warning: Found NA value in set:", s)

    print(f"Generated {len(sets)} unique sets.")

    # Create a DataFrame from the quizzes generated
    quizzes_df = pd.DataFrame(sets, columns=[f'MCQ_{i+1}' for i in range(quiz_size)])
    # Change type of columns to int
    for i in range(quiz_size):
        quizzes_df[f'MCQ_{i+1}'] = quizzes_df[f'MCQ_{i+1}'].astype(int)

    # Get sorted unique topics for a fixed-size vector
    unique_topics = sorted(mcqs_df['topic'].unique())
    topic_index = {topic: i for i, topic in enumerate(unique_topics)}  # Map topic to index
    # Compute topic coverage for each row in quizzes_df
    quizzes_df['topic_coverage'] = quizzes_df.apply(lambda row: compute_topic_coverage(row.tolist()), axis=1)
    quizzes_df['difficulty_coverage'] = quizzes_df.apply(lambda row: compute_difficulty_distribution(row.tolist()), axis=1)
    quizzes_df.to_csv(f"{output_name}.csv", index=False)
    return quizzes_df

def generate_mcqs(output_name, file_path="../data/mcqs.csv", num_topics=10, test_num="test10"):
    """Generate MCQs with topic and difficulty information."""
    # One-hot encoding for difficulty (values 1-5)
    def one_hot_difficulty(difficulty):
        vec = np.zeros(5)
        if 1 <= difficulty <= 5:
            vec[int(difficulty) - 1] = 1
        return vec

    try:
        mcqs_df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

    # Rename the id column to id
    mcqs_df.rename(columns={'id': 'mcq_id'}, inplace=True)
    mcqs_df['topic'] = mcqs_df['mcq_id'].str.extract(r'OIC-(\d+)-\d+-[A-Z]')
    distinct_count = mcqs_df['topic'].nunique()

    print("Number of distinct topics values:", distinct_count)

    mcqs_df['id'] = mcqs_df.index
    mcqs_df['difficulty'] = mcqs_df['difficulty'].apply(one_hot_difficulty)
    mcqs_df = mcqs_df[['id', 'mcq_id', 'topic', 'question', 'option_a', 'option_b', 'option_c', 'option_d', 'correct_option', 'difficulty']]
    
    topics = mcqs_df['topic'].unique()
    # Create a dictionary with the topics and their indexes
    topics = random.sample(list(topics), num_topics)
    mcqs_df = mcqs_df[mcqs_df['topic'].isin(topics)].reset_index(drop=True)
    save_to_log(f"Topics selected: {topics}, Number of unique MCQs: {mcqs_df.shape[0]}", f'../logs/{test_num}/training')
    # Save the dataframe to a csv file
    mcqs_df.to_csv(f"{output_name}.csv", index=False)

    return mcqs_df

class BaseDataGenerator(ABC):
    """Base class for ../data generators."""
    
    @abstractmethod
    def generate_universe(self, num_states, num_topics, num_difficulties):
        """Generate a universe of states."""
        pass
    
    @abstractmethod
    def generate_targets(self, num_topics, num_difficulties):
        """Generate target vectors."""
        pass

class UniformDataGenerator(BaseDataGenerator):
    """Generator for completely random synthetic ../data using Dirichlet distribution."""
    
    def generate_universe(self, num_states, num_topics, num_difficulties):
        """Generate a synthetic universe with random topic and difficulty distributions."""
        # Generate random topic distributions
        topic_distributions = np.random.dirichlet(np.ones(num_topics), size=num_states)
        
        # Generate random difficulty distributions
        difficulty_distributions = np.random.dirichlet(np.ones(num_difficulties), size=num_states)
        
        # Combine into universe
        universe = np.concatenate([topic_distributions, difficulty_distributions], axis=1)
        return universe
    
    def generate_targets(self, num_topics, num_difficulties):
        """Generate synthetic targets using Dirichlet distribution."""
        target1 = np.random.dirichlet(np.ones(num_topics))
        target2 = np.random.dirichlet(np.ones(num_difficulties))
        return [target1, target2]

class TopicFocusedGenerator(BaseDataGenerator):
    """Generator where topic distributions are very similar to each other."""
    
    def generate_universe(self, num_states, num_topics, num_difficulties):
        """Generate a universe with similar topic distributions."""
        # Generate a base topic distribution
        base_topic = np.random.dirichlet(np.ones(num_topics))
        # Add small noise to create similar distributions
        topic_distributions = np.tile(base_topic, (num_states, 1)) + np.random.normal(0, 0.05, (num_states, num_topics))
        topic_distributions = np.clip(topic_distributions, 0.01, 1)  # Ensure no zeros
        topic_distributions /= topic_distributions.sum(axis=1, keepdims=True)
        
        # Handle any remaining NaN values
        nan_mask = np.isnan(topic_distributions)
        if np.any(nan_mask):
            topic_distributions[nan_mask] = 1.0 / num_topics
        
        # Generate random difficulty distributions
        difficulty_distributions = np.random.dirichlet(np.ones(num_difficulties), size=num_states)
        
        universe = np.concatenate([topic_distributions, difficulty_distributions], axis=1)
        return universe
    
    def generate_targets(self, num_topics, num_difficulties):
        """Generate targets close to the base topic distribution."""
        target1 = np.random.dirichlet(np.ones(num_topics))  # Similar to universe topics
        target2 = np.random.dirichlet(np.ones(num_difficulties))
        return [target1, target2]

class DifficultyFocusedGenerator(BaseDataGenerator):
    """Generator where difficulty distributions are very similar to each other."""
    
    def generate_universe(self, num_states, num_topics, num_difficulties):
        """Generate a universe with similar difficulty distributions."""
        # Generate random topic distributions
        topic_distributions = np.random.dirichlet(np.ones(num_topics), size=num_states)
        
        # Generate a base difficulty distribution
        base_difficulty = np.random.dirichlet(np.ones(num_difficulties))
        # Add small noise to create similar distributions
        difficulty_distributions = np.tile(base_difficulty, (num_states, 1)) + np.random.normal(0, 0.05, (num_states, num_difficulties))
        difficulty_distributions = np.clip(difficulty_distributions, 0.01, 1)  # Ensure no zeros
        difficulty_distributions /= difficulty_distributions.sum(axis=1, keepdims=True)
        
        # Handle any remaining NaN values
        nan_mask = np.isnan(difficulty_distributions)
        if np.any(nan_mask):
            difficulty_distributions[nan_mask] = 1.0 / num_difficulties
        
        universe = np.concatenate([topic_distributions, difficulty_distributions], axis=1)
        return universe
    
    def generate_targets(self, num_topics, num_difficulties):
        """Generate targets close to the base difficulty distribution."""
        target1 = np.random.dirichlet(np.ones(num_topics))
        target2 = np.random.dirichlet(np.ones(num_difficulties))  # Similar to universe difficulties
        return [target1, target2]

class TopicDiverseGenerator(BaseDataGenerator):
    """Generator where topic distributions are very different from each other."""
    
    def generate_universe(self, num_states, num_topics, num_difficulties):
        """Generate a universe with diverse topic distributions."""
        # Split states into groups with different topic patterns
        group_size = num_states // 3
        remaining = num_states - 2 * group_size
        
        # Group 1: High focus on first few topics
        topics1 = np.random.dirichlet(np.ones(num_topics) * 0.5, size=group_size)
        topics1 = np.sort(topics1, axis=1)
        
        # Group 2: High focus on last few topics
        topics2 = np.random.dirichlet(np.ones(num_topics) * 0.5, size=group_size)
        topics2 = -np.sort(-topics2, axis=1)
        
        # Group 3: Mixed patterns
        topics3 = np.random.dirichlet(np.ones(num_topics) * 0.5, size=remaining)
        topics3 = np.roll(topics3, shift=np.random.randint(0, num_topics, size=remaining), axis=1)
        
        # Combine and shuffle
        topic_distributions = np.concatenate([topics1, topics2, topics3], axis=0)
        np.random.shuffle(topic_distributions)
        
        # Generate random difficulty distributions
        difficulty_distributions = np.random.dirichlet(np.ones(num_difficulties), size=num_states)
        
        universe = np.concatenate([topic_distributions, difficulty_distributions], axis=1)
        return universe
    
    def generate_targets(self, num_topics, num_difficulties):
        """Generate targets that match one of the diverse patterns."""
        target1 = np.random.dirichlet(np.ones(num_topics) * 0.5)
        if np.random.random() > 0.5:
            target1 = np.sort(target1)
        else:
            target1 = -np.sort(-target1)
        target2 = np.random.dirichlet(np.ones(num_difficulties))
        return [target1, target2]

class TopicDifficultyCorrelatedGenerator(BaseDataGenerator):
    """Generator where topic and difficulty distributions are correlated."""
    
    def generate_universe(self, num_states, num_topics, num_difficulties):
        """Generate a universe with correlated topic and difficulty distributions."""
        # Generate topic distributions
        topic_distributions = np.random.dirichlet(np.ones(num_topics), size=num_states)
        
        # Generate correlated difficulty distributions
        # Higher topic values lead to higher difficulty values
        difficulty_distributions = topic_distributions[:, :num_difficulties] + np.random.normal(0, 0.1, (num_states, num_difficulties))
        difficulty_distributions = np.clip(difficulty_distributions, 0.01, 1)  # Ensure no zeros
        difficulty_distributions /= difficulty_distributions.sum(axis=1, keepdims=True)
        
        # Handle any remaining NaN values
        nan_mask = np.isnan(difficulty_distributions)
        if np.any(nan_mask):
            difficulty_distributions[nan_mask] = 1.0 / num_difficulties
        
        universe = np.concatenate([topic_distributions, difficulty_distributions], axis=1)
        return universe
    
    def generate_targets(self, num_topics, num_difficulties):
        """Generate correlated targets."""
        target1 = np.random.dirichlet(np.ones(num_topics))
        target2 = target1[:num_difficulties] + np.random.normal(0, 0.1, num_difficulties)
        target2 = np.clip(target2, 0.01, 1)  # Ensure no zeros
        target2 /= target2.sum()
        
        # Handle any remaining NaN values
        if np.any(np.isnan(target2)):
            target2 = np.ones(num_difficulties) / num_difficulties
            
        return [target1, target2]

class DifficultyDiverseGenerator(BaseDataGenerator):
    """Generator where difficulty distributions are very different from each other."""
    
    def generate_universe(self, num_states, num_topics, num_difficulties):
        """Generate a universe with diverse difficulty distributions."""
        # Generate random topic distributions
        topic_distributions = np.random.dirichlet(np.ones(num_topics), size=num_states)
        
        # Split states into groups with different difficulty patterns
        group_size = num_states // 3
        remaining = num_states - 2 * group_size
        
        # Group 1: High focus on first few difficulties
        difficulties1 = np.random.dirichlet(np.ones(num_difficulties) * 0.5, size=group_size)
        difficulties1 = np.sort(difficulties1, axis=1)
        
        # Group 2: High focus on last few difficulties
        difficulties2 = np.random.dirichlet(np.ones(num_difficulties) * 0.5, size=group_size)
        difficulties2 = -np.sort(-difficulties2, axis=1)
        
        # Group 3: Mixed patterns
        difficulties3 = np.random.dirichlet(np.ones(num_difficulties) * 0.5, size=remaining)
        difficulties3 = np.roll(difficulties3, shift=np.random.randint(0, num_difficulties, size=remaining), axis=1)
        
        # Combine and shuffle
        difficulty_distributions = np.concatenate([difficulties1, difficulties2, difficulties3], axis=0)
        np.random.shuffle(difficulty_distributions)
        
        # Ensure no zeros and handle NaN values
        difficulty_distributions = np.clip(difficulty_distributions, 0.01, 1)
        difficulty_distributions /= difficulty_distributions.sum(axis=1, keepdims=True)
        nan_mask = np.isnan(difficulty_distributions)
        if np.any(nan_mask):
            difficulty_distributions[nan_mask] = 1.0 / num_difficulties
        
        universe = np.concatenate([topic_distributions, difficulty_distributions], axis=1)
        return universe
    
    def generate_targets(self, num_topics, num_difficulties):
        """Generate targets that match one of the diverse difficulty patterns."""
        target1 = np.random.dirichlet(np.ones(num_topics))
        target2 = np.random.dirichlet(np.ones(num_difficulties) * 0.5)
        if np.random.random() > 0.5:
            target2 = np.sort(target2)
        else:
            target2 = -np.sort(-target2)
        
        # Ensure no zeros and handle NaN values
        target2 = np.clip(target2, 0.01, 1)
        target2 /= target2.sum()
        if np.any(np.isnan(target2)):
            target2 = np.ones(num_difficulties) / num_difficulties
            
        return [target1, target2]

class RealDataGenerator(BaseDataGenerator):
    """Generator for ../data based on real MCQ ../data."""
    
    def generate_universe(self, test_num, num_states, num_topics, num_difficulties, quiz_size=10):
        """Generate universe from real MCQ ../data."""
        # Create ../data directory if it doesn't exist
        os.makedirs(f"../data/{test_num}", exist_ok=True)
        os.makedirs(f"../jsons/{test_num}/universes", exist_ok=True)
        
        # Generate MCQs
        mcqs_df = generate_mcqs(f"../data/{test_num}/test_mcqs", file_path="../data/mcqs.csv", num_topics=num_topics, test_num=test_num)
        if mcqs_df is None:
            raise ValueError("Failed to generate MCQs")
        
        # Generate quizzes
        quizzes_df = generate_quizzes(mcqs_df, f"../data/{test_num}/test_quizzes", universe_size=num_states, quiz_size=quiz_size)
        
        # Create universe from quizzes
        universe = quizzes_df[['topic_coverage', 'difficulty_coverage']].values
        universe = np.array([flatten_inhomogeneous_array(row) for row in universe], dtype=np.float32)
        return universe
    
    def generate_targets(self, num_topics, num_difficulties):
        """Generate targets based on real ../data distributions."""
        # Load or generate targets based on real ../data
        target1 = np.random.uniform(0, 1, size=num_topics)
        target1 = target1 / np.sum(target1)  # Normalize
        target2 = np.random.uniform(0, 1, size=num_difficulties)
        target2 = target2 / np.sum(target2)  # Normalize
        return [target1, target2]

class UniformTargetGenerator:
    """Generator for uniform target distributions."""
    
    @staticmethod
    def generate_targets(num_topics, num_difficulties):
        """Generate uniform target distributions for both topic and difficulty."""
        target1 = np.ones(num_topics) / num_topics  # Uniform topic distribution
        target2 = np.ones(num_difficulties) / num_difficulties  # Uniform difficulty distribution
        return [target1, target2]

class SparseTopicTargetGenerator:
    """Generator for sparse topic and uniform difficulty distributions."""
    
    @staticmethod
    def generate_targets(num_topics, num_difficulties):
        """Generate sparse topic distribution and uniform difficulty distribution."""
        # Create sparse topic distribution (most weight on a few topics)
        target1 = np.zeros(num_topics)
        selected_topics = np.random.choice(num_topics, size=max(2, num_topics // 4), replace=False)
        target1[selected_topics] = 1
        target1 = target1 / np.sum(target1)  # Normalize
        
        # Create uniform difficulty distribution
        target2 = np.ones(num_difficulties) / num_difficulties
        return [target1, target2]

class SparseDifficultyTargetGenerator:
    """Generator for uniform topic and sparse difficulty distributions."""
    
    @staticmethod
    def generate_targets(num_topics, num_difficulties):
        """Generate uniform topic distribution and sparse difficulty distribution."""
        # Create uniform topic distribution
        target1 = np.ones(num_topics) / num_topics
        
        # Create sparse difficulty distribution (most weight on a few difficulties)
        target2 = np.zeros(num_difficulties)
        selected_difficulties = np.random.choice(num_difficulties, size=max(1, num_difficulties // 2), replace=False)
        target2[selected_difficulties] = 1
        target2 = target2 / np.sum(target2)  # Normalize
        return [target1, target2]

def generate_targets_with_distribution(distribution_type, num_topics, num_difficulties):
    """Generate targets based on the specified distribution type."""
    generators = {
        'uniform': UniformTargetGenerator,
        'sparse_topic': SparseTopicTargetGenerator,
        'sparse_difficulty': SparseDifficultyTargetGenerator
    }
    
    generator = generators.get(distribution_type)
    if generator is None:
        raise ValueError(f"Unknown distribution type: {distribution_type}")
    
    return generator.generate_targets(num_topics, num_difficulties)

def generate_synthetic_universe(num_states=10000, num_topics=10, num_difficulties=5, generator_type='uniform'):
    """Generate a synthetic universe with specified generator type."""
    if generator_type == 'uniform':
        generator = UniformDataGenerator()
    elif generator_type == 'topic_focused':
        generator = TopicFocusedGenerator()
    elif generator_type == 'difficulty_focused':
        generator = DifficultyFocusedGenerator()
    elif generator_type == 'topic_diverse':
        generator = TopicDiverseGenerator()
    elif generator_type == 'difficulty_diverse':
        generator = DifficultyDiverseGenerator()
    elif generator_type == 'correlated':
        generator = TopicDifficultyCorrelatedGenerator()
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")
    
    universe = generator.generate_universe(num_states, num_topics, num_difficulties)
    targets = generator.generate_targets(num_topics, num_difficulties)
    
    return universe, targets

def generate_universe_from_real_data(test_num, num_topics=10, universe_size=10000, quiz_size=10):
    """Generate universe from real MCQ ../data."""
    generator = RealDataGenerator()
    universe = generator.generate_universe(test_num, universe_size, num_topics, 5, quiz_size)
    targets = generator.generate_targets(num_topics, 5)
    return universe, targets

def save_universe(universe, targets, test_num):
    """Save universe and targets to JSON files."""
    with open(f'../jsons/{test_num}/universes/universe.json', 'w') as f:
        json.dump(universe.tolist(), f)
    
    with open(f'../jsons/{test_num}/universes/targets.json', 'w') as f:
        json.dump([targets[0].tolist(), targets[1].tolist()], f)

def load_universe(test_num):
    """Load universe and targets from JSON files."""
    with open(f"../jsons/{test_num}/universes/universe.json", "r") as f:
        universe = np.array(json.load(f), dtype=np.float32)
    
    with open(f"../jsons/{test_num}/universes/targets.json", "r") as f:
        targets = [np.array(t, dtype=np.float32) for t in json.load(f)]
    
    return universe, targets 