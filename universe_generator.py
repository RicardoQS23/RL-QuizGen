import os
import json
import random
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from utils.logging import save_to_log

def generate_mcqs(test_num, csv_file, num_topics, topic_column='topic', difficulty_column='difficulty', file_separator=','):
    """
    Generate MCQs from a CSV file.
    
    Args:
        test_num (str): Test number for saving results
        csv_file (str): Path to the CSV file
        num_topics (int): Number of topics to use
        topic_column (str): Name of the topic column
        difficulty_column (str): Name of the difficulty column
        file_separator (str): CSV file separator
    
    Returns:
        tuple: (list of MCQs in the format required by RealDataGenerator, num_topics)
    """
    # Try different encodings
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(csv_file, sep=file_separator, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    
    if df is None:
        raise ValueError(f"Could not read file {csv_file} with any of the tried encodings: {encodings}")
    
    df.rename(columns={topic_column: 'topic', difficulty_column: 'difficulty'}, inplace=True)

    # For medical dataset, extract topic from mcq_id
    if topic_column != 'topic_id':
        df.rename(columns={'id': 'mcq_id'}, inplace=True)
        df['topic'] = df['mcq_id'].str.extract(r'OIC-(\d+)-\d+-[A-Z]')
        #transform topic to int
        df['topic'] = df['topic'].astype(int)
    else:
        df.rename(columns={'id': 'mcq_id'}, inplace=True)
        df.rename(columns={'difficulty_level': 'difficulty'}, inplace=True)
        df['option_a'] = df['correct_answer']
        df.rename(columns={'correct_answer': 'correct_option', 'answer2': 'option_b', 'answer3': 'option_c', 'answer4': 'option_d'}, inplace=True)
        df['topic'] = df['topic'].astype(int)

    df['id'] = df.index
    df['difficulty'] = df['difficulty']
    
    # Get unique topics
    unique_topics = df['topic'].unique()
    
    # Select topics
    if num_topics > len(unique_topics):
        save_to_log(f"Warning: Requested {num_topics} topics but only {len(unique_topics)} available. Using all available topics.", 'training')
        num_topics = len(unique_topics)
    
    selected_topics = np.random.choice(unique_topics, num_topics, replace=False)
    save_to_log(f"Selected topics: {selected_topics}", f'../logs/{test_num}/training')
    
    #filter mcqs with difficulty 6

    # Filter MCQs by selected topics
    mcqs = df[df['topic'].isin(selected_topics)]

    mcqs = mcqs[mcqs['difficulty'] != 6]
    #mcqs.to_csv(f"../data/{test_num}/mcqs_{csv_file.split('/')[-1]}", index=False)

    save_to_log(f"Generated {len(mcqs)} unique MCQs", f'../logs/{test_num}/training')
    # Convert to list of dictionaries in the format required by RealDataGenerator
    mcqs_list = []
    for _, row in mcqs.iterrows():
        mcq_dict = {
            'id': row['id'],
            'topic': row['topic'],
            'difficulty': row['difficulty']
        }
        mcqs_list.append(mcq_dict)
    
    return mcqs_list, num_topics

class BaseDataGenerator(ABC):
    """Base class for ../data generators."""
    
    @abstractmethod
    def generate_universe(self, universe_size, num_topics, num_difficulties):
        """Generate a universe of states."""
        pass


class UniformDataGenerator(BaseDataGenerator):
    """Generator for completely random synthetic ../data using Dirichlet distribution."""
    
    def generate_universe(self, universe_size, num_topics, num_difficulties):
        """Generate a synthetic universe with random topic and difficulty distributions."""
        # Generate random topic distributions
        topic_distributions = np.random.dirichlet(np.ones(num_topics), size=universe_size)
        
        # Generate random difficulty distributions
        difficulty_distributions = np.random.dirichlet(np.ones(num_difficulties), size=universe_size)
        
        # Combine into universe
        universe = np.concatenate([topic_distributions, difficulty_distributions], axis=1)
        return universe
    
class TopicFocusedGenerator(BaseDataGenerator):
    """Generator where topic distributions are very similar to each other."""
    
    def generate_universe(self, universe_size, num_topics, num_difficulties):
        """Generate a universe with similar topic distributions."""
        # Generate a base topic distribution
        base_topic = np.random.dirichlet(np.ones(num_topics))
        # Add small noise to create similar distributions
        topic_distributions = np.tile(base_topic, (universe_size, 1)) + np.random.normal(0, 0.05, (universe_size, num_topics))
        topic_distributions = np.clip(topic_distributions, 0.01, 1)  # Ensure no zeros
        topic_distributions /= topic_distributions.sum(axis=1, keepdims=True)
        
        # Handle any remaining NaN values
        nan_mask = np.isnan(topic_distributions)
        if np.any(nan_mask):
            topic_distributions[nan_mask] = 1.0 / num_topics
        
        # Generate random difficulty distributions
        difficulty_distributions = np.random.dirichlet(np.ones(num_difficulties), size=universe_size)
        
        universe = np.concatenate([topic_distributions, difficulty_distributions], axis=1)
        return universe
    

class DifficultyFocusedGenerator(BaseDataGenerator):
    """Generator where difficulty distributions are very similar to each other."""
    
    def generate_universe(self, universe_size, num_topics, num_difficulties):
        """Generate a universe with similar difficulty distributions."""
        # Generate random topic distributions
        topic_distributions = np.random.dirichlet(np.ones(num_topics), size=universe_size)
        
        # Generate a base difficulty distribution
        base_difficulty = np.random.dirichlet(np.ones(num_difficulties))
        # Add small noise to create similar distributions
        difficulty_distributions = np.tile(base_difficulty, (universe_size, 1)) + np.random.normal(0, 0.05, (universe_size, num_difficulties))
        difficulty_distributions = np.clip(difficulty_distributions, 0.01, 1)  # Ensure no zeros
        difficulty_distributions /= difficulty_distributions.sum(axis=1, keepdims=True)
        
        # Handle any remaining NaN values
        nan_mask = np.isnan(difficulty_distributions)
        if np.any(nan_mask):
            difficulty_distributions[nan_mask] = 1.0 / num_difficulties
        
        universe = np.concatenate([topic_distributions, difficulty_distributions], axis=1)
        return universe
    

class TopicDiverseGenerator(BaseDataGenerator):
    """Generator where topic distributions are very different from each other."""
    
    def generate_universe(self, universe_size, num_topics, num_difficulties):
        """Generate a universe with diverse topic distributions."""
        # Split states into groups with different topic patterns
        group_size = universe_size // 3
        remaining = universe_size - 2 * group_size
        
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
        difficulty_distributions = np.random.dirichlet(np.ones(num_difficulties), size=universe_size)
        
        universe = np.concatenate([topic_distributions, difficulty_distributions], axis=1)
        return universe
    

class TopicDifficultyCorrelatedGenerator(BaseDataGenerator):
    """Generator where topic and difficulty distributions are correlated."""
    
    def generate_universe(self, universe_size, num_topics, num_difficulties):
        """Generate a universe with correlated topic and difficulty distributions."""
        # Generate topic distributions
        topic_distributions = np.random.dirichlet(np.ones(num_topics), size=universe_size)
        
        # Generate correlated difficulty distributions
        # Higher topic values lead to higher difficulty values
        difficulty_distributions = topic_distributions[:, :num_difficulties] + np.random.normal(0, 0.1, (universe_size, num_difficulties))
        difficulty_distributions = np.clip(difficulty_distributions, 0.01, 1)  # Ensure no zeros
        difficulty_distributions /= difficulty_distributions.sum(axis=1, keepdims=True)
        
        # Handle any remaining NaN values
        nan_mask = np.isnan(difficulty_distributions)
        if np.any(nan_mask):
            difficulty_distributions[nan_mask] = 1.0 / num_difficulties
        
        universe = np.concatenate([topic_distributions, difficulty_distributions], axis=1)
        return universe

class DifficultyDiverseGenerator(BaseDataGenerator):
    """Generator where difficulty distributions are very different from each other."""
    
    def generate_universe(self, universe_size, num_topics, num_difficulties):
        """Generate a universe with diverse difficulty distributions."""
        # Generate random topic distributions
        topic_distributions = np.random.dirichlet(np.ones(num_topics), size=universe_size)
        
        # Split states into groups with different difficulty patterns
        group_size = universe_size // 3
        remaining = universe_size - 2 * group_size
        
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
    
class RealDataGenerator(BaseDataGenerator):
    """
    Generator for real MCQ data.
    
    Args:
        mcqs (list): List of MCQs
        num_topics (int): Number of topics
        num_difficulties (int): Number of difficulty levels
        quiz_size (int): Number of MCQs per quiz
    """
    def __init__(self, mcqs, num_topics, num_difficulties, quiz_size):
        self.mcqs = mcqs
        self.num_topics = num_topics
        self.num_difficulties = num_difficulties
        self.quiz_size = quiz_size
        
        # Create topic and difficulty mappings
        self.topic_to_idx = {topic: i for i, topic in enumerate(sorted(set(m['topic'] for m in mcqs)))}
        self.difficulty_to_idx = {i: i for i in range(num_difficulties)}
        
        # Group MCQs by topic and difficulty
        self.mcqs_by_topic_diff = {}
        for mcq in mcqs:
            topic = mcq['topic']
            difficulty = int(mcq['difficulty']) - 1  # Convert 1-based to 0-based
            if topic not in self.mcqs_by_topic_diff:
                self.mcqs_by_topic_diff[topic] = {}
            if difficulty not in self.mcqs_by_topic_diff[topic]:
                self.mcqs_by_topic_diff[topic][difficulty] = []
            self.mcqs_by_topic_diff[topic][difficulty].append(mcq)
    
    def generate_universe(self, size, test_num):
        """
        Generate a universe of quizzes.
        
        Args:
            size (int): Size of the universe
            
        Returns:
            list: List of quizzes
        """
        universe = []
        seen_quizzes = set()  # Track unique quizzes
        
        max_attempts = 1000  # Maximum attempts to generate a unique quiz
        attempts = 0
        
        while len(universe) < size and attempts < max_attempts:
            quiz = self._generate_quiz()
            # Create a unique key for the quiz by sorting MCQ IDs
            quiz_key = tuple(sorted(mcq['id'] for mcq in quiz))
            
            if quiz_key not in seen_quizzes:
                universe.append(quiz)
                seen_quizzes.add(quiz_key)
                attempts = 0  # Reset attempts counter on success
            else:
                attempts += 1
        
        if len(universe) < size:
            save_to_log(f"Warning: Could only generate {len(universe)} unique quizzes out of requested {size}", f'../logs/{test_num}/training')
        
        return universe
    
    def _generate_quiz(self):
        """
        Generate a single quiz with diverse topic distribution.
        Returns:
            list: List of MCQs in the quiz
        """
        quiz = []
        topics = list(self.mcqs_by_topic_diff.keys())
        # Randomly sample topics for the quiz (with replacement)
        sampled_topics = np.random.choice(topics, size=self.num_topics, replace=False)
        # Distribute remaining MCQs randomly across sampled topics
        while len(quiz) < self.quiz_size:
            topic = np.random.choice(sampled_topics)
            difficulties = list(self.mcqs_by_topic_diff[topic].keys())
            difficulty = np.random.choice(difficulties)
            mcq = np.random.choice(self.mcqs_by_topic_diff[topic][difficulty])
            quiz.append(mcq)
        return quiz

class RealTargetGenerator:
    """Generator for target distributions from real MCQ data."""
    
    def __init__(self, mcqs, num_topics, num_difficulties):
        self.mcqs = mcqs
        self.num_topics = num_topics
        self.num_difficulties = num_difficulties
        
        # Create topic and difficulty mappings
        self.topic_to_idx = {topic: i for i, topic in enumerate(sorted(set(m['topic'] for m in mcqs)))}
        self.difficulty_to_idx = {i: i for i in range(num_difficulties)}
    
    def generate_targets(self):
        """
        Generate target distributions.
        
        Returns:
            dict: Target distributions
        """
        targets = {
            'topic': np.zeros(self.num_topics),
            'difficulty': np.zeros(self.num_difficulties)
        }
        
        # Calculate topic distribution
        topic_counts = {}
        for mcq in self.mcqs:
            topic = mcq['topic']
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        total_mcqs = len(self.mcqs)
        for topic, count in topic_counts.items():
            targets['topic'][self.topic_to_idx[topic]] = count / total_mcqs
        
        # Calculate difficulty distribution
        difficulty_counts = np.zeros(self.num_difficulties)
        for mcq in self.mcqs:
            difficulty = int(mcq['difficulty']) - 1
            difficulty_counts[difficulty] += 1
        
        targets['difficulty'] = difficulty_counts / total_mcqs
        
        return targets

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
    
class SparseTopicDifficultyTargetGenerator:
    """Generator for sparse topic and sparse difficulty distributions."""
    
    @staticmethod
    def generate_targets(num_topics, num_difficulties):
        """Generate sparse topic and sparse difficulty distributions."""
        # Create sparse topic distribution (most weight on a few topics)
        target1 = np.zeros(num_topics)
        selected_topics = np.random.choice(num_topics, size=max(2, num_topics // 4), replace=False)
        target1[selected_topics] = 1
        target1 = target1 / np.sum(target1)  # Normalize
        
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
        'sparse_difficulty': SparseDifficultyTargetGenerator,
        'sparse_topic_difficulty': SparseTopicDifficultyTargetGenerator
    }
    
    generator = generators.get(distribution_type)
    if generator is None:
        raise ValueError(f"Unknown distribution type: {distribution_type}")
    
    return generator.generate_targets(num_topics, num_difficulties)

def generate_synthetic_universe(test_num, num_topics, universe_size, num_difficulties=5, universe_distribution='uniform', target_distribution='uniform'):
    """Generate a synthetic universe with specified generator type."""
    if universe_distribution == 'uniform':
        generator = UniformDataGenerator()
    elif universe_distribution == 'topic_focused':
        generator = TopicFocusedGenerator()
    elif universe_distribution == 'difficulty_focused':
        generator = DifficultyFocusedGenerator()
    elif universe_distribution == 'topic_diverse':
        generator = TopicDiverseGenerator()
    elif universe_distribution == 'difficulty_diverse':
        generator = DifficultyDiverseGenerator()
    elif universe_distribution == 'correlated':
        generator = TopicDifficultyCorrelatedGenerator()
    else:
        raise ValueError(f"Unknown generator type: {universe_distribution}")
    
    universe = generator.generate_universe(universe_size, num_topics, num_difficulties)
    targets = generate_targets_with_distribution(target_distribution, num_topics, num_difficulties)
    save_universe(universe, targets, test_num)
    
    return universe, targets

#def generate_universe_from_real_data(test_num, num_topics, universe_size, quiz_size, dataset='medical', target_distribution='uniform'):
#    """
#    Generate a universe from real MCQ data.
#    
#    Args:
#        test_num (str): Test number for saving results
#        num_topics (int): Number of topics to use
#        universe_size (int): Size of the universe
#        quiz_size (int): Number of MCQs per quiz
#        dataset (str): Dataset to use ('medical' or 'math')
#        target_distribution (str): Type of target distribution to use ('uniform', 'sparse_topic', 'sparse_difficulty')
#    
#    Returns:
#        tuple: (universe, targets)
#    """
#    # Set parameters based on dataset
#    num_difficulties = 6 if dataset == 'math' else 5
#    sep = ';' if dataset == 'math' else ','
#    topic_column = 'topic_id' if dataset == 'math' else 'topic'
#    difficulty_column = 'Level' if dataset == 'math' else 'difficulty'
#
#    mcqs, num_topics = generate_mcqs(test_num, f'../data/{dataset}.csv', num_topics, topic_column, difficulty_column, sep)
#    # Create universe generator
#    universe_generator = RealDataGenerator(mcqs, num_topics, num_difficulties, quiz_size)
#    
#    # Generate universe
#    universe = universe_generator.generate_universe(universe_size, test_num)
    
#    # Convert universe to array of arrays with size (num_topics + num_difficulties)
#    universe_array = []
#    for quiz in universe:
#        # Calculate topic distribution
#        topic_dist = np.zeros(num_topics)
#        difficulty_dist = np.zeros(num_difficulties)
#        
#        for mcq in quiz:
#            topic = mcq['topic']
#            difficulty = mcq['difficulty'] - 1  # Convert to 0-based
#            topic_dist[universe_generator.topic_to_idx[topic]] += 1
#            difficulty_dist[difficulty] += 1
#        
#        # Normalize distributions
#        topic_dist = topic_dist / len(quiz)
#        difficulty_dist = difficulty_dist / len(quiz)
#        
#        # Combine into single array
#        combined_dist = np.concatenate([topic_dist, difficulty_dist])
#        universe_array.append(combined_dist)
#    
#    universe_array = np.array(universe_array)
    
#    # Generate targets using the specified distribution
#    targets = generate_targets_with_distribution(target_distribution, num_topics, num_difficulties)
#    # Save universe and targets
#    save_universe(universe_array, targets, test_num)
    
#    return universe_array, targets, num_topics

def generate_universe_from_real_data(test_num, num_topics, universe_size, quiz_size, dataset='medical', target_distribution='uniform'):
    """
    Generate a universe from real MCQ data.
    
    Args:
        test_num (str): Test number for saving results
        num_topics (int): Number of topics to use
        universe_size (int): Size of the universe
        quiz_size (int): Number of MCQs per quiz
        dataset (str): Dataset to use ('medical' or 'math')
        target_distribution (str): Type of target distribution to use ('uniform', 'sparse_topic', 'sparse_difficulty')
    
    Returns:
        tuple: (universe, targets)
    """
    # Set parameters based on dataset
    num_difficulties = 6 if dataset == 'math' else 5
    sep = ';' if dataset == 'math' else ','
    topic_column = 'topic_id' if dataset == 'math' else 'topic'
    difficulty_column = 'Level' if dataset == 'math' else 'difficulty'

    mcqs, num_topics = generate_mcqs(test_num, f'../data/{dataset}.csv', num_topics, topic_column, difficulty_column, sep)
    # Create universe generator
    universe_generator = RealDataGenerator(mcqs, num_topics, num_difficulties, quiz_size)
    
    # Generate universe
    universe = universe_generator.generate_universe(universe_size, test_num)
    
    # Create lists to store data for DataFrame
    quiz_data = []
    universe_array = []
    
    for quiz_idx, quiz in enumerate(universe):
        # Calculate topic and difficulty distributions
        topic_dist = np.zeros(num_topics)
        difficulty_dist = np.zeros(num_difficulties)
        
        # Get MCQ IDs and calculate distributions
        mcq_ids = [mcq['id'] for mcq in quiz]
        for mcq in quiz:
            topic = mcq['topic']
            difficulty = mcq['difficulty'] - 1  # Convert to 0-based
            topic_dist[universe_generator.topic_to_idx[topic]] += 1
            difficulty_dist[difficulty] += 1
        
        # Normalize distributions
        topic_dist = topic_dist / len(quiz)
        difficulty_dist = difficulty_dist / len(quiz)
        
        # Create row data for DataFrame
        row_data = {
            'quiz_id': quiz_idx,
            **{f'mcq_{i+1}': mcq_id for i, mcq_id in enumerate(mcq_ids)},
            **{f'topic_coverage_{i}': cov for i, cov in enumerate(topic_dist)},
            **{f'difficulty_coverage_{i}': cov for i, cov in enumerate(difficulty_dist)}
        }
        quiz_data.append(row_data)
        
        # Add to universe array
        combined_dist = np.concatenate([topic_dist, difficulty_dist])
        universe_array.append(combined_dist)
    
    # Create and save DataFrame
    quizzes_df = pd.DataFrame(quiz_data)
    quizzes_df.to_csv(f'../data/{test_num}/quizzes_{dataset}.csv', index=False)
    
    universe_array = np.array(universe_array)
    
    # Generate targets using the specified distribution
    targets = generate_targets_with_distribution(target_distribution, num_topics, num_difficulties)
    # Save universe and targets
    save_universe(universe_array, targets, test_num)
    
    return universe_array, targets, num_topics

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