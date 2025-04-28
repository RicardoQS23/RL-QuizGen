import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from utils.utilities import load_data

def plot_universe(universe, targets, title, save_path=None):
    """Plot universe using both PCA and t-SNE."""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(title, fontsize=16)
    
    # Combine topic and difficulty targets into a single vector
    combined_target = np.concatenate(targets)
    
    # PCA visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(universe)
    
    # Plot PCA
    scatter = ax1.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5, s=10)
    # Plot target in PCA space
    target_pca = pca.transform(combined_target.reshape(1, -1))
    ax1.scatter(target_pca[:, 0], target_pca[:, 1], color='red', s=100, marker='*', label='Target')
    ax1.set_title('PCA Visualization')
    ax1.set_xlabel('Principal Component 1')
    ax1.set_ylabel('Principal Component 2')
    ax1.legend()
    
    # t-SNE visualization
    # Combine universe and target for t-SNE
    combined_data = np.vstack([universe, combined_target.reshape(1, -1)])
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(combined_data)
    
    # Plot t-SNE (excluding the last point which is the target)
    ax2.scatter(tsne_result[:-1, 0], tsne_result[:-1, 1], alpha=0.5, s=10)
    # Plot target in t-SNE space (last point)
    ax2.scatter(tsne_result[-1, 0], tsne_result[-1, 1], color='red', s=100, marker='*', label='Target')
    ax2.set_title('t-SNE Visualization')
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.legend()
    
    if save_path:
        plt.savefig(save_path)
    #plt.show()

def plot_universe_distributions(universe, title, save_path=None):
    """Plot distributions of topic and difficulty coverage."""
    # Split universe into topics and difficulties
    num_topics = 10  # Assuming 10 topics
    num_difficulties = 5  # Assuming 5 difficulty levels
    topic_dist = universe[:, :num_topics]
    difficulty_dist = universe[:, num_topics:]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(title, fontsize=16)
    
    # Plot topic distribution
    topic_means = np.mean(topic_dist, axis=0)
    topic_stds = np.std(topic_dist, axis=0)
    ax1.bar(range(num_topics), topic_means, yerr=topic_stds, capsize=5)
    ax1.set_title('Topic Distribution')
    ax1.set_xlabel('Topic')
    ax1.set_ylabel('Mean Coverage')
    ax1.set_xticks(range(num_topics))
    
    # Plot difficulty distribution
    difficulty_means = np.mean(difficulty_dist, axis=0)
    difficulty_stds = np.std(difficulty_dist, axis=0)
    ax2.bar(range(num_difficulties), difficulty_means, yerr=difficulty_stds, capsize=5)
    ax2.set_title('Difficulty Distribution')
    ax2.set_xlabel('Difficulty Level')
    ax2.set_ylabel('Mean Coverage')
    ax2.set_xticks(range(num_difficulties))
    
    if save_path:
        plt.savefig(save_path)
    #plt.show()

def plot_correlation(universe, title, save_path=None):
    """Plot correlation between topics and difficulties."""
    num_topics = 10
    num_difficulties = 5
    topic_dist = universe[:, :num_topics]
    difficulty_dist = universe[:, num_topics:]
    
    # Calculate correlation matrix
    correlation = np.zeros((num_topics, num_difficulties))
    for i in range(num_topics):
        for j in range(num_difficulties):
            correlation[i, j] = np.corrcoef(topic_dist[:, i], difficulty_dist[:, j])[0, 1]
    
    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0,
                xticklabels=[f'Diff {i+1}' for i in range(num_difficulties)],
                yticklabels=[f'Topic {i+1}' for i in range(num_topics)])
    plt.title(f'Topic-Difficulty Correlation\n{title}')
    
    if save_path:
        plt.savefig(save_path)
    #plt.show()

def get_labels(test_num):
    if test_num == "test1":
        return 'Uniform Distribution'
    elif test_num == "test2":
        return 'Similar Topic Distribution'
    elif test_num == "test3":
        return 'Similar Difficulty Distribution'
    elif test_num == "test4":
        return 'Different Topic Distribution'
    elif test_num == "test5":
        return 'Different Difficulty Distribution'
    elif test_num == "test6":
        return 'Topic-Difficulty Correlated Distribution'
    else:
        return 'Real Data Distribution'

def main():
    test_nums = ["test1", "test2", "test3", "test4", "test5", "test6", "test7", "test8", "test9", "test10", "test11"]
    # Generate and plot real data universe
    for test_num in test_nums:
        try:
            print(f"Loading Universe Data for {test_num}...")
            universe, targets = load_data(test_num)
            
            # Create directory for real data
            output_dir = f'../images/{test_num}'
            os.makedirs(output_dir, exist_ok=True)
            
            # Plot visualizations
            plot_universe(
                universe, targets,
                f'{get_labels(test_num)}',
                f'{output_dir}/universe.png'
            )
            
            plot_universe_distributions(
                universe,
                f'{get_labels(test_num)}',
                f'{output_dir}/distributions.png'
            )
            
            plot_correlation(
                universe,
                f'{get_labels(test_num)}',
                f'{output_dir}/correlation.png'
            )
        except Exception as e:
            print(f"Could not load universe data: {e}")

if __name__ == "__main__":
    main() 