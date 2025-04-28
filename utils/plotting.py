import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.utilities import load_data
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

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

def plot_reward_landscape(test_num, alfa_values, save=True, show=False):
    """Plot the reward landscape for different alfa values.
    
    Args:
        test_num (int): The test number
        alfa_values (list): List of alfa values to plot
        save (bool): Whether to save the plot
        show (bool): Whether to show the plot
    """
    # Load data
    universe, targets = load_data(f"test{test_num}")
    D = universe.shape[1]
    D1 = D - 5
    D2 = 5
    
    # Compute rewards for all universe states
    vec1s = universe[:, :D1]
    vec2s = universe[:, D1:D1+D2]
    
    sim1 = cosine_similarity(vec1s, targets[0].reshape(1, -1)).flatten()
    sim2 = cosine_similarity(vec2s, targets[1].reshape(1, -1)).flatten()
    
    # Set up the figure
    n_plots = len(alfa_values)
    n_cols = 2
    n_rows = (n_plots + 1) // 2
    fig = plt.figure(figsize=(15, 5*n_rows))
    gs = plt.GridSpec(n_rows, n_cols, figure=fig)
    
    # Initialize PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(universe)
    
    # Load baseline data
    with open(f"../jsons/test{test_num}/baseline_inference/baseline_states.json", "r") as f:
        baseline_states = json.load(f)
        baseline_states = np.array(baseline_states, dtype=np.int32)
    
    # Define colors for special points
    colors = ['#2077B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD']
    norm = plt.Normalize(0, 1)  # Normalize rewards to [0,1] range
    
    for j, alfa in enumerate(alfa_values):
        # Create subplot
        ax = fig.add_subplot(gs[j // n_cols, j % n_cols])
        
        # Compute rewards for current alfa
        rewards = alfa * sim1 + (1 - alfa) * sim2
        
        # Plot universe points with reward-based coloring
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], 
                           c=rewards, 
                           cmap='viridis',
                           norm=norm,
                           s=5,
                           alpha=0.8)
        
        # Plot baseline point
        baseline_state = baseline_states[j]
        baseline_coords = X_2d[baseline_state]
        ax.scatter(baseline_coords[0], baseline_coords[1], 
                  color=colors[3], 
                  marker="*", 
                  label='Baseline', 
                  s=150,
                  edgecolor='black',
                  linewidth=0.5)
        
        # Customize subplot
        ax.set_title(f"α = {alfa}", pad=10, fontsize=12)
        ax.set_xlabel("Topic Dimension", fontsize=10)
        ax.set_ylabel("Difficulty Dimension", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Add legend
        ax.legend(loc="upper right", 
                 frameon=True,
                 fancybox=True,
                 framealpha=0.8,
                 fontsize=9)
        # Add a single colorbar for all plots
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = plt.colorbar(scatter, cax=cbar_ax)
        cbar.set_label("Reward", fontsize=10)
        
    
    # Add main title
    plt.suptitle(f"Reward Landscape - {get_labels(f'test{test_num}')}", 
                y=1.02, 
                fontsize=14)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Make room for the colorbar
    
    # Save or show
    if save:
        plt.savefig(f"../images/test{test_num}/universe.png", 
                   dpi=300, 
                   bbox_inches='tight',
                   facecolor='white')
    if show:
        plt.show()
    plt.close() 

def project_to_1d(data, pca):
    """Project data to 1D using PCA."""
    return pca.transform(data.reshape(1, -1)).flatten()[0]

def plot_agent_comparison(universe, targets, alfa, alfa_values, test_number, output_name, save=True, show=False):
    """Create a single plot for a specific agent and alfa value."""
    # Parameters
    D = universe.shape[1]
    D1 = D - 5
    D2 = 5
    
    # Compute rewards for all universe states
    vec1s = universe[:, :D1]
    vec2s = universe[:, D1:D1+D2]
    
    sim1 = cosine_similarity(vec1s, targets[0].reshape(1, -1)).flatten()
    sim2 = cosine_similarity(vec2s, targets[1].reshape(1, -1)).flatten()
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define colors for special points
    colors = ["#0d6be3", "#d7129b", "#581845", "#d62728", "#9467bd", 
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    
    # Initialize PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(universe)
    
    # Compute rewards for current alfa
    rewards = alfa * sim1 + (1 - alfa) * sim2
    
    # Plot universe points with reward-based coloring
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=rewards, cmap='viridis', s=5)
    
    # Load and process agent data
    with open(f"../jsons/{test_number}/agent_inference/inference_states_alfa_{alfa}.json", "r") as f:
        visited_states = json.load(f)
        visited_states = np.array(visited_states, dtype=np.int32)
    visited_coords = X_2d[visited_states]
    
    # Load baseline data
    with open(f"../jsons/{test_number}/baseline_inference/baseline_states.json", "r") as f:
        baseline_states = json.load(f)
        baseline_states = np.array(baseline_states, dtype=np.int32)
    baseline_state = baseline_states[alfa_values.index(alfa)]
    baseline_coords = X_2d[baseline_state]
    
    # Plot baseline point
    ax.scatter(baseline_coords[0], baseline_coords[1], color=colors[2], 
              marker="*", label='Baseline', s=150)
    
    # Plot the agent's path with increasing intensity
    for idx in range(len(visited_coords) - 1):
        ax.plot(
            visited_coords[idx:idx+2, 0], 
            visited_coords[idx:idx+2, 1], 
            color=plt.cm.Reds(idx / len(visited_coords)), 
            linewidth=2, 
            alpha=0.8
        )
    
    # Plot start and end points
    ax.scatter(visited_coords[0, 0], visited_coords[0, 1], 
              color=colors[0], label='Start', s=100)
    ax.scatter(visited_coords[-1, 0], visited_coords[-1, 1], 
              color=colors[1], label='End', s=100)
    
    # Add grid and labels
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Alfa = {alfa}", pad=10)
    ax.set_ylabel("Difficulty Dim")
    ax.set_xlabel("Topic Dim")
    
    # Add legend
    ax.legend(loc="upper right", framealpha=0.8)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Reward")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show the figure
    if save:
        plt.savefig(f"../images/{output_name}_alfa_{alfa}.png", dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def plot_comparison(universe, targets, alfa_values, test_number, output_name, save=True, show=False):
    """Create comparison plots for different alfa values using PCA."""
    # Create a separate plot for each alfa value
    for alfa in alfa_values:
        plot_agent_comparison(
            universe=universe,
            targets=targets,
            alfa=alfa,
            alfa_values=alfa_values,
            test_number=test_number,
            output_name=output_name,
            save=save,
            show=show
        )

def moving_average(data, window_size=64):
    """Compute the moving average using a window."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_agent_data(input_path, alfa_values, y_label, title, output_name, window_size=64, flag=False, save=True, show=False):
    """Plot the agent_data with a moving average using seaborn for better visualization."""
    # Set the style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Create figure
    fig = plt.figure(figsize=(10, 6))
    
    # Use a clear and distinct color palette
    colors = ['#2077B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B']
    
    for idx, alfa in enumerate(alfa_values):
        with open(f'{input_path}_alfa_{alfa}.json', 'r') as f:
            agent_data = json.load(f)
            
        if flag:
            data_array = flatten_nested_array(agent_data)
        else:
            data_array = np.array(agent_data).flatten()
            
        # Calculate moving average
        smoothed_data = moving_average(data_array, window_size)
        x_values = np.arange(len(smoothed_data))
        
        # Plot with seaborn styling
        plt.plot(x_values, smoothed_data, 
                label=f'α = {alfa}',
                color=colors[idx % len(colors)],
                linewidth=1.5)

    # Customize plot
    plt.title(title, pad=20, fontsize=12)
    plt.ylabel(y_label, fontsize=10)
    
    # Set x-label based on the metric
    x_label = 'Iterations' if y_label in ['Q_value', 'Loss'] else 'Episodes'
    plt.xlabel(x_label, fontsize=10)
    
    # Customize legend
    plt.legend(frameon=True, 
              fancybox=True, 
              framealpha=0.8, 
              fontsize=9,
              loc='best')
    
    # Customize grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Customize ticks
    plt.tick_params(labelsize=8)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save and show
    if save:
        plt.savefig(f'../images/{output_name}.png', 
                    bbox_inches='tight',
                    dpi=300,
                    facecolor='white')
    if show:
        plt.show()
    plt.close()

def plot_action_distribution(input_path, alfa_values, title, output_name, window_size=32, save=True, show=False):
    """Plot the action distribution using seaborn for better visualization."""
    # Set the style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Create a figure with a more efficient layout
    n_plots = len(alfa_values)
    n_cols = 2
    n_rows = (n_plots + 1) // 2  # Ceiling division
    
    fig = plt.figure(figsize=(12, 3*n_rows))
    gs = plt.GridSpec(n_rows, n_cols, figure=fig)
    
    # Use a clear and distinct color palette for actions
    colors = ['#2077B4', '#FF7F0E', '#2CA02C', '#D62728']
    
    for j, alfa in enumerate(alfa_values):
        with open(f'{input_path}_alfa_{alfa}.json', 'r') as f:
            data = json.load(f)

        # Count occurrences of each action in every episode
        episodes = len(data)
        action_counts = {i: np.zeros(episodes) for i in range(4)}
        
        for i, episode in enumerate(data):
            if not isinstance(episode, list):
                continue
            for action in episode:
                action_counts[action][i] += 1

        # Normalize counts per episode
        normalized_values = np.array([action_counts[action] for action in range(4)])
        episode_sums = normalized_values.sum(axis=0)
        episode_sums[episode_sums == 0] = 1
        normalized_values /= episode_sums

        # Smooth the values with proper edge handling
        smoothed_values = []
        
        for action_vals in normalized_values:
            # Calculate moving average
            kernel_size = min(window_size, len(action_vals))
            kernel = np.ones(kernel_size) / kernel_size
            smoothed = np.convolve(action_vals, kernel, mode='same')
            # Ensure the smoothed values have the same length as episodes
            smoothed = smoothed[:episodes]
            smoothed_values.append(smoothed)
        
        smoothed_values = np.array(smoothed_values)

        # Create subplot
        ax = fig.add_subplot(gs[j // n_cols, j % n_cols])
        
        # Plot stacked area with no edges between areas
        ax.stackplot(range(episodes), 
                    *smoothed_values,
                    labels=[f"Action {i}" for i in range(4)],
                    colors=colors,
                    linewidth=0,  # Remove edge lines
                    edgecolor=None)  # Remove edge colors
        
        # Customize subplot
        ax.set_title(f"α = {alfa}", pad=10, fontsize=10)
        ax.set_xlabel("Episodes", fontsize=9)
        ax.set_ylabel("Operator Occurrences", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(False)  # Remove grid
        
        # Only show legend for the first plot
        if j == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    # Adjust layout and add main title
    plt.suptitle(title, y=1.02, fontsize=12)
    plt.tight_layout()
    
    # Save and show
    if save:
        plt.savefig(f'../images/{output_name}.png', bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    plt.close()

def flatten_nested_array(nested_array):
    """Recursively flatten a nested array of lists into a single 1D NumPy array."""
    flat_list = []

    def recurse(item):
        if isinstance(item, (list, tuple, np.ndarray)):
            for subitem in item:
                recurse(subitem)
        else:
            flat_list.append(item)

    recurse(nested_array)
    return np.array(flat_list, dtype=np.float32)

def plot_all_results(test_num, alfa_values, save=True, show=False, window_size=64):
    """Plot all results for a given test number and alfa values.
    
    Args:
        test_num (str): The test number identifier
        alfa_values (list): List of alfa values to plot
    """
    # Plot success rates
    plot_agent_data(f'../jsons/{test_num}/success/all_success', alfa_values, 'Success', 
                    f'Episode Success', f'{test_num}/all_success', window_size=window_size, save=save, show=show)
    
    # Plot rewards
    plot_agent_data(f'../jsons/{test_num}/reward/all_rewards', alfa_values, 'Reward', 
                   f'Episode Total Rewards', f'{test_num}/all_rewards', window_size=window_size, save=save, show=show)
    
    # Plot dimension-specific rewards
    plot_agent_data(f'../jsons/{test_num}/reward_dim1/all_rewards_dim1', alfa_values, 'Reward topic component', 
                   f'Episode Rewards for Dimension 1', f'{test_num}/all_rewards_dim1', window_size=window_size, save=save, show=show)
    plot_agent_data(f'../jsons/{test_num}/reward_dim2/all_rewards_dim2', alfa_values, 'Reward difficulty component', 
                   f'Episode Rewards for Dimension 2', f'{test_num}/all_rewards_dim2', window_size=window_size, save=save, show=show)
    
    # Plot Q-values
    plot_agent_data(f'../jsons/{test_num}/qvalues/all_qvalues', alfa_values, 'Q_value', 
                   f'Q-Values', f'{test_num}/all_qvalues', window_size=window_size, flag=True, save=save, show=show)
    
    # Plot losses
    plot_agent_data(f'../jsons/{test_num}/loss/all_losses', alfa_values, 'Loss', 
                   f'Losses', f'{test_num}/all_losses', window_size=window_size, save=save, show=show)
    
    # Plot exploration ratio
    plot_agent_data(f'../jsons/{test_num}/exploration_ratio/all_exploration_ratio', alfa_values, 
                   'Exploration Ratio', f'Exploration Ratio', f'{test_num}/all_exploration_ratio', window_size=window_size, save=save, show=show)
    
    # Plot action distribution
    plot_action_distribution(f'../jsons/{test_num}/actions/all_actions', alfa_values, 
                           f'Action Distribution', f'{test_num}/all_actions', window_size=int(window_size/4), save=save, show=show) 