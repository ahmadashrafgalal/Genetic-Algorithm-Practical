import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
import networkx as nx

class VisualizationUtils:
    """Utility class for creating professional visualizations"""
    
    @staticmethod
    def setup_professional_style():
        """Configure matplotlib for professional appearance"""
        plt.style.use('seaborn-v0_8-darkgrid')
        
    @staticmethod
    def get_color_palette():
        """Return a professional color palette"""
        return {
            'primary': '#2196F3',
            'success': '#4CAF50',
            'warning': '#FF9800',
            'danger': '#F44336',
            'accent': '#9C27B0',
            'light': '#E0E0E0',
            'dark': '#424242'
        }
    
    @staticmethod
    def plot_convergence_curve(history, title="Convergence Curve"):
        """Create a convergence curve visualization"""
        ax = plt.gca()
        colors = VisualizationUtils.get_color_palette()
        
        ax.plot(history, linewidth=2.5, color=colors['primary'], marker='o', 
                markersize=3, markevery=max(1, len(history)//20))
        ax.fill_between(range(len(history)), history, alpha=0.2, color=colors['primary'])
        
        ax.set_xlabel('Generation/Iteration', fontsize=11, fontweight='bold')
        ax.set_ylabel('Best Fitness', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        return ax
    
    @staticmethod
    def plot_comparative_bars(categories, values, title="Comparison"):
        """Create a professional bar chart"""
        ax = plt.gca()
        colors_palette = VisualizationUtils.get_color_palette()
        
        colors = [colors_palette['primary']] * len(categories)
        bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}' if isinstance(height, (int, float)) else str(height),
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Value', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        return ax
    
    @staticmethod
    def plot_heatmap(data, title="Heatmap", x_labels=None, y_labels=None):
        """Create a professional heatmap"""
        ax = plt.gca()
        
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto')
        
        if x_labels:
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
        
        if y_labels:
            ax.set_yticks(range(len(y_labels)))
            ax.set_yticklabels(y_labels)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax)
        
        return ax, im
    
    @staticmethod
    def add_stats_box(ax, stats_dict, location='upper right'):
        """Add a statistics box to plot"""
        textstr = '\n'.join([f'{k}: {v}' for k, v in stats_dict.items()])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props, family='monospace')

class DataGenerator:
    """Utility class for generating test data"""
    
    @staticmethod
    def generate_random_cities(num_cities, seed=42):
        """Generate random cities for TSP"""
        np.random.seed(seed)
        cities_coords = np.random.rand(num_cities, 2) * 10
        city_labels = [f"C{i}" for i in range(num_cities)]
        return cities_coords, city_labels
    
    @staticmethod
    def generate_graph(num_nodes, edge_prob=0.4, seed=42):
        """Generate a random graph"""
        G = nx.erdos_renyi_graph(num_nodes, edge_prob, seed=seed)
        return G
    
    @staticmethod
    def generate_random_knapsack_items(num_items):
        """Generate random knapsack items"""
        np.random.seed(42)
        values = np.random.randint(10, 100, num_items)
        weights = np.random.randint(5, 50, num_items)
        return values, weights
