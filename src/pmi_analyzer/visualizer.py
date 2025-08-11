"""
Visualization module for PMI analysis
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class PMIVisualizer:
    """Handles all visualization tasks"""
    
    def __init__(self, output_dir: str = 'results/figures'):
        """
        Initialize visualizer
        
        Args:
            output_dir: Directory to save figures
        """
        self.output_dir = output_dir
        self.figsize_default = (12, 8)
        
    def plot_process_distribution(self, process_counts: Dict[str, int], 
                                 save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot distribution of processes
        
        Args:
            process_counts: Dictionary with process counts
            save_name: Filename to save plot
            
        Returns:
            Matplotlib figure
        """
        # Separate process counts and combination counts
        processes = {k: v for k, v in process_counts.items() if not k.endswith('_processes')}
        combinations = {k: v for k, v in process_counts.items() if k.endswith('_processes')}
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Process distribution
        ax1.bar(processes.keys(), processes.values())
        ax1.set_xlabel('Process Type')
        ax1.set_ylabel('Number of Parts')
        ax1.set_title('Distribution of Manufacturing Processes')
        
        # Add percentages on bars
        total = sum(processes.values())
        for i, (k, v) in enumerate(processes.items()):
            ax1.text(i, v + 5, f'{v/total*100:.1f}%', ha='center')
        
        # Combinations distribution
        combo_labels = [k.replace('_processes', ' processes') for k in combinations.keys()]
        ax2.bar(combo_labels, combinations.values())
        ax2.set_xlabel('Number of Processes')
        ax2.set_ylabel('Number of Parts')
        ax2.set_title('Parts by Process Count')
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
            
        return fig
    
    def plot_feature_importance_heatmap(self, importance_df: pd.DataFrame, 
                                      process_names: List[str],
                                      top_n: int = 20,
                                      save_name: Optional[str] = None) -> plt.Figure:
        """
        Create feature importance heatmap
        
        Args:
            importance_df: DataFrame with feature importances
            process_names: List of process names
            top_n: Number of top features to show
            save_name: Filename to save plot
            
        Returns:
            Matplotlib figure
        """
        # Get top features
        top_features = importance_df.nlargest(top_n, 'average')
        
        # Create proper column names
        column_mapping = {f'process_{i}': name for i, name in enumerate(process_names)}
        top_features_renamed = top_features.rename(columns=column_mapping)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(top_features_renamed[process_names], 
                   annot=True, fmt='.3f', cmap='YlOrRd',
                   cbar_kws={'label': 'Importance'},
                   ax=ax)
        
        ax.set_title(f'Feature Importance Heatmap (Top {top_n})', fontsize=14)
        ax.set_xlabel('Manufacturing Process')
        ax.set_ylabel('PMI Features')
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
            
        return fig
    
    def plot_top_features_per_process(self, importance_df: pd.DataFrame,
                                    process_names: List[str],
                                    top_n: int = 10,
                                    save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot top features for each process
        
        Args:
            importance_df: DataFrame with feature importances
            process_names: List of process names
            top_n: Number of top features to show
            save_name: Filename to save plot
            
        Returns:
            Matplotlib figure
        """
        n_processes = len(process_names)
        fig, axes = plt.subplots(1, n_processes, figsize=(5*n_processes, 8))
        
        if n_processes == 1:
            axes = [axes]
        
        for i, process in enumerate(process_names):
            col_name = f'process_{i}'
            top_features = importance_df.nlargest(top_n, col_name)
            
            ax = axes[i]
            top_features[col_name].plot(kind='barh', ax=ax, color=f'C{i}')
            ax.set_xlabel('Importance')
            ax.set_title(f'Top {top_n} Features for {process}')
            ax.invert_yaxis()
            
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
            
        return fig
    
    def plot_cv_scores(self, cv_scores: Dict[str, np.ndarray],
                      save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot cross-validation scores
        
        Args:
            cv_scores: Dictionary with CV scores for each process
            save_name: Filename to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize_default)
        
        # Prepare data for box plot
        data = []
        labels = []
        for process, scores in cv_scores.items():
            data.append(scores)
            labels.append(process)
        
        # Create box plot
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        
        # Color boxes
        for patch, color in zip(bp['boxes'], plt.cm.Set3.colors):
            patch.set_facecolor(color)
        
        ax.set_xlabel('Process')
        ax.set_ylabel('F1 Score')
        ax.set_title('Cross-Validation F1 Scores by Process')
        ax.grid(True, alpha=0.3)
        
        # Add mean values
        for i, (process, scores) in enumerate(cv_scores.items()):
            ax.text(i+1, scores.mean() + 0.02, f'{scores.mean():.3f}', 
                   ha='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
            
        return fig
    
    def plot_feature_correlation(self, X: np.ndarray, feature_names: List[str],
                               top_n: int = 20,
                               save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot correlation matrix of top features
        
        Args:
            X: Feature matrix
            feature_names: List of feature names
            top_n: Number of top features to include
            save_name: Filename to save plot
            
        Returns:
            Matplotlib figure
        """
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X.T)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Select top features based on variance
        variances = X.var(axis=0)
        top_indices = np.argsort(variances)[-top_n:]
        
        # Create subset correlation matrix
        corr_subset = corr_matrix[np.ix_(top_indices, top_indices)]
        feature_subset = [feature_names[i] for i in top_indices]
        
        # Plot heatmap
        sns.heatmap(corr_subset, 
                   xticklabels=feature_subset,
                   yticklabels=feature_subset,
                   annot=True, fmt='.2f', 
                   cmap='coolwarm',
                   center=0, vmin=-1, vmax=1,
                   ax=ax)
        
        ax.set_title(f'Feature Correlation Matrix (Top {top_n} by Variance)', fontsize=14)
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
            
        return fig
    
    def _save_figure(self, fig: plt.Figure, filename: str):
        """Save figure to output directory"""
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {filepath}")