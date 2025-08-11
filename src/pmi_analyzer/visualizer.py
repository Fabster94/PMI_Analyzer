
"""
visualization module
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class PMIVisualizer:
    
    def __init__(self, output_dir: str = 'results/figures'):
        self.output_dir = output_dir
        self.figsize_default = (12, 8)
    
    def plot_confusion_matrices(self, confusion_matrices: Dict[str, np.ndarray],
                              process_names: List[str],
                              save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot confusion matrices for all processes
        
        Args:
            confusion_matrices: Dict mapping process names to confusion matrices
            process_names: List of process names
            save_name: Filename to save plot
            
        Returns:
            Matplotlib figure
        """
        n_processes = len(process_names)
        fig, axes = plt.subplots(1, n_processes, figsize=(5*n_processes, 4))
        
        if n_processes == 1:
            axes = [axes]
        
        for i, process in enumerate(process_names):
            if process in confusion_matrices:
                cm = confusion_matrices[process]
                
                # Calculate percentages
                cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
                
                # Create heatmap
                ax = axes[i]
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['No', 'Yes'], 
                           yticklabels=['No', 'Yes'],
                           ax=ax, cbar=True)
                
                # Add percentage annotations
                for j in range(2):
                    for k in range(2):
                        ax.text(k + 0.5, j + 0.7, f'({cm_percent[j, k]:.1f}%)',
                               ha='center', va='center', fontsize=9, color='gray')
                
                ax.set_title(f'Confusion Matrix: {process}')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
            
        return fig
    
    def plot_feature_importance_heatmap(self, importance_df: pd.DataFrame, 
                                      process_names: List[str],
                                      top_n: int = 20,
                                      save_name: Optional[str] = None) -> plt.Figure:
        """Create feature importance heatmap"""
        # Get top features
        top_features = importance_df.nlargest(top_n, 'average')
        
        # Create proper column names
        column_mapping = {f'process_{i}': name for i, name in enumerate(process_names)}
        top_features_renamed = top_features.rename(columns=column_mapping)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(top_features_renamed[process_names] if all(p in top_features_renamed.columns for p in process_names) else top_features[process_names], 
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
        """Plot top features for each process"""
        n_processes = len(process_names)
        fig, axes = plt.subplots(1, n_processes, figsize=(5*n_processes, 8))
        
        if n_processes == 1:
            axes = [axes]
        
        for i, process in enumerate(process_names):
            # Use the actual process name instead of process_i
            if process in importance_df.columns:
                top_features = importance_df.nlargest(top_n, process)
                
                ax = axes[i]
                top_features[process].plot(kind='barh', ax=ax, color=f'C{i}')
                ax.set_xlabel('Importance')
                ax.set_title(f'Top {top_n} Features for {process}')
                ax.invert_yaxis()
            else:
                logger.warning(f"Process '{process}' not found in importance_df columns")
            
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
            
        return fig
    
    def plot_cv_scores(self, cv_scores: Dict[str, np.ndarray],
                      save_name: Optional[str] = None) -> plt.Figure:
        """Plot cross-validation scores"""
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
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
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
        """Plot correlation matrix of top features"""
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
    
    def plot_feature_stability(self, cv_importances: Dict[str, pd.DataFrame],
                             process_names: List[str],
                             top_n: int = 15,
                             save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot feature importance stability across CV folds
        
        Args:
            cv_importances: Dict of DataFrames with CV importance statistics
            process_names: List of process names
            top_n: Number of top features to show
            save_name: Filename to save plot
            
        Returns:
            Matplotlib figure
        """
        n_processes = len(process_names)
        fig, axes = plt.subplots(n_processes, 1, figsize=(12, 5*n_processes))
        
        if n_processes == 1:
            axes = [axes]
        
        for i, process in enumerate(process_names):
            if process in cv_importances:
                importance_df = cv_importances[process]
                
                # Get top features by mean importance
                top_features = importance_df.nlargest(top_n, 'mean')
                
                ax = axes[i]
                
                # Create error bar plot
                x_pos = np.arange(len(top_features))
                
                ax.barh(x_pos, top_features['mean'], 
                       xerr=top_features['std'], 
                       capsize=5, color=f'C{i}', alpha=0.7)
                
                # Add feature names
                ax.set_yticks(x_pos)
                ax.set_yticklabels(top_features.index)
                ax.invert_yaxis()
                
                ax.set_xlabel('Feature Importance (Mean ± Std)')
                ax.set_title(f'Feature Importance Stability: {process}')
                ax.grid(True, alpha=0.3)
                
                # Add coefficient of variation as text
                for j, (idx, row) in enumerate(top_features.iterrows()):
                    cv = row['std'] / row['mean'] if row['mean'] > 0 else 0
                    ax.text(row['mean'] + row['std'] + 0.005, j, 
                           f'CV={cv:.2f}', va='center', fontsize=8)
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
            
        return fig
    
    def plot_ablation_results(self, ablation_results: pd.DataFrame,
                            process_names: List[str],
                            save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot ablation study results
        
        Args:
            ablation_results: DataFrame with ablation study results
            process_names: List of process names
            save_name: Filename to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Absolute F1 scores
        x_pos = np.arange(len(ablation_results))
        width = 0.25
        
        for i, process in enumerate(process_names):
            if f'f1_{process}' in ablation_results.columns:
                ax1.bar(x_pos + i*width, ablation_results[f'f1_{process}'], 
                       width, label=process, alpha=0.8)
        
        ax1.set_xlabel('Feature Configuration')
        ax1.set_ylabel('F1 Score')
        ax1.set_title('Ablation Study: F1 Scores by Feature Configuration')
        ax1.set_xticks(x_pos + width)
        ax1.set_xticklabels(ablation_results.index, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add baseline line
        if 'all_features' in ablation_results.index and 'f1_average' in ablation_results.columns:
            baseline_avg = ablation_results.loc['all_features', 'f1_average']
            # Ensure it's a scalar value
            if hasattr(baseline_avg, 'item'):
                baseline_avg = baseline_avg.item()
            ax1.axhline(y=baseline_avg, color='red', linestyle='--', 
                       label='Baseline (all features)', alpha=0.5)
        
        # Plot 2: Relative performance
        relative_data = []
        for col in ablation_results.columns:
            if col.endswith('_relative'):
                process = col.replace('f1_', '').replace('_relative', '')
                relative_data.append({
                    'config': ablation_results.index.tolist(),
                    'relative_f1': ablation_results[col].tolist(),
                    'process': [process] * len(ablation_results)
                })
        
        if relative_data:
            relative_df = pd.concat([pd.DataFrame(d) for d in relative_data])
            
            # Create grouped bar plot
            for i, config in enumerate(ablation_results.index):
                config_data = relative_df[relative_df['config'] == config]
                for j, row in enumerate(config_data.itertuples()):
                    ax2.bar(i + j*0.25, row.relative_f1, 0.25, 
                           label=row.process if i == 0 else "")
            
            ax2.set_xlabel('Feature Configuration')
            ax2.set_ylabel('Relative F1 Score (%)')
            ax2.set_title('Relative Performance vs. All Features Baseline')
            ax2.set_xticks(range(len(ablation_results)))
            ax2.set_xticklabels(ablation_results.index, rotation=45, ha='right')
            ax2.axhline(y=100, color='red', linestyle='--', alpha=0.5)
            ax2.grid(True, alpha=0.3)
            if i == 0:
                ax2.legend()
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
            
        return fig
    
    def plot_redundancy_analysis(self, redundancy_df: pd.DataFrame,
                               save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot feature redundancy analysis
        
        Args:
            redundancy_df: DataFrame with redundancy analysis results
            save_name: Filename to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize_default)
        
        # Plot F1 average vs number of features
        ax.plot(redundancy_df.index + 1, redundancy_df['f1_average'], 
               'o-', linewidth=2, markersize=8, label='Average F1')
        
        # Add individual process lines
        for col in redundancy_df.columns:
            if col.startswith('f1_') and not col.endswith('_std') and col != 'f1_average':
                process = col.replace('f1_', '')
                ax.plot(redundancy_df.index + 1, redundancy_df[col], 
                       '--', alpha=0.7, label=process)
        
        ax.set_xlabel('Number of Top Features')
        ax.set_ylabel('F1 Score')
        ax.set_title('Performance vs. Number of Features (Redundancy Analysis)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add annotations for plateaus
        if len(redundancy_df) > 5:
            # Find where performance plateaus (change < 1%)
            f1_values = redundancy_df['f1_average'].values
            for i in range(5, len(f1_values)):
                if abs(f1_values[i] - f1_values[i-1]) < 0.01:
                    ax.annotate(f'Plateau at {i+1} features',
                              xy=(i+1, f1_values[i]),
                              xytext=(i+1, f1_values[i] - 0.05),
                              arrowprops=dict(arrowstyle='->', color='red', alpha=0.5))
                    break
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
            
        return fig
    
    def plot_importance_comparison(self, mdi_importance: pd.DataFrame,
                                 perm_importance: pd.DataFrame,
                                 top_n: int = 20,
                                 save_name: Optional[str] = None) -> plt.Figure:
        """
        Compare MDI and Permutation Importance
        
        Args:
            mdi_importance: DataFrame with MDI importance
            perm_importance: DataFrame with permutation importance
            top_n: Number of top features to show
            save_name: Filename to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Get top features from MDI
        top_mdi = mdi_importance.nlargest(top_n, 'average')
        
        # Get same features from permutation importance
        top_perm = perm_importance.loc[top_mdi.index]
        
        # Plot 1: Side-by-side comparison
        ax1 = axes[0]
        x_pos = np.arange(len(top_mdi))
        width = 0.35
        
        ax1.barh(x_pos - width/2, top_mdi['average'], width, label='MDI', alpha=0.7)
        ax1.barh(x_pos + width/2, top_perm['average'], width, label='Permutation', alpha=0.7)
        
        ax1.set_yticks(x_pos)
        ax1.set_yticklabels(top_mdi.index)
        ax1.invert_yaxis()
        ax1.set_xlabel('Importance Score')
        ax1.set_title(f'Top {top_n} Features: MDI vs Permutation Importance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Scatter plot for correlation
        ax2 = axes[1]
        
        # Get all features present in both
        common_features = list(set(mdi_importance.index) & set(perm_importance.index))
        mdi_values = mdi_importance.loc[common_features, 'average'].values
        perm_values = perm_importance.loc[common_features, 'average'].values
        
        ax2.scatter(mdi_values, perm_values, alpha=0.6)
        
        # Add correlation line
        z = np.polyfit(mdi_values, perm_values, 1)
        p = np.poly1d(z)
        ax2.plot(sorted(mdi_values), p(sorted(mdi_values)), "r--", alpha=0.8)
        
        # Calculate correlation
        corr = np.corrcoef(mdi_values, perm_values)[0, 1]
        ax2.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                transform=ax2.transAxes, verticalalignment='top')
        
        ax2.set_xlabel('MDI Importance')
        ax2.set_ylabel('Permutation Importance')
        ax2.set_title('MDI vs Permutation Importance Correlation')
        ax2.grid(True, alpha=0.3)
        
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
    
    # Include all visualization methods - either inherit or copy from PMIVisualizer
    def plot_process_distribution(self, process_counts: Dict[str, int], 
                                 save_name: Optional[str] = None) -> plt.Figure:
        """Plot distribution of processes"""
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