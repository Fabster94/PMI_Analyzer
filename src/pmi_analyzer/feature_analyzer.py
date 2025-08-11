"""
Feature importance and analysis module
"""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureAnalyzer:
    """Advanced feature analysis beyond basic importance scores"""
    
    def __init__(self, model, feature_names: List[str], process_names: List[str]):
        """
        Initialize feature analyzer
        
        Args:
            model: Trained model
            feature_names: List of feature names
            process_names: List of process names
        """
        self.model = model
        self.feature_names = feature_names
        self.process_names = process_names
        
    def calculate_permutation_importance(self, X: np.ndarray, y: np.ndarray,
                                       n_repeats: int = 10) -> pd.DataFrame:
        """
        Calculate permutation importance for more robust feature importance
        
        Args:
            X: Feature matrix
            y: Target matrix
            n_repeats: Number of permutation repeats
            
        Returns:
            DataFrame with permutation importance scores
        """
        logger.info("Calculating permutation importance...")
        
        perm_importance_data = {}
        
        for i, process in enumerate(self.process_names):
            # Get single estimator
            estimator = self.model.estimators_[i]
            
            # Calculate permutation importance
            perm_importance = permutation_importance(
                estimator, X, y[:, i], 
                n_repeats=n_repeats,
                random_state=42,
                n_jobs=-1
            )
            
            perm_importance_data[f'{process}_mean'] = perm_importance.importances_mean
            perm_importance_data[f'{process}_std'] = perm_importance.importances_std
        
        # Create DataFrame
        perm_df = pd.DataFrame(perm_importance_data, index=self.feature_names)
        
        # Add average importance
        mean_cols = [col for col in perm_df.columns if col.endswith('_mean')]
        perm_df['average'] = perm_df[mean_cols].mean(axis=1)
        
        return perm_df
    
    def identify_feature_groups(self) -> Dict[str, List[str]]:
        """
        Categorize features into logical groups
        
        Returns:
            Dictionary mapping group names to feature lists
        """
        groups = {
            'dimensions': [],
            'tolerances': [],
            'surface_finish': [],
            'fits': [],
            'datums': [],
            'other': []
        }
        
        for feature in self.feature_names:
            if 'dimension' in feature:
                groups['dimensions'].append(feature)
            elif feature.startswith('has_') and any(tol in feature for tol in 
                ['concentricity', 'position', 'flatness', 'perpendicularity',
                 'cylindricity', 'runout', 'roundness', 'straightness',
                 'parallelism', 'angularity', 'profile', 'symmetry']):
                groups['tolerances'].append(feature)
            elif 'surface' in feature or 'roughness' in feature:
                groups['surface_finish'].append(feature)
            elif 'fit' in feature:
                groups['fits'].append(feature)
            elif 'datum' in feature:
                groups['datums'].append(feature)
            else:
                groups['other'].append(feature)
        
        # Remove empty groups
        groups = {k: v for k, v in groups.items() if v}
        
        logger.info("Feature groups identified:")
        for group, features in groups.items():
            logger.info(f"  {group}: {len(features)} features")
            
        return groups
    
    def calculate_group_importance(self, importance_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate average importance by feature group
        
        Args:
            importance_df: DataFrame with feature importances
            
        Returns:
            DataFrame with group-level importance scores
        """
        groups = self.identify_feature_groups()
        
        # Pivot to have groups as rows and processes as columns
        reshaped_data = {}
        for group_name, features in groups.items():
            # Get features that exist in importance_df
            existing_features = [f for f in features if f in importance_df.index]
            
            if existing_features:
                row_data = {}
                # Calculate mean importance for each process
                for process in self.process_names:
                    if process in importance_df.columns:
                        row_data[process] = importance_df.loc[existing_features, process].mean()
                
                # Overall average
                if 'average' in importance_df.columns:
                    row_data['average'] = importance_df.loc[existing_features, 'average'].mean()
                
                if row_data:
                    reshaped_data[group_name] = row_data
        
        return pd.DataFrame(reshaped_data).T
    
    def find_redundant_features(self, X: np.ndarray, 
                              correlation_threshold: float = 0.95) -> List[Tuple[str, str, float]]:
        """
        Find highly correlated features that might be redundant
        
        Args:
            X: Feature matrix
            correlation_threshold: Threshold for considering features redundant
            
        Returns:
            List of tuples (feature1, feature2, correlation)
        """
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X.T)
        
        redundant_pairs = []
        
        # Find high correlations
        for i in range(len(self.feature_names)):
            for j in range(i+1, len(self.feature_names)):
                corr = corr_matrix[i, j]
                if abs(corr) > correlation_threshold:
                    redundant_pairs.append((
                        self.feature_names[i],
                        self.feature_names[j],
                        corr
                    ))
        
        # Sort by correlation
        redundant_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        if redundant_pairs:
            logger.info(f"Found {len(redundant_pairs)} redundant feature pairs:")
            for feat1, feat2, corr in redundant_pairs[:5]:
                logger.info(f"  {feat1} <-> {feat2}: {corr:.3f}")
                
        return redundant_pairs
    
    def generate_insights(self, importance_df: pd.DataFrame, 
                         metrics: Dict[str, float]) -> Dict[str, str]:
        """
        Generate interpretable insights from the analysis
        
        Args:
            importance_df: Feature importance DataFrame
            metrics: Model performance metrics
            
        Returns:
            Dictionary of insights
        """
        insights = {}
        
        # Overall model performance
        avg_f1 = metrics.get('f1_average', 0)
        if avg_f1 > 0.8:
            insights['performance'] = f"Excellent model performance (F1={avg_f1:.3f}). PMI features are highly predictive."
        elif avg_f1 > 0.6:
            insights['performance'] = f"Good model performance (F1={avg_f1:.3f}). PMI features provide useful signals."
        else:
            insights['performance'] = f"Moderate performance (F1={avg_f1:.3f}). Consider additional features."
        
        # Most important features
        top_features = importance_df.nlargest(5, 'average').index.tolist()
        insights['top_features'] = f"Top 5 features: {', '.join(top_features)}"
        
        # Process-specific insights
        for process in self.process_names:
            if process in importance_df.columns:
                top_process_feature = importance_df[process].idxmax()
                insights[f'{process}_key'] = f"Key feature for {process}: {top_process_feature}"
        
        # Feature group insights
        group_importance = self.calculate_group_importance(importance_df)
        if 'average' in group_importance.columns:
            top_group = group_importance['average'].idxmax()
            insights['important_group'] = f"Most important feature group: {top_group}"
        
        return insights