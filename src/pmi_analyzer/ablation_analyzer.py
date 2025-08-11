"""
Ablation study module for feature group analysis
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class AblationAnalyzer:
    """Performs ablation studies to understand feature group importance"""
    
    def __init__(self, feature_names: List[str], process_names: List[str], 
                 model_config: Optional[Dict] = None):
        """
        Initialize ablation analyzer
        
        Args:
            feature_names: List of all feature names
            process_names: List of process names
            model_config: Model configuration
        """
        self.feature_names = feature_names
        self.process_names = process_names
        self.model_config = model_config or self._get_default_config()
        self.feature_groups = self._identify_feature_groups()
        
    def _get_default_config(self) -> Dict:
        """Get default model configuration"""
        return {
            'n_estimators': 50,  # Fewer trees for faster ablation
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'random_state': 42,
            'n_jobs': -1
        }
    
    def _identify_feature_groups(self) -> Dict[str, List[str]]:
        """Categorize features into logical groups"""
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
        
        return groups
    
    def perform_ablation_study(self, X: np.ndarray, y: np.ndarray, 
                             ablation_config: Dict[str, List[str]] = None) -> pd.DataFrame:
        """
        Perform ablation study with specified feature group combinations
        
        Args:
            X: Feature matrix
            y: Target matrix
            ablation_config: Dictionary mapping study names to lists of feature groups
            
        Returns:
            DataFrame with ablation results
        """
        logger.info("Starting ablation study...")
        
        # Handle None or empty config
        if not ablation_config:
            logger.warning("No ablation config provided. Using default configuration.")
            ablation_config = {
                'all_features': ['dimensions', 'tolerances', 'surface_finish', 'fits', 'datums', 'other']
            }
        
        results = []
        
        # Check if 'all_features' is already in config
        has_all_features_in_config = 'all_features' in ablation_config
        
        # Baseline with all features (only if not in config)
        if not has_all_features_in_config:
            baseline_scores = self._evaluate_feature_set(X, y, list(range(X.shape[1])), "all_features")
            results.append(baseline_scores)
        
        # Perform ablation for each configuration
        for study_name, group_names in ablation_config.items():
            # Get feature indices for these groups
            feature_indices = self._get_feature_indices_for_groups(group_names)
            
            if not feature_indices:
                logger.warning(f"No features found for ablation config: {study_name}")
                continue
            
            # Evaluate this feature set
            scores = self._evaluate_feature_set(X, y, feature_indices, study_name)
            results.append(scores)
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        results_df = results_df.set_index('config')
        
        # Remove duplicates if any (keep first)
        results_df = results_df[~results_df.index.duplicated(keep='first')]
        
        # Calculate relative performance
        for col in results_df.columns:
            if col.startswith('f1_') and not col.endswith('_relative') and not col.endswith('_std'):
                if 'all_features' in results_df.index:
                    baseline_val = results_df.loc['all_features', col]
                    # Handle Series case
                    if hasattr(baseline_val, '__len__') and len(baseline_val) > 1:
                        baseline_val = baseline_val.iloc[0]
                    results_df[f'{col}_relative'] = results_df[col] / baseline_val * 100
                else:
                    results_df[f'{col}_relative'] = 100.0  # No baseline = 100%
        
        return results_df
    
    def _get_feature_indices_for_groups(self, group_names: List[str]) -> List[int]:
        """Get feature indices for specified groups"""
        feature_indices = []
        
        for group_name in group_names:
            if group_name in self.feature_groups:
                for feature in self.feature_groups[group_name]:
                    if feature in self.feature_names:
                        idx = self.feature_names.index(feature)
                        feature_indices.append(idx)
        
        return sorted(list(set(feature_indices)))
    
    def _evaluate_feature_set(self, X: np.ndarray, y: np.ndarray, 
                            feature_indices: List[int], config_name: str) -> Dict:
        """Evaluate a specific set of features"""
        logger.info(f"Evaluating {config_name} with {len(feature_indices)} features...")
        
        # Subset features
        X_subset = X[:, feature_indices]
        
        # Filter out non-RandomForest parameters
        rf_params = {k: v for k, v in self.model_config.items() 
                     if k not in ['test_size', 'cv_folds']}
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, 
                             random_state=self.model_config.get('random_state', 42))
        
        scores = {'config': config_name, 'n_features': len(feature_indices)}
        
        # Evaluate each process
        for i, process in enumerate(self.process_names):
            clf = RandomForestClassifier(**rf_params)
            cv_scores = cross_val_score(clf, X_subset, y[:, i], cv=skf, scoring='f1')
            scores[f'f1_{process}'] = cv_scores.mean()
            scores[f'f1_{process}_std'] = cv_scores.std()
        
        # Average F1
        f1_cols = [col for col in scores.keys() if col.startswith('f1_') and not col.endswith('_std')]
        if f1_cols:
            f1_values = [scores[col] for col in f1_cols]
            scores['f1_average'] = np.mean(f1_values)
        
        logger.info(f"{config_name}: Average F1 = {scores.get('f1_average', 0):.3f}")
        
        return scores
    
    def analyze_feature_redundancy(self, X: np.ndarray, y: np.ndarray, 
                                 top_n: int = 20) -> pd.DataFrame:
        """
        Analyze redundancy by progressively adding features
        
        Args:
            X: Feature matrix
            y: Target matrix
            top_n: Number of top features to analyze
            
        Returns:
            DataFrame showing performance as features are added
        """
        logger.info(f"Analyzing feature redundancy for top {top_n} features...")
        
        # First, get feature importance with all features
        rf_params = {k: v for k, v in self.model_config.items() 
                     if k not in ['test_size', 'cv_folds']}
        
        # Train a quick model to get importances
        base_estimator = RandomForestClassifier(**rf_params)
        model = MultiOutputClassifier(base_estimator)
        model.fit(X, y)
        
        # Get average importance across processes
        importances = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
        
        # Get top feature indices
        top_indices = np.argsort(importances)[-top_n:][::-1]
        
        results = []
        
        # Progressively add features
        for n_features in range(1, min(top_n + 1, len(top_indices) + 1)):
            feature_subset = top_indices[:n_features]
            
            # Evaluate with this subset
            scores = self._evaluate_feature_set(X, y, feature_subset.tolist(), 
                                              f"top_{n_features}_features")
            scores['features_added'] = self.feature_names[top_indices[n_features-1]]
            results.append(scores)
        
        return pd.DataFrame(results)
    
    def get_feature_groups_summary(self) -> pd.DataFrame:
        """Get summary of feature groups"""
        summary_data = []
        
        for group_name, features in self.feature_groups.items():
            summary_data.append({
                'group': group_name,
                'n_features': len(features),
                'features': ', '.join(features[:5]) + ('...' if len(features) > 5 else '')
            })
        
        return pd.DataFrame(summary_data)