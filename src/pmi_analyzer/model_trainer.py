"""
 model training module with cross-validation importance and confusion matrix
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import hamming_loss, f1_score, confusion_matrix
from sklearn.inspection import permutation_importance
from typing import Tuple, Dict, List, Optional
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ModelTrainer:
    
    def __init__(self, model_config: Optional[Dict] = None):
        """Initialize model trainer with configuration"""
        self.model_config = model_config or self._get_default_config()
        self.model = None
        self.cv_results = {}
        self.cv_importances = {}
        self.confusion_matrices = {}
        
    def _get_default_config(self) -> Dict:
        """Get default model configuration"""
        return {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'random_state': 42,
            'n_jobs': -1
        }
    
    def train_model(self, X: np.ndarray, y: np.ndarray, 
                   test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Train Random Forest model"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, 
            random_state=self.model_config.get('random_state', 42),
            stratify=y.sum(axis=1)
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # Filter out non-RandomForest parameters
        rf_params = {k: v for k, v in self.model_config.items() 
                     if k not in ['test_size', 'cv_folds']}
        
        # Create and train model
        base_estimator = RandomForestClassifier(**rf_params)
        self.model = MultiOutputClassifier(base_estimator)
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        return X_test, y_test, y_pred
    
    def calculate_confusion_matrices(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   process_names: List[str]) -> Dict[str, np.ndarray]:
        """Calculate confusion matrix for each process"""
        confusion_matrices = {}
        
        for i, process in enumerate(process_names):
            cm = confusion_matrix(y_true[:, i], y_pred[:, i])
            confusion_matrices[process] = cm
            
            # Log confusion matrix
            logger.info(f"\nConfusion Matrix for {process}:")
            logger.info(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
            logger.info(f"FN: {cm[1,0]}, TP: {cm[1,1]}")
            
        self.confusion_matrices = confusion_matrices
        return confusion_matrices
    
    def cross_validate_with_importance(self, X: np.ndarray, y: np.ndarray, 
                                     process_names: List[str], 
                                     feature_names: List[str],
                                     cv_folds: int = 5) -> Tuple[Dict[str, np.ndarray], Dict[str, pd.DataFrame]]:
        """
        Perform cross-validation and calculate feature importance for each fold
        
        Returns:
            Tuple of (cv_scores, cv_importances)
        """
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, 
                             random_state=self.model_config.get('random_state', 42))
        
        cv_scores = {process: [] for process in process_names}
        cv_importances = {process: [] for process in process_names}
        
        # Filter out non-RandomForest parameters
        rf_params = {k: v for k, v in self.model_config.items() 
                     if k not in ['test_size', 'cv_folds']}
        
        logger.info("Performing cross-validation with importance tracking...")
        
        # For each fold
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y.sum(axis=1))):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Train and evaluate each process
            for i, process in enumerate(process_names):
                clf = RandomForestClassifier(**rf_params)
                
                # Train on fold
                clf.fit(X_train_fold, y_train_fold[:, i])
                
                # Evaluate
                y_pred_fold = clf.predict(X_val_fold)
                score = f1_score(y_val_fold[:, i], y_pred_fold)
                cv_scores[process].append(score)
                
                # Get feature importance for this fold
                cv_importances[process].append(clf.feature_importances_)
            
            logger.info(f"Fold {fold_idx + 1}/{cv_folds} complete")
        
        # Convert scores to numpy arrays
        for process in process_names:
            cv_scores[process] = np.array(cv_scores[process])
            
            # Create DataFrame with importance statistics
            importance_array = np.array(cv_importances[process])  # shape: (n_folds, n_features)
            importance_df = pd.DataFrame({
                'mean': importance_array.mean(axis=0),
                'std': importance_array.std(axis=0),
                'min': importance_array.min(axis=0),
                'max': importance_array.max(axis=0)
            }, index=feature_names)
            cv_importances[process] = importance_df
            
            logger.info(f"CV {process}: {cv_scores[process].mean():.3f} (+/- {cv_scores[process].std() * 2:.3f})")
        
        self.cv_results = cv_scores
        self.cv_importances = cv_importances
        return cv_scores, cv_importances
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                      process_names: List[str], cv_folds: int = 5) -> Dict[str, np.ndarray]:
        """
        Perform standard cross-validation (without importance tracking)
        
        Args:
            X: Feature matrix
            y: Target matrix
            process_names: List of process names
            cv_folds: Number of CV folds
            
        Returns:
            Dictionary with CV scores for each process
        """
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, 
                             random_state=self.model_config.get('random_state', 42))
        
        cv_scores = {}
        
        # Filter out non-RandomForest parameters
        rf_params = {k: v for k, v in self.model_config.items() 
                     if k not in ['test_size', 'cv_folds']}
        
        for i, process in enumerate(process_names):
            # Get single process classifier
            clf = RandomForestClassifier(**rf_params)
            scores = cross_val_score(clf, X, y[:, i], cv=skf, scoring='f1')
            cv_scores[process] = scores
            
            logger.info(f"CV {process}: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
        
        self.cv_results = cv_scores
        return cv_scores
    
    def calculate_permutation_importance(self, X: np.ndarray, y: np.ndarray,
                                       process_names: List[str],
                                       feature_names: List[str],
                                       n_repeats: int = 10) -> pd.DataFrame:
        """Calculate permutation importance for robustness"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        logger.info(f"Calculating permutation importance with {n_repeats} repeats...")
        
        perm_importance_data = {}
        
        for i, process in enumerate(tqdm(process_names, desc="Permutation importance")):
            # Get single estimator
            estimator = self.model.estimators_[i]
            
            # Calculate permutation importance
            perm_importance = permutation_importance(
                estimator, X, y[:, i], 
                n_repeats=n_repeats,
                random_state=self.model_config.get('random_state', 42),
                n_jobs=-1
            )
            
            perm_importance_data[f'{process}_mean'] = perm_importance.importances_mean
            perm_importance_data[f'{process}_std'] = perm_importance.importances_std
        
        # Create DataFrame
        perm_df = pd.DataFrame(perm_importance_data, index=feature_names)
        
        # Add average importance
        mean_cols = [col for col in perm_df.columns if col.endswith('_mean')]
        perm_df['average'] = perm_df[mean_cols].mean(axis=1)
        
        return perm_df
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      process_names: List[str]) -> Dict[str, float]:
        """Evaluate model performance with additional metrics"""
        metrics = {
            'hamming_loss': hamming_loss(y_true, y_pred)
        }
        
        # Per-process metrics
        for i, process in enumerate(process_names):
            metrics[f'f1_{process}'] = f1_score(y_true[:, i], y_pred[:, i])
            
            # Additional metrics from confusion matrix
            cm = confusion_matrix(y_true[:, i], y_pred[:, i])
            tn, fp, fn, tp = cm.ravel()
            
            # Precision, Recall, Specificity
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            metrics[f'precision_{process}'] = precision
            metrics[f'recall_{process}'] = recall
            metrics[f'specificity_{process}'] = specificity
            
        # Average F1
        metrics['f1_average'] = np.mean([v for k, v in metrics.items() 
                                        if k.startswith('f1_')])
        
        logger.info("Model Performance:")
        logger.info(f"Hamming Loss: {metrics['hamming_loss']:.3f}")
        for process in process_names:
            logger.info(f"{process}: F1={metrics[f'f1_{process}']:.3f}, "
                       f"Prec={metrics[f'precision_{process}']:.3f}, "
                       f"Rec={metrics[f'recall_{process}']:.3f}")
            
        return metrics
    
    def get_feature_importances(self, feature_names: List[str]) -> pd.DataFrame:
        """Extract feature importances from trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        importance_data = {}
        
        # Get importance for each process
        for i, estimator in enumerate(self.model.estimators_):
            process_name = f"process_{i}"
            importance_data[process_name] = estimator.feature_importances_
        
        importance_df = pd.DataFrame(importance_data, index=feature_names)
        importance_df['average'] = importance_df.mean(axis=1)
        
        return importance_df