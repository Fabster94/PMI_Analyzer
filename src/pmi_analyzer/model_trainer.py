"""
Model training and evaluation module
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import hamming_loss, f1_score, classification_report
from typing import Tuple, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self, model_config: Optional[Dict] = None):
        """
        Initialize model trainer
        
        Args:
            model_config: Dictionary with model configuration
        """
        self.model_config = model_config or self._get_default_config()
        self.model = None
        self.cv_results = {}
        
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
        """
        Train Random Forest model
        
        Args:
            X: Feature matrix
            y: Target matrix
            test_size: Proportion of data for testing
            
        Returns:
            Tuple of (X_test, y_test, y_pred)
        """
        # Split data - stratify by number of processes
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, 
            random_state=self.model_config['random_state'],
            stratify=y.sum(axis=1)
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # Create and train model
        base_estimator = RandomForestClassifier(**self.model_config)
        self.model = MultiOutputClassifier(base_estimator)
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        return X_test, y_test, y_pred
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      process_names: List[str]) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            process_names: List of process names
            
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {
            'hamming_loss': hamming_loss(y_true, y_pred)
        }
        
        # Per-process metrics
        for i, process in enumerate(process_names):
            metrics[f'f1_{process}'] = f1_score(y_true[:, i], y_pred[:, i])
            
        # Average F1
        metrics['f1_average'] = np.mean([v for k, v in metrics.items() 
                                        if k.startswith('f1_')])
        
        logger.info("Model Performance:")
        logger.info(f"Hamming Loss: {metrics['hamming_loss']:.3f}")
        for process in process_names:
            logger.info(f"F1-Score for {process}: {metrics[f'f1_{process}']:.3f}")
            
        return metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                      process_names: List[str], cv_folds: int = 5) -> Dict[str, np.ndarray]:
        """
        Perform cross-validation
        
        Args:
            X: Feature matrix
            y: Target matrix
            process_names: List of process names
            cv_folds: Number of CV folds
            
        Returns:
            Dictionary with CV scores for each process
        """
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, 
                             random_state=self.model_config['random_state'])
        
        cv_scores = {}
        
        for i, process in enumerate(process_names):
            # Get single process classifier
            clf = RandomForestClassifier(**self.model_config)
            scores = cross_val_score(clf, X, y[:, i], cv=skf, scoring='f1')
            cv_scores[process] = scores
            
            logger.info(f"CV {process}: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
        
        self.cv_results = cv_scores
        return cv_scores
    
    def get_feature_importances(self, feature_names: List[str]) -> pd.DataFrame:
        """
        Extract feature importances from trained model
        
        Args:
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importances for each process
        """
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