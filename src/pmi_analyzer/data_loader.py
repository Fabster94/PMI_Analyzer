"""
Data loading and preprocessing module for PMI analysis
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
from sklearn.preprocessing import MultiLabelBinarizer
import logging

logger = logging.getLogger(__name__)


class PMIDataLoader:
    """Handles loading and preprocessing of PMI data"""
    
    def __init__(self):
        self.mlb = MultiLabelBinarizer()
        self.feature_columns = None
        self.process_names = ['drehen', 'fraesen', 'bohren']
        self.binary_mode = False  # Track if using binary columns
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load PMI data from CSV file
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with PMI data
        """
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} parts from {filepath}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def parse_processes(self, process_str: str) -> List[str]:
        """
        Parse process string to list
        
        Args:
            process_str: String representation of processes
            
        Returns:
            List of process names
        """
        if pd.isna(process_str) or process_str == '[]':
            return []
        
        # Remove brackets and quotes, split by comma
        process_str = process_str.strip('[]').replace("'", "").replace('"', '')
        if not process_str:
            return []
        
        processes = [p.strip() for p in process_str.split(',')]
        return processes
    
    def prepare_features_and_targets(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare feature matrix and target labels
        
        Args:
            df: DataFrame with PMI data
            
        Returns:
            Tuple of (features, targets, feature_names)
        """
        # Check if we have binary process columns or original Processes column
        binary_columns = [col for col in df.columns if col.startswith('has_') 
                         and any(proc in col for proc in ['drehen', 'fraesen', 'bohren'])]
        
        if binary_columns:
            # Use existing binary columns
            logger.info("Using existing binary process columns")
            self.binary_mode = True
            y = df[binary_columns].values
            self.mlb.classes_ = np.array([col.replace('has_', '') for col in binary_columns])
            self.process_names = list(self.mlb.classes_)
            
            # Prepare features (exclude identification and target columns)
            self.feature_columns = [col for col in df.columns 
                                   if col not in ['part_name', 'Processes', 'processes_list'] + binary_columns]
        else:
            # Parse processes from original format
            logger.info("Parsing processes from 'Processes' column")
            self.binary_mode = False
            df['processes_list'] = df['Processes'].apply(self.parse_processes)
            
            # Create binary encoding for processes
            y_lists = df['processes_list'].tolist()
            y = self.mlb.fit_transform(y_lists)
            self.process_names = list(self.mlb.classes_)
            
            # Prepare features (exclude identification and target columns)
            self.feature_columns = [col for col in df.columns 
                                   if col not in ['part_name', 'Processes', 'processes_list']]
        
        X = df[self.feature_columns].values
        
        logger.info(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
        logger.info(f"Process types: {self.mlb.classes_}")
        
        return X, y, self.feature_columns
    
    def get_process_distribution(self, y: np.ndarray) -> Dict[str, int]:
        """
        Get distribution of processes in the dataset
        
        Args:
            y: Binary encoded target matrix
            
        Returns:
            Dictionary with process counts
        """
        process_counts = {}
        
        # Check if we have classes
        if hasattr(self.mlb, 'classes_') and len(self.mlb.classes_) > 0:
            for i, process in enumerate(self.mlb.classes_):
                process_counts[process] = int(y[:, i].sum())
        else:
            # Fallback if classes not set (shouldn't happen)
            for i in range(y.shape[1]):
                process_counts[f'process_{i}'] = int(y[:, i].sum())
        
        # Add combination statistics
        processes_per_part = y.sum(axis=1)
        for i in range(4):
            process_counts[f'{i}_processes'] = int((processes_per_part == i).sum())
            
        return process_counts
    
    def get_feature_statistics(self, X: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
        """
        Calculate basic statistics for features
        
        Args:
            X: Feature matrix
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature statistics
        """
        df_features = pd.DataFrame(X, columns=feature_names)
        
        stats = pd.DataFrame({
            'mean': df_features.mean(),
            'std': df_features.std(),
            'min': df_features.min(),
            'max': df_features.max(),
            'non_zero_count': (df_features > 0).sum(),
            'non_zero_pct': (df_features > 0).sum() / len(df_features) * 100
        })
        
        return stats.round(3)