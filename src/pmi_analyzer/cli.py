"""
Command line interface for PMI Analyzer
"""
import argparse
import logging
import os
from pathlib import Path
import yaml

from pmi_analyzer.data_loader import PMIDataLoader
from pmi_analyzer.model_trainer import ModelTrainer
from pmi_analyzer.visualizer import PMIVisualizer
from pmi_analyzer.feature_analyzer import FeatureAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file or use defaults"""
    # Default configuration
    default_config = {
        'data': {
            'input_path': 'data/raw/pmi_features_with_processes.csv',
            'output_dir': 'results'
        },
        'model': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'random_state': 42,
            'test_size': 0.2,
            'cv_folds': 5
        },
        'visualization': {
            'top_n_features': 20,
            'figure_dir': 'results/figures'
        }
    }
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
            # Merge configs
            for key in user_config:
                if key in default_config:
                    default_config[key].update(user_config[key])
                else:
                    default_config[key] = user_config[key]
    
    return default_config


def run_analysis(config: dict):
    """Run the complete PMI analysis pipeline"""
    
    print("🚀 PMI Feature Importance Analysis")
    print("="*60)
    
    # Create output directories
    os.makedirs(config['data']['output_dir'], exist_ok=True)
    os.makedirs(config['visualization']['figure_dir'], exist_ok=True)
    
    # Step 1: Load and prepare data
    print("\n📊 Step 1: Loading and preparing data...")
    loader = PMIDataLoader()
    df = loader.load_data(config['data']['input_path'])
    X, y, feature_names = loader.prepare_features_and_targets(df)
    
    # Get process distribution
    process_dist = loader.get_process_distribution(y)
    feature_stats = loader.get_feature_statistics(X, feature_names)
    
    # Step 2: Exploratory visualization
    print("\n📈 Step 2: Creating exploratory visualizations...")
    visualizer = PMIVisualizer(output_dir=config['visualization']['figure_dir'])
    
    # Plot process distribution
    visualizer.plot_process_distribution(
        process_dist, 
        save_name='process_distribution.png'
    )
    
    # Step 3: Train model
    print("\n🤖 Step 3: Training Random Forest model...")
    trainer = ModelTrainer(model_config=config['model'])
    X_test, y_test, y_pred = trainer.train_model(
        X, y, 
        test_size=config['model']['test_size']
    )
    
    # Evaluate model
    metrics = trainer.evaluate_model(y_test, y_pred, loader.mlb.classes_)
    
    # Cross-validation
    cv_scores = trainer.cross_validate(
        X, y, 
        loader.mlb.classes_, 
        cv_folds=config['model']['cv_folds']
    )
    
    # Plot CV scores
    visualizer.plot_cv_scores(cv_scores, save_name='cv_scores.png')
    
    # Step 4: Feature importance analysis
    print("\n🎯 Step 4: Analyzing feature importance...")
    importance_df = trainer.get_feature_importances(feature_names)
    
    # Rename columns to use process names
    column_mapping = {f'process_{i}': name for i, name in enumerate(loader.mlb.classes_)}
    importance_df = importance_df.rename(columns=column_mapping)
    
    # Visualize feature importance
    visualizer.plot_feature_importance_heatmap(
        importance_df,
        list(loader.mlb.classes_),
        top_n=config['visualization']['top_n_features'],
        save_name='feature_importance_heatmap.png'
    )
    
    visualizer.plot_top_features_per_process(
        importance_df,
        list(loader.mlb.classes_),
        top_n=10,
        save_name='top_features_per_process.png'
    )
    
    # Plot feature correlations
    visualizer.plot_feature_correlation(
        X, feature_names,
        top_n=20,
        save_name='feature_correlation.png'
    )
    
    # Step 5: Generate report
    print("\n📝 Step 5: Generating analysis report...")
    
    # Initialize feature analyzer
    analyzer = FeatureAnalyzer(trainer.model, feature_names, list(loader.mlb.classes_))
    
    # Get insights
    insights = analyzer.generate_insights(importance_df, metrics)
    
    # Save results
    import pandas as pd
    
    # Save feature importance
    importance_df.to_csv(
        os.path.join(config['data']['output_dir'], 'feature_importance.csv')
    )
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(
        os.path.join(config['data']['output_dir'], 'model_metrics.csv'),
        index=False
    )
    
    # Print summary
    print("\n" + "="*60)
    print("📊 ANALYSIS SUMMARY")
    print("="*60)
    print(f"\nDataset: {len(df)} parts")
    print(f"Features: {len(feature_names)}")
    print(f"Processes: {', '.join(loader.mlb.classes_)}")
    
    print("\n🎯 Model Performance:")
    print(f"Average F1 Score: {metrics['f1_average']:.3f}")
    print(f"Hamming Loss: {metrics['hamming_loss']:.3f}")
    
    print("\n🌟 Top 5 Most Important Features (Overall):")
    top_features = importance_df.nlargest(5, 'average')
    for i, (feature, row) in enumerate(top_features.iterrows(), 1):
        print(f"{i}. {feature}: {row['average']:.4f}")
    
    print("\n💡 Key Insights:")
    for key, insight in insights.items():
        print(f"- {insight}")
    
    print("\n✅ Analysis complete!")
    print(f"Results saved to: {config['data']['output_dir']}")
    print(f"Figures saved to: {config['visualization']['figure_dir']}")
    
    return importance_df, metrics


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='PMI Feature Importance Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  pmi-analyze
  
  # Run with custom data file
  pmi-analyze --input data.csv
  
  # Run with custom config
  pmi-analyze --config my_config.yaml
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        help='Input CSV file path'
    )
    parser.add_argument(
        '--config', '-c',
        default='config.yaml',
        help='Configuration file path'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.input:
        config['data']['input_path'] = args.input
    if args.output:
        config['data']['output_dir'] = args.output
        config['visualization']['figure_dir'] = os.path.join(args.output, 'figures')
    
    # Run analysis
    try:
        run_analysis(config)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()