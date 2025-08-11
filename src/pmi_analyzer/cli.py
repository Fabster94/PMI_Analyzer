"""
command line interface
"""
import argparse
import logging
import os
import yaml
import pandas as pd
from datetime import datetime

from pmi_analyzer.data_loader import PMIDataLoader
from pmi_analyzer.model_trainer import ModelTrainer
from pmi_analyzer.visualizer import PMIVisualizer
from pmi_analyzer.feature_analyzer import FeatureAnalyzer
from pmi_analyzer.ablation_analyzer import AblationAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file or use defaults"""
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
        },
        'analysis': {
            'use_permutation_importance': True,
            'permutation_repeats': 10,
            'calculate_cv_importance': True,
            'create_confusion_matrix': True,
            'perform_ablation': False,
            'ablation_groups': {
                'dimensions_only': ['dimensions'],
                'tolerances_only': ['tolerances'],
                'surface_only': ['surface_finish'],
                'dims_and_tolerances': ['dimensions', 'tolerances'],
                'all_features': ['dimensions', 'tolerances', 'surface_finish', 'fits', 'datums', 'other']
            }
        }
    }
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
            # Deep merge configs
            for key in user_config:
                if key in default_config and isinstance(default_config[key], dict):
                    default_config[key].update(user_config[key])
                else:
                    default_config[key] = user_config[key]
    
    return default_config


def run_standard_analysis(config: dict, output_suffix: str = None):
    """Run the standard PMI analysis pipeline"""
    
    print("🚀 PMI Feature Importance Analysis (Extended)")
    print("="*60)
    
    # Modify output paths if suffix provided (for ablation)
    if output_suffix:
        config['data']['output_dir'] = os.path.join(config['data']['output_dir'], output_suffix)
        config['visualization']['figure_dir'] = os.path.join(config['data']['output_dir'], 'figures')
    
    # Create output directories
    os.makedirs(config['data']['output_dir'], exist_ok=True)
    os.makedirs(config['visualization']['figure_dir'], exist_ok=True)
    
    # Step 1: Load and prepare data
    print("\n📊 Step 1: Loading and preparing data...")
    loader = PMIDataLoader()
    df = loader.load_data(config['data']['input_path'])
    X, y, feature_names = loader.prepare_features_and_targets(df)
    
    # Get process distribution and statistics
    process_dist = loader.get_process_distribution(y)
    feature_stats = loader.get_feature_statistics(X, feature_names)
    
    # Save feature statistics
    feature_stats.to_csv(
        os.path.join(config['data']['output_dir'], 'feature_statistics.csv')
    )
    
    # Step 2: Exploratory visualization
    print("\n📈 Step 2: Creating exploratory visualizations...")
    visualizer = PMIVisualizer(output_dir=config['visualization']['figure_dir'])
    
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
    metrics = trainer.evaluate_model(y_test, y_pred, list(loader.mlb.classes_))
    
    # Calculate confusion matrices if requested
    if config['analysis'].get('create_confusion_matrix', True):
        confusion_matrices = trainer.calculate_confusion_matrices(
            y_test, y_pred, list(loader.mlb.classes_)
        )
        visualizer.plot_confusion_matrices(
            confusion_matrices,
            list(loader.mlb.classes_),
            save_name='confusion_matrices.png'
        )
    
    # Step 4: Cross-validation with importance tracking
    if config['analysis'].get('calculate_cv_importance', True):
        print("\n🔄 Step 4: Cross-validation with importance tracking...")
        cv_scores, cv_importances = trainer.cross_validate_with_importance(
            X, y, 
            list(loader.mlb.classes_),
            feature_names,
            cv_folds=config['model']['cv_folds']
        )
        
        # Plot CV scores
        visualizer.plot_cv_scores(cv_scores, save_name='cv_scores.png')
        
        # Plot feature stability
        visualizer.plot_feature_stability(
            cv_importances,
            list(loader.mlb.classes_),
            top_n=15,
            save_name='feature_stability.png'
        )
        
        # Save CV importance data
        for process, importance_df in cv_importances.items():
            importance_df.to_csv(
                os.path.join(config['data']['output_dir'], f'cv_importance_{process}.csv')
            )
    else:
        # Regular cross-validation
        cv_scores = trainer.cross_validate(
            X, y, 
            list(loader.mlb.classes_), 
            cv_folds=config['model']['cv_folds']
        )
    
    # Step 5: Feature importance analysis
    print("\n🎯 Step 5: Analyzing feature importance...")
    
    # MDI importance
    mdi_importance_df = trainer.get_feature_importances(feature_names)
    column_mapping = {f'process_{i}': name for i, name in enumerate(loader.mlb.classes_)}
    mdi_importance_df = mdi_importance_df.rename(columns=column_mapping)
    
    # Permutation importance if requested
    perm_importance_df = None
    if config['analysis'].get('use_permutation_importance', True):
        print("\n🔀 Calculating permutation importance...")
        perm_importance_df = trainer.calculate_permutation_importance(
            X, y,
            list(loader.mlb.classes_),
            feature_names,
            n_repeats=config['analysis'].get('permutation_repeats', 10)
        )
        
        # Save permutation importance
        perm_importance_df.to_csv(
            os.path.join(config['data']['output_dir'], 'permutation_importance.csv')
        )
        
        # Compare MDI vs Permutation importance
        visualizer.plot_importance_comparison(
            mdi_importance_df,
            perm_importance_df,
            top_n=20,
            save_name='importance_comparison.png'
        )
    
    # Use MDI for standard visualizations
    visualizer.plot_feature_importance_heatmap(
        mdi_importance_df,
        list(loader.mlb.classes_),
        top_n=config['visualization']['top_n_features'],
        save_name='feature_importance_heatmap.png'
    )
    
    visualizer.plot_top_features_per_process(
        mdi_importance_df,
        list(loader.mlb.classes_),
        top_n=10,
        save_name='top_features_per_process.png'
    )
    
    visualizer.plot_feature_correlation(
        X, feature_names,
        top_n=20,
        save_name='feature_correlation.png'
    )
    
    # Step 6: Advanced analysis
    print("\n📊 Step 6: Advanced feature analysis...")
    
    analyzer = FeatureAnalyzer(trainer.model, feature_names, list(loader.mlb.classes_))
    
    # Calculate group importance
    group_importance = analyzer.calculate_group_importance(mdi_importance_df)
    group_importance.to_csv(
        os.path.join(config['data']['output_dir'], 'feature_group_importance.csv')
    )
    
    # Find redundant features
    redundant_features = analyzer.find_redundant_features(X, correlation_threshold=0.9)
    if redundant_features:
        redundant_df = pd.DataFrame(redundant_features, 
                                   columns=['feature_1', 'feature_2', 'correlation'])
        redundant_df.to_csv(
            os.path.join(config['data']['output_dir'], 'redundant_features.csv'),
            index=False
        )
    
    # Generate insights
    insights = analyzer.generate_insights(mdi_importance_df, metrics)
    
    # Save all results
    mdi_importance_df.to_csv(
        os.path.join(config['data']['output_dir'], 'mdi_feature_importance.csv')
    )
    
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(
        os.path.join(config['data']['output_dir'], 'model_metrics.csv'),
        index=False
    )
    
    # Print summary
    print_analysis_summary(df, feature_names, loader, metrics, mdi_importance_df, 
                         insights, group_importance, feature_stats, config)
    
    return {
        'importance_df': mdi_importance_df,
        'perm_importance_df': perm_importance_df,
        'metrics': metrics,
        'cv_scores': cv_scores,
        'feature_names': feature_names,
        'X': X,
        'y': y,
        'loader': loader
    }


def run_ablation_study(config: dict, base_results: dict):
    """Run ablation study with different feature groups"""
    
    print("\n" + "="*60)
    print("🧪 ABLATION STUDY")
    print("="*60)
    
    # Get data from base results
    X = base_results['X']
    y = base_results['y']
    feature_names = base_results['feature_names']
    process_names = list(base_results['loader'].mlb.classes_)
    
    # Initialize ablation analyzer
    ablation_analyzer = AblationAnalyzer(
        feature_names, 
        process_names,
        model_config=config['model']
    )
    
    # Show feature groups
    groups_summary = ablation_analyzer.get_feature_groups_summary()
    print("\n📋 Feature Groups:")
    print(groups_summary.to_string(index=False))
    
    # Perform ablation study
    ablation_config = config['analysis'].get('ablation_groups', {})
    
    if not ablation_config:
        logger.warning("No ablation groups defined in config. Using default groups.")
        # Use default ablation groups if none defined
        ablation_config = {
            'dimensions_only': ['dimensions'],
            'tolerances_only': ['tolerances'],
            'surface_only': ['surface_finish'],
            'all_features': ['dimensions', 'tolerances', 'surface_finish', 'fits', 'datums', 'other']
        }
    
    ablation_results = ablation_analyzer.perform_ablation_study(X, y, ablation_config)
    
    # Save ablation results
    ablation_output_dir = os.path.join(config['data']['output_dir'], 'ablation_study')
    os.makedirs(ablation_output_dir, exist_ok=True)
    
    ablation_results.to_csv(
        os.path.join(ablation_output_dir, 'ablation_results.csv')
    )
    
    # Visualize ablation results
    visualizer = PMIVisualizer(output_dir=ablation_output_dir)
    visualizer.plot_ablation_results(
        ablation_results,
        process_names,
        save_name='ablation_study.png'
    )
    
    # Perform redundancy analysis
    print("\n🔍 Performing redundancy analysis...")
    redundancy_results = ablation_analyzer.analyze_feature_redundancy(X, y, top_n=30)
    redundancy_results.to_csv(
        os.path.join(ablation_output_dir, 'redundancy_analysis.csv')
    )
    
    visualizer.plot_redundancy_analysis(
        redundancy_results,
        save_name='redundancy_analysis.png'
    )
    
    # Run separate analyses for each feature group if requested
    if config['analysis'].get('run_separate_group_analyses', False):
        print("\n📂 Running separate analyses for each feature group...")
        
        for group_name, group_list in ablation_config.items():
            if group_name != 'all_features':
                print(f"\n→ Analyzing {group_name}...")
                
                # Modify config for this subset
                subset_config = config.copy()
                subset_config['analysis']['feature_groups'] = group_list
                
                # Run analysis with subset
                run_standard_analysis(subset_config, output_suffix=f'ablation/{group_name}')
    
    print("\n✅ Ablation study complete!")
    
    return ablation_results


def print_analysis_summary(df, feature_names, loader, metrics, importance_df, 
                          insights, group_importance, feature_stats, config):
    """Print comprehensive analysis summary"""
    
    print("\n" + "="*60)
    print("📊 ANALYSIS SUMMARY")
    print("="*60)
    print(f"\nDataset: {len(df)} parts")
    print(f"Features: {len(feature_names)}")
    print(f"Processes: {', '.join(loader.mlb.classes_)}")
    
    # Feature statistics
    print("\n📈 Feature Statistics:")
    active_features = feature_stats[feature_stats['non_zero_pct'] > 10].sort_values('non_zero_pct', ascending=False)
    print(f"Features present in >10% of parts: {len(active_features)}")
    if len(active_features) > 0:
        print("Most common features:")
        for feat, row in active_features.head(5).iterrows():
            print(f"  - {feat}: {row['non_zero_pct']:.1f}% of parts")
    
    # Model performance
    print("\n🎯 Model Performance:")
    print(f"Average F1 Score: {metrics['f1_average']:.3f}")
    print(f"Hamming Loss: {metrics['hamming_loss']:.3f}")
    
    for process in loader.mlb.classes_:
        print(f"\n{process}:")
        print(f"  F1 Score: {metrics[f'f1_{process}']:.3f}")
        print(f"  Precision: {metrics[f'precision_{process}']:.3f}")
        print(f"  Recall: {metrics[f'recall_{process}']:.3f}")
    
    # Top features
    print("\n🌟 Top 5 Most Important Features (MDI):")
    top_features = importance_df.nlargest(5, 'average')
    for i, (feature, row) in enumerate(top_features.iterrows(), 1):
        print(f"{i}. {feature}: {row['average']:.4f}")
    
    # Insights
    print("\n💡 Key Insights:")
    for key, insight in insights.items():
        print(f"- {insight}")
    
    # Feature group importance
    if not group_importance.empty and 'average' in group_importance.columns:
        print("\n🔧 Feature Group Importance:")
        for group, avg_imp in group_importance['average'].sort_values(ascending=False).items():
            print(f"  - {group}: {avg_imp:.3f}")
    
    print("\n✅ Analysis complete!")
    print(f"Results saved to: {config['data']['output_dir']}")
    print(f"Figures saved to: {config['visualization']['figure_dir']}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='PMI Feature Importance Analysis (Extended)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard analysis
  pmi-analyze --input data.csv
  
  # With ablation study
  pmi-analyze --input data.csv --ablation
  
  # With custom config
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
    parser.add_argument(
        '--ablation', '-a',
        action='store_true',
        help='Perform ablation study'
    )
    parser.add_argument(
        '--timestamp',
        action='store_true',
        help='Add timestamp to output directory'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.input:
        config['data']['input_path'] = args.input
    if args.output:
        config['data']['output_dir'] = args.output
    if args.ablation:
        config['analysis']['perform_ablation'] = True
    
    # Add timestamp if requested
    if args.timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config['data']['output_dir'] = f"{config['data']['output_dir']}_{timestamp}"
    
    # Update figure directory
    config['visualization']['figure_dir'] = os.path.join(config['data']['output_dir'], 'figures')
    
    # Run standard analysis
    try:
        results = run_standard_analysis(config)
        
        # Run ablation study if requested
        if config['analysis'].get('perform_ablation', False):
            run_ablation_study(config, results)
            
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()