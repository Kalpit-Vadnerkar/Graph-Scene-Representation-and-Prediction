from Risk_Assessment.RiskManager import EnhancedRiskManager
from Risk_Assessment.FeatureExtractor import TemporalFeatureExtractor
from Prediction_Model.model_utils import load_model


import os
import argparse


def main():
    parser = argparse.ArgumentParser(description="Enhanced fault detection pipeline")
    parser.add_argument('--components', type=int, default=None,
                       help='Number of PCA components to use (default: None = determine automatically)')
    parser.add_argument('--estimators', type=int, default=None,
                       help='Number of estimators for RandomForest (default: None = determine automatically)')
    parser.add_argument('--max_estimators', type=int, default=200,
                       help='Maximum number of estimators to test when optimizing')
    parser.add_argument('--mode', type=str, choices=['optimize', 'run', 'estimator_analysis'], 
                        default='optimize',
                        help='Mode: optimize (full optimization), run (with specified parameters), or estimator_analysis')
    args = parser.parse_args()

    # Import config after argparse to avoid circular imports
    from model_config import CONFIG
    
    # Create results directory
    os.makedirs('Results/Enhanced', exist_ok=True)
    
    # Initialize manager
    manager = EnhancedRiskManager(CONFIG)
    
    # Load model
    model = load_model(CONFIG)
    
    # Load data for all conditions
    print("Loading data for all conditions...")
    loaded_data_dict = {}
    for condition in CONFIG['conditions']:
        print(f"  Loading data for condition: {condition}")
        loaded_data_dict[condition] = manager.data_loader.load_data_and_predictions(model, condition)
    
    # Run selected mode
    if args.mode == 'optimize':
        # Full optimization pipeline
        print("\nRunning full optimization pipeline...")
        results = manager.optimize_and_run_fault_detection(
            loaded_data_dict, 
            max_components=args.components,
            max_estimators=args.max_estimators
        )
    elif args.mode == 'run':
        # Run with specified parameters
        if args.components is None:
            raise ValueError("Number of components must be specified in 'run' mode")
        if args.estimators is None:
            raise ValueError("Number of estimators must be specified in 'run' mode")
            
        print(f"\nRunning with specified parameters: {args.components} components, {args.estimators} estimators")
        results = manager.run_optimized_fault_detection(
            loaded_data_dict,
            n_components=args.components,
            n_estimators=args.estimators
        )
    elif args.mode == 'estimator_analysis':
        # Analyze estimator performance
        print("\nAnalyzing estimator performance...")
        if args.components is None:
            # First determine optimal components
            manager.feature_extractor = TemporalFeatureExtractor()
            manager.generate_all_residuals(loaded_data_dict)
            transformed_features_full = manager.feature_extractor.fit_transform(manager.all_residuals)
            optimal_components = manager.plot_and_analyze_explained_variance()
            components = optimal_components
        else:
            components = args.components
            
        results_df, optimal_estimators = manager.analyze_estimator_performance(
            loaded_data_dict,
            n_components=components,
            max_estimators=args.max_estimators
        )


if __name__ == "__main__":
    main()