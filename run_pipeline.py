#!/usr/bin/env python3
"""
IOMICS Causal Pipeline (IoCP) - Full Pipeline Runner

This script demonstrates how to run the complete causal inference pipeline
with multiple algorithms and comprehensive result analysis.

Usage:
    python run_pipeline.py
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Import pipeline modules
from util import preproc
from config import CausalConfig
from CausalModules import EstimateEffect

def load_and_preprocess_data():
    """
    Load and preprocess the data for causal analysis.
    """
    print("=== Loading and Preprocessing Data ===")
    
    try:
        # Load data files
        print("Loading data files...")
        betas_df = pd.read_csv('data/betas.csv')
        pds_df = pd.read_csv('data/pds.csv')
        
        # Preprocess data
        print("Preprocessing data...")
        df = preproc(betas_df, pds_df)
        
        print(f"Data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Target variable 'Group' distribution:\n{df['Group'].value_counts()}")
        
        return df
        
    except FileNotFoundError as e:
        print(f"Error: Data files not found. Please ensure 'data/betas.csv' and 'data/pds.csv' exist.")
        print(f"Error details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during data preprocessing: {e}")
        sys.exit(1)

def setup_configuration():
    """
    Set up the pipeline configuration.
    """
    print("\n=== Setting Up Configuration ===")
    
    config = CausalConfig()
    
    # Configure for multiple algorithms
    config.default_algorithms = ['pc', 'ges', 'icalingam']
    config.n_permutations = 100  # Adjust based on your needs
    config.confidence_level = 0.95
    
    print(f"Algorithms to test: {config.default_algorithms}")
    print(f"Number of permutations: {config.n_permutations}")
    print(f"Confidence level: {config.confidence_level}")
    
    return config

def run_causal_pipeline(df, config):
    """
    Run the complete causal inference pipeline.
    """
    print("\n=== Running Causal Inference Pipeline ===")
    
    # Initialize the pipeline
    print("Initializing pipeline...")
    estimator = EstimateEffect(df, config=config)
    
    # Define treatment and outcome variables
    treatment = 'CACNA1S'
    outcome = 'Group'
    
    print(f"Treatment variable: {treatment}")
    print(f"Outcome variable: {outcome}")
    
    # Create output directory for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")
    
    # Run the full pipeline
    print("\nStarting pipeline execution...")
    results = estimator.run_full_pipeline(
        treatment=treatment,
        outcome=outcome,
        refute_graph=True,
        refute_estimate=True,
        visualize=False,  # Set to True if you want plots
        export_path=f"{output_dir}/results.json",
        export_format='json'
    )
    
    return estimator, results, output_dir

def analyze_results(results, output_dir):
    """
    Analyze and display the pipeline results.
    """
    print("\n=== Analyzing Results ===")
    
    # Check if we have multiple algorithms or single algorithm
    if isinstance(results, dict) and len(results) > 1:
        print("Multiple algorithms were executed. Analyzing each:")
        
        successful_algorithms = []
        failed_algorithms = []
        
        for algo, result in results.items():
            if result is not None:
                successful_algorithms.append(algo)
                print(f"\n✅ {algo.upper()} Algorithm Results:")
                print(f"  - Graph nodes: {len(result['graph'].nodes())}")
                print(f"  - Graph edges: {len(result['graph'].edges())}")
                if 'effect_estimate' in result and result['effect_estimate'] is not None:
                    print(f"  - Effect estimate: {result['effect_estimate']}")
                else:
                    print(f"  - Effect estimate: Not available")
            else:
                failed_algorithms.append(algo)
                print(f"\n❌ {algo.upper()} Algorithm: Failed")
        
        print(f"\n📊 Summary:")
        print(f"  Successful algorithms: {successful_algorithms}")
        if failed_algorithms:
            print(f"  Failed algorithms: {failed_algorithms}")
        
        # Compare effect estimates if available
        effect_comparison = {}
        for algo, result in results.items():
            if result and 'effect_estimate' in result and result['effect_estimate'] is not None:
                effect_comparison[algo] = result['effect_estimate']
        
        if effect_comparison:
            print(f"\n📈 Effect Estimate Comparison:")
            for algo, effect in effect_comparison.items():
                print(f"  {algo}: {effect}")
    
    else:
        # Single algorithm result
        print("Single algorithm was executed.")
        if results and 'effect_estimate' in results:
            print(f"Effect estimate: {results['effect_estimate']}")
            print(f"Graph nodes: {len(results['graph'].nodes())}")
            print(f"Graph edges: {len(results['graph'].edges())}")
    
    print(f"\n📁 Results saved to: {output_dir}")
    print(f"   - Individual algorithm files: {output_dir}/results_*.json")

def cleanup_pipeline(estimator):
    """
    Clean up the pipeline to free memory.
    """
    print("\n=== Cleaning Up ===")
    estimator.cleanup()
    print("Pipeline cleanup completed.")

def main():
    """
    Main function to run the complete pipeline.
    """
    print("🚀 IOMICS Causal Pipeline (IoCP) - Full Pipeline Runner")
    print("=" * 60)
    
    try:
        # Step 1: Load and preprocess data
        df = load_and_preprocess_data()
        
        # Step 2: Set up configuration
        config = setup_configuration()
        
        # Step 3: Run the pipeline
        estimator, results, output_dir = run_causal_pipeline(df, config)
        
        # Step 4: Analyze results
        analyze_results(results, output_dir)
        
        # Step 5: Cleanup
        cleanup_pipeline(estimator)
        
        print("\n🎉 Pipeline execution completed successfully!")
        print(f"📁 Check the results in: {output_dir}")
        
    except KeyboardInterrupt:
        print("\nPipeline execution interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nPipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 