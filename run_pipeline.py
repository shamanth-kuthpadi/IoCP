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
from sklearn.feature_selection import mutual_info_regression

import warnings
warnings.filterwarnings("ignore")

import logging

logging.basicConfig(
    filename="pipeline_debug_output.txt",
    filemode="w",  # Overwrite each run; use "a" to append
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s: %(message)s"
)

# Import pipeline modules
from util import preproc
from config import CausalConfig
from CausalModules import EstimateEffect

from visualization_utils import load_and_visualize_graph

import argparse
from colorama import init, Fore, Style


def load_and_preprocess_data():
    """
    Load and preprocess the data for causal analysis.
    """
    logging.info("=== Loading and Preprocessing Data ===")
    
    try:
        # Load data files
        logging.info("Loading data files...")
    
        # Preprocess data
        logging.info("Preprocessing data...")
        data_url = "https://raw.githubusercontent.com/FenTechSolutions/CausalDiscoveryToolbox/master/cdt/data/resources/cyto_full_data.csv"
        df = pd.read_csv(data_url)   
           
        logging.info(f"Data shape: {df.shape}")
        logging.info(f"Columns: {list(df.columns)}")
        logging.info(f"Outcome variable 'pmek' summary:\n{df['pmek'].describe()}")
        
        return df
        
    except FileNotFoundError as e:
        logging.error(f"Error: Data files not found. Please ensure 'data/betas.csv' and 'data/pds.csv' exist.")
        logging.error(f"Error details: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error during data preprocessing: {e}")
        sys.exit(1)

def setup_configuration():
    """
    Set up the pipeline configuration.
    """
    logging.info("\n=== Setting Up Configuration ===")
    
    config = CausalConfig()
    
    # Configure for multiple algorithms
    config.default_algorithms = ['pc']
    config.n_permutations = 30  # Adjust based on your needs
    config.confidence_level = 0.95
    
    logging.info(f"Algorithms to test: {config.default_algorithms}")
    logging.info(f"Number of permutations: {config.n_permutations}")
    logging.info(f"Confidence level: {config.confidence_level}")
    
    return config

def run_causal_pipeline(df, config):
    """
    Run the complete causal inference pipeline.
    """
    logging.info("\n=== Running Causal Inference Pipeline ===")
    
    # Initialize the pipeline
    logging.info("Initializing pipeline...")
    estimator = EstimateEffect(df, config=config)
    
    # Define treatment and outcome variables
    treatment = 'PIP3'
    outcome = 'pmek'
    
    logging.info(f"Treatment variable: {treatment}")
    logging.info(f"Outcome variable: {outcome}")
    
    # Create output directory for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Results will be saved to: {output_dir}")
    
    # Run the full pipeline
    logging.info("\nStarting pipeline execution...")
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
    logging.info("\n=== Analyzing Results ===")
    if isinstance(results, dict):
        logging.info("Multiple algorithms were executed. Analyzing each:")
        successful_algorithms = []
        failed_algorithms = []
        for algo, result in results.items():
            logging.info(f"\n{algo.upper()} Algorithm Results:")
            if isinstance(result, dict):
                graph = result.get('graph')
                if graph is not None:
                    logging.info(f"  - Graph nodes: {len(graph.nodes())}")
                    logging.info(f"  - Graph edges: {len(graph.edges())}")
                else:
                    logging.info("  - Graph: Not available")
                effect = result.get('effect_estimate')
                logging.info(f"  - Effect estimate: {effect if effect is not None else 'Not available'}")
                successful_algorithms.append(algo)
            else:
                logging.info("  - Result: Not available or failed")
                failed_algorithms.append(algo)
        logging.info(f"\nSummary:")
        logging.info(f"  Successful algorithms: {successful_algorithms}")
        if failed_algorithms:
            logging.info(f"  Failed algorithms: {failed_algorithms}")
    else:
        logging.info("Result: Not available or failed")
    logging.info(f"\n📁 Results saved to: {output_dir}")
    logging.info(f"   - Individual algorithm files: {output_dir}/results_*.json")

def extract_graph_refutation_metrics(graph_ref_str):
    import re
    if not isinstance(graph_ref_str, str):
        graph_ref_str = str(graph_ref_str)
    logging.debug("DEBUG: graph_ref_str =", graph_ref_str)  # Debug print
    # Updated regex for TPa and LMC with DOTALL flag
    tpa_match = re.search(r"informative because (\d+) / (\d+).*?\(p-value: ([0-9.]+)\)", graph_ref_str, re.DOTALL)
    lmc_match = re.search(r"violates (\d+)/(\d+) LMCs.*?\(p-value: ([0-9.]+)\)", graph_ref_str, re.DOTALL)
    tpa_num, tpa_total, tpa_p = (tpa_match.group(1), tpa_match.group(2), tpa_match.group(3)) if tpa_match else (None, None, None)
    lmc_num, lmc_total, lmc_p = (lmc_match.group(1), lmc_match.group(2), lmc_match.group(3)) if lmc_match else (None, None, None)
    return tpa_num, tpa_total, tpa_p, lmc_num, lmc_total, lmc_p

def extract_refuter_metrics(refuter_result):
    import re
    if not refuter_result:
        return None, None
    if not isinstance(refuter_result, str):
        refuter_result = str(refuter_result)
    logging.debug("DEBUG: refuter_result =", refuter_result)  # Debug print
    # p value
    pval_match = re.search(r"p value:([0-9.eE+-]+)", refuter_result)
    pval = pval_match.group(1).strip() if pval_match else None
    # new effect
    neweff_match = re.search(r"New effect:([0-9.eE+-]+)", refuter_result)
    neweff = neweff_match.group(1).strip() if neweff_match else None
    return pval, neweff

def write_summary_csv(results, output_dir):
    import csv
    summary_path = os.path.join(output_dir, "summary.csv")
    header = [
        "Algorithm",
        "TPa (#)", "TPa (total)", "TPa (p-value)",
        "LMC (#)", "LMC (total)", "LMC (p-value)",
        "Placebo Refuter (p-value)", "Random Common Cause Refuter (p-value)", "Data Subsample Refuter (p-value)",
        "Placebo Refuter (new effect)", "Random Common Cause Refuter (new effect)", "Data Subsample Refuter (new effect)"
    ]
    rows = []
    for algo, result in results.items():
        if not isinstance(result, dict):
            continue
        tpa_num, tpa_total, tpa_p, lmc_num, lmc_total, lmc_p = extract_graph_refutation_metrics(result.get("graph_refutation_res", ""))
        refuters = result.get("estimate_refutation_res", [])
        if not isinstance(refuters, list):
            refuters = [refuters]
        placebo_p, placebo_eff = extract_refuter_metrics(refuters[0]) if len(refuters) > 0 else (None, None)
        randcc_p, randcc_eff = extract_refuter_metrics(refuters[1]) if len(refuters) > 1 else (None, None)
        datasub_p, datasub_eff = extract_refuter_metrics(refuters[2]) if len(refuters) > 2 else (None, None)
        row = [
            algo,
            tpa_num, tpa_total, tpa_p,
            lmc_num, lmc_total, lmc_p,
            placebo_p, randcc_p, datasub_p,
            placebo_eff, randcc_eff, datasub_eff
        ]
        rows.append(row)
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    logging.info(f"\nSummary CSV written to: {summary_path}")

def cleanup_pipeline(estimator):
    """
    Clean up the pipeline to free memory.
    """
    logging.info("\n=== Cleaning Up ===")
    estimator.cleanup()
    logging.info("Pipeline cleanup completed.")

def main():
    """
    Main function to run the complete pipeline.
    """
    init(autoreset=True)
    parser = argparse.ArgumentParser(
        description="IOMICS Causal Pipeline (IoCP) - Full Pipeline Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data', type=str, default=None, help='Path or URL to input data (CSV).')
    parser.add_argument('--treatment', type=str, default='PIP3', help='Treatment variable name.')
    parser.add_argument('--outcome', type=str, default='pmek', help='Outcome variable name.')
    parser.add_argument('--algorithms', type=str, default='pc', help='Comma-separated list of algorithms to use.')
    parser.add_argument('--output', type=str, default=None, help='Output directory for results.')
    parser.add_argument('--n_permutations', type=int, default=30, help='Number of permutations for statistical tests.')
    parser.add_argument('--confidence', type=float, default=0.95, help='Confidence level.')
    parser.add_argument('--verbose', action='store_true', help='Print progress and results to console.')
    args = parser.parse_args()

    logging.info("IOMICS Causal Pipeline (IoCP) - Full Pipeline Runner")
    logging.info("=" * 60)

    try:
        # Step 1: Load and preprocess data
        if args.data:
            data_url = args.data
        else:
            data_url = "https://raw.githubusercontent.com/FenTechSolutions/CausalDiscoveryToolbox/master/cdt/data/resources/cyto_full_data.csv"
        if args.verbose:
            print(Fore.BLUE + f"[INFO] Loading data from: {data_url}")
        df = pd.read_csv(data_url)
        if args.verbose:
            print(Fore.BLUE + f"[INFO] Data shape: {df.shape}")
            print(Fore.BLUE + f"[INFO] Columns: {list(df.columns)}")
            print(Fore.BLUE + f"[INFO] Outcome variable '{args.outcome}' summary:\n{df[args.outcome].describe()}")

        # Step 2: Set up configuration
        config = CausalConfig()
        config.default_algorithms = [a.strip() for a in args.algorithms.split(',')]
        config.n_permutations = args.n_permutations
        config.confidence_level = args.confidence
        if args.verbose:
            print(Fore.BLUE + f"[INFO] Algorithms to test: {config.default_algorithms}")
            print(Fore.BLUE + f"[INFO] Number of permutations: {config.n_permutations}")
            print(Fore.BLUE + f"[INFO] Confidence level: {config.confidence_level}")

        # Step 3: Run the pipeline
        estimator = EstimateEffect(df, config=config)
        treatment = args.treatment
        outcome = args.outcome
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = args.output or f"results_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        if args.verbose:
            print(Fore.BLUE + f"[INFO] Results will be saved to: {output_dir}")
        results = estimator.run_full_pipeline(
            treatment=treatment,
            outcome=outcome,
            refute_graph=True,
            refute_estimate=True,
            visualize=False,
            export_path=f"{output_dir}/results.json",
            export_format='json'
        )

        # Step 4: Analyze results
        analyze_results(results, output_dir)
        write_summary_csv(results, output_dir)
        cleanup_pipeline(estimator)
        if args.verbose:
            print(Fore.GREEN + Style.BRIGHT + "\n[SUCCESS] Pipeline execution completed successfully!")
            print(Fore.GREEN + f"[SUCCESS] Check the results in: {output_dir}")
    except KeyboardInterrupt:
        logging.info("\nPipeline execution interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"\nPipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        print(Fore.RED + Style.BRIGHT + f"[ERROR] Pipeline execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 