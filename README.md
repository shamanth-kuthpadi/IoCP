# IOMICS Causal Pipeline (IoCP)

## Overview

This repository provides a robust, modular, and extensible pipeline for causal inference using Python. The pipeline supports:

- Causal graph discovery (multiple algorithms)
- Graph refutation/falsification
- Causal effect identification and estimation
- Estimate refutation
- Visualization of graphs and results
- Exporting results
- Memory management and cleanup
- Full pipeline orchestration with a single method call
- **Multi-algorithm comparison** - automatically run and compare multiple causal discovery algorithms

The codebase is organized into clear modules:

- `CausalModules.py`: Main pipeline logic and orchestration
- `util.py`: Utility functions for data and graph handling
- `config.py`: Configuration management
- `logging_utils.py`: Logging setup and utilities
- `visualization_utils.py`: Visualization helpers

---

## Installation & Requirements

### Prerequisites

- Python 3.8+
- Required packages (install with pip):
  - pandas
  - numpy
  - networkx
  - matplotlib
  - pydot
  - scikit-learn
  - dowhy
  - causallearn
  - cdt
  - tqdm (optional, for progress bars)

```
pip install pandas numpy networkx matplotlib pydot scikit-learn dowhy causal-learn cdt tqdm
```

> **Note:** Some causal discovery algorithms require R and `Rscript` installed. Adjust `cdt.SETTINGS.rpath` in `CausalModules.py` if needed.

---

## Step-by-Step Usage

### 1. Prepare Your Data

- Data should be a pandas DataFrame with variables as columns.
- No missing values allowed (clean or impute before use).

### 2. Configure the Pipeline (Optional)

You can use the default configuration or customize it:

```python
from config import CausalConfig
config = CausalConfig()

# Single algorithm
config.default_algorithms = ['icalingam']  # or ['pc'], ['ges']

# Multiple algorithms (NEW!)
config.default_algorithms = ['pc', 'ges', 'icalingam']  # Will run all three

config.n_permutations = 200  # For more robust refutation
```

### 3. Initialize the Pipeline

```python
from CausalModules import EstimateEffect
estimator = EstimateEffect(df, config=config)
```

### 4. Run the Full Pipeline

#### Single Algorithm

```python
results = estimator.run_full_pipeline(
    treatment='CACNA1S',
    outcome='Group',
    algo='icalingam',           # or None to use config default
    refute_graph=True,          # Set False to skip graph refutation
    refute_estimate=True,       # Set False to skip estimate refutation
    visualize=True,             # Show all visualizations
    export_path='results.json', # Save results to file
    export_format='json'        # or 'pickle'
)
```

#### Multiple Algorithms

```python
# If config.default_algorithms has multiple algorithms, all will be run automatically
results = estimator.run_full_pipeline(
    treatment='CACNA1S',
    outcome='Group',
    refute_graph=True,
    refute_estimate=True,
    visualize=False,            # Set False to avoid too many plots
    export_path='results.json'  # Creates: results_pc.json, results_ges.json, results_icalingam.json
)
```

### 5. Access Results

#### Single Algorithm Result

```python
# results is a dictionary with pipeline outputs
effect_estimate = results['effect_estimate']
causal_graph = results['graph']
```

#### Multiple Algorithm Results

```python
# results is a dictionary with results for each algorithm
for algo, result in results.items():
    if result:  # Check if algorithm succeeded
        print(f"{algo}: Effect = {result['effect_estimate']}")
        print(f"{algo}: Graph nodes = {len(result['graph'].nodes())}")
```

### 6. Visualize Results (Manual Option)

```python
# For single algorithm
estimator.visualize_graph()
estimator.visualize_effect_estimate()
estimator.visualize_refutation()

# For multiple algorithms, you'd need to access individual estimators
# (The pipeline handles this automatically when visualize=True)
```

### 7. Export Results (Manual Option)

```python
# Single algorithm
estimator.export_results('results.json', format='json')

# Multiple algorithms create separate files automatically
# results_pc.json, results_ges.json, results_icalingam.json
```

### 8. Cleanup

```python
estimator.cleanup()
```

---

## Understanding the Pipeline Outputs

When you run the pipeline, you will see several key outputs in your terminal and in the exported files. Here’s what they mean:

### **1. Group Distribution**

```
Target variable 'Group' distribution:
0    50
1    50
Name: Group, dtype: int64
```

- **What it means:** This shows the count of each class in your outcome variable (e.g., 0 = Control, 1 = Preeclampsia). A balanced distribution is often desirable for causal analysis.

### **2. Effect Estimate**

```
Effect estimate: 0.234
```

- **What it means:** This is the estimated causal effect of the treatment variable (e.g., 'CACNA1S') on the outcome (e.g., 'Group'). The interpretation depends on your data and model (e.g., difference in probability, mean, or regression coefficient).

### **3. Graph Nodes and Edges**

```
Graph nodes: 11
Graph edges: 15
```

- **What it means:**
  - **Nodes:** Number of variables included in the discovered causal graph.
  - **Edges:** Number of directed relationships inferred between variables.
  - More edges can indicate a more complex model; fewer edges may indicate a sparser, possibly more interpretable model.

### **4. Exported Files**

```
results_20241201_143022/
├── results_pc.json
├── results_ges.json
└── results_icalingam.json
```

- **What it means:**
  - Each file contains the full pipeline results for one algorithm.
  - The files include the graph structure, effect estimate, estimand expression, and refutation results.
  - You can load these files in Python for further analysis or reporting.

---

## **Summary CSV Output**

After running the pipeline, a `summary.csv` file is generated in the results directory. This file provides a concise, algorithm-by-algorithm summary of key metrics for model quality and robustness.

### **CSV Columns Explained**

| Column Name                                  | Description                                                                                               |
| -------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| **Algorithm**                                | The name of the causal discovery algorithm used (e.g., `pc`, `ges`, `icalingam`).                         |
| **TPa (#)**                                  | Number of permutations (out of total) that lie in the same Markov equivalence class as the learned DAG.   |
| **TPa (total)**                              | Total number of permutations tested for Markov equivalence.                                               |
| **TPa (p-value)**                            | The p-value for the TPa test: probability that a random permutation is as informative as the learned DAG. |
| **LMC (#)**                                  | Number of Local Markov Condition (LMC) violations in the learned DAG.                                     |
| **LMC (total)**                              | Total number of LMCs tested.                                                                              |
| **LMC (p-value)**                            | The p-value for the LMC test: probability that a random permutation has as few or fewer LMC violations.   |
| **Placebo Refuter (p-value)**                | P-value from the Placebo Treatment Refuter (tests robustness by randomizing treatment).                   |
| **Random Common Cause Refuter (p-value)**    | P-value from the Random Common Cause Refuter (tests robustness by adding a random confounder).            |
| **Data Subsample Refuter (p-value)**         | P-value from the Data Subsample Refuter (tests robustness by subsampling the data).                       |
| **Placebo Refuter (new effect)**             | The new estimated effect after applying the Placebo Treatment Refuter.                                    |
| **Random Common Cause Refuter (new effect)** | The new estimated effect after applying the Random Common Cause Refuter.                                  |
| **Data Subsample Refuter (new effect)**      | The new estimated effect after applying the Data Subsample Refuter.                                       |

### **What Each Metric Means**

#### **Algorithm**

- The causal discovery method used to learn the graph structure from your data.

#### **TPa (Markov Equivalence Class Test)**

- **TPa (#):** How many of the random permutations of the graph structure are in the same Markov equivalence class as your learned DAG.
- **TPa (total):** The total number of permutations tested.
- **TPa (p-value):** The probability that a random permutation is as “informative” as your learned DAG. Lower values mean your DAG is more unique/informative.

#### **LMC (Local Markov Condition Test)**

- **LMC (#):** Number of local Markov conditions violated by your learned DAG (i.e., where the implied conditional independencies do not hold in the data).
- **LMC (total):** Total number of local Markov conditions tested.
- **LMC (p-value):** The probability that a random permutation has as few or fewer LMC violations as your learned DAG. Lower values mean your DAG fits the data’s conditional independencies better.

#### **Refuters**

These are robustness checks for the estimated causal effect:

- **Placebo Treatment Refuter:** Randomizes the treatment variable. If the effect estimate changes drastically or the p-value is low, your original estimate may not be robust.
- **Random Common Cause Refuter:** Adds a random confounder. If the effect estimate changes, your result may be sensitive to unobserved confounding.
- **Data Subsample Refuter:** Repeats the estimation on a random subsample of the data. Large changes suggest instability.

For each refuter:

- **(p-value):** The probability that the observed effect could be due to chance under the refuter’s scenario.
- **(new effect):** The new estimated effect after applying the refuter. Large deviations from the original effect suggest lack of robustness.

### **How to Use This CSV**

- **Compare algorithms:** See which algorithm produces the most robust and informative graph and effect estimate.
- **Diagnose issues:** High LMC violations or low p-values in refuters may indicate problems with the model or data.
- **Report results:** The CSV provides a concise, reproducible summary for publication or further analysis.

---

### **5. Summary Statistics**

```
📊 Summary:
  Successful algorithms: ['pc', 'ges', 'icalingam']
  Failed algorithms: []

📈 Effect Estimate Comparison:
  pc: 0.234
  ges: 0.198
  icalingam: 0.267
```

- **What it means:**
  - **Successful algorithms:** List of algorithms that completed without error.
  - **Failed algorithms:** List of algorithms that encountered an error (with error messages shown above).
  - **Effect Estimate Comparison:** Quick comparison of the estimated effects from each algorithm, useful for model selection or sensitivity analysis.

### **6. Warnings and Errors**

- If you see warnings or errors (e.g., about missing data, failed algorithms, or export issues), read the message for troubleshooting tips. The pipeline is designed to continue running other algorithms even if one fails.

### **7. Latency Outputs (Timing Information)**

```
[LATENCY] Causal graph discovery (pc): 2.13 seconds
[LATENCY] Graph refutation: 5.42 seconds
[LATENCY] Model creation: 0.01 seconds
[LATENCY] Effect identification: 0.02 seconds
[LATENCY] Effect estimation: 0.03 seconds
[LATENCY] Estimate refutation: 0.12 seconds
[LATENCY] Results export: 0.01 seconds
[LATENCY] Total pipeline time (pc): 7.72 seconds
[LATENCY] ges total time: 8.01 seconds
[LATENCY] Total time for all algorithms: 24.12 seconds
```

- **What it means:**
  - Each `[LATENCY]` line shows how long a particular step or the entire pipeline took to run (in seconds).
  - **Causal graph discovery:** Time to learn the causal structure from data for the specified algorithm.
  - **Graph refutation:** Time to statistically test and possibly update the graph.
  - **Model creation:** Time to instantiate the causal model.
  - **Effect identification:** Time to identify the estimand (causal query).
  - **Effect estimation:** Time to compute the causal effect.
  - **Estimate refutation:** Time to test the robustness of the effect estimate.
  - **Results export:** Time to save results to disk.
  - **Total pipeline time (algo):** Total time for the full pipeline for a single algorithm.
  - **Total time for all algorithms:** Total time spent running all algorithms in a multi-algorithm run.
- **Why it's useful:**
  - Helps you identify bottlenecks or slow steps in your analysis.
  - Useful for benchmarking, optimization, and resource planning.
  - Lets you compare the computational cost of different algorithms.

---

## Example: Full Workflow

### Single Algorithm

```python
from util import preproc
from config import CausalConfig
from CausalModules import EstimateEffect

# Load and preprocess data
betas_df = pd.read_csv('data/betas.csv')
pds_df = pd.read_csv('data/pds.csv')
df = preproc(betas_df, pds_df)

# Configure for single algorithm
config = CausalConfig()
config.default_algorithms = ['icalingam']

# Initialize pipeline
estimator = EstimateEffect(df, config=config)

# Run everything
results = estimator.run_full_pipeline(
    treatment='CACNA1S',
    outcome='Group',
    visualize=True,
    export_path='results.json'
)

# Access results
print(f"Effect estimate: {results['effect_estimate']}")
```

### Multiple Algorithms

```python
# Configure for multiple algorithms
config = CausalConfig()
config.default_algorithms = ['pc', 'ges', 'icalingam']

# Initialize pipeline
estimator = EstimateEffect(df, config=config)

# Run everything (all algorithms automatically)
results = estimator.run_full_pipeline(
    treatment='CACNA1S',
    outcome='Group',
    visualize=False,  # Avoid too many plots
    export_path='results.json'
)

# Compare results
for algo, result in results.items():
    if result:
        print(f"\n{algo}:")
        print(f"  Effect estimate: {result['effect_estimate']}")
        print(f"  Graph nodes: {len(result['graph'].nodes())}")
        print(f"  Graph edges: {len(result['graph'].edges())}")
    else:
        print(f"\n{algo}: Failed")

# Cleanup
estimator.cleanup()
```

---

## Method Reference

### `EstimateEffect` Methods

- `find_causal_graph(algo=None, pk=None)`: Discover causal graph. `algo` can be 'pc', 'ges', or 'icalingam'. `pk` is prior knowledge.
- `refute_cgm(n_perm=None, ...)`: Falsify/refute the graph. `n_perm` is number of permutations.
- `create_model(treatment, outcome)`: Create a DoWhy causal model.
- `identify_effect(method=None)`: Identify the causal effect (estimand).
- `estimate_effect(method_cat=None, ctrl_val=0, trtm_val=1)`: Estimate the effect.
- `refute_estimate(method_name=None, ...)`: Refute the effect estimate.
- `get_all_information()`: Get all results as a dictionary.
- `visualize_graph(title="Causal Graph")`: Visualize the current graph.
- `visualize_effect_estimate(title="Effect Estimate")`: Visualize the effect estimate.
- `visualize_refutation(title="Refutation Result")`: Visualize the refutation result.
- `export_results(filepath, format='json')`: Export results to file.
- `cleanup()`: Free memory.
- `run_full_pipeline(...)`: Run the entire pipeline in one call. **NEW**: Automatically runs multiple algorithms if specified in config.

### `CausalConfig` Parameters

- `default_algorithms`: List of algorithms to try (first is default for single runs, all are used for multi-algorithm runs).
- `default_estimation_methods`: List of estimation methods.
- `default_refutation_methods`: List of refutation methods.
- `n_permutations`: Number of permutations for refutation.
- `confidence_level`: Confidence level for estimates.
- `max_vars`: Max variables for graph learning (feature selection).

---

## Multi-Algorithm Features

### Automatic Multi-Algorithm Execution

When `config.default_algorithms` contains multiple algorithms, the pipeline automatically:

- Runs each algorithm separately
- Creates separate export files for each algorithm
- Handles failures gracefully (continues with other algorithms)
- Provides progress tracking and summary reports

### Export File Naming

For multiple algorithms with `export_path='results.json'`:

- `results_pc.json` - PC algorithm results
- `results_ges.json` - GES algorithm results
- `results_icalingam.json` - ICALiNGAM algorithm results

### Result Structure

```python
# Single algorithm
results = {
    'graph': nx.DiGraph,
    'effect_estimate': object,
    'estimand_expression': object,
    # ... other pipeline outputs
}

# Multiple algorithms
results = {
    'pc': { 'graph': nx.DiGraph, 'effect_estimate': object, ... },
    'ges': { 'graph': nx.DiGraph, 'effect_estimate': object, ... },
    'icalingam': { 'graph': nx.DiGraph, 'effect_estimate': object, ... }
}
```

---

## Pipeline Result Structure

The pipeline returns a dictionary with the following structure for each algorithm:

| Key                     | Type                     | Description                                                        |
| ----------------------- | ------------------------ | ------------------------------------------------------------------ |
| graph                   | networkx.DiGraph or None | The discovered causal graph, or None if not available              |
| graph_refutation_res    | object or None           | Result of graph falsification/refutation (may be EvaluationResult) |
| estimand_expression     | object or None           | The identified estimand (causal effect expression)                 |
| effect_estimate         | object or None           | The estimated causal effect                                        |
| estimate_refutation_res | list or None             | List of refuter results (one per refuter), or None if not run      |

If multiple algorithms are run, the top-level result is a dictionary mapping algorithm names to the above structure. If an algorithm fails, its value is None or a dict with all keys set to None.

This structure ensures robust downstream analysis and reporting.

---

## Troubleshooting & Tips

- **Missing values:** Clean or impute your data before using the pipeline.
- **Algorithm errors:** Ensure all dependencies are installed. Some algorithms require R and `Rscript`.
- **Serialization errors:** If exporting to JSON fails, use `pickle` format.
- **Performance:** For large datasets, increase memory and consider feature selection.
- **Multi-algorithm runs:** Set `visualize=False` to avoid too many plots when running multiple algorithms.
- **Custom algorithms:** Extend `CausalConfig` and add new methods to `EstimateEffect` as needed.

---

## Extending & Customizing the Pipeline

- Add new algorithms by extending `find_causal_graph` and updating `CausalConfig`.
- Add new estimation or refutation methods similarly.
- Customize logging by editing `logging_utils.py`.
- Add new visualizations in `visualization_utils.py`.
- Use the modular structure to swap or extend any component.

---

## Support

For questions, open an issue or contact sk@iomics.us.

---

## Running the Pipeline via CLI

You can now run the full causal pipeline directly from the command line using the `run_pipeline.py` script. This is the easiest way to get started and automate your analyses.

### Basic Usage

```sh
python run_pipeline.py --verbose
```

This will run the pipeline with default settings and print progress and results to your terminal.

### CLI Arguments

- `--data` : Path or URL to input data (CSV). Defaults to a sample dataset if not provided.
- `--treatment` : Name of the treatment variable. Default: `PIP3`
- `--outcome` : Name of the outcome variable. Default: `pmek`
- `--algorithms` : Comma-separated list of algorithms to use (e.g., `pc,ges`). Default: `pc`
- `--output` : Output directory for results. Default: `results_<timestamp>`
- `--n_permutations` : Number of permutations for statistical tests. Default: `30`
- `--confidence` : Confidence level (e.g., `0.95`). Default: `0.95`
- `--verbose` : Print progress and results to the console.

### Example: Custom Data and Settings

```sh
python run_pipeline.py --data data/mydata.csv --treatment MY_TREAT --outcome MY_OUTCOME --algorithms pc,ges --output results_test --n_permutations 10 --confidence 0.9 --verbose
```

### See All Options

```sh
python run_pipeline.py --help
```

This will print a full list of available arguments and their descriptions.

### Output

- Results and logs will be saved in the specified output directory.
- Key progress and results will be printed to the terminal if `--verbose` is set.

---
