# MOCP
Multi Omics Causal Pipeline (MOCP) for modeling and inference


## Streamlining DoWhy for Effect Estimation

DoWhy suggests that we perform effect estimation tasks using the following pipeline:

1. Establish a causal graph that encapsulates assumptions about given data
2. Identify possible causal questions (estimands) that we can find an estimate for
3. Estimate the identified estimand
4. Refute the estimate to check for validity and sensitivity

While I will not go into the motivation or reasoning behind this pipeline, I will modify the pipeline to something like this:

1. Establish a causal graph that encapsulates assumptions about given data
2. <span style="color:blue">Refute the causal graph using statistical methods to check if the conditional independence statements dictated by our model align with the data.</span>
3. Identify possible causal questions (estimands) that we can find an estimate for
4. Estimate the identified estimand
5. <span style="color:blue">Refute the estimate to check for validity and sensitivity</span>

## **EstimateEffect Class Documentation**

The `EstimateEffect` class is designed to estimate causal effects from a dataset by leveraging various causal discovery algorithms such as PC, GES, and LiNGAM. The class includes methods to discover causal graphs, falsify graphs, and estimate causal effects.

Before using the class, ensure you have the following dependencies installed:

- `causallearn`
- `dowhy`
- `pandas` (assuming the data is passed in as a pandas DataFrame)

You can install these dependencies using pip (ensure that you are using a python virtual environment with a python version == 3.10.14):

```bash
pip install causallearn dowhy pandas
```

### Attributes:
* `data`: The dataset (typically a pandas DataFrame) on which causal discovery and effect estimation will be performed.
* `graph`: The discovered causal graph.
* `graph_ref`: A reference to the causal graph, possibly with modifications.
* `model`: The dowhy.CausalModel object for estimating causal effects.
* `estimand`: The causal estimand, which represents the causal query.
* `estimate`: The estimated causal effect.
* `est_ref`: A reference to the estimated effect, potentially after adjustments.

### Initialization:

```python
estimator = EstimateEffect(data)
```

* `data`: A pandas DataFrame that contains the dataset you want to analyze.

### Methods:

`find_causal_graph(algo='pc', pk=None)`

This method discovers the causal graph from the dataset using a specified algorithm.

* Parameters:

    * `algo`: The algorithm to use for causal discovery. Options include 'pc', 'ges', and 'lingam'. The default is 'pc'.
    * `pk`: Prior knowledge, which can include required or forbidden edges in the graph. This is optional and should be passed as a dictionary:

```python
pk = {
    'required': [('A', 'B'), ('C', 'D')],
    'forbidden': [('E', 'F')]
}
```

* Returns: The discovered causal graph, which is stored in the graph attribute.

* Usage Example:

```python
causal_graph = estimator.find_causal_graph(algo='ges', pk={'required': [('X', 'Y')]})
```

---

`refute_cgm(self, n_perm=100, indep_test=gcm, cond_indep_test=gcm, apply_sugst=True, show_plt=False)`

This method refutes the previously discovered causal graphical model and applies suggestions that are returned by the algorithm (if it is set to true).

* Parameters:
   * `n_perm`: number of graphs to randomly permute
   * `indep_test`: for now, just leave this at default (used to perform independent tests between two random variables within our model)
   * `cond_indep_test`: for now, just leave this at default (used to perform conditional independent tests for a set of random variables within our model)
   * `apply_sugst`: the falsification procedure performed by DoWhy returns a set of suggestions, the parameter is a boolean indicating if we would like to apply those suggestions or not 
   * `show_plt`: once the falsification is done, this boolean checks if we would like to present a graph showcasing the p-val

* Returns: a new graph that either has the suggestions applied or not

* Usage Example:
```python
rcgm = estimator.refute_cgm(n_perm=100, show_plt=True)
```

---

`create_model(self, treatment, outcome)`

Once we have a statistically corraborated causal graph, we need to represent in a way that DoWhy can understand. Thus, the purpose of this method is to create a DoWhy-interpretable model.

* Parameters:
   * `treatment`: the variable that we will be intervening on
   * `outcome`: the variable that we will checking to see if the intervention had any effect on

* Returns: a CausalModel

*  Usage Example:
```python
model = create_model(treatment='pka', outcome='pip2')
```

---

`identify_effect(self)`

This method simply identifies an estimand expression from our causal model.

---
