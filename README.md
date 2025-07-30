# IOMICS Causal Pipeline (IoCP)

This repository contains the source code for the DoWhy-based pipeline to perform effect estimation using the following general methodology:

<img width="1000" height="135" alt="image" src="https://github.com/user-attachments/assets/3c2fc122-bcdb-4c5a-b6c2-49bb3b5ac780" />

1. Model a causal inference problem using assumptions → defining a causal graphical model
2. Refute the causal graph once discovered or given
3. Identify an expression for the causal effect under those assumptions
4. Estimate the effect using the estimand expression found in the previous step
5. Finally, verify the validity of the estimate using a variety of robustness checks.

### Motivation

To understand the causal effect of a (treatment) variable on an (outcome) variable, we need to first construct a causal graphical model that explains the underlying data generating process. In doing so, the aim is to evaluate interventions (manipulations of a variable) and ultimately estimate the effect of the treatment variable on the outcome.
It would be simple if we could just perform a randomized control trial (RCT) to control for confounding variables and thereby determine a measure of causality between a treatment and outcome. However, there are many domains where only observational data exists - data that already exists via records.

### Preliminary Code Setup

```bash
conda create --name iocp python=3.11
conda activate iocp
python -m pip install -r requirements.txt
```

### Source Code Functionality

`CausalModule.py` contains all the function calls required to run the pipeline.

A boiler-plate version to run the full pipeline for a given dataset is presented in `run_effect_estimation.py`. Feel free to make changes as needed.

Below is a table to showcase the function calls -- the name of the function, purpose, input, output:

