"""
WHAT: This script runs effect estimation using the CausalModule from CausalModule.py
WHY: A straightforward script that compiles all the necessary steps to run effect estimation.
ASSUMES: The CausalModule is implemented correctly and the necessary libraries are installed.
FUTURE IMPROVEMENTS: Allowing for command line arguments to specify the necessary parameters
VARIABLES:
- data: Pandas DataFrame containing the dataset.
- discovery_algorithm: Causal discovery algorithm to discover the causal graph.
- treatment_variable: The variable to be treated.
- outcome_variable: The outcome variable to be measured.
- treatment_value: The value of the treatment variable for the treatment group.
- control_value: The value of the treatment variable for the control group.
WHO: S.K.S 2025/08/01
"""


from CausalModule import CausalModule
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def run_effect_estimation(data, discovery_algorithm, treatment_variable, outcome_variable, treatment_value=None, control_value=None):
    
    """
    Function to run effect estimation using the CausalModule.
    :param data: Pandas DataFrame containing the dataset.
    :param discovery_algorithm: Causal discovery algorithm to discover the causal graph.
    :param treatment_variable: The variable to be treated.
    :param outcome_variable: The outcome variable to be measured.
    :param treatment_value: The value of the treatment variable for the treatment group.
    :param control_value: The value of the treatment variable for the control group.
    :return: All information from the CausalModule.
    """
    
    # Initialize the CausalModule with the provided parameters
    causal_module = CausalModule(
        data=data,
        discovery_algorithm=discovery_algorithm,
        treatment_variable=treatment_variable,
        outcome_variable=outcome_variable,
        treatment_value=treatment_value,
        control_value=control_value
    )
    
    # Find the causal graph
    causal_module.find_causal_graph()
    # Refute the causal graph
    causal_module.refute_cgm(n_perm=10)
    # Create a causal graph model
    causal_module.create_model()
    # Identify the estimand
    causal_module.identify_effect()
    # Estimate the effect
    causal_module.estimate_effect()
    # Refute the estimate
    causal_module.refute_estimate()
    # Visualize the causal graph
    causal_module.see_graph()
    # Getting graph metrics
    causal_module.see_graph_properties()
    # Getting graph refutation metrics
    causal_module.see_graph_refutation()
    # Getting effect estimation metrics
    causal_module.see_effect_estimation()
    # Getting effect estimation refutation metrics
    causal_module.see_estimate_refutation()
    # Save results to CSV (default directory is 'outputs/results')
    causal_module.save_into_csv()
    
def __main__():
    data_url = "https://raw.githubusercontent.com/FenTechSolutions/CausalDiscoveryToolbox/master/cdt/data/resources/cyto_full_data.csv"
    data = pd.read_csv(data_url) 
    discovery_algorithm = "icalingam" 
    treatment_variable = "PIP3"
    outcome_variable = "pmek"
    treatment_value = data["PIP3"].quantile(0.75)
    control_value = data["PIP3"].min()
    
    results = run_effect_estimation(
        data=data,
        discovery_algorithm=discovery_algorithm,
        treatment_variable=treatment_variable,
        outcome_variable=outcome_variable,
        treatment_value=treatment_value,
        control_value=control_value
    )
        
if __name__ == "__main__":
    __main__()