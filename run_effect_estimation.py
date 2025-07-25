from CausalModule import CausalModule
from utilities.visualization_utils import *
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def run_effect_estimation(data, discovery_algorithm, treatment_variable, outcome_variable, treatment_value=None, control_value=None):
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
    
    # Print the estimate
    print(f"Causal Estimate:", causal_module.estimate)
    
    # Return all information
    return causal_module.get_all_information()

def __main__():
    data_url = "https://raw.githubusercontent.com/FenTechSolutions/CausalDiscoveryToolbox/master/cdt/data/resources/cyto_full_data.csv"
    data = pd.read_csv(data_url) 
    discovery_algorithm = "icalingam" 
    treatment_variable = "PIP3"
    outcome_variable = "pmek"
    treatment_value = data["PIP3"].max()
    control_value = data["PIP3"].min()
    
    results = run_effect_estimation(
        data=data,
        discovery_algorithm=discovery_algorithm,
        treatment_variable=treatment_variable,
        outcome_variable=outcome_variable,
        treatment_value=treatment_value,
        control_value=control_value
    )
    
    print(results)
    
if __name__ == "__main__":
    __main__()