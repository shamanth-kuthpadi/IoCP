"""
WHAT: CausalModule.py is a module for performing effect estimation with causal discovery and inference.
WHY: It provides a structured way to discover causal relationships, estimate effects, and refute those estimates using various algorithms and methods.
ASSUMES: Requires pandas, causallearn, dowhy, and utilities for data manipulation and causal inference. Assumes input data is in a suitable format (e.g., pandas DataFrame).
FUTURE IMPROVEMENTS: Could include more advanced causal discovery algorithms, support for more complex prior knowledge, and enhanced error handling. *NOTE* If you can't think of anything put "See work packages"
VARIABLES:
- data: Input data as a pandas DataFrame.
- discovery_algorithm: Algorithm used for causal discovery (e.g., 'pc', 'ges', 'icalingam').
- treatment_variable: The variable representing the treatment in the causal analysis.
- outcome_variable: The variable representing the outcome in the causal analysis.
- treatment_value: The value of the treatment variable for effect estimation.
- control_value: The value of the treatment variable for control in effect estimation.
WHO: S.K.S 2025/07/28

PREREQUISITES:
- All required packages must be installed: pandas, causallearn, dowhy, pgmpy, networkx, sklearn
- Data must be in pandas DataFrame format with appropriate column names
- Treatment and outcome variables must exist as columns in the data
- For effect estimation, treatment_value and control_value should be specified
"""

# causal-learn imports
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.CDNOD import cdnod
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.PermutationBased.GRaSP import grasp
from causallearn.search.PermutationBased.BOSS import boss
from causallearn.search.FCMBased import lingam
from causallearn.utils.PDAG2DAG import pdag2dag
from causallearn.search.FCMBased.lingam.utils import make_dot
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.PCUtils.BackgroundKnowledgeOrientUtils import orient_by_background_knowledge

# dowhy imports
import dowhy.gcm.falsify
from dowhy.gcm.falsify import falsify_graph
from dowhy.gcm.falsify import apply_suggestions
from dowhy import CausalModel
from dowhy.gcm import ProbabilisticCausalModel
from dowhy.gcm import InvertibleStructuralCausalModel
from dowhy.gcm import fit
from dowhy.gcm import auto, interventional_samples, counterfactual_samples

# pgmpy imports
from pgmpy.base import DAG
from pgmpy.metrics import correlation_score
from sklearn.metrics import confusion_matrix

# utility imports
from utilities.utils import *

# https://stackoverflow.com/questions/79673823/dowhy-python-library-module-networkx-algorithms-has-no-attribute-d-separated
import networkx as nx
nx.algorithms.d_separated = nx.algorithms.d_separation.is_d_separator
nx.d_separated = nx.algorithms.d_separation.is_d_separator

import re
import os
import pickle

# This is for logging the pipeline intermediary outputs. 
# used AI to generate the logging code, so it might not be perfect.
import logging

# Remove all handlers associated with the root logger object (to avoid duplicate logs if re-imported)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
    
os.makedirs("outputs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("outputs/pipeline_output.txt", mode="w"),
        logging.StreamHandler()  # This sends logs to the console
    ]
)

class CausalModule:
    """
    CausalModule is a class that encapsulates the entire causal discovery and effect estimation pipeline.
    It allows users to input data, specify discovery algorithms, treatment and outcome variables, and perform
    causal discovery, effect estimation, and refutation of estimates.
    It also provides methods to handle prior knowledge and visualize results.
    
    PREREQUISITES:
    - Data must be provided during initialization or before calling discovery methods -- ensure also that the data is all numeric
    - Treatment and outcome variables must be specified before effect estimation
    - Treatment and control values should be set for effect estimation
    - All required packages (pandas, causallearn, dowhy, pgmpy, networkx, sklearn) must be installed
    """
    def __init__(self, 
                 data = None, 
                 discovery_algorithm = None, 
                 treatment_variable = None, 
                 outcome_variable = None,
                 treatment_value = None,
                 control_value = None):
        """
        Initializes the CausalModule with data and parameters for causal analysis.
        
        PREREQUISITES:
        - Data should be a pandas DataFrame with appropriate column names and numeric values -- this is important as some discovery algorithms may not work
        - Treatment and outcome variables should exist as columns in the data
        - Treatment and control values should be valid values for the treatment variable
        
        Parameters:
            - data: Input data as a pandas DataFrame.
            - discovery_algorithm: Algorithm used for causal discovery (e.g., 'pc', 'ges', 'icalingam').
            - treatment_variable: The variable representing the treatment in the causal analysis.
            - outcome_variable: The variable representing the outcome in the causal analysis.
            - treatment_value: The value of the treatment variable for effect estimation.
            - control_value: The value of the treatment variable for control in effect estimation.
        """
        
        self.data = data
        self.discovery_algorithm = discovery_algorithm
        self.treatment_variable = treatment_variable
        self.outcome_variable = outcome_variable
        self.treatment_value = treatment_value
        self.control_value = control_value
        
        
        self.graph = None
        self.graph_ref = None
        self.model = None
        self.estimand = None
        self.estimate = None
        self.est_ref = None
        self.graph_quality_score = None
        self.graph_quality_summary = None
        self.node_quality_score = None
        self.interventional_samples = None
        
        self.results = {}  # Initialize results dictionary to store outputs
        logging.info("CausalModule initialized with provided parameters.")
        
    # For now, the only prior knowledge that the prototype will allow is required/forbidden edges
    # pk must be of the type => {'required': [list of edges to require], 'forbidden': [list of edges to forbid]}
    def find_causal_graph(self, algo='pc', pk=None):
        """
        Finds the causal graph using the specified discovery algorithm.
        
        PREREQUISITES:
        - Data must be provided during initialization (self.data must not be None)
        - Data should be a pandas DataFrame with appropriate column names
        - If prior knowledge (pk) is provided, it must be a dictionary with 'required' and/or 'forbidden' keys
        
        Parameters:
            - algo: The discovery algorithm to use (default is 'pc'). Options: 'pc', 'ges', 'icalingam'.
            - pk: Prior knowledge in the form of required and forbidden edges. Format: {'required': [list of edges], 'forbidden': [list of edges]}.
        
        Returns:
            - The discovered causal graph as a networkx DiGraph.
        """

        if self.discovery_algorithm:
            algo = self.discovery_algorithm
        
        logging.info(f"Finding causal graph using {algo} algorithm")
        
        df = self.data.to_numpy()
        labels = list(self.data.columns)
        
        try:
            match algo:
                case 'pc':
                    cg = pc(data=df, show_progress=True, node_names=labels, verbose=False)
                    cg = pdag2dag(cg.G)
                    predicted_graph = genG_to_nx(cg, labels)
                    self.graph = predicted_graph
                case 'grasp':
                    cg = grasp(X=df, node_names=labels, verbose=False)
                    cg = pdag2dag(cg)
                    predicted_graph = genG_to_nx(cg, labels)
                    self.graph = predicted_graph
                case 'boss':
                    cg = boss(X=df, node_names=labels, verbose=False)
                    cg = pdag2dag(cg)
                    predicted_graph = genG_to_nx(cg, labels)
                    self.graph = predicted_graph
                case 'ges':
                    cg = ges(X=df, node_names=labels)
                    cg = pdag2dag(cg['G'])
                    predicted_graph = genG_to_nx(cg, labels)
                    self.graph = predicted_graph
                case 'icalingam':
                    model = lingam.ICALiNGAM()
                    model.fit(df)
                    pyd_lingam = make_dot(model.adjacency_matrix_, labels=labels)
                    pyd_lingam = pyd_lingam.pipe(format='dot').decode('utf-8')
                    pyd_lingam = (pyd_lingam,) = graph_from_dot_data(pyd_lingam)
                    dot_data_lingam = pyd_lingam.to_string()
                    pydot_graph_lingam = graph_from_dot_data(dot_data_lingam)[0]
                    predicted_graph = nx.drawing.nx_pydot.from_pydot(pydot_graph_lingam)
                    predicted_graph = nx.DiGraph(predicted_graph)
                    self.graph = predicted_graph
            
            if pk is not None:
                # ensuring that pk is indeed of the right type
                if not isinstance(pk, dict):
                    logging.info(f"Please ensure that the prior knowledge is of the right form")
                    raise
                # are there any edges to require
                if 'required' in pk.keys():
                    eb = pk['required']
                    self.graph.add_edges_from(eb)
                # are there any edges to remove
                if 'forbidden' in pk.keys():
                    eb = pk['forbidden']
                    self.graph.remove_edges_from(eb)
        
        except Exception as e:
            logging.error(f"Error in creating causal graph: {e}")
            raise

        return self.graph

    # What if user already has a graph they would like to input
    def input_causal_graph(self, graph):
        """
        Allows users to input their own causal graph as an alternative to automatic discovery.
        
        PREREQUISITES:
        - Graph must be a valid networkx DiGraph
        - Graph nodes should correspond to column names in the data
        - Treatment and outcome variables should be present in the graph
        
        Parameters:
            - graph: A networkx DiGraph representing the causal graph.
        
        Returns:
            - The input causal graph.
        """
        self.graph = graph

    def refute_cgm(self, n_perm=100, apply_sugst=True, show_plt=False):
        """
        Refutes the discovered causal graph using permutation tests.
        
        PREREQUISITES:
        - A causal graph must be available (call find_causal_graph or input_causal_graph first)
        - Data must be provided and accessible
        - Treatment and outcome variables must be specified
        
        Parameters:
            - n_perm: Number of permutations for the refutation test (default is 100).
            - apply_sugst: Whether to apply suggestions to the graph after refutation (default is True).
            - show_plt: Whether to show the plot of the refutation results (default is False).
        
        Returns:
            - The refuted causal graph.
        """
        
        indep_test = gcm
        cond_indep_test = gcm
        
        logging.info("Refuting the discovered/given causal graph")
        
        try:
            result = falsify_graph(self.graph, self.data, n_permutations=n_perm,
                                  independence_test=indep_test,
                                  conditional_independence_test=cond_indep_test, plot_histogram=show_plt)
            
            self.graph_ref = result
            
            if apply_sugst:
                self.graph = apply_suggestions(self.graph, result)
            
        except Exception as e:
            logging.error(f"Error in refuting graph: {e}")
            raise

        return self.graph
    
    def create_model(self):
        """
        Creates a DoWhy causal model from the discovered or given causal graph.
        
        PREREQUISITES:
        - A causal graph must be available (call find_causal_graph or input_causal_graph first)
        - Data must be provided and accessible
        - Treatment and outcome variables must be specified
        - Graph must be a valid networkx DiGraph compatible with DoWhy
        
        Returns:
            - The causal model as a dowhy CausalModel object.
        """
        
        logging.info("Creating a causal model from the discovered/given causal graph")
        
        model_est = CausalModel(
                data=self.data,
                treatment=self.treatment_variable,
                outcome=self.outcome_variable,
                graph=self.graph
            )
        self.model = model_est
        return self.model

    def identify_effect(self, method=None):
        """
        Identifies the effect of the treatment on the outcome variable using the causal model.
        
        PREREQUISITES:
        - A causal model must be created (call create_model first)
        - Treatment and outcome variables must be specified
        - The causal graph must have a valid path from treatment to outcome
        
        Parameters:
            - method: Method to use for effect identification (default is None, which uses the default method set by DoWhy).
                     Options: 'maximal-adjustment', 'minimal-adjustment', 'exhaustive-search', 'default'
        
        Returns:
            - The identified estimand as a dowhy IdentifiedEstimand object.
        """
        
        logging.info("Identifying the effect estimand of the treatment on the outcome variable")
        
        try:
            if method is None:
                identified_estimand = self.model.identify_effect()
            else:
                identified_estimand = self.model.identify_effect(method=method)

            self.estimand = identified_estimand
            # Add logging if estimand is None or not identified
            if self.estimand is None or not hasattr(self.estimand, 'estimand_type'):
                logging.warning("Warning: Could not identify a valid estimand from the discovered causal graph. Please check the graph structure or variable selection.")
        except Exception as e:
            logging.error(f"Error in identifying effect: {e}")
            raise

        logging.info("Note that you can also use other methods for the identification process. Below are method descriptions taken directly from DoWhy's documentation")
        logging.info("maximal-adjustment: returns the maximal set that satisfies the backdoor criterion. This is usually the fastest way to find a valid backdoor set, but the set may contain many superfluous variables.")
        logging.info("minimal-adjustment: returns the set with minimal number of variables that satisfies the backdoor criterion. This may take longer to execute, and sometimes may not return any backdoor set within the maximum number of iterations.")
        logging.info("exhaustive-search: returns all valid backdoor sets. This can take a while to run for large graphs.")
        logging.info("default: This is a good mix of minimal and maximal adjustment. It starts with maximal adjustment which is usually fast. It then runs minimal adjustment and returns the set having the smallest number of variables.")
        return self.estimand
    
    def estimate_effect(self, method_cat='backdoor.linear_regression', ctrl_val=None, trtm_val=None):
        """
        Estimates the effect of the treatment on the outcome variable using the identified estimand.
        
        PREREQUISITES:
        - An estimand must be identified (call identify_effect first)
        - Treatment and control values must be specified (either during initialization or as parameters)
        - The causal model must be valid and accessible
        
        Parameters:
            - method_cat: The method category to use for effect estimation (default is 'backdoor.linear_regression').
            - ctrl_val: The control value for the treatment variable (default is None, which uses the control value set during initialization).
            - trtm_val: The treatment value for the treatment variable (default is None, which uses the treatment value set during initialization).
        
        Returns:
            - The estimated effect as a dowhy EffectEstimate object.
        """
        
        logging.info("Estimating the effect of the treatment on the outcome variable")
        
        if ctrl_val is None:
            ctrl_val = self.control_value
        if trtm_val is None:
            trtm_val = self.treatment_value
                    
        estimate = None
        try:
            match method_cat:
                case 'backdoor.linear_regression':
                    estimate = self.model.estimate_effect(self.estimand,
                                                  method_name=method_cat,
                                                  control_value=ctrl_val,
                                                  treatment_value=trtm_val,
                                                  confidence_intervals=True,
                                                  test_significance=True)
                # there are other estimation methods that I can add later on, however parameter space will increase immensely
            self.estimate = estimate
        except Exception as e:
            logging.error(f"Error in estimating the effect: {e}. Assuming that all previous steps in the pipeline are completed, the most probable reason for this error is that an estimand wasn't identified (i.e. either change the identify_effect parameter for method or discover a new causal graph).")
            raise
        
        logging.info("Note that it is ok for your treatment to be a continuous variable, DoWhy automatically discretizes at the backend.")
        return self.estimate
    
    # should give a warning to users if the estimate is to be refuted

    def refute_estimate(self,  method_name="ALL", placebo_type='permute', subset_fraction=0.9):
        """
        Refutes the estimated effect of the treatment on the outcome variable using various methods.
        
        PREREQUISITES:
        - An effect must be estimated (call estimate_effect first)
        - The causal model must be valid and accessible
        - Data must be sufficient for the refutation methods
        
        Parameters:
            - method_name: The method to use for refutation (default is "ALL", which applies all methods).
                         Options: "placebo_treatment_refuter", "random_common_cause", "data_subset_refuter", "ALL"
            - placebo_type: The type of placebo treatment to use for refutation (default is 'permute').
            - subset_fraction: The fraction of the data to use for the data subset refuter (default is 0.9).
        
        Returns:
            - The refuted estimate as a dowhy RefutationResult object or a list of results if multiple methods are applied.
        """
        
        logging.info("Refuting the estimated effect of the treatment on the outcome variable")
        
        ref = None
        
        def placebo_treatment_refuter(model):
            return model.refute_estimate(
                self.estimand,
                self.estimate,
                method_name="placebo_treatment_refuter",
                placebo_type=placebo_type
            )
        def random_common_cause_refuter(model):
            return model.refute_estimate(
                self.estimand,
                self.estimate,
                method_name="random_common_cause"
            )
        def data_subset_refuter(model):
            return model.refute_estimate(
                self.estimand,
                self.estimate,
                method_name="data_subset_refuter",
                subset_fraction=subset_fraction
            )
        
        try:
            match method_name:
                case "placebo_treatment_refuter":
                    ref = placebo_treatment_refuter(self.model)
                
                case "random_common_cause":
                    ref = random_common_cause_refuter(self.model)

                case "data_subset_refuter":
                    ref = data_subset_refuter(self.model)
                
                case "ALL":
                    ref_placebo = placebo_treatment_refuter(self.model)
                    ref_rand_cause = random_common_cause_refuter(self.model)
                    ref_subset = data_subset_refuter(self.model)
                    ref = [ref_placebo, ref_rand_cause, ref_subset]
                    
            if not isinstance(ref, list) and ref.refutation_result['is_statistically_significant']:
                logging.warning("Please make sure to take a revisit the pipeline as the refutation p-val is significant: ", ref.refutation_result['p_value'])
    
            self.est_ref = ref
        
        except Exception as e:
            logging.error(f"Error in refuting estimate: {e}")
            raise
            
        return self.est_ref
    
    def see_graph_properties(self):
        """
        Extracts properties from the causal graph.
        
        PREREQUISITES:
        - A causal graph must be available (call find_causal_graph or input_causal_graph first)
        - Treatment and outcome variables must be specified
        
        Returns:
            - A dictionary containing various properties of the graph including:
              - num_nodes: Number of nodes in the graph
              - num_edges: Number of edges in the graph
              - edge_weights: Dictionary of edge weights
              - all_paths: All paths from treatment to outcome
              - treatment_mb: Markov blanket of treatment variable
              - outcome_mb: Markov blanket of outcome variable
        """
        graph = self.graph
        treatment = self.treatment_variable
        outcome = self.outcome_variable
        
        metrics = {}
        
        G = DAG(graph)
        # number of nodes
        logging.info("==========================================")
        metrics['num_nodes'] = len(G)
        logging.info(f"Number of nodes: {metrics['num_nodes']}")
        # number of edges
        logging.info("==========================================")
        metrics['num_edges'] = G.number_of_edges()
        logging.info(f"Number of edges: {metrics['num_edges']}")
        # edge weights
        logging.info("==========================================")
        metrics['edge_weights'] = {f"{u}->{v}": data.get('label', 1) for u, v, data in G.edges(data=True)}
        for u, v, data in G.edges(data=True):
            logging.info(f"Edge: {u} -> {v}, Weight: {data.get('label', 1)}")
        # paths from treatment to outcome
        logging.info("==========================================")
        metrics['all_paths'] = list(nx.all_simple_paths(G, source=treatment, target=outcome))
        logging.info(f"Paths from {treatment} [treatment] to {outcome} [outcome]: {len(metrics['all_paths'])}")
        for path in metrics['all_paths']:
            logging.info(" -> ".join(path))
        # Markov blanket of treatment and outcome
        logging.info("==========================================")
        metrics['treatment_mb'] = G.get_markov_blanket(treatment)
        logging.info(f"Markov blanket of {treatment}: {metrics['treatment_mb']}")
        metrics['outcome_mb'] = G.get_markov_blanket(outcome)
        logging.info(f"Markov blanket of {outcome}: {metrics['outcome_mb']}")
        
        return metrics
    
    def see_graph(self):
        """
        Visualizes the causal graph.
        
        PREREQUISITES:
        - A causal graph must be available (call find_causal_graph or input_causal_graph first)
        
        Returns:
            - None (displays the graph visualization)
        """
        if self.graph is not None:
            disp_graph_nx(self.graph)
        else:
            logging.warning("No causal graph available to visualize.")
        
    def _extract_graph_refutation_metrics(self, graph_ref_str):
        """
        Extracts metrics from the graph refutation result string.
        
        PREREQUISITES:
        - Graph refutation must be performed (call refute_cgm first)
        - Graph refutation result must be available
        
        Parameters:
            - graph_ref_str: The graph refutation result string.
        
        Returns:
            - A tuple containing the number of informative TPA, total TPA, p-value for TPA, 
              number of violated LMCs, total LMCs, and p-value for LMCs.
        """
        if not isinstance(graph_ref_str, str):
            graph_ref_str = str(graph_ref_str)
            
        tpa_match = re.search(r"informative because (\d+) / (\d+).*?\(p-value: ([0-9.]+)\)", graph_ref_str, re.DOTALL)
        lmc_match = re.search(r"violates (\d+)/(\d+) LMCs.*?\(p-value: ([0-9.]+)\)", graph_ref_str, re.DOTALL)
        tpa_num, tpa_total, tpa_p = (tpa_match.group(1), tpa_match.group(2), tpa_match.group(3)) if tpa_match else (None, None, None)
        lmc_num, lmc_total, lmc_p = (lmc_match.group(1), lmc_match.group(2), lmc_match.group(3)) if lmc_match else (None, None, None)
        return tpa_num, tpa_total, tpa_p, lmc_num, lmc_total, lmc_p
    
    def _extract_graph_quality_score(self, graph, data, test='pearsonr', significance_level=0.05, score=confusion_matrix):
        """
        Extracts the graph quality score based on the specified test and significance level.
        This function directly uses the function from pgmpy.metrics to calculate the correlation score.
        
        PREREQUISITES:
        - A causal graph must be available
        - Data must be provided and accessible
        - Graph must be compatible with pgmpy DAG format
        
        Parameters:
            - graph: The causal graph as a networkx DiGraph.
            - data: The data used for testing the graph quality.
            - test: The statistical test to use (default is 'pearsonr').
            - significance_level: The significance level for the test (default is 0.05).
            - score: The scoring function to use (default is confusion_matrix).
        
        Returns:
            - The graph quality score.
        """
        # Implement the logic to calculate the graph quality score
        # This is a placeholder implementation
        pggraph = DAG(graph)

        results = correlation_score(pggraph, data, test, significance_level, score, return_summary=False)
        summary = correlation_score(pggraph, data, test, significance_level, score, return_summary=True)
        
        self.graph_quality_score = results
        self.graph_quality_summary = summary
        
        return results
    
    def see_graph_quality_score(self):
        """
        Shows the graph quality score/measure.
        
        PREREQUISITES:
        - A causal graph must be available (call find_causal_graph or input_causal_graph first)
        - Data must be provided and accessible
        
        Returns:
            - None (logs the graph quality score)
        """
        if self.graph is not None:
            score = self._extract_graph_quality_score(self.graph, self.data)
            logging.info(f"Graph Quality Score: {score}")
        else:
            logging.warning("No causal graph available to visualize quality score.")
    
    def see_graph_refutation(self):
        """
        Shows the graph refutation results.
        
        PREREQUISITES:
        - Graph refutation must be performed (call refute_cgm first)
        - Graph refutation result must be available
        
        Returns:
            - None (logs the graph refutation metrics)
        """
        if self.graph_ref is not None:
            tpa_num, tpa_total, tpa_p, lmc_num, lmc_total, lmc_p = self._extract_graph_refutation_metrics(self.graph_ref)
            logging.info(f"Graph refutation metrics: TPA: {tpa_num}/{tpa_total} (p-value: {tpa_p}), LMC: {lmc_num}/{lmc_total} (p-value: {lmc_p})")
        else:
            logging.warning("No graph refutation available to see.")
    
    def _extract_refuter_metrics(self, refuter_result):
        """
        Extracts metrics from the refuter result string.
        
        PREREQUISITES:
        - Estimate refutation must be performed (call refute_estimate first)
        - Refuter result must be available
        
        Parameters:
            - refuter_result: The refuter result string.
        
        Returns:
            - A tuple containing the p-value and new effect.
        """
        if not refuter_result:
            return None, None
        if not isinstance(refuter_result, str):
            refuter_result = str(refuter_result)

        # p value
        pval_match = re.search(r"p value:([0-9.eE+-]+)", refuter_result)
        pval = pval_match.group(1).strip() if pval_match else None
        # new effect
        neweff_match = re.search(r"New effect:([0-9.eE+-]+)", refuter_result)
        neweff = neweff_match.group(1).strip() if neweff_match else None
        return pval, neweff 

    def see_estimate_refutation(self):
        """
        Shows the estimate refutation results.
        
        PREREQUISITES:
        - Estimate refutation must be performed (call refute_estimate first)
        - Refuter result must be available
        
        Returns:
            - None (logs the estimate refutation metrics)
        """
        if self.est_ref is not None:
            if isinstance(self.est_ref, list):
                for ref in self.est_ref:
                    pval, neweff = self._extract_refuter_metrics(ref)
                    logging.info(f"Refutation result: p-value: {pval}, New effect: {neweff}")
            else:
                pval, neweff = self._extract_refuter_metrics(self.est_ref)
                logging.info(f"Refutation result: p-value: {pval}, New effect: {neweff}")
        else:
            logging.warning("No estimate refutation available to see.")
    
    def _extract_effect_estimation(self, estimate_obj):
        """
        Extracts the effect estimation metrics from the estimate object.
        
        PREREQUISITES:
        - Effect estimation must be performed (call estimate_effect first)
        - Estimate object must be available
        
        Parameters:
            - estimate_obj: The effect estimate object from DoWhy.
        
        Returns:
            - A dictionary containing the effect estimate metrics.
        """
        estimate_metrics = {
                'Effect Estimate': estimate_obj.value,
                'Realized Estimand Expression': estimate_obj.realized_estimand_expr,
                'Treatment Value': estimate_obj.treatment_value,
                'Control Value': estimate_obj.control_value,
            }
        
        return estimate_metrics
    
    def see_effect_estimation(self):
        """
        Visualizes the effect estimation results.
        
        PREREQUISITES:
        - Effect estimation must be performed (call estimate_effect first)
        - Estimate object must be available
        
        Returns:
            - None (logs the effect estimation metrics)
        """
        if self.estimate is not None:
            estimate_metrics = self._extract_effect_estimation(self.estimate)
            logging.info(f"Effect Estimate: {estimate_metrics['Effect Estimate']}")
            logging.info(f"Realized Estimand Expression: {estimate_metrics['Realized Estimand Expression']}")
            logging.info(f"Treatment Value: {estimate_metrics['Treatment Value']}")
            logging.info(f"Control Value: {estimate_metrics['Control Value']}")
        else:
            logging.warning("No effect estimation available to see.")
    
    def _extract_node_quality_score(self, node_name):
        """
        Extracts the quality for a selected node/feature in the causal graph.
        
        PREREQUISITES:
        - Graph quality score must be calculated (call see_graph_quality_score first)
        - Node name must exist in the causal graph
        - Graph quality summary must be available
        
        Parameters:
            - node_name: String representing the name/label of a node in the causal graph.
        
        Returns:
            - A float that represents the fraction of statistically uncorroborated relations in the graph.
        """
        summary = self.graph_quality_summary
        filtered_summary = summary[(summary['var1'] == node_name) | (summary['var2'] == node_name)]
        count_mismatches = (filtered_summary['stat_test'] != filtered_summary['d_connected']).sum()
        
        self.node_quality_score = count_mismatches / filtered_summary.shape[0]
        
        return self.node_quality_score
    
    def see_node_quality_score(self, node_name):
        """
        Shows the node quality score.
        
        PREREQUISITES:
        - Graph quality score must be calculated (call see_graph_quality_score first)
        - Node name must exist in the causal graph
        - Node name must be provided as parameter
        
        Parameters:
            - node_name: String representing the name/label of a node in the causal graph.
        
        Returns:
            - None (logs the node quality score)
        """
        if self.node_quality_score is not None:
            score = self._extract_node_quality_score(node_name)
            logging.info(f"Node Quality Score: {score}")
        else:
            logging.warning("No node quality score to see.") 
            
    # https://www.pywhy.org/dowhy/v0.11/user_guide/causal_tasks/what_if/interventions.html    
    def simulate_intervention(self, variable_to_intervene_dict, num_samples_to_draw=100):
        """
        Simulates an intervention on a specified variable and returns samples from the interventional distribution.
        This is also synonymous to a classifier in the case that all features (besides the target/outcome feature) are intervened on.
        
        PREREQUISITES:
        - A causal graph must be available (call find_causal_graph or input_causal_graph first)
        - Data must be provided and accessible
        - Variable to intervene on must exist in the causal graph
        - Intervention value must be appropriate for the variable type
        
        Parameters:
            - variable_to_intervene: The variable(s) to intervene on, should be a dict that contains variable to value mapping {'v_name': lambda x: #value} (default is None, which uses the treatment variable).
            - num_samples_to_draw: The number of samples to draw from the interventional distribution (default is 100).
        
        Returns:
            - Samples from the interventional distribution as a pandas DataFrame.
        """
        causal_model = ProbabilisticCausalModel(self.graph)
        auto.assign_causal_mechanisms(causal_model, self.data)
        fit(causal_model, self.data)
        samples = interventional_samples(
            causal_model,
            variable_to_intervene_dict,
            num_samples_to_draw=num_samples_to_draw
        )
        
        self.interventional_samples = samples
        
        return samples
        

    def store_results(self, dir_path='outputs/results'):
        """
        Stores various causal inference outputs as a Python object
        in the instance and also saves them to CSV files.
        
        PREREQUISITES:
        - At least one of the following must be available:
          - Causal graph (call find_causal_graph or input_causal_graph first)
          - Graph refutation results (call refute_cgm first)
          - Effect estimation results (call estimate_effect first)
          - Estimate refutation results (call refute_estimate first)
          - Interventional samples (call simulate_intervention first)
        - Directory path must be writable
        
        Parameters:
            - dir_path: Directory to save the CSV files (default is 'outputs/results').
        
        Returns:
            - None (saves results to files and stores in self.results)
        """
        os.makedirs(dir_path, exist_ok=True)
        self.results = {}  # Reset or initialize

        # Graph properties
        if self.graph is not None:
            metrics = self.see_graph_properties()
            self.results['graph_properties'] = metrics
            pd.DataFrame([metrics]).to_csv(os.path.join(dir_path, 'graph_properties.csv'), index=False)
            logging.info(f"Graph properties saved to {os.path.join(dir_path, 'graph_properties.csv')}")
        else:
            logging.warning("No causal graph available to save properties.")

        # Graph refutation metrics
        if self.graph_ref is not None:
            tpa_num, tpa_total, tpa_p, lmc_num, lmc_total, lmc_p = self._extract_graph_refutation_metrics(self.graph_ref)
            ref_metrics = {
                'TPA': f"{tpa_num}/{tpa_total}",
                'TPA p-value': tpa_p,
                'LMC': f"{lmc_num}/{lmc_total}",
                'LMC p-value': lmc_p
            }
            self.results['graph_refutation'] = ref_metrics
            pd.DataFrame([ref_metrics]).to_csv(os.path.join(dir_path, 'graph_refutation.csv'), index=False)
            logging.info(f"Graph refutation metrics saved to {os.path.join(dir_path, 'graph_refutation.csv')}")

        # Graph quality score
        if self.graph is not None:
            self.results['node_quality_score'] = self.node_quality_score
            self.results['graph_quality_score'] = self.graph_quality_score
            self.results['graph_quality_summary'] = self.graph_quality_summary
            pd.DataFrame({'Graph Quality Score': [self.graph_quality_score]}).to_csv(os.path.join(dir_path, 'graph_quality_score.csv'), index=False)
            logging.info(f"Graph quality score saved to {os.path.join(dir_path, 'graph_quality_score.csv')}")
            pd.DataFrame({'Graph Quality Summary': [self.graph_quality_summary]}).to_csv(os.path.join(dir_path, 'graph_quality_summary.csv'), index=False)
            logging.info(f"Graph quality summary saved to {os.path.join(dir_path, 'graph_quality_summary.csv')}")            
            pd.DataFrame({'Node Quality Score': [self.node_quality_score]}).to_csv(os.path.join(dir_path, 'node_quality_score.csv'), index=False)
            logging.info(f"Node quality score saved to {os.path.join(dir_path, 'node_quality_score.csv')}")

        # Effect estimate
        if self.estimate is not None:
            estimate_metrics = self._extract_effect_estimation(self.estimate)
            self.results['effect_estimate'] = estimate_metrics
            pd.DataFrame([estimate_metrics]).to_csv(os.path.join(dir_path, 'effect_estimate.csv'), index=False)
            logging.info(f"Effect estimate saved to {os.path.join(dir_path, 'effect_estimate.csv')}")

        # Estimate refutation metrics
        if self.est_ref is not None:
            ref_results = []
            if isinstance(self.est_ref, list):
                for ref in self.est_ref:
                    pval, neweff = self._extract_refuter_metrics(ref)
                    ref_results.append({'p-value': pval, 'New effect': neweff})
            else:
                pval, neweff = self._extract_refuter_metrics(self.est_ref)
                ref_results = [{'p-value': pval, 'New effect': neweff}]
            self.results['estimate_refutation'] = ref_results
            pd.DataFrame(ref_results).to_csv(os.path.join(dir_path, 'estimate_refutation.csv'), index=False)
            logging.info(f"Estimate refutation metrics saved to {os.path.join(dir_path, 'estimate_refutation.csv')}")
        
        if self.interventional_samples is not None:
            self.results['interventional_samples'] = self.interventional_samples
            self.interventional_samples.to_csv(os.path.join(dir_path, 'interventional_samples.csv'), index=False)
            logging.info(f"Interventional samples saved to {os.path.join(dir_path, 'interventional_samples.csv')}")            

