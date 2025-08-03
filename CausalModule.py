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
"""

# causal-learn imports
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.FCMBased import lingam
from causallearn.utils.PDAG2DAG import pdag2dag
from causallearn.search.FCMBased.lingam.utils import make_dot

# dowhy imports
import dowhy.gcm.falsify
from dowhy.gcm.falsify import falsify_graph
from dowhy.gcm.falsify import apply_suggestions
from dowhy import CausalModel

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

# This is for logging the pipeline intermediary outputs. 
# used AI to generate the logging code, so it might not be perfect.
import logging

# Remove all handlers associated with the root logger object (to avoid duplicate logs if re-imported)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("pipeline_output.txt", mode="w"),
        logging.StreamHandler()  # This sends logs to the console
    ]
)


class CausalModule:
    """
    CausalModule is a class that encapsulates the entire causal discovery and effect estimation pipeline.
    It allows users to input data, specify discovery algorithms, treatment and outcome variables, and perform
    causal discovery, effect estimation, and refutation of estimates.
    It also provides methods to handle prior knowledge and visualize results.
    """
    def __init__(self, 
                 data = None, 
                 discovery_algorithm = None, 
                 treatment_variable = None, 
                 outcome_variable = None,
                 treatment_value = None,
                 control_value = None):
        """
        Initialization parameters:
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
    
    # For now, the only prior knowledge that the prototype will allow is required/forbidden edges
    # pk must be of the type => {'required': [list of edges to require], 'forbidden': [list of edges to forbid]}
    def find_causal_graph(self, algo='pc', pk=None):
        """
        Finds the causal graph using the specified discovery algorithm.
        :param algo: The discovery algorithm to use (default is 'pc').
        :param pk: Prior knowledge in the form of required and forbidden edges.
        :return: The discovered causal graph as a networkx DiGraph.
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
        User can input their own causal graph.
        :param graph: A networkx DiGraph representing the causal graph.
        :return: The input causal graph.
        """
        self.graph = graph

    def refute_cgm(self, n_perm=100, indep_test=gcm, cond_indep_test=gcm, apply_sugst=True, show_plt=False):
        """
        Refutes the discovered causal graph using permutation tests.
        :param n_perm: Number of permutations for the refutation test (default is 100).
        :param indep_test: Independence test to use for refutation (default is gcm).
        :param cond_indep_test: Conditional independence test to use for refutation (default is gcm).
        :param apply_sugst: Whether to apply suggestions to the graph after refutation (default is True).
        :param show_plt: Whether to show the plot of the refutation results (default is False).
        :return: The refuted causal graph.
        """
        
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
        :return: The causal model as a dowhy CausalModel object.
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
        :param method: Method to use for effect identification (default is None, which uses the default method set by DoWhy).
        :return: The identified estimand as a dowhy IdentifiedEstimand object.
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
        :param method_cat: The method category to use for effect estimation (default is 'backdoor.linear_regression').
        :param ctrl_val: The control value for the treatment variable (default is None, which uses the control value set during initialization).
        :param trtm_val: The treatment value for the treatment variable (default is None, which uses the treatment value set during initialization).
        :return: The estimated effect as a dowhy EffectEstimate object.
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
            logging.error(f"Error in estimating the effect: {e}")
            raise
        
        logging.info("Note that it is ok for your treatment to be a continuous variable, DoWhy automatically discretizes at the backend.")
        return self.estimate
    
    # should give a warning to users if the estimate is to be refuted

    def refute_estimate(self,  method_name="ALL", placebo_type='permute', subset_fraction=0.9):
        """
        Refutes the estimated effect of the treatment on the outcome variable using various methods.
        :param method_name: The method to use for refutation (default is "ALL", which applies all methods).
        :param placebo_type: The type of placebo treatment to use for refutation (default is 'permute').
        :param subset_fraction: The fraction of the data to use for the data subset refuter (default is 0.9).
        :return: The refuted estimate as a dowhy RefutationResult object or a list of results if multiple methods are applied.
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
        :param graph: The causal graph as a networkx DiGraph.
        :param treatment: The treatment variable in the graph.
        :param outcome: The outcome variable in the graph.
        :return: A dictionary containing various properties of the graph.
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
        :return: None
        """
        if self.graph is not None:
            disp_graph_nx(self.graph)
        else:
            logging.warning("No causal graph available to visualize.")
        
    def _extract_graph_refutation_metrics(self, graph_ref_str):
        """
        Extracts metrics from the graph refutation result string.
        :param graph_ref_str: The graph refutation result string.
        :return: A tuple containing the number of informative TPA, total TPA, p-value for TPA, number of violated LMCs, total LMCs, and p-value for LMCs.
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
        :param graph: The causal graph as a networkx DiGraph.
        :param data: The data used for testing the graph quality.
        :param test: The statistical test to use (default is 'pearsonr').
        :param significance_level: The significance level for the test (default is 0.05).
        :param score: The scoring function to use (default is confusion_matrix).
        :return: The graph quality score.
        """
        # Implement the logic to calculate the graph quality score
        # This is a placeholder implementation
        pggraph = DAG(graph)

        return correlation_score(pggraph, data, test, significance_level, score)
    
    def see_graph_quality_score(self):
        """
        Shows the graph quality score/measure
        :return: None
        """
        if self.graph is not None:
            score = self._extract_graph_quality_score(self.graph, self.data)
            logging.info(f"Graph Quality Score: {score}")
        else:
            logging.warning("No causal graph available to visualize quality score.")
    
    def see_graph_refutation(self):
        """
        Shows the graph refutation results.
        :return: None
        """
        if self.graph_ref is not None:
            tpa_num, tpa_total, tpa_p, lmc_num, lmc_total, lmc_p = self._extract_graph_refutation_metrics(self.graph_ref)
            logging.info(f"Graph refutation metrics: TPA: {tpa_num}/{tpa_total} (p-value: {tpa_p}), LMC: {lmc_num}/{lmc_total} (p-value: {lmc_p})")
        else:
            logging.warning("No graph refutation available to see.")
    
    def _extract_refuter_metrics(self, refuter_result):
        """
        Extracts metrics from the refuter result string.
        :param refuter_result: The refuter result string.
        :return: A tuple containing the p-value and new effect.
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
        :return: None
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
        :param estimate_obj: The effect estimate object from DoWhy.
        :return: A dictionary containing the effect estimate metrics.
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
        :return: None
        """
        if self.estimate is not None:
            estimate_metrics = self._extract_effect_estimation(self.estimate)
            logging.info(f"Effect Estimate: {estimate_metrics['Effect Estimate']}")
            logging.info(f"Realized Estimand Expression: {estimate_metrics['Realized Estimand Expression']}")
            logging.info(f"Treatment Value: {estimate_metrics['Treatment Value']}")
            logging.info(f"Control Value: {estimate_metrics['Control Value']}")
        else:
            logging.warning("No effect estimation available to see.")
        
    
    def save_into_csv(self, csv_path):
        """
        Saves the discovered causal graph into a CSV file.
        :param csv_path: Path to save the CSV file.
        """
        # Save the graph properties into a CSV file
        if self.graph is not None:
            metrics = self.see_graph_properties()
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(csv_path.replace('.csv', '_graph_properties.csv'), index=False)
            logging.info(f"Graph properties saved to {csv_path}")
        else:
            logging.warning("No causal graph available to save properties.")
        
        # Save the graph refutation metrics into a CSV file
        if self.graph_ref is not None:
            tpa_num, tpa_total, tpa_p, lmc_num, lmc_total, lmc_p = self._extract_graph_refutation_metrics(self.graph_ref)
            ref_metrics = {
                'TPA': f"{tpa_num}/{tpa_total}",
                'TPA p-value': tpa_p,
                'LMC': f"{lmc_num}/{lmc_total}",
                'LMC p-value': lmc_p
            }
            ref_df = pd.DataFrame([ref_metrics])
            ref_df.to_csv(csv_path.replace('.csv', '_graph_refutation.csv'), index=False)
            logging.info(f"Graph refutation metrics saved to {csv_path.replace('.csv', '_graph_refutation.csv')}")
        
        # Save the graph quality score into a CSV file
        if self.graph is not None:
            score = self._extract_graph_quality_score(self.graph, self.data)
            score_df = pd.DataFrame({'Graph Quality Score': [score]})
            score_df.to_csv(csv_path.replace('.csv', '_graph_quality_score.csv'), index=False)
            logging.info(f"Graph quality score saved to {csv_path.replace('.csv', '_graph_quality_score.csv')}")
        
        # Save the effect estimate into a CSV file
        if self.estimate is not None:
            estimate_metrics = self._extract_effect_estimation(self.estimate)
            est_df = pd.DataFrame([estimate_metrics])
            est_df.to_csv(csv_path.replace('.csv', '_effect_estimate.csv'), index=False)
            logging.info(f"Effect estimate saved to {csv_path.replace('.csv', '_effect_estimate.csv')}")
        
        # Save the estimate refutation metrics into a CSV file
        if self.est_ref is not None:
            if isinstance(self.est_ref, list):
                ref_results = []
                for ref in self.est_ref:
                    pval, neweff = self._extract_refuter_metrics(ref)
                    ref_results.append({'p-value': pval, 'New effect': neweff})
                ref_df = pd.DataFrame(ref_results)
            else:
                pval, neweff = self._extract_refuter_metrics(self.est_ref)
                ref_df = pd.DataFrame([{'p-value': pval, 'New effect': neweff}])
            ref_df.to_csv(csv_path.replace('.csv', '_estimate_refutation.csv'), index=False)
            logging.info(f"Estimate refutation metrics saved to {csv_path.replace('.csv', '_estimate_refutation.csv')}")
        
