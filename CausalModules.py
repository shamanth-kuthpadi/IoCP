from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.FCMBased import lingam
from causallearn.utils.PDAG2DAG import pdag2dag
from causallearn.search.FCMBased.lingam.utils import make_dot
from util import *
import dowhy.gcm.falsify
from dowhy.gcm.falsify import falsify_graph
from dowhy.gcm.falsify import apply_suggestions
from dowhy import CausalModel
import cdt
from cdt.causality.graph import PC
from cdt.causality.graph import CCDr
from cdt.causality.graph import GES
from cdt.causality.graph import CAM
from cdt.utils.graph import dagify_min_edge
from logging_utils import setup_logging, get_logger
from config import CausalConfig
from visualization_utils import visualize_graph, visualize_effect_estimate, visualize_refutation
import time

cdt.SETTINGS.rpath = '/usr/local/bin/Rscript'

class EstimateEffect:
    """
    A class to perform end-to-end causal inference analysis using various algorithms and the DoWhy library.
    
    Attributes:
        data (pd.DataFrame): The input dataset for causal analysis.
        graph (networkx.DiGraph): The learned or provided causal graph.
        graph_ref (object): Result of graph falsification/refutation.
        model (dowhy.CausalModel): The DoWhy causal model object.
        estimand (object): The identified estimand (causal effect expression).
        estimate (object): The estimated causal effect.
        est_ref (object): The result of refuting the causal estimate.
    """
    def __init__(self, data, config=None):
        """
        Initialize the EstimateEffect object with a dataset and optional configuration.
        
        Args:
            data (pd.DataFrame): The input data for causal analysis.
        """
        setup_logging()
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info("Initializing EstimateEffect with input data.")
        self.validate_data(data)
        self.data = data
        self.config = config if config is not None else CausalConfig()
        self.graph = None
        self.graph_ref = None
        self.model = None
        self.estimand = None
        self.estimate = None
        self.est_ref = None
    
    def validate_data(self, data):
        """
        Validate input data for causal analysis.
        Raises ValueError if data is not a DataFrame, is empty, or contains missing values.
        """
        import pandas as pd
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")
        if data.empty:
            raise ValueError("Input data cannot be empty.")
        if data.isnull().any().any():
            raise ValueError("Input data contains missing values. Please clean or impute your data.")

    def validate_treatment_outcome(self, treatment, outcome):
        """
        Validate that treatment and outcome variables exist in the data columns.
        Raises ValueError if any are missing.
        Returns treatment, outcome as lists.
        """
        if isinstance(treatment, str):
            treatment = [treatment]
        if isinstance(outcome, str):
            outcome = [outcome]
        missing = [var for var in treatment + outcome if var not in self.data.columns]
        if missing:
            raise ValueError(f"Variables not found in data columns: {missing}")
        return treatment, outcome

    # For now, the only prior knowledge that the prototype will allow is required/forbidden edges
    # pk must be of the type => {'required': [list of edges to require], 'forbidden': [list of edges to forbid]}
    def find_causal_graph(self, algo=None, pk=None):
        """
        Learn a causal graph from the data using the specified algorithm.
        Optionally, incorporate prior knowledge by requiring or forbidding certain edges.
        
        Args:
            algo (str): The algorithm to use ('pc', 'ges', 'icalingam').
            pk (dict, optional): Prior knowledge with 'required' and/or 'forbidden' edge lists.
        
        Returns:
            networkx.DiGraph: The learned causal graph.
        """
        algo = algo if algo is not None else self.config.default_algorithms[0]
        self.logger.info(f"Starting causal graph discovery with algorithm: {algo}")
        print(f"[INFO] Discovering causal graph using algorithm: {algo}...")
        df = self.data.to_numpy()
        labels = list(self.data.columns)
        try:
            match algo:
                # case 'pc':
                #     cg = pc(data=df, show_progress=False, node_names=labels)
                #     cg = pdag2dag(cg.G)
                #     predicted_graph = genG_to_nx(cg, labels)
                #     self.graph = predicted_graph
                case 'ges':
                    # GES: Greedy Equivalence Search (score-based)
                    cg = ges(X=df, node_names=labels)
                    cg = pdag2dag(cg['G'])
                    predicted_graph = genG_to_nx(cg, labels)
                    self.graph = predicted_graph
                case 'icalingam':
                    # ICALiNGAM: Independent Component Analysis LiNGAM (linear non-Gaussian)
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
                case 'pc':
                    # PC: Constraint-based algorithm
                    model = PC()
                    predicted_graph = model.predict(self.data)
                    predicted_graph = dagify_min_edge(predicted_graph)
                    self.graph = nx.DiGraph(predicted_graph)
            
            if pk is not None:
                # ensuring that pk is indeed of the right type
                if not isinstance(pk, dict):
                    raise TypeError("[find_causal_graph] Prior knowledge (pk) must be a dict with 'required' and/or 'forbidden' keys.")
                # are there any edges to require
                if 'required' in pk.keys():
                    eb = pk['required']
                    self.graph.add_edges_from(eb)
                # are there any edges to remove
                if 'forbidden' in pk.keys():
                    eb = pk['forbidden']
                    self.graph.remove_edges_from(eb)
        except Exception as e:
            self.logger.error(f"[find_causal_graph] {e}")
            print(f"[ERROR] Causal graph discovery failed: {e}")
            raise RuntimeError(f"[find_causal_graph] Error in creating causal graph with algo '{algo}': {e}") from e
        self.logger.info("Causal graph discovery complete.")
        print("[INFO] Causal graph discovery complete.")
        return self.graph

    # What if user already has a graph they would like to input
    def input_causal_graph(self, graph):
        """
        Set a user-provided causal graph as the current graph.
        
        Args:
            graph (networkx.DiGraph): The user-supplied causal graph.
        """
        self.logger.info("Setting user-provided causal graph.")
        try:
            if graph is None:
                raise ValueError("[input_causal_graph] Provided graph is None.")
            self.graph = graph
        except Exception as e:
            self.logger.error(f"[input_causal_graph] {e}")
            raise RuntimeError(f"[input_causal_graph] Error setting user graph: {e}") from e

    def refute_cgm(self, n_perm=None, indep_test=gcm, cond_indep_test=gcm, apply_sugst=True, show_plt=False):
        """
        Falsify/refute the learned or provided causal graph using DoWhy's falsification tools.
        Optionally, apply suggested changes to the graph.
        
        Args:
            n_perm (int): Number of permutations for statistical tests.
            indep_test (callable): Independence test function.
            cond_indep_test (callable): Conditional independence test function.
            apply_sugst (bool): Whether to apply suggested changes to the graph.
            show_plt (bool): Whether to plot histograms of test statistics.
        
        Returns:
            networkx.DiGraph: The (possibly updated) causal graph.
        """
        n_perm = n_perm if n_perm is not None else self.config.n_permutations
        self.logger.info("Starting graph refutation/falsification.")
        print(f"[INFO] Refuting/falsifying the causal graph with {n_perm} permutations. This may take a while...")
        try:
            if self.graph is None:
                raise ValueError("[refute_cgm] No graph to refute. Please create or input a graph first.")
            result = falsify_graph(self.graph, self.data, n_permutations=n_perm,
                                  independence_test=indep_test,
                                  conditional_independence_test=cond_indep_test, plot_histogram=show_plt)
            self.graph_ref = result
            if apply_sugst is True:
                self.graph = apply_suggestions(self.graph, result)
            
        except Exception as e:
            self.logger.error(f"[refute_cgm] {e}")
            print(f"[ERROR] Graph refutation failed: {e}")
            raise RuntimeError(f"[refute_cgm] Error in refuting graph: {e}") from e
        self.logger.info("Graph refutation/falsification complete.")
        print("[INFO] Graph refutation/falsification complete.")
        return self.graph
    
    def create_model(self, treatment, outcome):
        """
        Create a DoWhy CausalModel object using the current graph, treatment, and outcome.
        
        Args:
            treatment (str or list): Treatment variable(s).
            outcome (str or list): Outcome variable(s).
        
        Returns:
            dowhy.CausalModel: The created causal model.
        """
        self.logger.info(f"Creating CausalModel with treatment: {treatment}, outcome: {outcome}")
        try:
            if self.graph is None:
                raise ValueError("[create_model] No graph available. Please create or input a graph first.")
            # Validate treatment and outcome
            treatment, outcome = self.validate_treatment_outcome(treatment, outcome)
            model_est = CausalModel(
                    data=self.data,
                    treatment=treatment,
                    outcome=outcome,
                    graph=self.graph
                )
            self.model = model_est
            self.logger.info("CausalModel created successfully.")
            return self.model
        except Exception as e:
            self.logger.error(f"[create_model] {e}")
            raise RuntimeError(f"[create_model] Error creating CausalModel: {e}") from e

    def identify_effect(self, method=None):
        """
        Identify the causal effect (estimand) using the created model.
        Optionally, specify the identification method.
        
        Args:
            method (str, optional): Identification method (see DoWhy docs for options).
        
        Returns:
            object: The identified estimand.
        """
        self.logger.info(f"Identifying effect with method: {method}")
        try:
            if self.model is None:
                raise ValueError("[identify_effect] No model available. Please create a model first.")
            if method is None:
                identified_estimand = self.model.identify_effect()
            else:
                identified_estimand = self.model.identify_effect(method=method)

            self.estimand = identified_estimand
            self.logger.info("Effect identification complete.")
        except Exception as e:
            self.logger.error(f"[identify_effect] {e}")
            raise RuntimeError(f"[identify_effect] Error identifying effect: {e}") from e

        print("Note that you can also use other methods for the identification process. Below are method descriptions taken directly from DoWhy's documentation")
        print("maximal-adjustment: returns the maximal set that satisfies the backdoor criterion. This is usually the fastest way to find a valid backdoor set, but the set may contain many superfluous variables.")
        print("minimal-adjustment: returns the set with minimal number of variables that satisfies the backdoor criterion. This may take longer to execute, and sometimes may not return any backdoor set within the maximum number of iterations.")
        print("exhaustive-search: returns all valid backdoor sets. This can take a while to run for large graphs.")
        print("default: This is a good mix of minimal and maximal adjustment. It starts with maximal adjustment which is usually fast. It then runs minimal adjustment and returns the set having the smallest number of variables.")
        return self.estimand
    
    def estimate_effect(self, method_cat=None, ctrl_val=0, trtm_val=1):
        """
        Estimate the causal effect using the identified estimand and specified estimation method.
        
        Args:
            method_cat (str): The estimation method (e.g., 'backdoor.linear_regression').
            ctrl_val (numeric): Value for the control group.
            trtm_val (numeric): Value for the treatment group.
        
        Returns:
            object: The estimated effect.
        """
        method_cat = method_cat if method_cat is not None else self.config.default_estimation_methods[0]
        self.logger.info(f"Estimating effect with method: {method_cat}")
        estimate = None
        try:
            if self.model is None or self.estimand is None:
                raise ValueError("[estimate_effect] Model or estimand not available. Please create model and identify effect first.")
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
            self.logger.info("Effect estimation complete.")
        except Exception as e:
            self.logger.error(f"[estimate_effect] {e}")
            raise RuntimeError(f"[estimate_effect] Error estimating the effect: {e}") from e
        
        print("Note that it is ok for your treatment to be a continuous variable, DoWhy automatically discretizes at the backend.")
        return self.estimate
    
    # should give a warning to users if the estimate is to be refuted

    def refute_estimate(self,  method_name=None, placebo_type='permute', subset_fraction=0.9):
        """
        Refute the estimated causal effect using various refutation strategies.
        
        Args:
            method_name (str): The refutation method ('placebo_treatment_refuter', 'random_common_cause', 'data_subset_refuter', or 'ALL').
            placebo_type (str): Type of placebo for placebo refuter.
            subset_fraction (float): Fraction of data to use for subset refuter.
        
        Returns:
            object or list: The refutation result(s).
        """
        method_name = method_name if method_name is not None else self.config.default_refutation_methods[0]
        self.logger.info(f"Refuting estimate with method: {method_name}")
        ref = None
        try:
            if self.model is None or self.estimand is None or self.estimate is None:
                raise ValueError("[refute_estimate] Model, estimand, or estimate not available. Please complete previous steps first.")
            match method_name:
                case "placebo_treatment_refuter":
                    ref = self.model.refute_estimate(
                        self.estimand,
                        self.estimate,
                        method_name=method_name,
                        placebo_type=placebo_type
                    )
                
                case "random_common_cause":
                    ref = self.model.refute_estimate(
                        self.estimand,
                        self.estimate,
                        method_name=method_name
                    )
                case "data_subset_refuter":
                    ref = self.model.refute_estimate(
                        self.estimand,
                        self.estimate,
                        method_name=method_name,
                        subset_fraction=subset_fraction
                    )
                case "ALL":
                    # Run all three refuters and return a list of results
                    ref_placebo = self.model.refute_estimate(
                        self.estimand,
                        self.estimate,
                        method_name=method_name,
                        placebo_type=placebo_type
                    )
                    ref_rand_cause = self.model.refute_estimate(
                        self.estimand,
                        self.estimate,
                        method_name=method_name
                    )
                    ref_subset = self.model.refute_estimate(
                        self.estimand,
                        self.estimate,
                        method_name=method_name,
                        subset_fraction=subset_fraction
                    )
                    ref = [ref_placebo, ref_rand_cause, ref_subset]
            if not isinstance(ref, list) and ref.refutation_result['is_statistically_significant']:
                print("Please make sure to take a revisit the pipeline as the refutation p-val is significant: ", ref.refutation_result['p_value'])
            self.est_ref = ref
            self.logger.info("Estimate refutation complete.")
        except Exception as e:
            self.logger.error(f"[refute_estimate] {e}")
            raise RuntimeError(f"[refute_estimate] Error in refuting estimate: {e}") from e
            
        return self.est_ref
    
    def get_all_information(self):
        """
        Retrieve all relevant information from the causal analysis pipeline.
        
        Returns:
            dict: Dictionary containing the graph, graph refutation result, estimand, effect estimate, and estimate refutation result.
        """
        return {'graph': self.graph, 
                'graph_refutation_res': self.graph_ref,
                'estimand_expression': self.estimand,
                'effect_estimate': self.estimate,
                'estimate_refutation_res': self.est_ref
                }

    def visualize_graph(self, title="Causal Graph"):
        """
        Visualize the current causal graph using the visualization utility.
        """
        self.logger.info("Visualizing causal graph.")
        if self.graph is not None:
            visualize_graph(self.graph, title=title)
        else:
            print("[WARN] No graph to visualize.")

    def visualize_effect_estimate(self, title="Effect Estimate"):
        """
        Visualize the current effect estimate using the visualization utility.
        """
        self.logger.info("Visualizing effect estimate.")
        if self.estimate is not None:
            visualize_effect_estimate(self.estimate, title=title)
        else:
            print("[WARN] No effect estimate to visualize.")

    def visualize_refutation(self, title="Refutation Result"):
        """
        Visualize the current refutation result using the visualization utility.
        """
        self.logger.info("Visualizing refutation result.")
        if self.est_ref is not None:
            visualize_refutation(self.est_ref, title=title)
        else:
            print("[WARN] No refutation result to visualize.")

    def export_results(self, filepath, format='json'):
        """
        Export pipeline results to a file in JSON or pickle format.
        Args:
            filepath (str): Path to the output file.
            format (str): 'json' or 'pickle'.
        """
        import json
        import pickle
        results = self.get_all_information()
        self.logger.info(f"Exporting results to {filepath} as {format}.")
        try:
            if format == 'json':
                try:
                    with open(filepath, 'w') as f:
                        json.dump(results, f, default=str, indent=2)
                    print(f"[INFO] Results exported to {filepath} (JSON format).")
                except TypeError as e:
                    print(f"[WARN] JSON export failed due to non-serializable objects: {e}. Try using 'pickle' format instead.")
                    self.logger.warning(f"JSON export failed: {e}")
            elif format == 'pickle':
                with open(filepath, 'wb') as f:
                    pickle.dump(results, f)
                print(f"[INFO] Results exported to {filepath} (pickle format).")
            else:
                print(f"[ERROR] Unsupported export format: {format}")
                self.logger.error(f"Unsupported export format: {format}")
        except Exception as e:
            print(f"[ERROR] Failed to export results: {e}")
            self.logger.error(f"Failed to export results: {e}")

    def cleanup(self):
        """
        Clean up large objects to free memory. Sets data, graph, model, estimand, estimate, and est_ref to None and runs garbage collection.
        """
        import gc
        self.logger.info("Cleaning up large objects and freeing memory.")
        self.data = None
        self.graph = None
        self.graph_ref = None
        self.model = None
        self.estimand = None
        self.estimate = None
        self.est_ref = None
        gc.collect()
        print("[INFO] Pipeline memory cleanup complete.")

    def run_full_pipeline(self, treatment, outcome, algo=None, refute_graph=True, refute_estimate=True,
                         visualize=True, export_path=None, export_format='json', **kwargs):
        """
        Run the complete causal inference pipeline.
        Args:
            treatment: Treatment variable(s)
            outcome: Outcome variable(s)
            algo: Causal discovery algorithm (default from config)
            refute_graph: Whether to refute the learned graph
            refute_estimate: Whether to refute the effect estimate
            visualize: Whether to visualize results
            export_path: Path to export results (optional)
            export_format: 'json' or 'pickle'
            **kwargs: Additional parameters for underlying methods
        Returns:
            dict: Complete pipeline results (or dict of results if multiple algorithms)
        """
        # Check if multiple algorithms should be run
        if algo is None and len(self.config.default_algorithms) > 1:
            return self._run_multiple_algorithms(treatment, outcome, refute_graph, refute_estimate,
                                               visualize, export_path, export_format, **kwargs)
        
        # Single algorithm run (original logic)
        algo = algo if algo is not None else self.config.default_algorithms[0]
        self.logger.info("Running full causal inference pipeline.")
        print("[INFO] Running full causal inference pipeline...")
        start_time = time.time()
        try:
            # Step 1: Learn causal graph
            step_start = time.time()
            graph = self.find_causal_graph(algo=algo)
            step_end = time.time()
            print(f"[LATENCY] Causal graph discovery ({algo}): {step_end - step_start:.2f} seconds")
            self.logger.info(f"Causal graph discovery ({algo}) took {step_end - step_start:.2f} seconds")
            if visualize:
                self.visualize_graph()
            # Step 2: Refute graph (optional)
            if refute_graph:
                step_start = time.time()
                graph = self.refute_cgm()
                step_end = time.time()
                print(f"[LATENCY] Graph refutation: {step_end - step_start:.2f} seconds")
                self.logger.info(f"Graph refutation took {step_end - step_start:.2f} seconds")
                if visualize:
                    self.visualize_graph(title="Refuted Causal Graph")
            # Step 3: Create model
            step_start = time.time()
            self.create_model(treatment, outcome)
            step_end = time.time()
            print(f"[LATENCY] Model creation: {step_end - step_start:.2f} seconds")
            self.logger.info(f"Model creation took {step_end - step_start:.2f} seconds")
            # Step 4: Identify effect
            step_start = time.time()
            self.identify_effect()
            step_end = time.time()
            print(f"[LATENCY] Effect identification: {step_end - step_start:.2f} seconds")
            self.logger.info(f"Effect identification took {step_end - step_start:.2f} seconds")
            # Step 5: Estimate effect
            step_start = time.time()
            self.estimate_effect()
            step_end = time.time()
            print(f"[LATENCY] Effect estimation: {step_end - step_start:.2f} seconds")
            self.logger.info(f"Effect estimation took {step_end - step_start:.2f} seconds")
            if visualize:
                self.visualize_effect_estimate()
            # Step 6: Refute estimate (optional)
            if refute_estimate:
                step_start = time.time()
                self.refute_estimate()
                step_end = time.time()
                print(f"[LATENCY] Estimate refutation: {step_end - step_start:.2f} seconds")
                self.logger.info(f"Estimate refutation took {step_end - step_start:.2f} seconds")
                if visualize:
                    self.visualize_refutation()
            # Step 7: Export results (optional)
            if export_path is not None:
                step_start = time.time()
                self.export_results(export_path, format=export_format)
                step_end = time.time()
                print(f"[LATENCY] Results export: {step_end - step_start:.2f} seconds")
                self.logger.info(f"Results export took {step_end - step_start:.2f} seconds")
            total_time = time.time() - start_time
            print(f"[LATENCY] Total pipeline time ({algo}): {total_time:.2f} seconds")
            self.logger.info(f"Total pipeline time ({algo}): {total_time:.2f} seconds")
            print("[INFO] Full pipeline completed successfully.")
            self.logger.info("Full pipeline completed successfully.")
            return self.get_all_information()
        except Exception as e:
            print(f"[ERROR] Full pipeline failed: {e}")
            self.logger.error(f"Full pipeline failed: {e}")
            raise

    def _run_multiple_algorithms(self, treatment, outcome, refute_graph=True, refute_estimate=True,
                                visualize=True, export_path=None, export_format='json', **kwargs):
        """
        Run the pipeline for multiple algorithms specified in config.default_algorithms.
        """
        import time
        algorithms = self.config.default_algorithms
        results = {}
        
        self.logger.info(f"Running pipeline for multiple algorithms: {algorithms}")
        print(f"[INFO] Running pipeline for {len(algorithms)} algorithms: {algorithms}")
        overall_start = time.time()
        for i, algo in enumerate(algorithms):
            print(f"\n=== Algorithm {i+1}/{len(algorithms)}: {algo} ===")
            algo_start = time.time()
            try:
                # Create a fresh estimator for each algorithm
                fresh_estimator = EstimateEffect(self.data, config=self.config)
                
                # Determine export path for this algorithm
                algo_export_path = None
                if export_path is not None:
                    # Split path and extension
                    if '.' in export_path:
                        base_path, ext = export_path.rsplit('.', 1)
                        algo_export_path = f"{base_path}_{algo}.{ext}"
                    else:
                        algo_export_path = f"{export_path}_{algo}"
                
                # Run pipeline for this algorithm
                result = fresh_estimator.run_full_pipeline(
                    treatment=treatment,
                    outcome=outcome,
                    algo=algo,
                    refute_graph=refute_graph,
                    refute_estimate=refute_estimate,
                    visualize=visualize,
                    export_path=algo_export_path,
                    export_format=export_format,
                    **kwargs
                )
                results[algo] = result
                algo_end = time.time()
                print(f"[LATENCY] {algo} total time: {algo_end - algo_start:.2f} seconds")
                self.logger.info(f"{algo} total time: {algo_end - algo_start:.2f} seconds")
                print(f"✅ {algo} completed successfully")
                
            except Exception as e:
                algo_end = time.time()
                print(f"❌ {algo} failed: {e}")
                print(f"[LATENCY] {algo} failed after {algo_end - algo_start:.2f} seconds")
                self.logger.error(f"Algorithm {algo} failed: {e}")
                results[algo] = None
        overall_end = time.time()
        print(f"[LATENCY] Total time for all algorithms: {overall_end - overall_start:.2f} seconds")
        self.logger.info(f"Total time for all algorithms: {overall_end - overall_start:.2f} seconds")
        # Summary
        successful = [algo for algo, result in results.items() if result is not None]
        failed = [algo for algo, result in results.items() if result is None]
        
        print(f"\n=== Summary ===")
        print(f"Successful algorithms: {successful}")
        if failed:
            print(f"Failed algorithms: {failed}")
        
        return results