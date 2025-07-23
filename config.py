class CausalConfig:
    """
    Configuration class for causal analysis pipeline parameters.
    """
    def __init__(self):
        self.default_algorithms = ['pc', 'ges', 'icalingam']
        self.default_estimation_methods = ['backdoor.linear_regression']
        self.default_refutation_methods = ['placebo_treatment_refuter', 'random_common_cause', 'data_subset_refuter']
        self.n_permutations = 100
        self.confidence_level = 0.95

    def run_multiple_algorithms(self, estimator, treatment, outcome, algorithms=None, **kwargs):
        """
        Run multiple causal discovery algorithms and compare results.
        
        Args:
            estimator: EstimateEffect instance
            treatment: Treatment variable(s)
            outcome: Outcome variable(s)
            algorithms: List of algorithms to try (defaults to self.default_algorithms)
            **kwargs: Additional parameters for run_full_pipeline
            
        Returns:
            dict: Results for each algorithm
        """
        if algorithms is None:
            algorithms = self.default_algorithms
            
        results = {}
        
        for algo in algorithms:
            print(f"\n=== Testing Algorithm: {algo} ===")
            try:
                from CausalModules import EstimateEffect
                fresh_estimator = EstimateEffect(estimator.data, config=self)
                
                result = fresh_estimator.run_full_pipeline(
                    treatment=treatment,
                    outcome=outcome,
                    algo=algo,
                    **kwargs
                )
                results[algo] = result
                print(f"{algo} completed successfully")
                
            except Exception as e:
                print(f"{algo} failed: {e}")
                results[algo] = None
                
        return results 