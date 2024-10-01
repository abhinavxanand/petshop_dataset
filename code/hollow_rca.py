from typing import List
import networkx as nx
import pandas as pd
import numpy as np
from rca_task import PotentialRootCause
from data_preprocessing import reduce_df, marginalize_node, impute_df

def make_hollow_rca(
    root_cause_top_k: int = 3,
    imputation_method: str = "mean",
):
    """
    Hollow RCA method that simulates the identification of potential root causes.

    Args:
        root_cause_top_k: The maximum number of root causes in the results.
        imputation_method: How NaNs should be imputed.
    """

    def analyze_root_causes(
        graph: nx.DiGraph,
        target_node: str,
        target_metric: str,
        target_statistic: str,
        normal_metrics: pd.DataFrame,
        abnormal_metrics: pd.DataFrame,
    ) -> List[PotentialRootCause]:
        """Simulated method to identify potential root causes.

        Args:
            graph: Call graph of microservice architecture.
            target_node: Node whose SLO violation to investigate.
            target_metric: Metric that is in violation with SLO.
            target_statistic: Statistic such as Average of the target_metric that is in violation with SLO.
            normal_metrics: Metrics of all microservices during previous normal operations.
            abnormal_metrics: Metrics of all microservices during SLO violation.

        Returns: List of simulated potential root causes.
        """
        # Simulate processing inputs
        normal_metrics = normal_metrics.copy()
        abnormal_metrics = abnormal_metrics.copy()

        # Example of placeholder processing
        normal_metrics = reduce_df(normal_metrics, target_metric, target_statistic)
        abnormal_metrics = reduce_df(abnormal_metrics, target_metric, target_statistic)

        # Impute missing values
        impute_df(normal_metrics, imputation_method)
        impute_df(abnormal_metrics, imputation_method)

        # Simulate root cause detection
        simulated_causes = []
        for _ in range(root_cause_top_k):
            simulated_causes.append(PotentialRootCause(target_node, target_metric, np.random.rand()))

        return simulated_causes

    return analyze_root_causes
