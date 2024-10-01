from typing import List
from scipy.stats import ks_2samp
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import networkx as nx
from rca_task import PotentialRootCause
from data_preprocessing import reduce_df, marginalize_node, impute_df

def make_kstest():
    def functional_analyze_root_causes(
        graph: nx.DiGraph,
        target_node: str,
        target_metric: str,
        target_statistic: str,
        normal_metrics: pd.DataFrame,
        abnormal_metrics: pd.DataFrame,
    ) -> List[PotentialRootCause]:
        """
        Functional RCA method that concatenates normal and abnormal metrics, scales the data,
        calculates percentage change or decrease, and ranks columns based on Kolmogorov-Smirnov test scores.
        Columns more similar to PetSite get higher scores, excluding PetSite itself.
        """

        # 1. Preprocess and find common columns
        statistic_of_interest = target_statistic
        normal_metrics = reduce_df(normal_metrics, target_metric, statistic_of_interest)
        abnormal_metrics = reduce_df(abnormal_metrics, target_metric, statistic_of_interest)

        # Find common columns between normal and abnormal metrics
        common_columns = list(set(normal_metrics.columns).intersection(abnormal_metrics.columns))

        # Print out the missing columns for better debugging
        missing_in_abnormal = list(set(normal_metrics.columns) - set(abnormal_metrics.columns))
        missing_in_normal = list(set(abnormal_metrics.columns) - set(normal_metrics.columns))
        
        if missing_in_abnormal:
            print(f"Columns in normal_metrics but missing in abnormal_metrics: {missing_in_abnormal}")
        if missing_in_normal:
            print(f"Columns in abnormal_metrics but missing in normal_metrics: {missing_in_normal}")

        # Exclude missing columns that are not in both DataFrames
        normal_metrics = normal_metrics.loc[:, common_columns]
        abnormal_metrics = abnormal_metrics.loc[:, common_columns]

        # 2. Concatenate and scale the data
        result_df = pd.concat([normal_metrics, abnormal_metrics], ignore_index=True)
        scaler = MinMaxScaler()
        result_df = pd.DataFrame(scaler.fit_transform(result_df), columns=result_df.columns)

        # 3. Select the last 5 rows
        result_df = result_df.tail(5)

        # 4. Calculate percentage change or decrease
        if target_metric == 'availability':
            # For availability, we calculate percentage decrease
            percentage_change = (result_df.iloc[0] - result_df.iloc[-1]) / result_df.iloc[0] * 100
            # print("Calculating percentage decrease for availability...")
        else:
            # For other metrics, calculate percentage increase
            percentage_change = (result_df.iloc[-1] - result_df.iloc[0]) / result_df.iloc[0] * 100
            # print("Calculating percentage change for metric...")

        # Sort columns by percentage change or decrease and get the top 10
        sorted_columns_percentage = percentage_change.sort_values(ascending=False)
        top_columns_percentage = sorted_columns_percentage.head(10).index

        # print(f"Top columns by percentage change/decrease: {top_columns_percentage}")

        # 5. Rank the top columns by Kolmogorov-Smirnov (KS) test with PetSite column
        petsite_column = 'PetSite'  # Assuming 'PetSite' is the column of interest

        # Ensure PetSite column exists in the data, otherwise skip this step
        if petsite_column not in result_df.columns:
            print(f"Column '{petsite_column}' not found in the data.")
            return []

        petsite_data = result_df[petsite_column]
        ks_scores = {}

        for column in top_columns_percentage:
            if column == petsite_column:
                continue  # Skip the PetSite column itself
            
            column_data = result_df[column]
            ks_statistic, _ = ks_2samp(petsite_data, column_data)
            
            # Invert KS score to prioritize columns similar to PetSite
            ks_scores[column] = 1 / (1 + ks_statistic)  # Higher score for more similar columns

        # Sort columns based on inverted KS test scores in descending order (higher scores = more similarity)
        sorted_columns_by_ks = sorted(ks_scores.items(), key=lambda item: item[1], reverse=True)

        # print(f"Sorted columns by similarity to PetSite (descending): {sorted_columns_by_ks}")

        # 6. Create PotentialRootCause objects from the sorted KS scores and print them
        potential_root_causes = [
            PotentialRootCause(root_cause, target_metric, score) for root_cause, score in sorted_columns_by_ks
        ]

        # Print root causes for debugging
        # print("Identified Potential Root Causes:")
        # for cause in potential_root_causes:
        #     print(f"Node: {cause.node}, Metric: {cause.metric}, Score: {cause.score}")

        return potential_root_causes

    return functional_analyze_root_causes
