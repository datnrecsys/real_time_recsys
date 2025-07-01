import os
from typing import Any, Generic, TypeVar

import pandas as pd
from evidently.metrics import (FBetaTopKMetric, NDCGKMetric,
                               PersonalizationMetric, PrecisionTopKMetric,
                               RecallTopKMetric)
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report

from src.algo.base.base_dl_model import BaseDLModel
from src.algo.sequence.model import SequenceRatingPrediction
from src.algo.sequence_two_tower.model import SequenceRatingPrediction as SequenceRatingPredictionTwoTower
from src.domain.model_request import ModelRequest, SequenceModelRequest
from src.utils.embedding_id_mapper import IDMapper
import warnings
from loguru import logger
from functools import lru_cache

# ignore all FutureWarnings
warnings.simplefilter("ignore", category=FutureWarning)

T = TypeVar("T", bound=BaseDLModel)

class RankingMetricComputer(Generic[T]):
    def __init__(self, 
                 rec_model: T, 
                 device: str = "cpu",
                 mlf_client: Any | None = None,
                 idm: IDMapper | None = None, 
                 rating_col: str = "rating",
                 timestamp_col: str = "timestamp",
                 top_k = 10,
                 top_K = 100,
                 batch_size: int = 1,
                 evidently_report_fp: str | None = None,
):
        self.device = device
        self.rec_model = rec_model
        self.idm = idm
        self.top_k = top_k
        self.top_K = top_K
        self.batch_size = batch_size
        self.rating_col = rating_col
        self.timestamp_col = timestamp_col
        self.mlf_client = mlf_client
        self.evidently_report_fp = evidently_report_fp
        assert top_K > top_k, "top_K must be greater than top_k"
        
    # @lru_cache
    def _get_rec_df(self, input_data: ModelRequest) -> pd.DataFrame:
        """
        Get the recommendation DataFrame from the model.
        
        Args:
            input_data (ModelRequest): The input data for generating recommendations.
        
        Returns:
            pd.DataFrame: A DataFrame containing the recommendations.
        """
        recommendations = self.rec_model.recommend(
            input_data,
            k=self.top_K,
            batch_size=self.batch_size,
        )
        
        recommendation_df = pd.DataFrame(recommendations)
        
        # print(recommendation_df)
        
        return recommendation_df
        

    def _create_rec_df(self, rec_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a DataFrame for recommendations.
        
        Args:
            df (pd.DataFrame): The DataFrame containing the recommendations.
            idm (IDMapper): Optional. The IDMapper object to map IDs. If provided, it will map user and item IDs.
            user_col (str): The name of the user column in the DataFrame.
            item_col (str): The name of the item column in the DataFrame.
            rating_col (str): The name of the rating column in the DataFrame.
            timestamp_col (str): The name of the timestamp column in the DataFrame.

        Returns:
            pd.DataFrame: A DataFrame with mapped user and item IDs and recommendation.
        """
            
        
        recommendation_df = rec_df.copy()
        
        # Assign rec 
        recommendation_df = recommendation_df.assign(
            rec_ranking = lambda df:(
                df.groupby("user_indice", as_index=False)["score"].rank(
                    ascending=False, method="first"
                )
            )
        )
        
        # Assign user_id and item_id in order to process the next step 
        
        recommendation_df = recommendation_df.assign(
            user_id = lambda df: df["user_indice"],
            item_id = lambda df: df["recommendation"])
        
        if self.idm:
            # print("Mapping user and item IDs using IDMapper...")
            recommendation_df = recommendation_df.assign(
                **{
                "user_id": lambda df: df["user_indice"].apply(
                    lambda user_indice: self.idm.get_user_id(user_indice)
                ),
                "item_id": lambda df: df["recommendation"].apply(
                    lambda parent_asin: self.idm.get_item_id(parent_asin)
                ),

            }
            )
        
        print(recommendation_df.head(10))
            
        return recommendation_df
    
    def _create_label_df(
        self,
        df: pd.DataFrame,
):
        """
        Create a DataFrame for labels.

        Args:
            df (pd.DataFrame): The DataFrame containing the labels.
            user_col (str): The name of the user column in the DataFrame.
            item_col (str): The name of the item column in the DataFrame.
            rating_col (str): The name of the rating column in the DataFrame.
            timestamp_col (str): The name of the timestamp column in the DataFrame.
        Returns:
            pd.DataFrame: A DataFrame with sorted user and item IDs and labels.
        """
        label_df = df.copy()
        
        user_col = "user_id" 
        
        if user_col not in df.columns or not self.idm:
            label_df = label_df.drop(columns=[user_col])
            label_df = label_df.rename(columns={"user_indice": user_col})
        
        label_df.rename(columns={"parent_asin": "item_id"}, inplace=True)
        item_col = "item_id"
        if item_col not in df.columns or not self.idm:
            label_df = label_df.drop(columns=[item_col])
            label_df = label_df.rename(columns={"item_indice": item_col})
        
        label_df = (
            label_df.sort_values([self.timestamp_col], ascending=False)
            .assign(
                rating_rank = lambda df: df.groupby(user_col)[self.rating_col].rank(
                    ascending=False, method="first"
                )
            )
            .sort_values(["rating_rank"], ascending=[True])[[user_col, item_col, self.rating_col, "rating_rank"]]
        )
        
        # print(label_df)
        return label_df
    
    def _merge_recs_with_target(
        self,
        recs_df: pd.DataFrame,
        label_df: pd.DataFrame,
    ):
        
        return (
            recs_df.pipe(
                lambda df: pd.merge(
                    df,
                    label_df[["user_id", "item_id", self.rating_col, "rating_rank"]],
                    on=["user_id", "item_id"],
                    how="outer",
                )
            )
            .assign(
                rating=lambda df: df[self.rating_col].fillna(0).astype(int),
                # Fill the recall with ranking = top_K + 1 so that the recall calculation is correct
                rec_ranking=lambda df: df["rec_ranking"].fillna(self.top_K + 1).astype(int),
            )
            .sort_values(['user_id', "rec_ranking"])
        )
        
    def _build_report(self, merge_df: pd.DataFrame) -> Report:
    
        report =  Report(
            metrics=[
                NDCGKMetric(k=self.top_k),
                RecallTopKMetric(k=self.top_K),
                PrecisionTopKMetric(k=self.top_k),
                FBetaTopKMetric(k=self.top_k),
                PersonalizationMetric(k=self.top_k),
            ],
        )

        report.run(
            reference_data=None,
            current_data=merge_df,
            column_mapping=ColumnMapping(
                recommendations_type="rank",
                target=self.rating_col,
                prediction="rec_ranking",
                item_id="item_id",
                user_id="user_id",
            ),
        )
        # print(report.as_dict())
        if self.evidently_report_fp is not None:
            import os
            os.makedirs(self.evidently_report_fp, exist_ok=True)
            report.save_html(f"{self.evidently_report_fp}/evidently_ranking_report.html")
            logger.info(f"Report saved to {self.evidently_report_fp}/evidently_ranking_report.html")
        return report
    
    def _log_to_mlf(self, run_id: str, report: Report):
        try:
            self.mlf_client.log_artifact(run_id, f"{self.evidently_report_fp}/evidently_ranking_report.html")
        except Exception as e:
            logger.warning(f"Failed to log artifact to MLflow: {e}")
            
        for metric_result in report.as_dict()["metrics"]:
            metric = metric_result["metric"]
            if metric == "PersonalizationMetric":
                metric_value = float(metric_result["result"]["current_value"])
                self.mlf_client.log_metric(run_id, f"val_{metric}", metric_value)
                continue
            result = metric_result["result"]["current"].to_dict()
            for kth, metric_value in result.items():
                self.mlf_client.log_metric(run_id, f"val_{metric}_at_k_as_step", metric_value, step=kth)
        return 
    # def log_ranking_metrics(self, eval_df, run_id: str = None, persist_dir: str | None = None, min_rel_score: int = 2):

    #     column_mapping = ColumnMapping(
    #         recommendations_type="rank",
    #         target=self.rating_col,
    #         prediction="rec_ranking",
    #         item_id=self.item_col,
    #         user_id=self.user_col,
    #     )

    #     report = Report(
    #         metrics=[
    #             NDCGKMetric(k=self.top_k),
    #             RecallTopKMetric(k=self.top_K),
    #             PrecisionTopKMetric(k=self.top_K),
    #             FBetaTopKMetric(k=self.top_k),
    #             PersonalizationMetric(k=self.top_k),
    #         ],
    #     )

    #     report.run(reference_data=None, current_data=eval_df, column_mapping=column_mapping)

    #     if persist_dir is not None:
    #         evidently_report_fp = f"{persist_dir}/evidently_ranking_report.html"
    #         os.makedirs(persist_dir, exist_ok=True)
    #         report.save_html(evidently_report_fp)

    #     if self.mlf_client:
    #         self.mlf_client.log_artifact(run_id, evidently_report_fp)
    #         for metric_result in report.as_dict()["metrics"]:
    #             metric = metric_result["metric"]
    #             if metric == "PersonalizationMetric":
    #                 metric_value = float(metric_result["result"]["current_value"])
    #                 self.mlf_client.log_metric(run_id,f"val_{metric}", metric_value)
    #                 continue
    #             result = metric_result["result"]["current"].to_dict()
    #             for kth, metric_value in result.items():
    #                 self.mlf_client.log_metric(run_id, f"val_{metric}_at_k_as_step", metric_value, step=kth)

    #     return report

    def _build_input_data(self, df: pd.DataFrame, device: str = "cpu") -> ModelRequest:
        """
        Build the input data for the model from a DataFrame.
        
        Args:
            df (pd.DataFrame): The DataFrame containing user and item information.
        
        Returns:
            ModelRequest: An instance of ModelRequest with user_id and target_item.
        """
        if isinstance(self.rec_model, SequenceRatingPrediction):
            return SequenceModelRequest.from_df_for_rec(df, device=device)
        elif isinstance(self.rec_model, SequenceRatingPredictionTwoTower):
            return SequenceModelRequest.from_df_for_rec(
                df, 
                device=device
            )
        else:
            return ModelRequest.from_df_for_rec(df)


    def calculate(self, label_df: pd.DataFrame, run_id = None, log_to_mlflow = False, device: str = "cpu") -> pd.DataFrame:
        """
        Generate recommendations for the given input data.
        
        Args:
            input_data (ModelRequest): The input data for generating recommendations.
        
        Returns:
            pd.DataFrame: A DataFrame containing the recommendations.
        """
        input_data = self._build_input_data(label_df, device=device)

        rec_df = self._get_rec_df(input_data)
        
        rec_df = self._create_rec_df(rec_df)
        
        label_df = self._create_label_df(label_df)
        
        merged_df = self._merge_recs_with_target(rec_df, label_df)
        
        report = self._build_report(merged_df)
        
        if log_to_mlflow:
            if not self.mlf_client:
                logger.warning("mlf_client must be provided to log the report.")
                return report
            if run_id is None:
                logger.warning("run_id must be provided to log the report.")
                return report
            
            self._log_to_mlf(run_id, report)

        return report

        