import os
import warnings

import mlflow
import numpy as np
import pandas as pd
from evidently.metric_preset import ClassificationPreset
from evidently.metrics import (
    FBetaTopKMetric,
    NDCGKMetric,
    PersonalizationMetric,
    PrecisionTopKMetric,
    RecallTopKMetric,
)
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report

from src.utils.math_utils import sigmoid

warnings.filterwarnings(
    action="ignore",
    category=FutureWarning,
    module=r"evidently.metrics.recsys.precision_recall_k",
)


def log_ranking_metrics(args, eval_df, min_rel_score: int = 2):

    column_mapping = ColumnMapping(
        recommendations_type="rank",
        target=args.rating_col,
        prediction="rec_ranking",
        item_id=args.item_col,
        user_id=args.user_col,
    )

    report = Report(
        metrics=[
            NDCGKMetric(k=args.top_k, min_rel_score= min_rel_score),
            RecallTopKMetric(k=args.top_K, min_rel_score= min_rel_score),
            PrecisionTopKMetric(k=args.top_K, min_rel_score= min_rel_score),
            FBetaTopKMetric(k=args.top_k, min_rel_score= min_rel_score),
            PersonalizationMetric(k=args.top_k, min_rel_score= min_rel_score),
        ],
    )

    report.run(reference_data=None, current_data=eval_df, column_mapping=column_mapping)

    evidently_report_fp = f"{args.notebook_persit_dp}/evidently_ranking_report.html"
    os.makedirs(args.notebook_persit_dp, exist_ok=True)
    report.save_html(evidently_report_fp)

    if args.log_to_mlflow:
        mlflow.log_artifact(evidently_report_fp)
        for metric_result in report.as_dict()["metrics"]:
            metric = metric_result["metric"]
            if metric == "PersonalizationMetric":
                metric_value = float(metric_result["result"]["current_value"])
                mlflow.log_metric(f"val_{metric}", metric_value)
                continue
            result = metric_result["result"]["current"].to_dict()
            for kth, metric_value in result.items():
                mlflow.log_metric(f"val_{metric}_at_k_as_step", metric_value, step=kth)

    return report


def log_classification_metrics(
    args,
    eval_classification_df,
    target_col="label",
    prediction_col="classification_proba",
):
    column_mapping = ColumnMapping(target=target_col, prediction=prediction_col)
    classification_performance_report = Report(
        metrics=[
            ClassificationPreset(probas_threshold=sigmoid(2)),
        ]
    )

    classification_performance_report.run(
        reference_data=None,
        current_data=eval_classification_df[[target_col, prediction_col]],
        column_mapping=column_mapping,
    )

    evidently_report_fp = (
        f"{args.notebook_persit_dp}/evidently_classification_report.html"
    )
    os.makedirs(args.notebook_persit_dp, exist_ok=True)
    classification_performance_report.save_html(evidently_report_fp)

    if args.log_to_mlflow:
        mlflow.log_artifact(evidently_report_fp)
        for metric_result in classification_performance_report.as_dict()["metrics"]:
            metric = metric_result["metric"]
            if metric == "ClassificationQualityMetric":
                roc_auc = float(metric_result["result"]["current"]["roc_auc"])
                mlflow.log_metric(f"val_roc_auc", roc_auc)
                continue
            if metric == "ClassificationPRTable":
                columns = [
                    "top_perc",
                    "count",
                    "prob",
                    "tp",
                    "fp",
                    "precision",
                    "recall",
                ]
                table = metric_result["result"]["current"][1]
                table_df = pd.DataFrame(table, columns=columns)
                for i, row in table_df.iterrows():
                    prob = int(row["prob"] * 100)  # MLflow step only takes int
                    precision = float(row["precision"])
                    recall = float(row["recall"])
                    mlflow.log_metric(
                        f"val_precision_at_prob_as_threshold_step", precision, step=prob
                    )
                    mlflow.log_metric(
                        f"val_recall_at_prob_as_threshold_step", recall, step=prob
                    )
                break

    return classification_performance_report