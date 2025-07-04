import os

import lightning as L
import pandas as pd
import torch
import torch.nn as nn
from evidently.metric_preset import ClassificationPreset
from evidently.metrics import (FBetaTopKMetric, NDCGKMetric,
                               PersonalizationMetric, PrecisionTopKMetric,
                               RecallTopKMetric)
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report
from loguru import logger
from pydantic import BaseModel

from src.algo.mf.dataset import UserItemRatingDFDataset
from src.algo.mf.model import MatrixFactorizationRating
from src.eval.log_metrics import (log_classification_metrics,
                                  log_ranking_metrics)
from src.eval.utils import (create_label_df, create_rec_df,
                            merge_recs_with_target)


class MFLitModule(L.LightningModule):
    def __init__(
        self,
        model: MatrixFactorizationRating,
        user_col: str = "user_id",
        item_col: str = "parent_asin",
        timestamp_col: str = "timestamp",
        rating_col: str = "rating", 
        learning_rate: float = 0.001,
        l2_reg: float = 0.0,   
        log_dir: str = ".",
        idm: BaseModel = None,
        top_K: int = 100,
        top_k: int = 10,
        accelerator: str = "cpu",

    ):
        super().__init__()
        self.model = model
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.timestamp_col = timestamp_col
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.log_dir = log_dir
        self.idm = idm
        self.top_K = top_K 
        self.top_k = top_k
        self.accelerator = accelerator

    def training_step(self, batch, batch_idx):
        if not isinstance(self.trainer.train_dataloader.dataset, self.model.get_default_dataset()):
            raise ValueError(
                "Training dataset must be an instance of UserItemRatingDFDataset"
            )
        
        input_user_ids = batch["user"]
        input_item_ids = batch["item"]

        labels = batch["rating"].float()
        
        predictions = self.model.forward(input_user_ids, input_item_ids).view(labels.shape)

        loss_fn = self._get_loss_fn()
        loss = loss_fn(predictions, labels)

        # https://lightning.ai/docs/pytorch/stable/visualize/logging_advanced.html#in-lightningmodule
        self.log("train_loss", loss, prog_bar=True, logger=True, sync_dist=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_user_ids = batch["user"]
        input_item_ids = batch["item"]

        labels = batch["rating"].float()

        predictions = self.model.forward(input_user_ids, input_item_ids).view(labels.shape)
        loss_fn = self._get_loss_fn()
        loss = loss_fn(predictions, labels)

        # https://lightning.ai/docs/pytorch/stable/visualize/logging_advanced.html#in-lightningmodule
        self.log("val_loss", loss, prog_bar=True, logger=True, sync_dist=True, on_epoch=True)

        return loss
    
    def configure_optimizers(self):
        # Create the optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_reg,
        )

        # Create the scheduler: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.3, patience=2
            ),
            "monitor": "val_loss",
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def on_validation_epoch_end(self):
        sch = self.lr_schedulers()

        # Decay the learning rate if the validation loss has not improved
        if sch is not None:
            self.log("learning_rate", sch.get_last_lr()[0], sync_dist=True)

    def on_fit_end(self):
        self.model = self.model.to(self._get_device())
        logger.info(f"Logging classification metrics...")
        self._log_classification_metrics()
        
        logger.info(f"Logging ranking metrics...")
        self._log_ranking_metrics()

        logger.info(f"Evidently metrics are available at: {os.path.abspath(self.log_dir)}")


    def _log_classification_metrics(
        self,
    ):
        # Need to call model.eval() here to disable dropout and batchnorm at prediction
        # Else the model output score would be severely affected
        self.model.eval()

        val_loader = self.trainer.val_dataloaders

        labels = []
        classifications = []

        for _, batch_input in enumerate(val_loader):
            _input_user_ids = batch_input["user"].to(self._get_device())
            _input_item_ids = batch_input["item"].to(self._get_device())
            _labels = batch_input["rating"].to(self._get_device())
            _classifications = self.model.predict(
                _input_user_ids, _input_item_ids
            ).view(_labels.shape)

            labels.extend(_labels.cpu().detach().numpy())
            classifications.extend(_classifications.cpu().detach().numpy())

        eval_classification_df = pd.DataFrame(
            {
                "labels": labels,
                "classification_proba": classifications,
            }
        ).assign(label=lambda df: df["labels"].gt(0).astype(int))

        self.eval_classification_df = eval_classification_df

        # Evidently
        target_col = "label"
        prediction_col = "classification_proba"
        column_mapping = ColumnMapping(target=target_col, prediction=prediction_col)
        classification_performance_report = Report(
            metrics=[
                ClassificationPreset(probas_threshold=0.731),
            ]
        )

        classification_performance_report.run(
            reference_data=None,
            current_data=eval_classification_df[[target_col, prediction_col]],
            column_mapping=column_mapping,
        )

        evidently_report_fp = f"{self.log_dir}/evidently_report_classification.html"
        os.makedirs(self.log_dir, exist_ok=True)
        classification_performance_report.save_html(evidently_report_fp)

        if "mlflow" in str(self.logger.__class__).lower():
            run_id = self.logger.run_id
            mlf_client = self.logger.experiment
            mlf_client.log_artifact(run_id, evidently_report_fp)
            for metric_result in classification_performance_report.as_dict()["metrics"]:
                metric = metric_result["metric"]
                if metric == "ClassificationQualityMetric":
                    roc_auc = float(metric_result["result"]["current"]["roc_auc"])
                    mlf_client.log_metric(run_id, f"val_roc_auc", roc_auc)
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
                        mlf_client.log_metric(
                            run_id,
                            f"val_precision_at_prob_as_threshold_step",
                            precision,
                            step=prob,
                        )
                        mlf_client.log_metric(
                            run_id,
                            f"val_recall_at_prob_as_threshold_step",
                            recall,
                            step=prob,
                        )
                    break
    
    def _log_ranking_metrics(self):
        self.model.eval()

        timestamp_col = self.timestamp_col
        rating_col = self.rating_col
        user_col = "user_indice"
        item_col = "item_indice"
        top_K = self.top_K
        top_k = self.top_k
        idm = self.idm

        val_df = self.trainer.val_dataloaders.dataset.df

        # Prepare input dataframe for prediction where user_indice is unique and item_sequence contains the last interaction in training data
        # Retain the first row of each user and use that as input for recommendations
        # We would compare that predictions with all the items this customer rates in val set
        to_rec_df = val_df.sort_values(timestamp_col, ascending=True).drop_duplicates(
            subset=["user_indice"]
        )
        recommendations = self.model.recommend(
            torch.tensor(to_rec_df["user_indice"].values, device=self._get_device()),
            top_k=top_K,
            batch_size=4,
        )
        # print(f"Recommendations: {recommendations}")

        recommendations_df = pd.DataFrame(recommendations).pipe(
            create_rec_df, idm
        ).rename(
            columns={
                "recommendation": item_col,
            }
        )
        # print(f"Recommendations_df: {recommendations_df}")

        label_df = create_label_df(
            val_df,
            user_col=user_col,
            item_col=item_col,
            rating_col=rating_col,
            timestamp_col=timestamp_col,
        )

        # print("Label_df: ", label_df)

        eval_df = merge_recs_with_target(
            recommendations_df,
            label_df,
            k=top_K,
            user_col=user_col,
            item_col=item_col,
            rating_col=rating_col,
        )

        # print("Eval_df: ", eval_df)

        self.eval_ranking_df = eval_df

        column_mapping = ColumnMapping(
            recommendations_type="rank",
            target=rating_col,
            prediction="rec_ranking",
            item_id=item_col,
            user_id=user_col,
        )

        report = Report(
            metrics=[
                NDCGKMetric(k=top_k),
                RecallTopKMetric(k=top_K),
                PrecisionTopKMetric(k=top_K),
                FBetaTopKMetric(k=top_k),
                PersonalizationMetric(k=top_k),
            ],
        )

        report.run(
            reference_data=None, current_data=eval_df, column_mapping=column_mapping
        )

        evidently_report_fp = f"{self.log_dir}/evidently_report_ranking.html"
        os.makedirs(self.log_dir, exist_ok=True)
        report.save_html(evidently_report_fp)
        # print(report.as_dict())

        if "mlflow" in str(self.logger.__class__).lower():
            run_id = self.logger.run_id
            mlf_client = self.logger.experiment
            mlf_client.log_artifact(run_id, evidently_report_fp)
            for metric_result in report.as_dict()["metrics"]:
                metric = metric_result["metric"]
                if metric == "PersonalizationMetric":
                    metric_value = float(metric_result["result"]["current_value"])
                    mlf_client.log_metric(run_id, f"val_{metric}", metric_value)
                    continue
                result = metric_result["result"]["current"].to_dict()
                for kth, metric_value in result.items():
                    mlf_client.log_metric(
                        run_id, f"val_{metric}_at_k_as_step", metric_value, step=kth
                    )


    def _get_loss_fn(self):
        """
        Get the default loss function for the model.
        """
        #https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
        return nn.MSELoss()
    
    def _get_device(self):
        return self.accelerator