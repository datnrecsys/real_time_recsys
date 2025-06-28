from typing import cast

import lightning as L
import torch
import torch.nn as nn
from pydantic import BaseModel
from torchmetrics import AUROC, AveragePrecision

from src.algo.sequence.model import SequenceRatingPrediction
from src.domain.model_request import SequenceModelRequest
from src.eval.recommendation import RankingMetricComputer


class SeqModellingLitModule(L.LightningModule):
    def __init__(
        self,
        model: SequenceRatingPrediction,
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
        negative_samples: int = 1,

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
        self.val_roc_auc_metric = AUROC(task="binary")
        self.val_pr_auc_metric = AveragePrecision(task="binary")
        self.negative_samples = negative_samples

    def training_step(self, batch, batch_idx):
        if not isinstance(self.trainer.train_dataloader.dataset, self.model.get_default_dataset()):
            raise ValueError(
                "Training dataset must be an instance of UserItemBinaryRatingDFDataset"
            )
        
        input_user_ids = batch["user"]
        input_item_ids = batch["item"]
        input_item_sequences = batch["item_sequence"].int()

        labels = batch["rating"].float()

        predictions = self.model.forward(SequenceModelRequest(
            user_id=input_user_ids,
            item_sequence=input_item_sequences,
            target_item=input_item_ids,
            recommendation=False
        )).view(labels.shape)
        # print(predictions)

        loss_fn = self._get_loss_fn()


        # print(predictions)
        # print(labels)

        loss = loss_fn(predictions, labels)
        # print(loss)

        # https://lightning.ai/docs/pytorch/stable/visualize/logging_advanced.html#in-lightningmodule
        self.log("train_loss", loss, prog_bar=True, logger=True, sync_dist=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_user_ids = batch["user"]
        input_item_ids = batch["item"]
        input_item_sequences = batch["item_sequence"].int()
        

        labels = cast(torch.Tensor, batch["rating"].float())

        predictions = self.model.forward(SequenceModelRequest(
            user_id=input_user_ids,
            item_sequence=input_item_sequences,
            target_item=input_item_ids,
            recommendation=False
        )).view(labels.shape)
        
        predictions = nn.Sigmoid()(predictions)
        loss_fn = nn.BCELoss()
        loss = loss_fn(predictions, labels)
        
        # Update AUROC with current batch predictions and labels
        self.val_roc_auc_metric.update(predictions, labels.int())
        # Update PR-AUC with current batch predictions and labels
        self.val_pr_auc_metric.update(predictions, labels.int())

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
    
    def on_train_epoch_end(self):
        # Log model weights
        for name, param in self.named_parameters():
            if param.requires_grad:
                self.log(f"weights/{name}", param.norm(), sync_dist=True)
                
    def on_after_backward(self) -> None:
        
        # this is called right after loss.backward()
        for name, param in self.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            # log gradient norm at each step
            # on_step=True logs it every training step
            # on_epoch=False since you'll already get it per step
            self.log(
                f"gradients/{name}",
                param.grad.norm(),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=True
            )

    def on_validation_epoch_end(self):
        sch = self.lr_schedulers()

        # Decay the learning rate if the validation loss has not improved
        if sch is not None:
            self.log("learning_rate", sch.get_last_lr()[0], sync_dist=True)
            
        # Compute and log the final ROC-AUC for the epoch
        roc_auc = self.val_roc_auc_metric.compute()
        pr_auc = self.val_pr_auc_metric.compute()

        self.log(
            "val_roc_auc",
            roc_auc,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_pr_auc",
            pr_auc,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # Reset the metric for the next epoch
        self.val_roc_auc_metric.reset()
        self.val_pr_auc_metric.reset()


    def _get_loss_fn(self):
        """
        Get the default loss function for the model.
        """
        #https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
        return nn.BCEWithLogitsLoss(pos_weight= torch.tensor(self.negative_samples, device=self._get_device()))

    def _get_device(self):
        return self.accelerator
    
    # def test_step(self, batch, batch_idx):
    #     """
    #     Test step for the model.
    #     """
    #     val_df = self.trainer.test_dataloaders.dataset.df
        
        
    #     val_recs_df = val_df.sort_values(by=self.timestamp_col).drop_duplicates(subset=[self.user_col], keep="first")
        
    #     item_rec = RankingMetricComputer(
    #         self.model,
    #         mlf_client=self.logger.experiment,
    #         batch_size=1024,
    #         evidently_report_fp = f"{self.log_dir}",
    #     )
    #     try: 
    #         run_id = self.logger.run_id
    #     except Exception :
    #         run_id = None
            
    #     report = item_rec.calculate(
    #         val_recs_df,
    #         run_id=run_id,
    #         log_to_mlflow=True,
    #         device=self._get_device()
    #     )
        
    #     return report 
        
    # def on_fit_end(self):
    def _log_ranking_report(self):
        """
        Log the ranking report to MLflow.
        """
        model = self.model.eval()
        val_df = self.trainer.val_dataloaders.dataset.df
        print(val_df)

        val_recs_df = val_df.sort_values(by=self.timestamp_col).drop_duplicates(subset=["user_indice"], keep="first")

        item_rec = RankingMetricComputer(
            model,
            mlf_client=self.logger.experiment,
            batch_size=1,
            evidently_report_fp = f"{self.log_dir}",
        )
        
        try: 
            run_id = self.logger.run_id
        except Exception :
            run_id = None
            
        report = item_rec.calculate(
            val_recs_df,
            run_id=run_id,
            log_to_mlflow=True,
            device=self._get_device()
        )
        
        return report
        