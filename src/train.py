"""
# Author: Yinghao Li
# Modified: September 13th, 2023
# ---------------------------------------
# Description: Trainer class for training BERT for sequence labeling
"""

import torch
import numpy as np
import logging
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_scheduler

from .args import Config
from .dataset import DataCollator, MASKED_LB_ID
from .utils.metric import get_ner_metrics
from .utils.container import CheckpointContainer

logger = logging.getLogger(__name__)


class Trainer:
    """
    Bert trainer used for training BERT for token classification (sequence labeling)
    """

    def __init__(
        self, config: Config, collate_fn=None, model=None, training_dataset=None, valid_dataset=None, test_dataset=None
    ):
        if not collate_fn:
            tokenizer = AutoTokenizer.from_pretrained(config.bert_model_name_or_path)
            collate_fn = DataCollator(tokenizer)
        self._config = config
        self._training_dataset = training_dataset
        self._valid_dataset = valid_dataset
        self._test_dataset = test_dataset
        self._collate_fn = collate_fn

        self._model = model
        self._optimizer = None
        self._scheduler = None
        self._loss = None
        self._device = config.device
        self._checkpoint_container = CheckpointContainer("metric-larger")
        self.initialize()

    def initialize(self):
        """
        Initialize the trainer's status and its key components including the model,
        optimizer, learning rate scheduler, and loss function.

        Returns
        -------
        self : Trainer
            Initialized Trainer instance.
        """
        self.initialize_model()
        self.initialize_optimizer()
        self.initialize_scheduler()
        self.initialize_loss()
        return self

    def initialize_model(self):
        self._model = AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=self._config.bert_model_name_or_path, num_labels=self._config.n_lbs
        )
        return self

    def initialize_optimizer(self):
        """
        Initialize training optimizer
        """
        self._optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=self._config.lr, weight_decay=self._config.weight_decay
        )
        return self

    def initialize_scheduler(self):
        """
        Initialize learning rate scheduler
        """
        num_update_steps_per_epoch = int(np.ceil(len(self._training_dataset) / self._config.batch_size))
        num_warmup_steps = int(np.ceil(num_update_steps_per_epoch * self._config.warmup_ratio * self._config.n_epochs))
        num_training_steps = int(np.ceil(num_update_steps_per_epoch * self._config.n_epochs))

        self._scheduler = get_scheduler(
            self._config.lr_scheduler_type,
            self._optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        return self

    def initialize_loss(self):
        """
        Initialize loss function
        """
        self._loss = torch.nn.CrossEntropyLoss(reduction="mean")
        return self

    def run(self):
        # ----- start training process -----
        logger.info("Start training...")
        for epoch_i in range(self._config.n_epochs):
            logger.info("")
            logger.info(f"Epoch {epoch_i + 1} of {self._config.n_epochs}")

            training_dataloader = self.get_dataloader(self._training_dataset, shuffle=True)
            train_loss = self.training_step(training_dataloader)
            logger.info(f"Training loss: {train_loss:.4f}")

            self.eval_and_save()

        best_valid_result = self.test(self._valid_dataset)
        logger.info("")
        logger.info("Best validation result:")
        self.log_results(best_valid_result, detailed=True)

        test_results = self.test()
        logger.info("")
        logger.info("Test results:")
        self.log_results(test_results, detailed=True)

        return None

    def training_step(self, data_loader):
        """
        For each training epoch
        """
        train_loss = 0
        n_tks = 1e-9

        self._model.to(self._device)
        self._model.train()
        self._optimizer.zero_grad()

        for batch in tqdm(data_loader):
            batch.to(self._device)

            outputs = self._model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, labels=batch.labels)
            loss = self.get_loss(outputs.logits, batch.labels)
            assert torch.abs(loss - outputs.loss) < 1e-6, ValueError("Loss mismatch!")

            # Do backpropagation and update the model parameters, and track the training loss.
            # `train_loss` is the summarized loss for all tokens involved in backpropagation.
            # --- TODO: start of your code ---

            # --- TODO: end of your code ---

        return train_loss / n_tks

    def get_loss(self, logits, lbs):
        """
        Get loss for a batch of data.

        Parameters
        ----------
        logits : torch.Tensor
            Output logits from the model.
        lbs : torch.Tensor
            Ground truth label ids.

        Returns
        -------
        loss : torch.Tensor
            Loss for the batch of data.
        """
        # Compute the loss for the batch of data.
        # Your result should match the result from `outputs.loss`.
        # --- TODO: start of your code ---

        # --- TODO: end of your code ---

    def eval_and_save(self):
        """
        Evaluate the model and save it if its performance exceeds the previous highest
        """
        valid_results = self.evaluate(self._valid_dataset)

        logger.info("Validation results:")
        self.log_results(valid_results)

        # ----- check model performance and update buffer -----
        if self._checkpoint_container.check_and_update(self._model, valid_results["f1"]):
            logger.info("Model buffer is updated!")

        return None

    def evaluate(self, dataset, detailed=False):
        self._model.to(self._device)
        self._model.eval()
        data_loader = self.get_dataloader(dataset)

        pred_lbs: list[list[str]] = None

        # TODO: Predicted labels for each sample in the dataset and stored in `pred_lbs`, a list of list of strings.
        # TODO: The string elements represent the enitity labels, such as "O" or "B-PER".
        # --- TODO: start of your code ---

        # --- TODO: end of your code ---

        metric = get_ner_metrics(dataset.lbs, pred_lbs, detailed=detailed)
        return metric

    def test(self, dataset=None):
        if dataset is None:
            dataset = self._test_dataset
        self._model.load_state_dict(self._checkpoint_container.state_dict)
        metrics = self.evaluate(dataset, detailed=True)
        return metrics

    @staticmethod
    def log_results(metrics, detailed=False):
        if detailed:
            for key, val in metrics.items():
                logger.info(f"[{key}]")
                for k, v in val.items():
                    logger.info(f"  {k}: {v:.4f}.")
        else:
            for k, v in metrics.items():
                logger.info(f"  {k}: {v:.4f}.")

    def get_dataloader(
        self,
        dataset,
        shuffle: bool = False,
        batch_size: int = 0,
    ):
        """
        Create a DataLoader for the provided dataset.

        Parameters
        ----------
        dataset : Dataset
            Dataset for which the DataLoader is to be created.
        shuffle : bool, optional
            Whether to shuffle the data. Defaults to False.
        batch_size : int, optional
            Batch size for the DataLoader. If not provided, will use the batch size from the configuration.

        Returns
        -------
        DataLoader
            Returns the created DataLoader for the provided dataset.
        """
        try:
            dataloader = DataLoader(
                dataset=dataset,
                collate_fn=self._collate_fn,
                batch_size=batch_size if batch_size else self._config.batch_size,
                num_workers=getattr(self._config, "num_workers", 0),
                pin_memory=getattr(self._config, "pin_memory", False),
                shuffle=shuffle,
                drop_last=False,
            )
        except Exception as e:
            logger.exception(e)
            raise e

        return dataloader
