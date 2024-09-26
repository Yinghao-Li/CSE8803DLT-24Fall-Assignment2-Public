"""
# Author: Yinghao Li
# Modified: September 26th, 2023
# ---------------------------------------
# Description: arguments and configurations
"""

import os.path as osp
import json
import torch
import logging
from functools import cached_property
from dataclasses import dataclass, field
from .utils.data import entity_to_bio_labels
from .utils.io import prettify_json

logger = logging.getLogger(__name__)


@dataclass
class Arguments:
    """
    Arguments regarding the training of Neural hidden Markov Model
    """

    # --- student arguments ---
    name: str = field(default=None, metadata={"help": "name of the student model", "nargs": "+"})
    gtid: str = field(default=None, metadata={"help": "GTID of the student"})

    # --- manage directories and IO ---
    data_dir: str = field(default="./data", metadata={"help": "Directory to datasets"})
    bert_model_name_or_path: str = field(
        default="distilbert-base-uncased",
        metadata={
            "help": "Path to pretrained BERT model or model identifier from huggingface.co/models; "
            "Used to construct BERT embeddings if not exist"
        },
    )
    log_path: str = field(
        default=osp.join("log", "record.log"),
        metadata={"help": "Path to log directory"},
    )

    # --- training arguments ---
    lr: float = field(default=5e-5, metadata={"help": "learning rate"})
    batch_size: int = field(default=16, metadata={"help": "model training batch size"})
    n_epochs: int = field(default=20, metadata={"help": "number of denoising model training epochs"})
    warmup_ratio: float = field(default=0.1, metadata={"help": "ratio of warmup steps"})
    lr_scheduler_type: str = field(
        default="linear",
        metadata={
            "help": "Learning rate scheduler with warm ups defined in `transformers`, Please refer to "
            "https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#schedules for details",
            "choices": [
                "linear",
                "cosine",
                "cosine_with_restarts",
                "polynomial",
                "constant",
                "constant_with_warmup",
            ],
        },
    )
    weight_decay: float = field(default=0.01, metadata={"help": "strength of weight decay"})
    seed: int = field(default=42, metadata={"help": "random seed"})

    # --- device arguments ---
    no_mps: bool = field(default=False, metadata={"help": "Disable MPS even when it is available"})
    no_cuda: bool = field(default=False, metadata={"help": "Disable CUDA even when it is available"})

    def __post_init__(self):
        assert osp.isfile(osp.join(self.data_dir, "train.json")), f"Training file does not exist!"

        if isinstance(self.name, list):
            self.name = " ".join(self.name)

    @cached_property
    def device(self) -> str:
        """
        The device used by this process.
        """
        if not self.no_cuda and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        return device


@dataclass
class Config(Arguments):
    entity_types = None
    bio_label_types = None

    # --- properties ---
    @property
    def n_ents(self):
        return len(self.entity_types)

    @property
    def n_lbs(self):
        return len(self.bio_label_types)

    # --- functions ---
    def get_meta(self):
        """
        Load meta file and update arguments
        """
        # Load meta if exist
        meta_dir = osp.join(self.data_dir, "meta.json")

        if not osp.isfile(meta_dir):
            raise FileNotFoundError("Meta file does not exist!")

        with open(meta_dir, "r", encoding="utf-8") as f:
            meta_dict = json.load(f)

        self.entity_types = meta_dict["entity_types"]
        self.bio_label_types = entity_to_bio_labels(meta_dict["entity_types"])

        return self

    def from_args(self, args):
        """
        Initialize configuration from arguments

        Parameters
        ----------
        args: arguments (parent class)

        Returns
        -------
        self (type: BertConfig)
        """
        arg_elements = {
            attr: getattr(args, attr)
            for attr in dir(args)
            if not callable(getattr(args, attr)) and not attr.startswith("__") and not attr.startswith("_")
        }
        for attr, value in arg_elements.items():
            try:
                setattr(self, attr, value)
            except AttributeError:
                pass
        return self

    def log(self):
        """
        Log all configurations
        """
        elements = {
            attr: getattr(self, attr)
            for attr in dir(self)
            if not callable(getattr(self, attr)) and not (attr.startswith("__") or attr.startswith("_"))
        }
        logger.info(f"Configurations:\n{prettify_json(json.dumps(elements, indent=2), collapse_level=2)}")

        return self
