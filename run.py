"""
# Author: Yinghao Li
# Modified: September 13th, 2023
# ---------------------------------------
# Description: Training and testing BERT for sequence labeling
"""

import logging
import os
import sys

from transformers import (
    HfArgumentParser,
    set_seed,
)

from src.args import Arguments, Config
from src.dataset import Dataset
from src.train import Trainer
from src.utils.io import set_logging

logger = logging.getLogger(__name__)


def main(args):
    config = Config().from_args(args).get_meta().log()
    assert config.name is not None, f"Student name is not specified!"
    assert config.gtid is not None, f"Student GTID is not specified!"
    logger.info(f"name: {config.name}")
    logger.info(f"GTID: {config.gtid}")

    logger.info("Loading datasets...")
    training_dataset = Dataset().prepare(config=config, partition="train")
    logger.info(f"Training dataset loaded, length={len(training_dataset)}")

    valid_dataset = Dataset().prepare(config=config, partition="valid")
    logger.info(f"Validation dataset loaded, length={len(valid_dataset)}")

    test_dataset = Dataset().prepare(config=config, partition="test")
    logger.info(f"Test dataset loaded, length={len(test_dataset)}")

    trainer = Trainer(
        config=config,
        training_dataset=training_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
    )

    trainer.run()

    return None


if __name__ == "__main__":
    # --- set up arguments ---
    parser = HfArgumentParser(Arguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        (arguments,) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (arguments,) = parser.parse_args_into_dataclasses()

    set_logging(log_path=arguments.log_path)
    set_seed(arguments.seed)

    main(args=arguments)
