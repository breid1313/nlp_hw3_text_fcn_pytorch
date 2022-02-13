# Copyright 2020 Vladislav Lialin and Skillfactory LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Train a neural network classifier."""

import argparse
import logging
import os
import sys

import torch

import tokenizers
import toml

from nn_classifier import data_utils
from nn_classifier.modelling import FcnBinaryClassifier


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(os.path.basename(__file__))


def parse_args(args=None):
    parser = argparse.ArgumentParser()

    # fmt: off
    parser.add_argument("--model_dir",
                        help="path to the directory with the model, tokenizer and a toml config")
    parser.add_argument("--device", default=None, type=str,
                        help="device to train on, use GPU if available by default")
    # fmt: on

    args = parser.parse_args(args)

    return args


def preprocess_text(text, tokenizer):
    """
    Args:
        text: str, a text to preprocess
        tokenizer: Tokenizer object

    Returns:
        torch.FloatTensor[tokenizer.get_vocab_size(),], a CountVector corresponding to the text
    """
    # TASK 3.1:
    # Write text preprocessing
    # 1. Convert text to ids using tokenizer
    # 2. Convert ids to torch.LongTensor
    # 3. Use data_utils.convert_text_ids_to_count_vector to get count vectors
    # 4. Use .unsqueeze_() to add a batch dimension to the vector
    # We need to add batch dimension, because our model works with batches,
    # but during inference, the batch size will always be one.
    # Our implementation is 4 lines
    # YOUR CODE STARTS

    # YOUR CODE ENDS

    return count_vector


def main(args):

    logger.info("Loading tokenizer and config")
    tokenizer_path = os.path.join(args.model_dir, "tokenizer.json")
    tokenizer = tokenizers.Tokenizer.from_file(tokenizer_path)

    with open(os.path.join(args.model_dir, "args.toml")) as f:
        train_args = toml.load(f)

    logger.info("Loading model")
    # TASK 3.2:
    # 1. initialize the model instance using the parameters from train_args
    # set dropout to 0 as we do not need the dropout for inference
    # 2. use torch.load to load the state_dict from model_dir/model_checkpoint.pt
    # 3. use model.load_state_dict(state_dict) to load trained weights into the model
    # 4. turn on model test mode to switch batch norm to the evaluation mode
    # Our implementation is 8 lines
    # YOUR CODE STARTS

    # YOUR CODE ENDS

    logger.info("This model is trained to classify movie reviews into positive and negative,"
                "to interact with it just write a text that is similar to a movie review and press ENTER."
                "Press CTRL+C to exit.")

    while True:
        input_text = input("Review text: ")
        text_tensor = preprocess_text(input_text, tokenizer)

        p = model(text_tensor)

        if p > 0.5:
            logger.info(f"The text is positive")
        else:
            logger.info(f"The text is negative")

        logger.info(f"Probability of a positive sentiment: {float(p)}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
