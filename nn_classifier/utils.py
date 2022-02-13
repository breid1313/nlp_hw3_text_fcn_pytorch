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
from collections import Counter

import torch
import torch.nn as nn

import tokenizers
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace


def make_whitespace_tokenizer(texts, max_vocab_size=10_000, unk_token="UNK"):
    pre_tokenizer = Whitespace()
    tokenized_texts = [[w for w, _ in pre_tokenizer.pre_tokenize_str(t)] for t in texts]

    c = Counter()
    for text in tokenized_texts:
        c.update(text)

    token2id = {word: i + 1 for i, (word, count) in enumerate(c.most_common(max_vocab_size))}
    # usually, UNK is assigned index 0 or 1
    token2id[unk_token] = 0

    tokenizer = tokenizers.Tokenizer(WordLevel(token2id, unk_token))
    tokenizer.pre_tokenizer = pre_tokenizer
    return tokenizer


def accuracy(probs, targets):
    """Computes accuracy given predicted probabilities and expected labels.

    Args:
        probs: torch.FloatTensor[batch_size, 1], probabilities of a positive class
        targets: torch.LongTensor[batch_size, 1], true classes

    Returns:
        0 <= float <= 1, proportion of correct predictions
    """
    predictions = (probs >= 0.5).flatten()
    targets = targets.flatten()
    acc = torch.sum(predictions == targets).float() / targets.shape[0]
    acc = float(acc)

    return acc


def evaluate_model(model: nn.Module, dataloader, device=None):
    """Compute accuracy on the test set.

    Args:
        model: torch.nn.Module, model to evaluate
        dataloader: torch.utils.data.Dataloader, a dataloader with a test set
        device: str or torch.device

    Returns:
        0 <= float <= 1, proportion of correct predictions
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    all_probs = []
    all_labels = []

    # we will use this flag to correctly restore model state after evaluation
    was_training = model.training
    model.eval()

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            probs = model(x)

            all_probs.append(probs)
            all_labels.append(y)

    # remember to turn your model training mode back
    if was_training:
        model.train()

    all_probs_tensor = torch.cat(all_probs)  # concatenate all tensors into one big tensor
    all_labels_tensor = torch.cat(all_labels)

    acc = accuracy(all_probs_tensor, all_labels_tensor)

    return acc
