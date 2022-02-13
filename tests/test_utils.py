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
import pytest

import torch

from nn_classifier import data_utils, modelling, utils


def test_make_whitespace_tokenizer():
    dataset = ["a list of sentences from my dataset", "this is a text with known words"]
    text_to_encode = "this is text"
    text_with_unk = "a text with unknown_word"

    tokenizer = utils.make_whitespace_tokenizer(dataset)
    text_ids = tokenizer.encode(text_to_encode).ids

    assert isinstance(text_ids, list)
    assert len(text_ids) == 3
    assert tokenizer.get_vocab_size() == 14
    assert tokenizer.encode(text_to_encode)

    unk_token_id = tokenizer.model.token_to_id(tokenizer.model.unk_token)
    unk_text_ids = tokenizer.encode(text_with_unk).ids
    assert unk_text_ids[-1] == unk_token_id


def test_accuracy():
    probs = torch.FloatTensor([0.6, 0.7, 0.8, 0.1, 0.1, 0]).reshape(-1, 1)
    targets = torch.LongTensor([1, 0, 1, 0, 1, 0]).reshape(-1, 1)
    expected_acc = 4.0 / 6
    acc = utils.accuracy(probs, targets)

    assert isinstance(acc, float)
    assert (
        round(acc - expected_acc, 7) == 0
    ), f"computed accuracy {acc} is incorrect, expected {expected_acc}, got {acc}"


def test_accuracy_rand():
    torch.manual_seed(42)
    probs = torch.randn(11, 1)
    targets = torch.randint(2, size=(11,))
    acc = utils.accuracy(probs, targets)

    assert isinstance(acc, float), "accuracy should be a float object (not a torch Tensor)"
    assert 0 < acc < 1, f"for this random seed accuracy shouldn't be exactly 0 or 1, but you got {acc}"


def test_evaluate_model():
    dataset_size, input_size, hidden_size = 5, 7, 11
    model = modelling.FcnBinaryClassifier(input_size, hidden_size)

    inputs = torch.randn(dataset_size, input_size)
    labels = torch.randint(2, size=(dataset_size,))
    dataset = torch.utils.data.TensorDataset(inputs, labels)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

    acc = utils.evaluate_model(model, dataloader, "cpu")
    assert 0 < acc < 1
    assert isinstance(acc, float)
