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
"""Test CountDataset methods.

Note that we repeat a piese of code quite a number of times here.
Usually, this is a questionable practice but it our opinion this makes the tests more independent and easier to modify.
"""
import pytest

import torch

from nn_classifier.data_utils import CountDataset
from nn_classifier.utils import make_whitespace_tokenizer


def test_dataset_init():
    # This test will fail if the dataset is unable to build
    texts = ["text number one", "text number two"]
    labels = [0, 1]
    tokenizer = make_whitespace_tokenizer(texts)

    count_dataset = CountDataset(texts, tokenizer, labels)


def test_convert_text_to_tensor():
    # Very simple and important tests to a function that returns a tensor test tensor shape and data type
    texts = ["text number one", "text number two"]
    labels = [0, 1]
    tokenizer = make_whitespace_tokenizer(texts)

    count_dataset = CountDataset(texts, tokenizer, labels)

    _t = count_dataset._convert_text_to_tensor(texts[0])
    assert isinstance(_t, torch.Tensor), f"the output should be a torch.Tensor, got {type(_t)} instead"
    assert _t.shape == (
        len(tokenizer.encode(texts[0]).ids),
    ), "the output tensor should be of the shape (n_tokens,)"
    _err_msg = (
        "each tensor value should correspond to a token id, found a value that is bigger than the maximum id"
    )
    assert torch.max(_t) < tokenizer.get_vocab_size(), _err_msg
    assert torch.max(_t) > 0, "tensor should have positive values for the text that has non-UNK tokens"
    assert torch.min(_t) >= 0, "tensor should only have nonnegative values"


def test_len():
    texts = ["text number one", "text number two"]
    labels = [0, 1]
    tokenizer = make_whitespace_tokenizer(texts)

    count_dataset = CountDataset(texts, tokenizer, labels)

    assert len(count_dataset) == len(
        texts
    ), f"the length computation is not correct, expected {len(texts)} got {len(count_dataset)} instead"


def test_getitem():
    texts = ["text number one", "text number two"]
    labels = [0, 1]
    tokenizer = make_whitespace_tokenizer(texts)

    count_dataset = CountDataset(texts, tokenizer, labels)

    item, label = count_dataset[0]
    _err_msg = (
        f"CountVector has wrong shape, expected {(tokenizer.get_vocab_size(),)} got {item.shape} instead"
    )
    assert item.shape == (tokenizer.get_vocab_size(),), _err_msg
    assert torch.all(item >= 0), "all elements of the vector should be nonnegative"
    assert label is not None
    assert torch.all(0 <= label <= 1), "all labels should be either 0 or 1"
