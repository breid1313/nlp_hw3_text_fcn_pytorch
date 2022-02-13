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
import torch.nn.functional as F

from nn_classifier.modelling import FcnBinaryClassifier


def test_model():
    batch_size, input_size, hidden_size = 3, 11, 7
    x = torch.randn(batch_size, input_size)

    model = FcnBinaryClassifier(input_size, hidden_size, use_batch_norm=False)
    out = model(x)

    assert (
        model.batch_norm is None
    ), "model should not have batch norm if it was not specified during initialization"
    assert out.shape == (batch_size, 1)

    model = FcnBinaryClassifier(input_size, hidden_size, use_batch_norm=True)
    out = model(x)

    _err_msg = "model should have batch norm if it was specified during initialization"
    assert isinstance(model.batch_norm, torch.nn.BatchNorm1d), _err_msg
    assert out.shape == (batch_size, 1)
