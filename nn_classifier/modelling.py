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
"""Fully-connected neural network classifier"""

import sys
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


class FcnBinaryClassifier(nn.Module):
    """
    A fully-connected neural network with a single hidden layer and batchnorm for binary classification.

    Architecture:
        Linear(input_size, hidden_size)
        ReLU()
        BatchNorm()
        Dropout()
        Linear(hidden_size, 1)

    Args:
        input_size: size of the input vector
        hidden_size: size of the hidden layer
        dropout_prob: dropout parameter
        use_batch_norm: if True, add BatchNorm between layers
    """

    def __init__(self, input_size, hidden_size, dropout_prob=0.5, use_batch_norm=False):
        super().__init__()
        # As we want a more flexible network than in the Module 2 assignment
        # (i.e., the network should have batch normalization if use_batch_norm is True),
        # using nn.Sequential is not a good idea.
        # Instead, we will create three attributes: input_layer, batch_norm and output_layer.
        # Note that we do not create attributes for nonlinearities (ReLU and Sigmoid)
        # as they can be used directly in .forward() using functional interface
        # (e.g., torch.relu and torch.sigmoid).
        # However, even though dropout can also be called using F.dropout, we do not recommend doing it
        # the nn.Dropout object automatically turns of dropout in test mode while F.dropout doesn't
        # TASK 1.1 (2 points):
        # 1. Create a Linear layer object that projects from input_size to hidden_size and assign it to .input_layer
        # 2. If use_batch_norm, create nn.BatchNorm1d object and assign it to .batch_norm
        # 3. Create dropout object
        # 4. Create a Linear layer object that projects from hidden_size to 1 and assign it to .output_layer
        # Our implementation is 6 lines of code (not counting the line breaks)
        # YOUR CODE STARTS

        super(FcnBinaryClassifier, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.batch_norm = nn.BatchNorm1d(input_size) if use_batch_norm else None
        self.dropout = nn.Dropout(p=dropout_prob)
        self.output_layer = nn.Linear(hidden_size, 1)

        # YOUR CODE ENDS

    def forward(self, x):
        """
        Args:
            x: torch.FloatTensor[batch_size, input_size]

        Returns:
            torch.FloatTensor[batch_size,] probabilities of a positive class for each example in the batch
        """
        # TASK 1.2 (1 point): call the layers in the right order and apply nonlinearities.
        # 1. Use torch.relu after input layer
        # 2. If batch norm is specified, apply it after ReLU
        # 3. Apply sigmoid to get probabilities after output_layer
        # YOUR CODE STARTS

        x = self.input_layer(x)
        x = F.relu(x)
        if self.batch_norm:
            x = self.dropout(x)
        x = self.output_layer(x)
        prob = F.sigmoid(x)

        # YOUR CODE ENDS

        return prob
