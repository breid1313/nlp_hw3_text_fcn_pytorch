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

import torch
from tqdm.auto import tqdm


class CountDataset(torch.utils.data.Dataset):
    """
    A Dataset object to handle turning numericalized text into count tensors.

    Args:
        texts: List[str], a list of texts
        tokenizer: tokenizers.Tokenizer object
        labels: List[int] or numpy array, optional - classes corresponding to the texts
    """

    def __init__(self, texts, tokenizer, labels=None):
        if labels is not None and len(texts) != len(labels):
            raise ValueError("labels and texts should have the same number of elements")

        self.texts = texts
        self.tokenizer = tokenizer
        self.labels = labels

        # in order to save time during the training we tokenize and tensorize
        # all texts in the __init__
        # this technique sometimes called prefetching and it is very typical in NLP
        self._text_ids = [
            self._convert_text_to_tensor(t) for t in tqdm(self.texts, desc="Preprocessing Dataset")
        ]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """Turn the text at index idx into count vector

        and return it along the corresponding label (if labels were provided to the __init__)

        Returns:
            torch.Tensor[vocab_size,], torch.FloatTensor[1,] - count vector and (optionally) a label

            if the labels were not provided
        """

        count_vector = convert_text_ids_to_count_vector(self._text_ids[idx], self.tokenizer.get_vocab_size())
        label = None if self.labels is None else torch.FloatTensor([self.labels[idx]])

        if label is None:
            return count_vector
        return count_vector, label

    def _convert_text_to_tensor(self, text):
        """
        Tokenizes the text and makes a torch.LongTensor object.

        Args:
            text: str, a text to encode

        Returns:
            torch.LongTensor[n_tokens,]
        """
        text_ids = self.tokenizer.encode(text).ids
        text_ids_tensor = torch.LongTensor(text_ids)

        return text_ids_tensor


def convert_text_ids_to_count_vector(text_ids, vector_size):
    """
    Args:
        text_ids: list[int], numericalized text
        vector_size: int, size of the CountVector

    Returns:
        torch.FloatTensor[vector_size]
    """
    count_vector = torch.bincount(text_ids, minlength=vector_size)
    count_vector = count_vector.float()

    return count_vector
