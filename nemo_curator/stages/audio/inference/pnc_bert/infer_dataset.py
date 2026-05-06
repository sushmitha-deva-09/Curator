# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
from typing import Any

import numpy as np
import torch
from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.core import Dataset
from nemo.core.neural_types import ChannelType, Index, MaskType, NeuralType
from nemo.core.neural_types.elements import BoolType
from nemo.utils import logging
from numpy import ndarray
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


def get_features_infer(
    queries: list[str],
    tokenizer: TokenizerSpec,
    max_seq_length: int = 64,
    step: int | None = 8,
    margin: int | None = 16,
) -> tuple[
    list[list[int]],
    list[list[int]],
    list[list[int]],
    list[list[int]],
    list[int],
    list[int],
    list[bool],
    list[bool],
]:
    """
    Processes the data and returns features.

    Args:
        queries: text sequences
        tokenizer: such as AutoTokenizer
        max_seq_length: max sequence length minus 2 for [CLS] and [SEP]
        step: relative shift of consequent segments into which long queries are split.
        margin: number of subtokens near edges of segments which are not used for prediction.

    Returns:
        Tuple of (all_input_ids, all_segment_ids, all_input_mask, all_subtokens_mask,
        all_quantities_of_preceding_words, all_query_ids, all_is_first, all_is_last)
    """
    st = []
    stm = []
    sent_lengths = []
    for query in queries:
        subtokens, subtokens_mask = _get_subtokens_and_subtokens_mask(query, tokenizer)
        sent_lengths.append(len(subtokens))
        st.append(subtokens)
        stm.append(subtokens_mask)

    _check_max_seq_length_and_margin_and_step(max_seq_length, margin, step)
    if max_seq_length > max(sent_lengths) + 2:
        max_seq_length = max(sent_lengths) + 2
        step = 1
        length = max_seq_length - 2
    else:
        length = max_seq_length - 2
        step = min(length - margin * 2, step)
    logging.info(f"Max length: {max_seq_length}")

    all_input_ids, all_segment_ids, all_subtokens_mask, all_input_mask = [], [], [], []
    all_quantities_of_preceding_words, all_query_ids, all_is_first, all_is_last = [], [], [], []
    for q_i, query_st in enumerate(st):
        q_inp_ids, q_segment_ids, q_subtokens_mask, q_inp_mask, q_quantities_of_preceding_words = [], [], [], [], []
        for i in range(0, max(len(query_st), length) - length + step, step):
            subtokens = [tokenizer.cls_token, *query_st[i : i + length], tokenizer.sep_token]
            q_inp_ids.append(tokenizer.tokens_to_ids(subtokens))
            q_segment_ids.append([0] * len(subtokens))
            q_subtokens_mask.append([False, *stm[q_i][i : i + length], False])
            q_inp_mask.append([True] * len(subtokens))
            q_quantities_of_preceding_words.append(np.count_nonzero(stm[q_i][:i]))
        all_input_ids.append(q_inp_ids)
        all_segment_ids.append(q_segment_ids)
        all_subtokens_mask.append(q_subtokens_mask)
        all_input_mask.append(q_inp_mask)
        all_quantities_of_preceding_words.append(q_quantities_of_preceding_words)
        all_query_ids.append([q_i] * len(q_inp_ids))
        all_is_first.append([True] + [False] * (len(q_inp_ids) - 1))
        all_is_last.append([False] * (len(q_inp_ids) - 1) + [True])
    return (
        list(itertools.chain(*all_input_ids)),
        list(itertools.chain(*all_segment_ids)),
        list(itertools.chain(*all_input_mask)),
        list(itertools.chain(*all_subtokens_mask)),
        list(itertools.chain(*all_quantities_of_preceding_words)),
        list(itertools.chain(*all_query_ids)),
        list(itertools.chain(*all_is_first)),
        list(itertools.chain(*all_is_last)),
    )


def _check_max_seq_length_and_margin_and_step(max_seq_length: int, margin: int, step: int) -> None:
    """Checks values of ``max_seq_length``, ``margin``, and ``step``."""
    min_seq_length = 3
    if max_seq_length < min_seq_length:
        msg = (
            f"Parameter `max_seq_length={max_seq_length}` cannot be less than 3 because `max_seq_length` is a length "
            f"of a segment with [CLS] and [SEP] tokens."
        )
        raise ValueError(msg)
    if (margin >= (max_seq_length - 2) // 2 and margin > 0) or margin < 0:
        msg = (
            f"Parameter `margin` has to be not negative and less than `(max_seq_length - 2) // 2`. Don't forget about "
            f"CLS and EOS tokens in the beginning and the end of segment. margin={margin}, "
            f"max_seq_length={max_seq_length}"
        )
        raise ValueError(msg)
    if step <= 0:
        msg = f"Parameter `step` has to be positive whereas step={step}"
        raise ValueError(msg)
    if step > max_seq_length - 2 - 2 * margin:
        logging.warning(
            f"Parameter step={step} is too big. It will be reduced to `min(max_seq_length, <maximum query length> + 2) "
            f"- 2 - 2 * margin`."
        )


def _get_subtokens_and_subtokens_mask(query: str, tokenizer: TokenizerSpec) -> tuple[list[str], list[bool]]:
    """Tokenizes a query into subtokens and produces a first-subtoken mask."""
    words = query.strip().split()
    subtokens = []
    subtokens_mask = []
    for word in words:
        word_tokens = tokenizer.text_to_tokens(word)
        subtokens.extend(word_tokens)
        subtokens_mask.append(True)
        subtokens_mask.extend([False] * (len(word_tokens) - 1))
    return subtokens, subtokens_mask


class BertPunctuationCapitalizationInferDataset(Dataset):
    """Creates dataset to use during inference for punctuation and capitalization tasks.

    Args:
        queries: text sequences
        tokenizer: such as AutoTokenizer
        max_seq_length: max sequence length minus 2 for [CLS] and [SEP]
        step: relative shift of consequent segments into which long queries are split.
        margin: number of subtokens near edges of segments which are not used for prediction.
    """

    @property
    def output_types(self) -> dict[str, NeuralType] | None:
        """Returns neural types of :meth:`collate_fn` output."""
        return {
            "input_ids": NeuralType(("B", "T"), ChannelType()),
            "segment_ids": NeuralType(("B", "T"), ChannelType()),
            "input_mask": NeuralType(("B", "T"), MaskType()),
            "subtokens_mask": NeuralType(("B", "T"), MaskType()),
            "quantities_of_preceding_words": NeuralType(("B",), Index()),
            "query_ids": NeuralType(("B",), Index()),
            "is_first": NeuralType(("B",), BoolType()),
            "is_last": NeuralType(("B",), BoolType()),
        }

    def __init__(
        self,
        queries: list[str],
        tokenizer: TokenizerSpec,
        max_seq_length: int = 64,
        step: int = 8,
        margin: int = 16,
    ) -> None:
        features = get_features_infer(
            queries=queries,
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            step=step,
            margin=margin,
        )
        self.all_input_ids: list[list[int]] = features[0]
        self.all_segment_ids: list[list[int]] = features[1]
        self.all_input_mask: list[list[int]] = features[2]
        self.all_subtokens_mask: list[list[int]] = features[3]
        self.all_quantities_of_preceding_words: list[int] = features[4]
        self.all_query_ids: list[int] = features[5]
        self.all_is_first: list[bool] = features[6]
        self.all_is_last: list[bool] = features[7]

    def __len__(self) -> int:
        return len(self.all_input_ids)

    def collate_fn(
        self,
        batch: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, bool, bool]],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Any, Any, Any, Any]:
        """Collates samples into batches."""
        inp_ids, segment_ids, inp_mask, st_mask, n_preceding, query_ids, is_first, is_last = zip(*batch, strict=False)
        return (
            pad_sequence([torch.tensor(x) for x in inp_ids], batch_first=True, padding_value=0),
            pad_sequence([torch.tensor(x) for x in segment_ids], batch_first=True, padding_value=0),
            pad_sequence([torch.tensor(x) for x in inp_mask], batch_first=True, padding_value=0),
            pad_sequence([torch.tensor(x) for x in st_mask], batch_first=True, padding_value=0),
            n_preceding,
            query_ids,
            is_first,
            is_last,
        )

    def __getitem__(self, idx: int) -> tuple[ndarray, ndarray, ndarray, ndarray, int, int, bool, bool]:
        return (
            np.array(self.all_input_ids[idx]),
            np.array(self.all_segment_ids[idx]),
            np.array(self.all_input_mask[idx], dtype=np.float32),
            np.array(self.all_subtokens_mask[idx]),
            self.all_quantities_of_preceding_words[idx],
            self.all_query_ids[idx],
            self.all_is_first[idx],
            self.all_is_last[idx],
        )
