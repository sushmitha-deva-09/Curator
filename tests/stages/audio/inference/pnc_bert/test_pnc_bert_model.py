# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for the ported BERT PNC inference module."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import pytest
import torch

from nemo_curator.stages.audio.inference.pnc_bert.classifier import Classifier
from nemo_curator.stages.audio.inference.pnc_bert.infer_dataset import (
    BertPunctuationCapitalizationInferDataset,
    _get_subtokens_and_subtokens_mask,
    get_features_infer,
)
from nemo_curator.stages.audio.inference.pnc_bert.model import (
    PunctuationCapitalizationModel,
    _load_label_ids,
)
from nemo_curator.stages.audio.inference.pnc_bert.token_classifier import TokenClassifier


class TestClassifier:
    def test_instantiation(self) -> None:
        clf = Classifier(hidden_size=64, dropout=0.1)
        assert clf._hidden_size == 64

    def test_dropout_layer(self) -> None:
        clf = Classifier(hidden_size=32)
        x = torch.randn(2, 10, 32)
        out = clf.dropout(x)
        assert out.shape == (2, 10, 32)


class TestTokenClassifier:
    def test_forward_shape(self) -> None:
        tc = TokenClassifier(hidden_size=128, num_classes=4, num_layers=1, dropout=0.1)
        x = torch.randn(3, 20, 128)
        out = tc(hidden_states=x)
        assert out.shape == (3, 20, 4)

    def test_multi_layer(self) -> None:
        tc = TokenClassifier(hidden_size=64, num_classes=8, num_layers=2)
        x = torch.randn(1, 5, 64)
        out = tc(hidden_states=x)
        assert out.shape == (1, 5, 8)

    def test_log_softmax_output(self) -> None:
        tc = TokenClassifier(hidden_size=32, num_classes=3, log_softmax=True)
        x = torch.randn(1, 5, 32)
        out = tc(hidden_states=x)
        assert out.shape == (1, 5, 3)
        assert (out <= 0).all()


class TestSubtokensAndMask:
    @pytest.fixture
    def tokenizer(self) -> MagicMock:
        """Create a mock NeMo TokenizerSpec-compatible tokenizer."""
        tok = MagicMock()
        tok.text_to_tokens.side_effect = lambda word: [word] if len(word) <= 4 else [word[:4], word[4:]]
        return tok

    def test_single_word(self, tokenizer: MagicMock) -> None:
        subtokens, mask = _get_subtokens_and_subtokens_mask("hi", tokenizer)
        assert subtokens == ["hi"]
        assert mask == [True]

    def test_multi_word(self, tokenizer: MagicMock) -> None:
        _subtokens, mask = _get_subtokens_and_subtokens_mask("hi there", tokenizer)
        assert mask[0] is True
        word_starts = [i for i, m in enumerate(mask) if m]
        assert len(word_starts) == 2

    def test_subword_split(self, tokenizer: MagicMock) -> None:
        _subtokens, mask = _get_subtokens_and_subtokens_mask("hello", tokenizer)
        assert mask[0] is True
        assert mask[1] is False
        assert sum(mask) == 1


class TestGetFeaturesInfer:
    @pytest.fixture
    def tokenizer(self) -> MagicMock:
        tok = MagicMock()
        tok.text_to_tokens.side_effect = lambda word: [word]
        tok.cls_token = "[CLS]"  # noqa: S105
        tok.sep_token = "[SEP]"  # noqa: S105
        tok.tokens_to_ids.side_effect = lambda tokens: list(range(len(tokens)))
        return tok

    def test_basic_segmentation(self, tokenizer: MagicMock) -> None:
        queries = ["hello world this is a test"]
        features = get_features_infer(queries, tokenizer, max_seq_length=64, step=8, margin=16)
        all_input_ids = features[0]
        assert len(all_input_ids) >= 1

    def test_multiple_queries(self, tokenizer: MagicMock) -> None:
        queries = ["first query", "second query here"]
        features = get_features_infer(queries, tokenizer, max_seq_length=64, step=8, margin=16)
        query_ids = features[5]
        assert 0 in query_ids
        assert 1 in query_ids


class TestInferDataset:
    @pytest.fixture
    def tokenizer(self) -> MagicMock:
        tok = MagicMock()
        tok.text_to_tokens.side_effect = lambda word: [word]
        tok.cls_token = "[CLS]"  # noqa: S105
        tok.sep_token = "[SEP]"  # noqa: S105
        tok.tokens_to_ids.side_effect = lambda tokens: list(range(len(tokens)))
        return tok

    def test_len_and_getitem(self, tokenizer: MagicMock) -> None:
        ds = BertPunctuationCapitalizationInferDataset(queries=["hello world"], tokenizer=tokenizer, max_seq_length=64)
        assert len(ds) >= 1
        item = ds[0]
        assert len(item) == 8
        assert isinstance(item[0], np.ndarray)

    def test_collate_fn(self, tokenizer: MagicMock) -> None:
        ds = BertPunctuationCapitalizationInferDataset(
            queries=["hello world", "test sentence"],
            tokenizer=tokenizer,
            max_seq_length=64,
        )
        batch = [ds[i] for i in range(len(ds))]
        collated = ds.collate_fn(batch)
        assert isinstance(collated[0], torch.Tensor)
        assert collated[0].dim() == 2


class TestModelHelpers:
    """Test inference helper methods that don't require full model instantiation."""

    def test_move_acc_probs_to_token_preds(self) -> None:
        acc_prob = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        pred: list[int] = []
        new_pred, new_acc = PunctuationCapitalizationModel._move_acc_probs_to_token_preds(pred, acc_prob, 2)
        assert new_pred == [1, 0]
        assert new_acc.shape == (1, 2)

    def test_update_accumulated_probabilities(self) -> None:
        acc = np.array([[0.5, 0.5], [0.3, 0.7]])
        update = np.array([[0.2, 0.8], [0.9, 0.1], [0.4, 0.6]])
        result = PunctuationCapitalizationModel._update_accumulated_probabilities(acc, update)
        assert result.shape == (3, 2)
        np.testing.assert_allclose(result[0], [0.1, 0.4])
        np.testing.assert_allclose(result[1], [0.27, 0.07])
        np.testing.assert_allclose(result[2], [0.4, 0.6])

    def test_remove_margins_keep_both(self) -> None:
        tensor = torch.arange(10).float()
        result = PunctuationCapitalizationModel._remove_margins(tensor, margin_size=2, keep_left=True, keep_right=True)
        assert result.shape == (10,)

    def test_remove_margins_remove_left(self) -> None:
        tensor = torch.arange(10).float()
        result = PunctuationCapitalizationModel._remove_margins(
            tensor, margin_size=2, keep_left=False, keep_right=True
        )
        assert result.shape == (7,)
        assert result[0] == 3.0

    def test_remove_margins_remove_right(self) -> None:
        tensor = torch.arange(10).float()
        result = PunctuationCapitalizationModel._remove_margins(
            tensor, margin_size=2, keep_left=True, keep_right=False
        )
        assert result.shape == (7,)


class TestLoadLabelIds:
    def test_load_from_file(self, tmp_path: Path) -> None:
        label_file = tmp_path / "labels.txt"
        label_file.write_text("O\n,\n.\n?\n")
        ids = _load_label_ids(str(label_file))
        assert ids == {"O": 0, ",": 1, ".": 2, "?": 3}


class TestListAvailableModels:
    def test_list_models(self) -> None:
        models = PunctuationCapitalizationModel.list_available_models()
        assert len(models) == 2
        names = [m.pretrained_model_name for m in models]
        assert "punctuation_en_bert" in names
        assert "punctuation_en_distilbert" in names
