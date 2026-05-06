# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""
Punctuation and Capitalization model ported from NeMo v2.4.1.

Inherits from ModelPT (still in nemo main) instead of NLPModel (removed).
Only inference code paths are retained; training code is stripped.
"""

from math import ceil
from typing import Any

import numpy as np
import torch
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.neural_types import LogitsType, NeuralType
from nemo.utils import logging
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import AutoModel

from nemo_curator.stages.audio.inference.pnc_bert.infer_dataset import (
    BertPunctuationCapitalizationInferDataset,
)
from nemo_curator.stages.audio.inference.pnc_bert.token_classifier import TokenClassifier

__all__ = ["PunctuationCapitalizationModel"]


def _load_label_ids(file_path: str) -> dict[str, int]:
    """Load label ids from a file. One label per line, id is line index."""
    ids = {}
    with open(file_path, encoding="utf_8") as f:
        for i, line in enumerate(f):
            ids[line.strip()] = i
    return ids


class PunctuationCapitalizationModel(ModelPT, Exportable):
    """
    A model for restoring punctuation and capitalization in text.

    The model consists of a language model and two multilayer perceptrons (MLP) on top the language model.
    The first MLP serves for punctuation prediction and the second is for capitalization prediction.

    Use method :meth:`~add_punctuation_capitalization` for model inference.

    Ported from NeMo v2.4.1 with NLPModel replaced by ModelPT (still available in nemo main).
    """

    @property
    def output_types(self) -> dict[str, NeuralType] | None:
        """Neural types of a :meth:`forward` method output."""
        return {
            "punct_logits": NeuralType(("B", "T", "C"), LogitsType()),
            "capit_logits": NeuralType(("B", "T", "C"), LogitsType()),
        }

    def __init__(self, cfg: DictConfig, trainer: Any = None) -> None:  # noqa: ANN401
        """Initializes BERT Punctuation and Capitalization model."""
        self.metrics = None
        self.label_ids_are_set: bool = False
        self.punct_label_ids: dict[str, int] | None = None
        self.capit_label_ids: dict[str, int] | None = None

        super().__init__(cfg=cfg, trainer=trainer)

        if not self.label_ids_are_set:
            self._set_label_ids()

        model_name = cfg.language_model.pretrained_model_name
        self.bert_model = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert_model.config.hidden_size

        tokenizer_name = cfg.tokenizer.tokenizer_name if cfg.get("tokenizer") else model_name
        self.tokenizer = AutoTokenizer(pretrained_model_name=tokenizer_name)

        punct_num_layers = cfg.punct_head.get("num_fc_layers", cfg.punct_head.get("punct_num_fc_layers", 1))
        capit_num_layers = cfg.capit_head.get("num_fc_layers", cfg.capit_head.get("capit_num_fc_layers", 1))

        self.punct_classifier = TokenClassifier(
            hidden_size=self.hidden_size,
            num_classes=len(self.punct_label_ids),
            activation=cfg.punct_head.activation,
            log_softmax=False,
            dropout=cfg.punct_head.fc_dropout,
            num_layers=punct_num_layers,
            use_transformer_init=cfg.punct_head.use_transformer_init,
        )

        self.capit_classifier = TokenClassifier(
            hidden_size=self.hidden_size,
            num_classes=len(self.capit_label_ids),
            activation=cfg.capit_head.activation,
            log_softmax=False,
            dropout=cfg.capit_head.fc_dropout,
            num_layers=capit_num_layers,
            use_transformer_init=cfg.capit_head.use_transformer_init,
        )

    @property
    def _pad_label(self) -> str:
        """Resolve pad_label from config (handles both old and new config formats)."""
        cdp = self._cfg.get("common_dataset_parameters")
        if cdp is not None and cdp.get("pad_label") is not None:
            return cdp.pad_label
        ds = self._cfg.get("dataset")
        if ds is not None and ds.get("pad_label") is not None:
            return ds.pad_label
        return "O"

    @typecheck()
    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Executes a forward pass through the model.

        Args:
            input_ids: an integer torch tensor of shape ``[Batch, Time]``.
            attention_mask: a boolean torch tensor of shape ``[Batch, Time]``.
            token_type_ids: an integer torch Tensor of shape ``[Batch, Time]``.

        Returns:
            Tuple of (punct_logits, capit_logits) each of shape ``[Batch, Time, NumLabels]``
        """
        hidden_states = self.bert_model(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )
        if hasattr(hidden_states, "last_hidden_state"):
            hidden_states = hidden_states.last_hidden_state
        elif isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        punct_logits = self.punct_classifier(hidden_states=hidden_states)
        capit_logits = self.capit_classifier(hidden_states=hidden_states)
        return punct_logits.float(), capit_logits.float()

    # ------------------------------------------------------------------
    # Label ID setup
    # ------------------------------------------------------------------

    def _set_label_ids(self) -> None:
        """
        Set model attributes ``punct_label_ids`` and ``capit_label_ids`` based on label ids
        passed in config. Handles both old-style configs (top-level punct_label_ids/capit_label_ids)
        and new-style (common_dataset_parameters.punct_label_ids).
        """
        self.punct_label_ids = self._resolve_label_ids("punct_label_ids")
        self.capit_label_ids = self._resolve_label_ids("capit_label_ids")
        self.label_ids_are_set = True

    def _resolve_label_ids(self, key: str) -> dict[str, int]:
        """Resolve label ids from config, checking top-level, common_dataset_parameters, and artifact files."""
        if self._cfg.get(key) is not None:
            return OmegaConf.to_container(self._cfg[key])
        cdp = self._cfg.get("common_dataset_parameters")
        if cdp is not None and cdp.get(key) is not None:
            return OmegaConf.to_container(cdp[key])
        artifact_key = "class_labels.punct_labels_file" if "punct" in key else "class_labels.capit_labels_file"
        class_labels = self._cfg.get("class_labels")
        if class_labels is not None:
            vocab_filename = class_labels.get(artifact_key.split(".")[-1])
            if vocab_filename is not None:
                try:
                    labels_file = self.register_artifact(artifact_key, str(vocab_filename))
                    if labels_file is not None:
                        return _load_label_ids(labels_file)
                except FileNotFoundError:
                    pass
        msg = (
            f"Could not set attribute `{key}`. Neither `model.{key}` nor "
            f"`model.common_dataset_parameters.{key}` is set in config, "
            f"and no artifact file could be loaded."
        )
        raise ValueError(msg)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _get_labels(self, punct_preds: list[int], capit_preds: list[int]) -> str:
        """Returns punctuation and capitalization labels in NeMo format."""
        assert len(capit_preds) == len(punct_preds), (  # noqa: S101
            f"len(capit_preds)={len(capit_preds)} len(punct_preds)={len(punct_preds)}"
        )
        punct_ids_to_labels = {v: k for k, v in self.punct_label_ids.items()}
        capit_ids_to_labels = {v: k for k, v in self.capit_label_ids.items()}
        result = ""
        for capit_pred, punct_pred in zip(capit_preds, punct_preds, strict=False):
            result += punct_ids_to_labels[punct_pred] + capit_ids_to_labels[capit_pred] + " "
        return result[:-1]

    def add_punctuation_capitalization(  # noqa: C901, PLR0913
        self,
        queries: list[str],
        batch_size: int | None = None,
        max_seq_length: int = 64,
        step: int = 8,
        margin: int = 16,
        return_labels: bool = False,
        dataloader_kwargs: dict[str, Any] | None = None,
    ) -> list[str]:
        """
        Adds punctuation and capitalization to the queries. Use this method for inference.

        Args:
            queries: lower cased text without punctuation.
            batch_size: batch size to use during inference.
            max_seq_length: maximum sequence length of a segment after tokenization.
            step: relative shift of consequent segments.
            margin: number of subtokens near edges of segments excluded from prediction.
            return_labels: whether to return labels in NeMo format instead of restored text.
            dataloader_kwargs: optional dictionary with PyTorch DataLoader parameters.

        Returns:
            List of queries with restored capitalization and punctuation.
        """
        if len(queries) == 0:
            return []
        if batch_size is None:
            batch_size = len(queries)
            logging.info(f"Using batch size {batch_size} for inference")
        result: list[str] = []
        mode = self.training
        try:
            self.eval()
            infer_datalayer = self._setup_infer_dataloader(
                queries, batch_size, max_seq_length, step, margin, dataloader_kwargs
            )
            all_punct_preds: list[list[int]] = [[] for _ in queries]
            all_capit_preds: list[list[int]] = [[] for _ in queries]
            acc_punct_probs: list[np.ndarray | None] = [None for _ in queries]
            acc_capit_probs: list[np.ndarray | None] = [None for _ in queries]
            d = self.device
            for _batch_i, batch in tqdm(
                enumerate(infer_datalayer), total=ceil(len(infer_datalayer.dataset) / batch_size), unit="batch"
            ):
                inp_ids, inp_type_ids, inp_mask, subtokens_mask, start_word_ids, query_ids, is_first, is_last = batch
                punct_logits, capit_logits = self.forward(
                    input_ids=inp_ids.to(d),
                    token_type_ids=inp_type_ids.to(d),
                    attention_mask=inp_mask.to(d),
                )
                _res = self._transform_logit_to_prob_and_remove_margins_and_extract_word_probs(
                    punct_logits, capit_logits, subtokens_mask, start_word_ids, margin, is_first, is_last
                )
                punct_probs, capit_probs, start_word_ids = _res
                for _i, (q_i, start_word_id, bpp_i, bcp_i) in enumerate(
                    zip(query_ids, start_word_ids, punct_probs, capit_probs, strict=False)
                ):
                    for all_preds, acc_probs, b_probs_i in [
                        (all_punct_preds, acc_punct_probs, bpp_i),
                        (all_capit_preds, acc_capit_probs, bcp_i),
                    ]:
                        if acc_probs[q_i] is None:
                            acc_probs[q_i] = b_probs_i
                        else:
                            all_preds[q_i], acc_probs[q_i] = self._move_acc_probs_to_token_preds(
                                all_preds[q_i],
                                acc_probs[q_i],
                                start_word_id - len(all_preds[q_i]),
                            )
                            acc_probs[q_i] = self._update_accumulated_probabilities(acc_probs[q_i], b_probs_i)
            for all_preds, acc_probs in [(all_punct_preds, acc_punct_probs), (all_capit_preds, acc_capit_probs)]:
                for q_i, (pred, prob) in enumerate(zip(all_preds, acc_probs, strict=False)):
                    if prob is not None:
                        all_preds[q_i], acc_probs[q_i] = self._move_acc_probs_to_token_preds(pred, prob, len(prob))
            for i, query in enumerate(queries):
                result.append(
                    self._get_labels(all_punct_preds[i], all_capit_preds[i])
                    if return_labels
                    else self._apply_punct_capit_predictions(query, all_punct_preds[i], all_capit_preds[i])
                )
        finally:
            self.train(mode=mode)
        return result

    def _setup_infer_dataloader(  # noqa: PLR0913
        self,
        queries: list[str],
        batch_size: int,
        max_seq_length: int,
        step: int,
        margin: int,
        dataloader_kwargs: dict[str, Any] | None,
    ) -> torch.utils.data.DataLoader:
        """Setup function for an infer data loader."""
        if dataloader_kwargs is None:
            dataloader_kwargs = {}
        dataset = BertPunctuationCapitalizationInferDataset(
            tokenizer=self.tokenizer,
            queries=queries,
            max_seq_length=max_seq_length,
            step=step,
            margin=margin,
        )
        return torch.utils.data.DataLoader(
            dataset=dataset,
            collate_fn=dataset.collate_fn,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            **dataloader_kwargs,
        )

    @staticmethod
    def _remove_margins(tensor: torch.Tensor, margin_size: int, keep_left: bool, keep_right: bool) -> torch.Tensor:
        tensor = tensor.detach().clone()
        if not keep_left:
            tensor = tensor[margin_size + 1 :]
        if not keep_right:
            tensor = tensor[: tensor.shape[0] - margin_size - 1]
        return tensor

    def _transform_logit_to_prob_and_remove_margins_and_extract_word_probs(  # noqa: PLR0913
        self,
        punct_logits: torch.Tensor,
        capit_logits: torch.Tensor,
        subtokens_mask: torch.Tensor,
        start_word_ids: tuple[int, ...],
        margin: int,
        is_first: tuple[bool, ...],
        is_last: tuple[bool, ...],
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
        """
        Applies softmax to get punctuation and capitalization probabilities, applies ``subtokens_mask`` to extract
        probabilities for words from probabilities for tokens, removes ``margin`` probabilities near segment edges.
        """
        new_start_word_ids = list(start_word_ids)
        subtokens_mask = subtokens_mask > 0.5  # noqa: PLR2004
        b_punct_probs, b_capit_probs = [], []
        for i, (first, last, pl, cl, stm) in enumerate(
            zip(is_first, is_last, punct_logits, capit_logits, subtokens_mask, strict=False)
        ):
            if not first:
                new_start_word_ids[i] += torch.count_nonzero(stm[: margin + 1]).numpy()
            stm = self._remove_margins(stm, margin, keep_left=first, keep_right=last)  # noqa: PLW2901
            for b_probs, logits in [(b_punct_probs, pl), (b_capit_probs, cl)]:
                p = torch.nn.functional.softmax(
                    self._remove_margins(logits, margin, keep_left=first, keep_right=last)[stm],
                    dim=-1,
                )
                b_probs.append(p.detach().cpu().numpy())
        return b_punct_probs, b_capit_probs, new_start_word_ids

    @staticmethod
    def _move_acc_probs_to_token_preds(
        pred: list[int], acc_prob: np.ndarray, number_of_probs_to_move: int
    ) -> tuple[list[int], np.ndarray]:
        """
        ``number_of_probs_to_move`` rows in the beginning are removed from ``acc_prob``. From every removed row
        the label with the largest probability is selected and appended to ``pred``.
        """
        if number_of_probs_to_move > acc_prob.shape[0]:
            msg = (
                f"Not enough accumulated probabilities. Number_of_probs_to_move={number_of_probs_to_move} "
                f"acc_prob.shape={acc_prob.shape}"
            )
            raise ValueError(msg)
        if number_of_probs_to_move > 0:
            pred = pred + list(np.argmax(acc_prob[:number_of_probs_to_move], axis=-1))
        return pred, acc_prob[number_of_probs_to_move:]

    @staticmethod
    def _update_accumulated_probabilities(acc_prob: np.ndarray, update: np.ndarray) -> np.ndarray:
        return np.concatenate([acc_prob * update[: acc_prob.shape[0]], update[acc_prob.shape[0] :]], axis=0)

    def _apply_punct_capit_predictions(self, query: str, punct_preds: list[int], capit_preds: list[int]) -> str:
        """Restores punctuation and capitalization in ``query``."""
        words = query.strip().split()
        assert len(words) == len(punct_preds), (  # noqa: S101
            f"len(query)={len(words)} len(punct_preds)={len(punct_preds)}, query[:30]={words[:30]}"
        )
        assert len(words) == len(capit_preds), (  # noqa: S101
            f"len(query)={len(words)} len(capit_preds)={len(capit_preds)}, query[:30]={words[:30]}"
        )
        punct_ids_to_labels = {v: k for k, v in self.punct_label_ids.items()}
        capit_ids_to_labels = {v: k for k, v in self.capit_label_ids.items()}
        query_with_punct_and_capit = ""
        for j, word in enumerate(words):
            punct_label = punct_ids_to_labels[punct_preds[j]]
            capit_label = capit_ids_to_labels[capit_preds[j]]

            if capit_label != self._pad_label:
                word = word.capitalize()  # noqa: PLW2901
            query_with_punct_and_capit += word
            if punct_label != self._pad_label:
                query_with_punct_and_capit += punct_label
            query_with_punct_and_capit += " "
        return query_with_punct_and_capit[:-1]

    # ------------------------------------------------------------------
    # ModelPT required methods
    # ------------------------------------------------------------------

    def setup_training_data(self, train_data_config: Any = None) -> None:  # noqa: ANN401
        """Not used for inference-only model."""

    def setup_validation_data(self, val_data_config: Any = None) -> None:  # noqa: ANN401
        """Not used for inference-only model."""

    @classmethod
    def list_available_models(cls) -> list[PretrainedModelInfo]:
        """Returns a list of pre-trained models which can be instantiated from NVIDIA's NGC cloud."""
        return [
            PretrainedModelInfo(
                pretrained_model_name="punctuation_en_bert",
                location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/punctuation_en_bert/versions/1.0.0rc1/"
                "files/punctuation_en_bert.nemo",
                description="The model was trained with NeMo BERT base uncased checkpoint on a subset of data from "
                "the following sources: Tatoeba sentences, books from Project Gutenberg, Fisher transcripts.",
            ),
            PretrainedModelInfo(
                pretrained_model_name="punctuation_en_distilbert",
                location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/punctuation_en_distilbert/versions/"
                "1.0.0rc1/files/punctuation_en_distilbert.nemo",
                description="The model was trained with DistilBERT base uncased checkpoint from HuggingFace on a "
                "subset of data from the following sources: Tatoeba sentences, books from Project Gutenberg, "
                "Fisher transcripts.",
            ),
        ]

    @classmethod
    def restore_from(  # type: ignore[override]  # noqa: PLR0913
        cls,
        restore_path: str,
        override_config_path: Any = None,  # noqa: ANN401
        map_location: Any = None,  # noqa: ANN401
        strict: bool = False,  # noqa: ARG003
        return_config: bool = False,
        save_restore_connector: Any = None,  # noqa: ANN401
        trainer: Any = None,  # noqa: ANN401
        validate_access_integrity: bool = True,
    ) -> "PunctuationCapitalizationModel":
        """Override to force strict=False for checkpoint compatibility across BERT versions."""
        return super().restore_from(
            restore_path=restore_path,
            override_config_path=override_config_path,
            map_location=map_location,
            strict=False,
            return_config=return_config,
            save_restore_connector=save_restore_connector,
            trainer=trainer,
            validate_access_integrity=validate_access_integrity,
        )

    @property
    def output_module(self) -> "PunctuationCapitalizationModel":
        return self
