# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Batched language identification with the official fastText lid.176 model."""

from __future__ import annotations

import math
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from loguru import logger

from nemo_curator.stages.audio.pipeline_utils import set_note
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from collections.abc import Sequence


def strip_fasttext_label(value: str) -> str:
    """Strip only fastText's leading label marker, matching the evaluation scripts."""
    prefix = "__label__"
    if value.startswith(prefix):
        return value[len(prefix) :]
    return value


@dataclass
class _FastTextLanguageIdentificationStage(ProcessingStage[AudioTask, AudioTask]):
    """Shared implementation for fastText-format language-ID checkpoints."""

    model_path: str = ""
    text_key: str = "abbreviated_text"
    source_lang_key: str = "source_lang"
    output_text_key: str = "llm_language_prediction"
    skip_me_key: str = "_skipme"
    notes_key: str = "additional_notes"
    backend_by_language: dict[str, str] | None = None
    batch_size: int = 1024
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    backend_name: ClassVar[str] = ""
    supported_backends: ClassVar[frozenset[str]] = frozenset({"llm", "fasttext", "indiclid"})

    _model: Any = field(default=None, init=False, repr=False)
    _model_labels: frozenset[str] = field(default_factory=frozenset, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.model_path:
            msg = f"model_path is required for {type(self).__name__}"
            raise ValueError(msg)
        if self.batch_size <= 0:
            msg = "batch_size must be positive"
            raise ValueError(msg)

        if not self.output_text_key:
            msg = "output_text_key must be non-empty"
            raise ValueError(msg)

        if self.backend_by_language is not None:
            normalized: dict[str, str] = {}
            for raw_language, raw_backend in self.backend_by_language.items():
                language = self._normalize_language_code(raw_language)
                backend = str(raw_backend).strip().lower()
                if backend not in self.supported_backends:
                    msg = (
                        f"Unsupported language-ID backend {raw_backend!r} for {raw_language!r}; "
                        f"expected one of {sorted(self.supported_backends)}"
                    )
                    raise ValueError(msg)
                if language in normalized:
                    msg = f"Duplicate language-ID backend config for {language!r}"
                    raise ValueError(msg)
                normalized[language] = backend
            if not normalized:
                msg = "backend_by_language must not be empty"
                raise ValueError(msg)
            self.backend_by_language = normalized

    @staticmethod
    def _normalize_language_code(value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            msg = f"source language must be a non-empty string, got {value!r}"
            raise ValueError(msg)
        return value.strip().lower()

    def _is_routed_to_this_stage(self, task: AudioTask) -> tuple[bool, str]:
        language = self._normalize_language_code(task.data.get(self.source_lang_key))
        if self.backend_by_language is None:
            return True, language
        if language not in self.backend_by_language:
            msg = f"No language-ID backend configured for source_lang={language!r}"
            raise ValueError(msg)
        return self.backend_by_language[language] == self.backend_name, language

    def setup(self, _worker_metadata: object | None = None) -> None:
        try:
            import fasttext
        except ImportError as exc:
            msg = (
                f"{type(self).__name__} requires fasttext. "
                "Install the tested CPU LID runtime with "
                "`pip install 'numpy==1.26.4' 'fasttext==0.9.3'`."
            )
            raise ImportError(msg) from exc

        resolved_path = Path(self.model_path).expanduser()
        if not resolved_path.is_file():
            msg = f"Language-ID model does not exist: {resolved_path}"
            raise ValueError(msg)

        self._model = fasttext.load_model(str(resolved_path))
        self._model_labels = frozenset(strip_fasttext_label(str(label)) for label in self._model.get_labels())
        self._validate_model_labels(self._model_labels)
        logger.info(
            "{}: loaded {} labels from {}",
            type(self).__name__,
            len(self._model_labels),
            resolved_path,
        )

    @abstractmethod
    def _validate_model_labels(self, labels: frozenset[str]) -> None:
        """Validate the loaded checkpoint's complete label inventory."""

    @abstractmethod
    def _sanitize_text(self, text: str) -> str:
        """Apply the exact preprocessing used by the corresponding evaluation."""

    @abstractmethod
    def _map_label_to_language(self, raw_label: str) -> str:
        """Map one stripped checkpoint label to the manifest language code."""

    @abstractmethod
    def _validate_source_language(self, language: str) -> None:
        """Fail when the selected model cannot represent the expected language."""

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.text_key, self.source_lang_key, self.skip_me_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.output_text_key]

    def _ensure_setup(self) -> None:
        if self._model is None:
            logger.warning(
                "{} ({}): setup() was not called before inference; calling setup() now",
                type(self).__name__,
                self.name,
            )
            self.setup()

    def process(self, task: AudioTask) -> AudioTask:
        return self.process_batch([task])[0]

    def _collect_model_inputs(self, tasks: list[AudioTask]) -> tuple[list[AudioTask], list[str]]:
        eligible_tasks: list[AudioTask] = []
        texts: list[str] = []
        for task in tasks:
            should_process, source_language = self._is_routed_to_this_stage(task)
            if not should_process:
                continue
            if task.data.get(self.skip_me_key, ""):
                task.data[self.output_text_key] = ""
                set_note(task.data, self.name, "skipped (flagged)", self.notes_key)
                continue
            text = task.data.get(self.text_key, "")
            if not isinstance(text, str) or not text.strip():
                task.data[self.output_text_key] = text
                set_note(task.data, self.name, "skipped (empty)", self.notes_key)
                continue

            self._validate_source_language(source_language)
            if "\x00" in text:
                msg = f"{type(self).__name__} does not accept NUL characters in {self.text_key!r}"
                raise ValueError(msg)
            eligible_tasks.append(task)
            texts.append(self._sanitize_text(text))
        return eligible_tasks, texts

    def _write_predictions(
        self,
        eligible_tasks: list[AudioTask],
        labels: Sequence[Sequence[object]],
        probabilities: Sequence[Sequence[object]],
    ) -> None:
        for index, task in enumerate(eligible_tasks):
            row_labels = labels[index]
            row_probabilities = probabilities[index]
            if len(row_labels) == 0 or len(row_probabilities) == 0:
                msg = f"{type(self).__name__} returned an empty top-1 prediction"
                raise ValueError(msg)
            raw_label = strip_fasttext_label(str(row_labels[0]))
            if raw_label not in self._model_labels:
                msg = f"{type(self).__name__} returned unknown label {raw_label!r}"
                raise ValueError(msg)
            probability = float(row_probabilities[0])
            if not math.isfinite(probability):
                msg = f"{type(self).__name__} returned non-finite probability {probability!r}"
                raise ValueError(msg)

            result_text = self._map_label_to_language(raw_label)
            input_text = task.data[self.text_key]
            task.data[self.output_text_key] = result_text
            note = "applied (modified)" if result_text != input_text else "applied (unchanged)"
            set_note(task.data, self.name, note, self.notes_key)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        # Ray Data supplies object columns as NumPy arrays. Normalize at the
        # stage boundary so both truth-value checks and the declared return
        # type are independent of the backend's batch container.
        task_list = list(tasks)
        if not task_list:
            return []
        self._ensure_setup()
        eligible_tasks, texts = self._collect_model_inputs(task_list)

        if not texts:
            return task_list

        labels, probabilities = self._model.predict(texts, k=1)
        if len(labels) != len(texts) or len(probabilities) != len(texts):
            msg = (
                f"{type(self).__name__} prediction count mismatch: "
                f"inputs={len(texts)}, labels={len(labels)}, probabilities={len(probabilities)}"
            )
            raise ValueError(msg)
        self._write_predictions(eligible_tasks, labels, probabilities)

        return task_list


@dataclass
class FastTextLanguageIdentificationStage(_FastTextLanguageIdentificationStage):
    """Run top-1 lid.176 inference with the exact local-evaluation behavior.

    Input whitespace is collapsed with ``" ".join(text.split())``. The model is
    called once per batch with ``k=1``; no confidence threshold, short-text
    bypass, label proxy, lowercasing, or probability clipping is applied.
    """

    name: str = "LanguageID"

    backend_name: ClassVar[str] = "fasttext"
    expected_label_count: ClassVar[int] = 176
    granary_source_languages: ClassVar[frozenset[str]] = frozenset(
        [
            "as",
            "bn",
            "brx",
            "doi",
            "gu",
            "hi",
            "kn",
            "kok",
            "ks",
            "mai",
            "ml",
            "mni",
            "mr",
            "ne",
            "or",
            "pa",
            "sa",
            "sat",
            "sd",
            "ta",
            "te",
            "ur",
        ]
    )
    granary_exact_languages: ClassVar[frozenset[str]] = frozenset(
        ["as", "bn", "gu", "hi", "kn", "mai", "ml", "mr", "ne", "or", "pa", "sa", "sd", "ta", "te", "ur"]
    )

    def _validate_model_labels(self, labels: frozenset[str]) -> None:
        if len(labels) != self.expected_label_count:
            msg = f"Expected {self.expected_label_count} lid.176 labels, found {len(labels)}"
            raise ValueError(msg)
        exact_languages = labels & self.granary_source_languages
        if exact_languages != self.granary_exact_languages or "gom" not in labels:
            msg = (
                "lid.176 Granary language inventory mismatch: "
                f"expected_exact={sorted(self.granary_exact_languages)}, "
                f"found_exact={sorted(exact_languages)}, has_konkani_proxy={'gom' in labels}"
            )
            raise ValueError(msg)

    def _sanitize_text(self, text: str) -> str:
        return " ".join(text.split())

    def _map_label_to_language(self, raw_label: str) -> str:
        return raw_label

    def _validate_source_language(self, language: str) -> None:
        if language not in self._model_labels:
            msg = (
                f"FastText lid.176 has no exact label for source_lang={language!r}; "
                "route this language to IndicLID instead"
            )
            raise ValueError(msg)
