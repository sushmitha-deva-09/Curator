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

"""AI4Bharat IndicConformer NeMo (.nemo) inference stage.

This stage is the Curator-style port of the upstream SDP processor at
``generic-sdp/generic_sdp/processors/indicconformer.py``. It loads the
AI4Bharat IndicConformer ``.nemo`` checkpoint via the AI4Bharat NeMo fork's
``EncDecCTCModel.restore_from()`` and runs CTC transcription using the
fork-specific ``transcribe(audio, batch_size=..., logprobs=False, language_id=...)``
API.

Two key constraints inherited from AI4Bharat's transcribe contract:

* CTC-only (``cur_decoder = 'ctc'``); RNNT and timestamps are not supported.
* ``language_id`` is a single string per ``transcribe()`` call. When a batch
contains tasks of multiple languages we group by language and call
``transcribe()`` once per group.

For an alternative path that uses the gated HuggingFace ONNX wrapper, see
``nemo_curator.stages.audio.inference.asr.indic_conformer.InferenceIndicConformerStage``.
"""

from __future__ import annotations

import contextlib
import gc
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger

from nemo_curator.stages.audio.pipeline_utils import set_note
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata


# ISO codes the AI4Bharat IndicConformer 600M multilingual checkpoint
# supports (https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual).
INDIC_CONFORMER_NEMO_LANGS: frozenset[str] = frozenset(
    {
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
    }
)

_DEFAULT_TARGET_SR = 16000


@dataclass
class InferenceIndicConformerNeMoStage(ProcessingStage[AudioTask, AudioTask]):
    """Audio transcription using AI4Bharat IndicConformer (.nemo via NeMo fork).

    Reads a per-task waveform tensor (and sample rate) from ``AudioTask.data``,
    resamples to the model's expected rate, groups by language, and calls
    AI4Bharat's CTC transcribe.

    Args:
        model_name: HuggingFace / NeMo pretrained ID, e.g.
            ``"ai4bharat/indicconformer_stt_hi_hybrid_rnnt_large"``. When set,
            the model is fetched via ``<concrete_class>.from_pretrained``.
        model_path: Local filesystem path to a ``.nemo`` checkpoint. Used when
            ``model_name`` is empty. Loaded via ``<concrete_class>.restore_from``.
        cache_dir: Optional cache directory for downloaded checkpoints. Surfaced
            via ``NEMO_CACHE_DIR`` since AI4Bharat NeMo's ``from_pretrained``
            does not accept a ``cache_dir`` keyword.
        model_class_name: Concrete NeMo ASR class name. AI4Bharat's fork does
            NOT auto-dispatch ``ASRModel.from_pretrained`` / ``restore_from``
            to a concrete subclass, and the wrong class produces
            ``Missing key num_classes`` (CTC class on hybrid config) or
            ``Missing key feat_in`` (hybrid class on CTC config). When empty,
            the class is inferred from the ``model_name`` / ``model_path``
            string (``"hybrid_rnnt"`` → ``EncDecHybridRNNTCTCBPEModel``,
            ``"ctc"`` without RNNT → ``EncDecCTCModel``); falls back to
            ``EncDecHybridRNNTCTCBPEModel`` (matches AI4Bharat's published
            ``indicconformer_stt_*_hybrid_rnnt_large`` checkpoints).
        hf_token: Optional HuggingFace access token. When set it is exported as
            ``HF_TOKEN`` in the worker environment so NeMo's downloader and
            ``huggingface_hub`` can authenticate against gated repos. Leave
            empty if the token is already provided via env / ``~/.cache/huggingface``.
        language: Default ISO language code used when ``source_lang_key`` is
            not configured or a task lacks the key.
        source_lang_key: Optional per-task language key. When non-empty,
            tasks are grouped by language so the single-language ``transcribe()``
            call can run per-group.
        waveform_key: Task data key for the mono float32 waveform tensor.
        sample_rate_key: Task data key for the integer sample rate.
        pred_text_key: Output key for the predicted transcription.
        language_key: Output key storing the resolved language code.
        notes_key: Top-level key used for ``additional_notes`` metadata.
        keep_waveform: When False (default) the (potentially large, non-JSON-
            serializable) waveform tensor is removed from ``task.data`` after
            transcription so a downstream ``ManifestWriterStage`` can serialize
            cleanly.
        transcribe_batch_size: Batch size passed to NeMo's ``transcribe()`` —
            controls GPU forward batching inside a same-language group.
        num_workers_override: Fixed Ray actor count. None = let the autoscaler
            decide.
    """

    model_name: str = ""
    model_path: str = ""
    cache_dir: str | None = None
    model_class_name: str = ""
    hf_token: str = field(default="", repr=False)
    language: str = "hi"
    name: str = "IndicConformerNeMo_inference"
    source_lang_key: str = ""
    waveform_key: str = "waveform"
    sample_rate_key: str = "sampling_rate"
    pred_text_key: str = "asr_prediction"
    language_key: str = "asr_language"
    notes_key: str = "additional_notes"
    keep_waveform: bool = False
    transcribe_batch_size: int = 32
    num_workers_override: int | None = None
    resources: Resources = field(default_factory=lambda: Resources(gpus=1.0))
    batch_size: int = 64

    _asr_model: Any = field(default=None, init=False, repr=False)
    _device: Any = field(default=None, init=False, repr=False)
    _target_sr: int = field(default=_DEFAULT_TARGET_SR, init=False, repr=False)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        if not self.model_name and not self.model_path:
            msg = (
                f"[{self.name}] one of `model_name` (HF / NeMo pretrained ID, e.g. "
                "'ai4bharat/indicconformer_stt_hi_hybrid_rnnt_large') or `model_path` "
                "(local .nemo checkpoint) is required."
            )
            raise ValueError(msg)
        if self.model_name and self.model_path:
            logger.warning(
                f"[{self.name}] both model_name={self.model_name!r} and "
                f"model_path={self.model_path!r} are set; model_name takes precedence."
            )

    # ------------------------------------------------------------------
    # Scaling hooks
    # ------------------------------------------------------------------

    def num_workers(self) -> int | None:
        return self.num_workers_override

    def xenna_stage_spec(self) -> dict[str, Any]:
        spec: dict[str, Any] = {}
        if self.num_workers_override is not None:
            spec["num_workers"] = self.num_workers_override
        return spec

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _export_hf_token(self) -> None:
        """If ``hf_token`` was provided, surface it to the env so NeMo's
        downloader and ``huggingface_hub`` can authenticate against gated repos.
        """
        if not self.hf_token:
            return
        for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
            os.environ.setdefault(var, self.hf_token)

    def _resolve_model_class(self) -> Any:  # noqa: ANN401 — returns a NeMo ASR class (concrete subclass of nemo.collections.asr.models.ASRModel) not exposed as a typed base
        """Pick a concrete NeMo ASR class.

        AI4Bharat's NeMo fork doesn't auto-dispatch ``ASRModel.from_pretrained``
        / ``restore_from`` to a concrete subclass, and the wrong concrete class
        crashes during config parsing:

        * ``EncDecCTCModel`` on a hybrid ``.nemo`` →
        ``ConfigAttributeError: Missing key num_classes``
        (the hybrid decoder config has ``feat_in`` + ``vocabulary``).
        * ``EncDecHybridRNNTCTCBPEModel`` on a pure CTC ``.nemo`` →
        ``Missing key feat_in`` (no RNNT decoder section).

        Resolution order:
        1. ``model_class_name`` (explicit user override).
        2. Inferred from the ``model_name`` / ``model_path`` filename
        (substring ``"hybrid_rnnt"`` / ``"ctc"``).
        3. Fallback: ``EncDecHybridRNNTCTCBPEModel`` (matches AI4Bharat's
        published ``indicconformer_stt_*_hybrid_rnnt_large`` checkpoints).
        """
        from nemo.collections.asr import models as nemo_asr_models

        if self.model_class_name:
            try:
                return getattr(nemo_asr_models, self.model_class_name)
            except AttributeError as e:
                msg = (
                    f"[{self.name}] model_class_name={self.model_class_name!r} not found "
                    "in nemo.collections.asr.models"
                )
                raise ValueError(msg) from e

        identifier = (self.model_name or os.path.basename(self.model_path or "")).lower()
        if "hybrid_rnnt" in identifier or "hybrid-rnnt" in identifier:
            return nemo_asr_models.EncDecHybridRNNTCTCBPEModel
        if "ctc" in identifier and "rnnt" not in identifier:
            return nemo_asr_models.EncDecCTCModel
        return nemo_asr_models.EncDecHybridRNNTCTCBPEModel

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        """Pre-fetch the model onto this node.

        - ``model_name``: download via ``EncDecCTCModel.from_pretrained`` so the
        checkpoint lands in the NeMo cache before per-replica setup.
        - ``model_path``: just verify the .nemo file exists.
        """
        if self.model_name:
            self._export_hf_token()
            if self.cache_dir:
                # AI4Bharat NeMo's from_pretrained does not accept cache_dir; route
                # via NeMo's env var instead so it lands in the requested location.
                os.environ.setdefault("NEMO_CACHE_DIR", self.cache_dir)
            try:
                model_cls = self._resolve_model_class()
                model_cls.from_pretrained(model_name=self.model_name)
            except Exception as e:
                msg = f"[{self.name}] failed to download model_name={self.model_name!r}"
                raise RuntimeError(msg) from e
            return

        if not os.path.exists(self.model_path):
            msg = (
                f"[{self.name}] model_path does not exist: {self.model_path}. "
                "AI4Bharat IndicConformer must be downloaded ahead of time."
            )
            raise FileNotFoundError(msg)

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        """Load the model onto this replica."""
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_cls = self._resolve_model_class()

        if self.model_name:
            self._export_hf_token()
            if self.cache_dir:
                os.environ.setdefault("NEMO_CACHE_DIR", self.cache_dir)
            logger.info(
                f"[{self.name}] Loading IndicConformer model_name={self.model_name!r} "
                f"as {model_cls.__name__} on {self._device}"
            )
            self._asr_model = model_cls.from_pretrained(
                model_name=self.model_name,
                map_location=self._device,
            )
        else:
            logger.info(
                f"[{self.name}] Loading IndicConformer .nemo from {self.model_path} "
                f"as {model_cls.__name__} on {self._device}"
            )
            self._asr_model = model_cls.restore_from(restore_path=self.model_path)

        self._asr_model.freeze()
        self._asr_model.to(self._device)
        self._asr_model.eval()
        # The .nemo can carry both CTC and RNNT heads; pin the CTC head per
        # AI4Bharat's reference SDP processor.
        self._asr_model.cur_decoder = "ctc"

        try:
            self._target_sr = int(self._asr_model.cfg.sample_rate)
        except Exception:  # noqa: BLE001
            self._target_sr = _DEFAULT_TARGET_SR

        logger.info(f"[{self.name}] Model loaded; target sample_rate={self._target_sr} Hz")

    def teardown(self) -> None:
        if self._asr_model is not None:
            del self._asr_model
            self._asr_model = None
        self._device = None
        gc.collect()
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # I/O contract
    # ------------------------------------------------------------------

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.waveform_key, self.sample_rate_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.pred_text_key, self.language_key]

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(self, task: AudioTask) -> AudioTask:
        msg = f"{type(self).__name__} only supports process_batch"
        raise NotImplementedError(msg)

    def _resolve_lang(self, task: AudioTask) -> str:
        """Pick the ISO language for a task. Per-task key wins over default."""
        if self.source_lang_key:
            raw = task.data.get(self.source_lang_key)
            if raw is not None and str(raw).strip():
                return str(raw).strip().lower()
        return self.language

    def _prepare_waveform(self, w: Any, sr: int) -> torch.Tensor | None:  # noqa: ANN401 — task waveform comes from upstream loader as torch.Tensor or numpy.ndarray
        """Coerce a task's waveform to a 1-D mono float32 tensor at the model's
        target sample rate. Returns None for empty inputs.
        """
        import torchaudio.functional as ta_F  # noqa: N812

        is_empty = (w.numel() == 0) if isinstance(w, torch.Tensor) else (w.size == 0)
        if is_empty:
            return None

        wav = w if isinstance(w, torch.Tensor) else torch.as_tensor(w)
        wav = wav.to(dtype=torch.float32)
        if wav.ndim > 1:
            # Torchaudio convention is (channels, T); collapse to mono.
            wav = wav.mean(dim=0)
        if int(sr) != self._target_sr:
            wav = ta_F.resample(wav, orig_freq=int(sr), new_freq=self._target_sr)
        return wav

    def _bucket_tasks_by_language(
        self, tasks: list[AudioTask]
    ) -> tuple[dict[str, list[int]], list[torch.Tensor | None], int, int]:
        """Resolve language + waveform per task; bucket eligible tasks by language
        so a single ``language_id`` covers each transcribe() call.

        Returns ``(lang_to_indices, prepped_waveforms, skipped_unsupported, skipped_no_audio)``.
        """
        lang_to_indices: dict[str, list[int]] = defaultdict(list)
        prepped: list[torch.Tensor | None] = [None] * len(tasks)
        skipped_unsupported = 0
        skipped_no_audio = 0

        for i, task in enumerate(tasks):
            lang = self._resolve_lang(task)
            if lang not in INDIC_CONFORMER_NEMO_LANGS:
                set_note(
                    task.data,
                    self.name,
                    f"skipped (unsupported language: {lang})",
                    self.notes_key,
                )
                set_note(
                    task.data,
                    self.pred_text_key,
                    f"lang_not_supported:{lang}",
                    self.notes_key,
                )
                skipped_unsupported += 1
                continue

            w = task.data.get(self.waveform_key)
            sr = task.data.get(self.sample_rate_key)
            if w is None or sr is None:
                skipped_no_audio += 1
                continue

            wav = self._prepare_waveform(w, int(sr))
            if wav is None:
                skipped_no_audio += 1
                continue

            prepped[i] = wav
            lang_to_indices[lang].append(i)

        return lang_to_indices, prepped, skipped_unsupported, skipped_no_audio

    def _transcribe_language_group(
        self,
        tasks: list[AudioTask],
        lang: str,
        indices: list[int],
        waveforms: list[torch.Tensor | None],
    ) -> None:
        """Run AI4Bharat's CTC ``transcribe()`` for a single-language bucket and
        write predictions / failure notes back onto the corresponding tasks.
        """
        try:
            raw = self._asr_model.transcribe(
                waveforms,
                batch_size=self.transcribe_batch_size,
                logprobs=False,
                language_id=lang,
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"[{self.name}] transcribe() failed for lang={lang}, batch_size={len(waveforms)}: {e}")
            for idx in indices:
                tasks[idx].data[self.pred_text_key] = ""
                tasks[idx].data[self.language_key] = lang
                set_note(
                    tasks[idx].data,
                    self.name,
                    f"transcribe_failed:{type(e).__name__}",
                    self.notes_key,
                )
            return

        # AI4Bharat's transcribe returns Tuple[List[str], ...]; the
        # first element is the list of greedy texts.
        texts = raw[0] if isinstance(raw, tuple) else raw
        for idx, text in zip(indices, texts, strict=True):
            tasks[idx].data[self.pred_text_key] = str(text) if text is not None else ""
            tasks[idx].data[self.language_key] = lang

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if len(tasks) == 0:
            return []
        if self._asr_model is None:
            msg = f"[{self.name}] Model not initialized — setup() was not called"
            raise RuntimeError(msg)

        t0 = time.perf_counter()

        for task in tasks:
            task.data.setdefault(self.pred_text_key, "")
            task.data.setdefault(self.language_key, "")

        lang_to_indices, prepped, skipped_unsupported, skipped_no_audio = self._bucket_tasks_by_language(tasks)

        total_eligible = sum(len(v) for v in lang_to_indices.values())
        with torch.inference_mode():
            for lang, indices in lang_to_indices.items():
                waveforms = [prepped[i] for i in indices]
                self._transcribe_language_group(tasks, lang, indices, waveforms)

        if not self.keep_waveform:
            for task in tasks:
                task.data.pop(self.waveform_key, None)

        self._log_metrics(
            {
                "process_time": time.perf_counter() - t0,
                "entries_processed": len(tasks),
                "predictions_generated": total_eligible,
                "skipped_unsupported_lang": skipped_unsupported,
                "skipped_no_audio": skipped_no_audio,
            }
        )
        logger.info(
            f"[{self.name}] generated {total_eligible} predictions, "
            f"skipped {skipped_unsupported} (unsupported lang), "
            f"{skipped_no_audio} (no audio)"
        )
        return tasks
