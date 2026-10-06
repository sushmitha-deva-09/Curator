# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Text post-processing pipeline for ASR output.

Reads JSONL manifests produced by the audio ASR pipeline and applies
text-only LLM stages (PnC, ITN, disfluency correction, captioning,
contextual ASR entity extraction, code-switching, speechQA).
Each stage runs as a separate Ray Data actor with its own vLLM engine.
All stages use the same model ID but load independently per actor.


Architecture:
    ALMManifestReader (CPU)
        → reads per-shard JSONL output from the ASR pipeline
    [if --enable_language_id and --language_id_first] LanguageID + verification
        → every backend runs on abbreviated_text before PnC
    [if --enable_pnc] TextLLMStage: PnC (GPU)
        → restores punctuation/capitalisation, writes pnc_text
    [if --enable_language_id without --language_id_first] LanguageID + verification
        → every backend runs on the configured PnC output after PnC
    LanguageID can use an LLM GPU, FastText CPU, or IndicLID CPU backend,
        or route rows by source_lang across all three
        → LLM preserves the existing prompt path; CPU backends reproduce the
          local evaluation preprocessing and batched top-1 inference
        → all backends write the same llm_language_prediction field
        → LLMLanguageVerificationStage (CPU): compares to source_lang; keeps
          every code-switched sample and flags single-language mismatches
    [if --enable_tn] TextLLMStage: TN (GPU)
        → text normalization (written→spoken), preserves disfluencies
        → writes tn_raw
    [if --enable_itn] TextLLMStage: ITN (GPU)
        → inverse text normalization of tn_raw (or pnc_text when TN disabled),
          preserves disfluencies
        → writes itn_raw
    [if --enable_itn_no-disfluencies] TextLLMStage: DisfluencyRemoval (GPU)
        → ITN + disfluency removal (fillers, repetitions)
        → writes itn_clean
    [if --enable_captioning] TextLLMStage: Captioning (GPU)
        → summarises transcript into a short caption
        → writes captioning_text
    [if --enable_context_asr] ContextualASRExtractionStage (GPU)
        → extracts domain, named-entity buckets, distractors
        → writes context_asr dict
    [if --enable_acoustic_distractor] AcousticDistractorStage (CPU)
        → G2Ps fine_context_terms and appends phonetically-similar
          words from the precomputed phoneme vocab to distractor_terms
    [if --enable_context_asr] ContextualASRPromptVariantStage (CPU)
        → appends five prompt variants to the context_asr dict
    ShardedManifestWriterStage (CPU)
        → writes per-shard JSONL output with .done markers
        → writes corrected_text
    [if --enable_code_switching] TextLLMStage: CodeSwitching (GPU)
        → restores English Latin spelling for English loanwords
          transliterated into the native script
        → writes code_switched_text
    [if --enable_speech_qa] TextLLMStage: SpeechQA (GPU)
        → either "SKIP" when the transcript is not suitable, or two lines
          "Q: <question>\nA: <answer>" grounded in the transcript
        → writes speech_qa_text
    [if --enable_instruction_packer] InstructionPackerStage (CPU)
        → collects every enabled text variant into a single
          'preference_instructions' field (list of {prompt, target} pairs) ready
          for instruction-tuning training
    ShardedManifestWriterStage (CPU)
        → writes per-shard JSONL output with .done markers

Each GPU stage runs as its own Ray Data actor (one GPU each) and loads its own
vLLM engine, so each stage may use its own max_model_len: the lightweight stages
use --max_model_len (2048) while contextual ASR uses --context_asr_max_model_len
(8192). They are NOT one shared engine.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.alm.alm_manifest_reader import ALMManifestReader
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask
from nemo_curator.stages.audio.alm.sharded_manifest_writer import ShardedManifestWriterStage
from nemo_curator.stages.audio.text_filtering.acoustic_distractor import AcousticDistractorStage
from nemo_curator.stages.audio.text_filtering.contextual_asr_extraction import ContextualASRExtractionStage
from nemo_curator.stages.audio.text_filtering.contextual_asr_prompt_variant import ContextualASRPromptVariantStage
from nemo_curator.stages.audio.text_filtering.fasttext_language_identification import (
    FastTextLanguageIdentificationStage,
)
from nemo_curator.stages.audio.text_filtering.fused_remote_text_llm_stage import FusedRemoteTextLLMStage
from nemo_curator.stages.audio.text_filtering.indiclid_language_identification import (
    IndicLIDLanguageIdentificationStage,
)
from nemo_curator.stages.audio.text_filtering.instruction_packer import InstructionPackerStage
from nemo_curator.stages.audio.text_filtering.llm_language_verification import LLMLanguageVerificationStage
from nemo_curator.stages.audio.text_filtering.pnc_language_rules import load_pnc_language_rules
from nemo_curator.stages.audio.text_filtering.remote_contextual_asr_extraction import (
    RemoteContextualASRExtractionStage,
)
from nemo_curator.stages.audio.text_filtering.remote_recover_entities import RemoteRecoverEntitiesStage
from nemo_curator.stages.audio.text_filtering.remote_text_llm_stage import RemoteTextLLMStage
from nemo_curator.stages.audio.text_filtering.text_llm_stage import TextLLMStage

_PROMPT_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "nemo_curator"
    / "stages"
    / "audio"
    / "text_filtering"
    / "prompts"
)
_ITN_PROMPT = _PROMPT_DIR / "itn_prompt.md"
_TN_PROMPT = _PROMPT_DIR / "tn_prompt.md"
_CORRECTION_PROMPT = _PROMPT_DIR / "correction_prompt.md"
_CAPTIONING_PROMPT = _PROMPT_DIR / "captioning_prompt.md"
_PNC_PROMPT = _PROMPT_DIR / "pnc_prompt.md"
_PNC_INDIC_PROMPT = _PROMPT_DIR / "pnc_prompt_indic.md"
_PNC_LANGUAGE_RULES = _PROMPT_DIR / "pnc_language_rules.json"
_CONTEXT_ASR_PROMPT = _PROMPT_DIR / "contextual_asr_prompt.md"

# Minimum recommended max_model_len when --enable_context_asr is used.
# The extraction prompt + output JSON can exceed shorter contexts.
_CONTEXT_ASR_MIN_MAX_MODEL_LEN = 4096
_CODE_SWITCHING_PROMPT = _PROMPT_DIR / "code_switching_prompt.md"
_SPEECH_QA_PROMPT = _PROMPT_DIR / "speech_qa_prompt.md"
_LANGUAGE_ID_PROMPT = _PROMPT_DIR / "language_id_prompt.md"
_RECOVER_ENTITIES_PROMPT = _PROMPT_DIR / "recover_entities_prompt.md"

@dataclass
class FieldCopyStage(ProcessingStage[AudioTask, AudioTask]):
    """Copy one manifest field to another (CPU, zero-cost).

    Useful for aliasing stage outputs — e.g. copying ``pnc_text`` → ``tn_raw``
    so downstream stages that read ``tn_raw`` receive the PnC result when the
    TN stage is disabled.  Records with a non-empty ``_skipme`` are passed
    through unchanged (dst_key is still written as empty string for schema
    consistency).
    """

    src_key: str = ""
    dst_key: str = ""
    name: str = "FieldCopy"
    skip_me_key: str = "_skipme"
    ignore_skip_me: bool = False
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def __post_init__(self) -> None:
        if not self.src_key or not self.dst_key:
            msg = "FieldCopyStage requires non-empty src_key and dst_key"
            raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.src_key, self.skip_me_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.dst_key]

    def _copy(self, task: AudioTask) -> AudioTask:
        if not self.ignore_skip_me and task.data.get(self.skip_me_key, ""):
            task.data.setdefault(self.dst_key, "")
        else:
            task.data[self.dst_key] = task.data.get(self.src_key, "")
        return task

    def process(self, task: AudioTask) -> AudioTask:
        return self._copy(task)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        return [self._copy(t) for t in tasks]


_LANGUAGE_ID_BACKENDS = frozenset({"llm", "fasttext", "indiclid", "config"})
_CPU_LANGUAGE_ID_BACKENDS = frozenset({"fasttext", "indiclid"})
_ROUTED_LANGUAGE_ID_BACKENDS = frozenset({"llm"}) | _CPU_LANGUAGE_ID_BACKENDS

# Evaluated Granary 22-language routing. Keep these sets next to the runner,
# matching the language-based model selection in the audio ASR example, so a
# production invocation does not depend on an external JSON file. Sindhi is
# deliberately absent from both CPU sets: the evaluated ASR text is Devanagari,
# while IndicLID-FTN only exposes ``snd_Arab`` and both CPU models scored 0%.
_FASTTEXT_LANGUAGE_ID_LANGS = frozenset({"bn", "hi", "kn", "mr", "pa", "ta", "te"})
_INDICLID_LANGUAGE_ID_LANGS = frozenset(
    {"as", "brx", "doi", "gu", "kok", "ks", "mai", "ml", "mni", "ne", "or", "sa", "sat", "ur"}
)
_DEFAULT_LANGUAGE_ID_BACKEND_BY_LANGUAGE = {
    **dict.fromkeys(sorted(_FASTTEXT_LANGUAGE_ID_LANGS), "fasttext"),
    **dict.fromkeys(sorted(_INDICLID_LANGUAGE_ID_LANGS), "indiclid"),
    "sd": "llm",
}
_SAMPLING_STAGE_KEYS = frozenset(
    {
        "recover_entities",
        "pnc",
        "language_id",
        "tn",
        "itn",
        "itn_no_disfluencies",
        "captioning",
        "context_asr",
        "code_switching",
        "speech_qa",
    }
)


def _load_language_id_backend_config(path: str) -> dict[str, str]:
    """Load a strict ``source_lang -> backend`` JSON routing override."""
    config_path = Path(path)
    try:
        parsed = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        msg = f"Unable to read language-ID backend config {config_path}: {exc}"
        raise ValueError(msg) from exc
    except json.JSONDecodeError as exc:
        msg = f"Invalid JSON in language-ID backend config {config_path}: {exc}"
        raise ValueError(msg) from exc

    if not isinstance(parsed, dict) or not parsed:
        msg = "Language-ID backend config must be a non-empty JSON object"
        raise ValueError(msg)

    normalized: dict[str, str] = {}
    for raw_language, raw_backend in parsed.items():
        if not isinstance(raw_language, str) or not raw_language.strip():
            msg = f"Language-ID backend config contains invalid language code {raw_language!r}"
            raise ValueError(msg)
        if not isinstance(raw_backend, str):
            msg = f"Language-ID backend for {raw_language!r} must be a string"
            raise TypeError(msg)
        language = raw_language.strip().lower()
        backend = raw_backend.strip().lower()
        if backend not in _ROUTED_LANGUAGE_ID_BACKENDS:
            msg = (
                f"Language-ID backend for {raw_language!r} must be one of "
                f"{sorted(_ROUTED_LANGUAGE_ID_BACKENDS)}, got {raw_backend!r}"
            )
            raise ValueError(msg)
        if language in normalized:
            msg = f"Language-ID backend config contains duplicate normalized code {language!r}"
            raise ValueError(msg)
        normalized[language] = backend
    return normalized


def _resolve_language_id_configuration(args: argparse.Namespace) -> tuple[dict[str, str] | None, str]:
    """Validate language-ID CLI combinations and resolve routing/output config."""
    backend = args.language_id_backend
    if not args.enable_language_id:
        configured_options = any(
            (
                backend != "llm",
                args.language_id_backend_config_file,
                args.fasttext_lid_model_path,
                args.indiclid_lid_model_path,
                args.language_id_first,
                args.language_id_text_key,
            )
        )
        if configured_options:
            msg = "Language-ID backend/model/config options require --enable_language_id"
            raise ValueError(msg)
        return None, "llm_language_prediction"

    if backend == "config":
        backend_by_language = (
            _load_language_id_backend_config(args.language_id_backend_config_file)
            if args.language_id_backend_config_file
            else dict(_DEFAULT_LANGUAGE_ID_BACKEND_BY_LANGUAGE)
        )
        selected_backends = set(backend_by_language.values())
        if selected_backends == {"llm"}:
            msg = "An all-LLM routing config is redundant; use --language_id_backend=llm instead"
            raise ValueError(msg)
    else:
        if args.language_id_backend_config_file:
            msg = "--language_id_backend_config_file requires --language_id_backend=config"
            raise ValueError(msg)
        backend_by_language = None
        selected_backends = {backend}

    if "fasttext" in selected_backends and not args.fasttext_lid_model_path:
        msg = "FastText language ID requires --fasttext_lid_model_path"
        raise ValueError(msg)
    if "indiclid" in selected_backends and not args.indiclid_lid_model_path:
        msg = "IndicLID language ID requires --indiclid_lid_model_path"
        raise ValueError(msg)
    if "fasttext" not in selected_backends and args.fasttext_lid_model_path:
        msg = "--fasttext_lid_model_path was provided but no language is routed to fasttext"
        raise ValueError(msg)
    if "indiclid" not in selected_backends and args.indiclid_lid_model_path:
        msg = "--indiclid_lid_model_path was provided but no language is routed to indiclid"
        raise ValueError(msg)

    return backend_by_language, "llm_language_prediction"


def _parse_stage_sampling_config(value: str) -> dict[str, dict[str, float]]:
    """Parse and validate per-stage temperature/top-p overrides."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        msg = f"Expected a JSON object, got invalid JSON: {exc}"
        raise argparse.ArgumentTypeError(msg) from exc
    if not isinstance(parsed, dict):
        msg = f"Expected a JSON object, got {type(parsed).__name__}"
        raise argparse.ArgumentTypeError(msg)

    unknown_stages = set(parsed) - _SAMPLING_STAGE_KEYS
    if unknown_stages:
        msg = f"Unknown sampling stage(s): {', '.join(sorted(unknown_stages))}"
        raise argparse.ArgumentTypeError(msg)

    normalized: dict[str, dict[str, float]] = {}
    for stage, config in parsed.items():
        normalized[stage] = _normalize_stage_sampling_config(stage, config)
    return normalized


def _normalize_stage_sampling_config(stage: str, config: object) -> dict[str, float]:
    """Validate and normalize one stage's sampling overrides."""
    if not isinstance(config, dict):
        msg = f"Sampling config for {stage!r} must be a JSON object"
        raise argparse.ArgumentTypeError(msg)
    unknown_fields = set(config) - {"temperature", "top_p"}
    if unknown_fields:
        msg = f"Unknown sampling field(s) for {stage!r}: {', '.join(sorted(unknown_fields))}"
        raise argparse.ArgumentTypeError(msg)

    normalized: dict[str, float] = {}
    for field_name, field_value in config.items():
        if isinstance(field_value, bool) or not isinstance(field_value, (int, float)):
            msg = f"{stage}.{field_name} must be a number"
            raise argparse.ArgumentTypeError(msg)
        numeric_value = float(field_value)
        if field_name == "temperature" and numeric_value < 0:
            msg = f"{stage}.temperature must be >= 0"
            raise argparse.ArgumentTypeError(msg)
        if field_name == "top_p" and not 0 < numeric_value <= 1:
            msg = f"{stage}.top_p must be in (0, 1]"
            raise argparse.ArgumentTypeError(msg)
        normalized[field_name] = numeric_value
    return normalized


def _resolve_stage_sampling(
    args: argparse.Namespace,
    stage: str,
    *,
    default_temperature: float | None = None,
    default_top_p: float | None = None,
) -> dict[str, float]:
    """Resolve a stage override over its backwards-compatible defaults."""
    if stage not in _SAMPLING_STAGE_KEYS:
        msg = f"Unknown sampling stage: {stage!r}"
        raise ValueError(msg)

    resolved_default_temperature = args.temperature if default_temperature is None else default_temperature
    resolved_default_top_p = args.top_p if default_top_p is None else default_top_p
    return {
        "temperature": resolved_default_temperature,
        "top_p": resolved_default_top_p,
        **args.stage_sampling_config.get(stage, {}),
    }


def _resolve_pnc_prompt_file(custom_prompt_file: str | None, *, use_indic_prompt: bool) -> str:
    if use_indic_prompt:
        return str(_PNC_INDIC_PROMPT)
    return custom_prompt_file or str(_PNC_PROMPT)


def _build_arg_parser() -> argparse.ArgumentParser:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description="Text post-processing pipeline for ASR output")

    ap.add_argument(
        "--input_manifest",
        type=str,
        required=True,
        help="Path to JSONL manifest(s) from the ASR pipeline output. Can be a directory or a glob pattern.",
    )
    ap.add_argument("--output_dir", type=str, required=True, help="Output directory for processed manifests.")

    ap.add_argument(
        "--enable_tn",
        action="store_true",
        default=False,
        help=(
            "Enable TN stage: written→spoken form (digits/symbols to number words), "
            "preserves disfluencies and all non-normalization wording. Output key: tn_raw"
        ),
    )
    ap.add_argument(
        "--disable_tn_validation",
        action="store_true",
        default=False,
        help=(
            "Disable TN output validation. By default the TN stage validates each "
            "output (validation_mode='tn'): it reverts to the input when the LLM "
            "translates to English or transliterates to another script instead of "
            "normalizing. Number expansion (word-count increase) is NOT penalized."
        ),
    )
    ap.add_argument(
        "--enable_itn",
        action="store_true",
        default=False,
        help="Enable ITN stage: spoken→written form, preserves disfluencies. Output key: itn_raw",
    )
    ap.add_argument(
        "--enable_itn_no-disfluencies",
        action="store_true",
        default=False,
        help="Enable ITN + disfluency removal stage: removes fillers/repetitions + ITN. Output key: itn_clean",
    )
    ap.add_argument(
        "--enable_captioning",
        action="store_true",
        default=False,
        help="Enable captioning stage: summarizes transcript into a short caption. Output key: captioning_text",
    )
    ap.add_argument(
        "--enable_pnc",
        action="store_true",
        default=False,
        help="Enable PnC stage: restores punctuation and capitalization. Output key: pnc_text",
    )
    ap.add_argument(
        "--enable_language_id",
        action="store_true",
        default=False,
        help=(
            "Enable language identification plus CPU verification. The default --language_id_backend=llm "
            "preserves the existing LLM behavior; fasttext and indiclid select one local CPU model, while config "
            "uses the built-in per-language FastText/IndicLID/LLM routing."
        ),
    )
    ap.add_argument(
        "--language_id_backend",
        choices=sorted(_LANGUAGE_ID_BACKENDS),
        default="llm",
        help=(
            "Language-ID implementation. 'llm' preserves the existing prompt stage; 'fasttext' and 'indiclid' "
            "use one local model for every row; 'config' routes each row by source_lang using "
            "the built-in evaluated 22-language split (FastText/IndicLID, with Sindhi routed to the LLM). "
            "--language_id_backend_config_file can optionally override that split. Direct fasttext fails closed "
            "if any row's source_lang is not an exact lid.176 label."
        ),
    )
    ap.add_argument(
        "--language_id_backend_config_file",
        type=str,
        default=None,
        help=(
            "Optional JSON override mapping each source_lang code to 'llm', 'fasttext', or 'indiclid'. Valid only "
            "with --language_id_backend=config; without it, the runner uses its built-in evaluated 22-language "
            "routing. A row whose source_lang is absent from the selected mapping fails closed. Example: "
            "examples/audio/text_processing/language_id_backends_indic_22.json."
        ),
    )
    ap.add_argument(
        "--fasttext_lid_model_path",
        type=str,
        default=None,
        help="Path to the official fastText lid.176.bin checkpoint.",
    )
    ap.add_argument(
        "--indiclid_lid_model_path",
        type=str,
        default=None,
        help="Path to AI4Bharat IndicLID-FTN v1.0 model_baseline_roman.bin.",
    )
    ap.add_argument(
        "--language_id_first",
        action="store_true",
        default=False,
        help=(
            "Run language ID and verification before PnC for every backend. Without this option, every backend "
            "runs after PnC. Language ID is always serial and is never included in stage fusion."
        ),
    )
    ap.add_argument(
        "--language_id_text_key",
        type=str,
        default=None,
        help=(
            "Input field for language ID, applied identically to every backend. Defaults to --pnc_output_key "
            "when PnC is enabled, existing pnc_text otherwise, or abbreviated_text with --language_id_first."
        ),
    )
    ap.add_argument(
        "--enable_context_asr",
        action="store_true",
        default=False,
        help=(
            "Enable contextual ASR stages: extract domain / named entities / distractors via LLM, "
            "then append five prompt variants. Output key: context_asr (nested dict)."
        ),
    )
    ap.add_argument(
        "--enable_acoustic_distractor",
        action="store_true",
        default=False,
        help=(
            "Enable acoustic distractor stage (CPU-only): G2Ps fine_context_terms via phonemizer "
            "and appends phonetically-similar words from --phoneme_vocab_path to context_asr.distractor_terms. "
            "Requires --enable_context_asr and --phoneme_vocab_path."
        ),
    )
    ap.add_argument(
        "--enable_code_switching",
        action="store_true",
        default=False,
        help="Enable code-switching stage: restores English Latin spelling for English loanwords "
        "transliterated into the native script. Output key: code_switched_text",
    )
    ap.add_argument(
        "--enable_speech_qa",
        action="store_true",
        default=False,
        help="Enable SpeechQA stage: generates a (Q, A) pair grounded in the transcript, or the "
        "single token SKIP when the transcript is not suitable. Output key: speech_qa_text",
    )
    ap.add_argument(
        "--ignore_skip_me",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Process ALL samples regardless of the '_skipme' flag (default: True). "
        "Pass --no_ignore_skip_me to respect the '_skipme' field and skip flagged records.",
    )
    ap.add_argument(
        "--enable_instruction_packer",
        action="store_true",
        default=False,
        help="Pack all enabled text outputs (pnc/itn/itn_no-disfluencies/captioning/code-switched/"
        "speech_qa/context_asr) into a single 'preference_instructions' field — a list of "
        "{prompt, target} pairs the trainer samples from one-per-epoch.",
    )
    ap.add_argument(
        "--instructions_output_key",
        type=str,
        default="preference_instructions",
        help="Output field name for the packed instruction-target list.",
    )
    ap.add_argument(
        "--instruction_packer_seed",
        type=int,
        default=42,
        help="Base RNG seed for per-sample prompt-template sampling in the instruction packer.",
    )
    ap.add_argument(
        "--instruction_packer_translations_key",
        type=str,
        default="translations",
        help="Source field holding the per-target-language translation dict packed as 'translation' instructions.",
    )
    ap.add_argument(
        "--instruction_packer_min_translation_score",
        type=float,
        default=0.7,
        help="Minimum translation_quality_score required to pack a translation; set to 0 to pack all non-empty translations.",
    )

    ap.add_argument("--text_key", type=str, default="pnc_text", help="Input text field from ASR pipeline output.")
    ap.add_argument("--tn_output_key", type=str, default="tn_raw", help="Output field for TN result.")
    ap.add_argument("--itn_output_key", type=str, default="itn_raw", help="Output field for ITN result.")
    ap.add_argument(
        "--itn_no_disfluencies_output_key",
        type=str,
        default="itn_clean",
        help="Output field for ITN + disfluency removal result.",
    )
    ap.add_argument(
        "--captioning_output_key", type=str, default="captioning_text", help="Output field for captioning result."
    )
    ap.add_argument("--pnc_output_key", type=str, default="pnc_text", help="Output field for PnC result.")
    ap.add_argument(
        "--code_switching_output_key",
        type=str,
        default="code_switched_text",
        help="Output field for code-switching restoration result.",
    )
    ap.add_argument(
        "--speech_qa_output_key",
        type=str,
        default="speech_qa_text",
        help="Output field for SpeechQA result. Either 'SKIP' (single line) or two lines 'Q: ...' + 'A: ...'.",
    )

    ap.add_argument(
        "--model_id", type=str, default="Qwen/Qwen3.5-35B-A3B-FP8", help="HuggingFace model ID for the text LLM."
    )
    ap.add_argument(
        "--tn_prompt_file", type=str, default=None, help="Path to TN prompt file. Defaults to bundled tn_prompt.md."
    )
    ap.add_argument(
        "--itn_prompt_file", type=str, default=None, help="Path to ITN prompt file. Defaults to bundled itn_prompt.md."
    )
    ap.add_argument(
        "--itn_no_disfluencies_prompt_file",
        type=str,
        default=None,
        help="Path to ITN + disfluency removal prompt file. Defaults to bundled correction_prompt.md.",
    )
    ap.add_argument(
        "--captioning_prompt_file",
        type=str,
        default=None,
        help="Path to captioning prompt file. Defaults to bundled captioning_prompt.md.",
    )
    pnc_prompt_group = ap.add_mutually_exclusive_group()
    pnc_prompt_group.add_argument(
        "--pnc_prompt_file",
        type=str,
        default=None,
        help="Path to a custom PnC prompt file. Defaults to the shared bundled pnc_prompt.md.",
    )
    pnc_prompt_group.add_argument(
        "--use_indic_pnc_prompt",
        action="store_true",
        help="Use the bundled row-scoped Indic P3 V6 prompt (pnc_prompt_indic.md).",
    )
    ap.add_argument(
        "--pnc_language_rules_file",
        type=str,
        default=None,
        help=(
            "JSON mapping used to resolve the PnC prompt's {language_rules} placeholder. "
            "Defaults to bundled pnc_language_rules.json and is loaded only when the selected prompt uses the placeholder."
        ),
    )
    ap.add_argument(
        "--language_id_prompt_file",
        type=str,
        default=None,
        help="Path to language-ID prompt file. Defaults to bundled language_id_prompt.md.",
    )
    ap.add_argument(
        "--code_switching_prompt_file",
        type=str,
        default=None,
        help="Path to code-switching prompt file. Defaults to bundled code_switching_prompt.md.",
    )
    ap.add_argument(
        "--speech_qa_prompt_file",
        type=str,
        default=None,
        help="Path to SpeechQA prompt file. Defaults to bundled speech_qa_prompt.md.",
    )

    ctx = ap.add_argument_group("contextual ASR")
    ctx.add_argument(
        "--context_asr_prompt_file",
        type=str,
        default=None,
        help="Path to context ASR extraction prompt file. Defaults to bundled contextual_asr_prompt.md.",
    )
    ctx.add_argument(
        "--context_asr_text_key",
        type=str,
        default="pnc_text",
        help="Input text field for context ASR extraction.",
    )
    ctx.add_argument(
        "--context_asr_high_quality_key",
        type=str,
        default=None,
        help=(
            "Optional manifest key (e.g. 'high_quality'); records whose value at this key is "
            "explicitly falsy are skipped (context_asr=None, note 'skipped (low_quality)'), so "
            "ctx only runs on the kept subset. Missing key => processed. Unset disables gating."
        ),
    )
    ctx.add_argument(
        "--context_asr_output_key",
        type=str,
        default="context_asr",
        help="Output field (nested dict) for context ASR extraction + prompt variants.",
    )
    ctx.add_argument(
        "--context_asr_max_output_tokens",
        type=int,
        default=2048,
        help=(
            "Max tokens generated per sample for context ASR extraction. This stage's vLLM engine "
            "max_model_len (--context_asr_max_model_len) must be ≥ this value plus prompt length."
        ),
    )
    ctx.add_argument(
        "--context_asr_max_model_len",
        type=int,
        default=8192,
        help=(
            "vLLM max_model_len for the contextual-ASR extraction stage ONLY. This stage runs as its "
            "own Ray Data actor (one GPU, its own vLLM engine), so it can use a larger context window "
            "than the lightweight stages without inflating their KV cache. The global --max_model_len "
            "stays small (2048) for PnC/ITN/etc. Must exceed the extraction prompt + "
            "--context_asr_max_output_tokens."
        ),
    )
    ctx.add_argument("--context_asr_seed", type=int, default=42, help="Base RNG seed for prompt-variant generation.")
    ctx.add_argument(
        "--context_asr_partial_keep_lo",
        type=float,
        default=0.5,
        help="Lower bound of random keep fraction for the partial-context variant.",
    )
    ctx.add_argument(
        "--context_asr_partial_keep_hi",
        type=float,
        default=0.8,
        help="Upper bound of random keep fraction for the partial-context variant.",
    )

    ad = ap.add_argument_group("acoustic distractors")
    ad.add_argument(
        "--phoneme_vocab_path",
        type=str,
        default=None,
        help=(
            "Path to a precomputed phoneme vocabulary JSON (produced by scripts/build_phoneme_vocab.py). "
            "Required when --enable_acoustic_distractor is set."
        ),
    )
    ad.add_argument(
        "--phoneme_vocab_language",
        type=str,
        default=None,
        help=(
            "Optional espeak-ng language code (e.g. en-us, fr, de, cmn). When set, used for all samples; "
            "otherwise the per-sample source_lang is mapped to an espeak code."
        ),
    )
    ad.add_argument(
        "--max_acoustic_distractors",
        type=int,
        default=8,
        help="Maximum acoustic distractors appended per sample.",
    )
    ad.add_argument(
        "--max_total_distractors",
        type=int,
        default=16,
        help="Cap on the combined (semantic + acoustic) distractor_terms list.",
    )
    ad.add_argument(
        "--acoustic_per_entity_top_k",
        type=int,
        default=3,
        help="Top-K acoustic candidates retained per source entity before cross-entity merging.",
    )
    ad.add_argument(
        "--min_npd",
        type=float,
        default=0.1,
        help="Lower NPD bound — entries closer than this are dropped (near-duplicates).",
    )
    ad.add_argument(
        "--max_npd",
        type=float,
        default=0.5,
        help="Upper NPD bound — entries farther than this are dropped (acoustically unrelated).",
    )
    ad.add_argument(
        "--acoustic_num_workers",
        type=int,
        default=None,
        help=(
            "Explicit Ray actor count for the CPU-only AcousticDistractor stage. It is cpus=1.0 "
            "and CPU-bound (G2P + phoneme NN search), so set this high to saturate idle CPUs "
            "rather than letting it autoscale low behind the cpus=8.0 LLM clients."
        ),
    )

    re = ap.add_argument_group("recover entities")
    re.add_argument(
        "--enable_recover_entities",
        action="store_true",
        default=False,
        help=(
            "Enable RecoverEntities stage (runs BEFORE all other LLM stages): rewrites the normalized "
            "transcription so its named entities use the ground-truth spelling/casing. Routed to the "
            "shared Dynamo inference server, so it REQUIRES --use_inference_server. Output key: "
            "--recover_entities_output_key (default entity_recovered_text)."
        ),
    )
    re.add_argument(
        "--recover_entities_prompt_file",
        type=str,
        default=None,
        help="Path to the RecoverEntities prompt file. Defaults to bundled recover_entities_prompt.md.",
    )
    re.add_argument(
        "--recover_entities_ground_truth_key",
        type=str,
        default="granary_v1_prediction",
        help="Manifest field holding the ground-truth/INITIAL transcription (source of correct entity forms).",
    )
    re.add_argument(
        "--recover_entities_normalized_key",
        type=str,
        default="abbreviated_text",
        help="Manifest field holding the normalized transcription (entities are recovered into a copy of this).",
    )
    re.add_argument(
        "--recover_entities_output_key",
        type=str,
        default="entity_recovered_text",
        help="Output field for the normalized text with recovered entity forms.",
    )
    re.add_argument(
        "--recover_entities_max_output_tokens",
        type=int,
        default=1024,
        help="Max tokens generated per sample for entity recovery.",
    )
    re.add_argument(
        "--recover_entities_max_model_len",
        type=int,
        default=4096,
        help=(
            "vLLM max_model_len required by RecoverEntities. When --use_inference_server is set, the "
            "shared server's max_model_len is raised to cover this value."
        ),
    )
    re.add_argument(
        "--recover_entities_num_workers",
        type=int,
        default=None,
        help="Explicit Ray actor count for the RecoverEntities stage. Overrides --num_workers for that stage only.",
    )
    re.add_argument(
        "--recover_entities_max_concurrent_requests",
        type=int,
        default=None,
        help=(
            "Max concurrent requests for the RecoverEntities stage. Overrides "
            "--inference_max_concurrent_requests for that stage only."
        ),
    )

    ap.add_argument(
        "--tensor_parallel_size", type=int, default=None, help="GPUs for tensor parallelism (default: auto-detect)."
    )
    ap.add_argument("--max_model_len", type=int, default=2048)
    ap.add_argument(
        "--max_num_batched_tokens",
        type=int,
        default=None,
        help="vLLM max_num_batched_tokens forwarded to the inference-server engine "
        "(None = vLLM default). Tuning knob for e.g. the gemma-4 recipe's max_batched_8k.",
    )
    ap.add_argument(
        "--language_model_only",
        action="store_true",
        help="Serve a multimodal model (e.g. gemma-4) as LANGUAGE-MODEL-ONLY: skip the vision/audio "
        "encoders (we only send text). Passes vLLM --language-model-only to the inference-server engine. "
        "Much lighter/faster/more-stable load and removes the multimodal max_num_batched_tokens>=2496 "
        "constraint. No effect on text-only models.",
    )
    ap.add_argument("--max_num_seqs", type=int, default=256)
    ap.add_argument("--max_output_tokens", type=int, default=512)
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the generic text stages (default: deterministic 0.0).",
    )
    ap.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Nucleus-sampling probability for the generic text stages (default: 1.0).",
    )
    ap.add_argument(
        "--stage_sampling_config",
        type=_parse_stage_sampling_config,
        default={},
        metavar="JSON",
        help=(
            "Per-stage sampling overrides. Keys: recover_entities, pnc, language_id, tn, itn, "
            "itn_no_disfluencies, captioning, context_asr, code_switching, speech_qa. Each value "
            "may set temperature and/or top_p, for example "
            '\'{"pnc":{"temperature":0},"speech_qa":{"temperature":0.7,"top_p":0.95}}\'. '
            "Global --temperature/--top_p remain fallbacks for generic text stages."
        ),
    )
    ap.add_argument(
        "--speculative_config",
        type=str,
        default=None,
        help=(
            "Value forwarded verbatim to vLLM --speculative-config, for example "
            '\'{"model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":4}\'.'
        ),
    )
    ap.add_argument(
        "--reuse_current_dynamo_environment",
        action="store_true",
        help=(
            "Skip Ray's isolated ai-dynamo[vllm] runtime-environment install and use the current "
            "container Python environment. The image must already contain compatible Dynamo, vLLM, "
            "and Ray packages. Useful for prebuilt production images and network filesystems where "
            "cloning the full driver virtualenv is prohibitively slow."
        ),
    )
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    ap.add_argument("--kv_cache_dtype", type=str, default="fp8")
    ap.add_argument("--num_workers", type=int, default=None, help="Explicit GPU worker count for Xenna.")
    ap.add_argument(
        "--context_asr_num_workers",
        type=int,
        default=None,
        help="Explicit Ray actor count for the ContextualASR stage. Overrides --num_workers for that stage only.",
    )
    ap.add_argument(
        "--context_asr_max_concurrent_requests",
        type=int,
        default=None,
        help="Max concurrent requests for the ContextualASR stage. Overrides --inference_max_concurrent_requests for that stage only.",
    )
    ap.add_argument("--batch_size", type=int, default=64)

    ap.add_argument("--execution_mode", type=str, default="streaming", choices=["streaming", "batch"])
    ap.add_argument(
        "--shards_per_batch",
        type=int,
        default=1,
        help=(
            "Process the input in batches of this many shard manifests, one pipeline.run() per "
            "batch (the inference server is started once and reused across batches). Default 1 = "
            "true shard-by-shard: each shard fully drains and is finalized with a .done marker "
            "before the next starts, so a mid-run kill loses at most one shard. Raise (e.g. 4-8) "
            "for very large shard counts to amortize per-run overhead. <=0 processes the whole "
            "input in a single run (legacy behaviour, weakest resumability)."
        ),
    )

    # ── Remote inference server (optional) ───────────────────────────
    # When enabled, the LLM stages send OpenAI-compatible requests to one
    # shared NVIDIA Dynamo (vLLM) server instead of each loading its own engine.
    srv = ap.add_argument_group("remote inference server")
    srv.add_argument(
        "--use_inference_server",
        action="store_true",
        default=False,
        help="Start a local RayClient + NVIDIA Dynamo InferenceServer and route all LLM "
        "stages to it as CPU-only HTTP clients. Without this flag, stages run in-process vLLM.",
    )
    srv.add_argument(
        "--inference_served_model_name",
        type=str,
        default=None,
        help="Model name sent as 'model=' in requests. Defaults to --model_id.",
    )
    srv.add_argument("--inference_api_key", type=str, default="EMPTY", help="API key forwarded to the server.")
    srv.add_argument(
        "--inference_max_replicas", type=int, default=1, help="Number of Dynamo vLLM worker replicas (num_replicas)."
    )
    srv.add_argument(
        "--inference_server_tp",
        type=int,
        default=None,
        help="tensor_parallel_size for the server engine (default: auto = visible GPU count).",
    )
    srv.add_argument(
        "--inference_router_mode",
        choices=["round_robin", "random", "kv", "direct"],
        default=None,
        help=(
            "Dynamo frontend routing policy for every request to the shared model. None preserves "
            "Dynamo's default (round-robin for aggregated serving); 'kv' enables cache-aware routing."
        ),
    )
    srv.add_argument("--inference_port", type=int, default=8000, help="Server HTTP port.")
    srv.add_argument(
        "--inference_max_concurrent_requests",
        type=int,
        default=64,
        help="Max in-flight requests per stage actor (async client semaphore bound).",
    )
    srv.add_argument("--inference_request_timeout", type=int, default=120, help="Per-request timeout (seconds).")
    srv.add_argument(
        "--inference_queue_max_waiting_requests",
        type=int,
        default=None,
        help=(
            "Enable shared vLLM queue admission at this per-worker threshold. "
            "Overload returns HTTP 429 and drives the model-wide AIMD window."
        ),
    )
    srv.add_argument(
        "--inference_admission_max_concurrent_requests",
        type=int,
        default=8192,
        help="Initial and maximum shared AIMD concurrency (default: 8192).",
    )
    srv.add_argument(
        "--inference_health_timeout",
        type=int,
        default=300,
        help="Seconds to wait for the inference server to become healthy before failing (default: 300).",
    )
    srv.add_argument(
        "--fuse_stages",
        action="store_true",
        default=False,
        help="Fuse independent LLM stages (Captioning, CodeSwitching, and SpeechQA) into a single actor that "
        "fires their prompts in parallel via asyncio.gather. Language ID is always serial so every backend has "
        "the same mutation order. Only active with --use_inference_server. TN always runs serially before the "
        "fused actor; dependent ITN stages run after it.",
    )

    return ap


def _resolve_shard_batches(input_manifest: str | list[str], shards_per_batch: int) -> list:
    """Split the input into batches of ``shards_per_batch`` manifest files, each a
    ``manifest_path`` for one ``pipeline.run()``. ``shards_per_batch <= 0`` returns the
    whole input unsplit (single run)."""
    import glob as _glob

    if not shards_per_batch or shards_per_batch <= 0:
        return [input_manifest]
    if isinstance(input_manifest, list):
        files = list(input_manifest)
    elif os.path.isdir(input_manifest):
        files = sorted(
            _glob.glob(os.path.join(input_manifest, "**", "*.jsonl"), recursive=True)
            + _glob.glob(os.path.join(input_manifest, "**", "*.json"), recursive=True)
        )
    elif any(c in input_manifest for c in "*?["):
        files = sorted(_glob.glob(input_manifest, recursive=True))
    else:
        files = [input_manifest]
    files = [f for f in files if not f.endswith(".done")]
    if not files:
        return [input_manifest]
    return [files[i : i + shards_per_batch] for i in range(0, len(files), shards_per_batch)]


def _finalize_done_markers(manifest_files: list, output_dir: str) -> None:
    """Deterministically write ``.done`` for every shard in a just-completed batch.

    A successful ``pipeline.run()`` over the batch guarantees all rows for these
    shards drained through the writer, so ``.done`` can be written by output line
    count instead of relying on the writer's in-stream ``count >= total`` check
    (which silently never fires if any row is dropped). Shards already finalized
    (skipped on resume) or producing no output rows are handled too.
    """
    from fsspec.core import url_to_fs

    from nemo_curator.stages.audio.alm.alm_manifest_reader import ALMManifestReaderStage

    for manifest in manifest_files:
        try:
            corpus = ""
            fs, resolved = url_to_fs(manifest)
            with fs.open(resolved, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        corpus = json.loads(line.strip()).get("corpus", "")
                        break
            shard_key = ALMManifestReaderStage._derive_shard_key(manifest, corpus)
            out_jsonl = os.path.join(output_dir, f"{shard_key}.jsonl")
            done_path = out_jsonl + ".done"
            if os.path.exists(done_path):
                continue
            if not os.path.exists(out_jsonl):
                # Shard processed but yielded no output rows — still mark it done so it
                # is not reprocessed on every resume.
                os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
                open(out_jsonl, "a").close()
            with open(out_jsonl, "rb") as f:
                n = sum(1 for _ in f)
            with open(done_path, "w") as f:
                f.write(f"{n}\n")
            logger.info(f"Finalized .done for shard {shard_key} ({n} utterances)")
        except Exception:
            logger.exception(f"Failed to finalize .done for manifest {manifest}")


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    args = _build_arg_parser().parse_args()
    language_id_backend_config, language_id_output_key = _resolve_language_id_configuration(args)

    if (
        not args.enable_tn
        and not args.enable_itn
        and not args.enable_itn_no_disfluencies
        and not args.enable_captioning
        and not args.enable_pnc
        and not args.enable_language_id
        and not args.enable_context_asr
        and not args.enable_code_switching
        and not args.enable_speech_qa
        and not args.enable_recover_entities
    ):
        logger.warning(
            "No stages enabled. Use --enable_recover_entities, --enable_pnc, --enable_tn, --enable_language_id, "
            "--enable_itn, --enable_itn_no-disfluencies, --enable_captioning, --enable_context_asr, "
            "--enable_code_switching, or --enable_speech_qa."
        )
        return

    # RecoverEntities is only implemented against the shared Dynamo inference
    # server (no in-process variant), so it requires --use_inference_server.
    if args.enable_recover_entities and not args.use_inference_server:
        msg = "--enable_recover_entities requires --use_inference_server (RecoverEntities runs on the Dynamo server)."
        raise ValueError(msg)
    if args.inference_queue_max_waiting_requests is not None and not args.use_inference_server:
        msg = "--inference_queue_max_waiting_requests requires --use_inference_server."
        raise ValueError(msg)

    # ── Optional Dynamo inference server ─────────────────────────────
    # Two modes:
    #   --use_inference_server -> start a local RayClient + NVIDIA Dynamo server
    #                             and route all LLM stages to it as HTTP clients
    #   (default)              -> in-process vLLM, one engine per stage
    # Resolved BEFORE building stages so the server is healthy at pipeline start.
    inference_server = None
    ray_client = None
    remote_base_url = None
    remote_model_name = None
    if args.use_inference_server:
        import torch

        from nemo_curator.core.client import RayClient
        from nemo_curator.core.serve import (
            DynamoAdmissionConfig,
            DynamoRouterConfig,
            DynamoServerConfig,
            DynamoVLLMModelConfig,
            InferenceServer,
        )

        # Count GPUs without Ray (ray.available_resources() requires a running
        # cluster). torch.cuda honours CUDA_VISIBLE_DEVICES. Start the cluster
        # exposing those GPUs BEFORE the backend deploys; the server is torn down
        # before the client at teardown (see finally).
        n_gpus = torch.cuda.device_count()
        ray_client = RayClient(num_gpus=n_gpus)
        ray_client.start()

        server_tp = args.inference_server_tp or n_gpus
        # One engine serves every stage, so its max_model_len must cover the
        # largest requirement (context_asr 8192 vs the text stages' 2048).
        server_max_len = args.max_model_len
        if args.enable_context_asr:
            server_max_len = max(server_max_len, args.context_asr_max_model_len)
        if args.enable_recover_entities:
            server_max_len = max(server_max_len, args.recover_entities_max_model_len)
        engine_kwargs = {
            "tensor_parallel_size": server_tp,
            "max_model_len": server_max_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "kv_cache_dtype": args.kv_cache_dtype,
            "trust_remote_code": True,
            "max_num_seqs": args.max_num_seqs,
        }
        # Fixed torch.compile cache dir (opt-in): vLLM's default key hashes traced-file paths, which
        # sit under Ray's per-session dir -> unique per lane -> never reused. An explicit cache_dir
        # makes vLLM reuse one graph across lanes (backends.py: `if not compilation_config.cache_dir`).
        _compile_cache = os.environ.get("VLLM_COMPILE_CACHE_DIR")
        if _compile_cache:
            engine_kwargs["compilation_config"] = {"cache_dir": _compile_cache}
        if args.max_num_batched_tokens is not None:
            engine_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
        if args.speculative_config is not None:
            engine_kwargs["speculative_config"] = args.speculative_config
        if args.language_model_only:
            # gemma-4 is multimodal; we only send text. LM-only skips the vision/audio towers
            # -> lighter, faster, stable load + no --disable_chunked_mm_input / mm-token constraint.
            engine_kwargs["language_model_only"] = True
        if server_tp == 1:
            # TP=1: force the uniprocessor executor so each replica skips
            # torch.distributed init (no TCPStore / rendezvous port). For TP>1 leave
            # it unset — vLLM picks 'mp' for single-node TP, and Dynamo drives
            # multi-node TP via --nnodes/--node-rank/--master-addr; forcing 'mp'
            # would break the multi-node case.
            engine_kwargs["distributed_executor_backend"] = "uni"

        # Dynamo uses a fixed replica count (no autoscaling). engine_kwargs are
        # forwarded to `python -m dynamo.vllm` as CLI flags. Needs the `dynamo`
        # package plus `etcd` + `nats-server` on PATH.
        model_cfg = DynamoVLLMModelConfig(
            model_identifier=args.model_id,
            model_name=args.inference_served_model_name,
            engine_kwargs=engine_kwargs,
            install_runtime_dependencies=not args.reuse_current_dynamo_environment,
            num_replicas=args.inference_max_replicas,
        )
        router_cfg = DynamoRouterConfig(
            mode=args.inference_router_mode.replace("_", "-") if args.inference_router_mode else None,
        )
        admission_cfg = (
            DynamoAdmissionConfig(
                max_waiting_requests=args.inference_queue_max_waiting_requests,
                max_concurrent_requests=args.inference_admission_max_concurrent_requests,
            )
            if args.inference_queue_max_waiting_requests is not None
            else None
        )
        backend_cfg = DynamoServerConfig(router=router_cfg, admission=admission_cfg)

        inference_server = InferenceServer(
            models=[model_cfg],
            backend=backend_cfg,
            port=args.inference_port,
            health_check_timeout_s=args.inference_health_timeout,
        )
        inference_server.start()
        # endpoint/port are finalized during start() (Dynamo binds to the infra
        # node IP and may reallocate the port), so read them after.
        remote_base_url = inference_server.endpoint
        remote_model_name = model_cfg.resolved_model_name

    remote_kwargs: dict = {}
    if remote_base_url:
        remote_kwargs = {
            "inference_base_url": remote_base_url,
            "served_model_name": remote_model_name,
            "inference_api_key": args.inference_api_key,
            "max_concurrent_requests": args.inference_max_concurrent_requests,
            "request_timeout": args.inference_request_timeout,
        }
        logger.info(f"Routing LLM stages to remote inference server at {remote_base_url} (model={remote_model_name})")
    # Pick in-process vs remote stage classes. The Remote* classes subclass
    # the in-process ones, inherit (and ignore) the GPU engine kwargs, and
    # force CPU-only resources in their __post_init__.
    text_stage_cls = RemoteTextLLMStage if remote_base_url else TextLLMStage
    ctx_stage_cls = RemoteContextualASRExtractionStage if remote_base_url else ContextualASRExtractionStage

    tn_prompt = args.tn_prompt_file or str(_TN_PROMPT)
    itn_prompt = args.itn_prompt_file or str(_ITN_PROMPT)
    itn_no_disfl_prompt = args.itn_no_disfluencies_prompt_file or str(_CORRECTION_PROMPT)
    captioning_prompt = args.captioning_prompt_file or str(_CAPTIONING_PROMPT)
    pnc_prompt = _resolve_pnc_prompt_file(
        args.pnc_prompt_file,
        use_indic_prompt=args.use_indic_pnc_prompt,
    )
    pnc_language_rules = None
    if args.enable_pnc and "{language_rules}" in Path(pnc_prompt).read_text(encoding="utf-8"):
        rules_file = args.pnc_language_rules_file or str(_PNC_LANGUAGE_RULES)
        pnc_language_rules = load_pnc_language_rules(rules_file)
        logger.info(
            "PnC per-row language rules enabled from {} (codes={})",
            rules_file,
            sorted(pnc_language_rules),
        )
    language_id_prompt = args.language_id_prompt_file or str(_LANGUAGE_ID_PROMPT)
    context_asr_prompt = args.context_asr_prompt_file or str(_CONTEXT_ASR_PROMPT)
    code_switching_prompt = args.code_switching_prompt_file or str(_CODE_SWITCHING_PROMPT)
    speech_qa_prompt = args.speech_qa_prompt_file or str(_SPEECH_QA_PROMPT)

    shared_model_kwargs = {
        "model_id": args.model_id,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_output_tokens": args.max_output_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "kv_cache_dtype": args.kv_cache_dtype,
        "num_workers_override": args.num_workers,
        "batch_size": args.batch_size,
        "ignore_skip_me": args.ignore_skip_me,
        **remote_kwargs,
    }

    # When --fuse_stages is active, parallel rewrite stages are collected here
    # and wrapped in one FusedRemoteTextLLMStage actor. Language ID stays serial
    # for every backend so prediction, verification, and _skipme mutations occur
    # at the same point in the pipeline. Fusion only has an effect with
    # --use_inference_server; it falls back to normal behaviour otherwise.
    use_fusing = bool(args.fuse_stages and remote_base_url)
    fuseable_sub_stages: list[RemoteTextLLMStage] = []

    # entity_recovered_text (when RecoverEntities ran) is the entity-corrected
    # normalized text. It supersedes the raw abbreviated_text/pnc_text as the
    # transcript the downstream LLM stages read. The "current best" transcript
    # threads through the chain:
    #   abbreviated_text/manifest -> (RecoverEntities) entity_recovered_text
    #                             -> (PnC) pnc_text -> TN/ITN/Captioning/...
    # so each stage reads the freshest key produced before it.
    recovered_key = args.recover_entities_output_key if args.enable_recover_entities else None
    # RecoverEntities (when on) reads abbreviated_text + granary_v1_prediction and
    # writes recovered_key. How that reaches the rest of the pipeline:
    #   - PnC ON : PnC reads recovered_key and republishes pnc_text, which the
    #              downstream stages already read by default → nothing else changes.
    #   - PnC OFF: no pnc_text is regenerated, so the downstream stages read
    #              recovered_key directly (via downstream_recovered below).
    # When RecoverEntities is OFF, downstream_recovered is None and every key below
    # falls back to its original value — the pipeline is byte-for-byte unchanged.
    downstream_recovered = recovered_key if (recovered_key and not args.enable_pnc) else None
    base_text_key = downstream_recovered or args.text_key  # TN / ITN(no-TN) / Captioning / CS / QA
    if args.language_id_text_key:
        langid_text_key = args.language_id_text_key
    elif args.language_id_first:
        langid_text_key = recovered_key or "abbreviated_text"
    else:
        langid_text_key = downstream_recovered or (args.pnc_output_key if args.enable_pnc else "pnc_text")

    # ITN reads tn_raw (TN's spoken form) when TN is enabled, else the base text.
    itn_input_key = args.tn_output_key if args.enable_tn else base_text_key
    # Captioning/CodeSwitching/SpeechQA read TN's output (tn_raw) when TN is on, else the base text.
    post_tn_text_key = itn_input_key

    stages = [
        ALMManifestReader(manifest_path=args.input_manifest, output_dir=args.output_dir, fanout=False),
    ]

    # RecoverEntities runs FIRST, before every other LLM stage: it rewrites the
    # normalized transcription so its named entities regain the ground-truth
    # spelling/casing. Only the Dynamo-server (remote) variant is wired here.
    if args.enable_recover_entities:
        recover_entities_prompt = args.recover_entities_prompt_file or str(_RECOVER_ENTITIES_PROMPT)
        stages.append(
            RemoteRecoverEntitiesStage(
                model_id=args.model_id,
                prompt_file=recover_entities_prompt,
                ground_truth_key=args.recover_entities_ground_truth_key,
                normalized_key=args.recover_entities_normalized_key,
                output_text_key=args.recover_entities_output_key,
                tensor_parallel_size=args.tensor_parallel_size,
                max_output_tokens=args.recover_entities_max_output_tokens,
                max_model_len=args.recover_entities_max_model_len,
                max_num_seqs=args.max_num_seqs,
                gpu_memory_utilization=args.gpu_memory_utilization,
                kv_cache_dtype=args.kv_cache_dtype,
                num_workers_override=(
                    args.recover_entities_num_workers
                    if args.recover_entities_num_workers is not None
                    else args.num_workers
                ),
                batch_size=args.batch_size,
                **_resolve_stage_sampling(
                    args,
                    "recover_entities",
                    default_temperature=0.0,
                    default_top_p=1.0,
                ),
                **{
                    **remote_kwargs,
                    **(
                        {"max_concurrent_requests": args.recover_entities_max_concurrent_requests}
                        if args.recover_entities_max_concurrent_requests is not None and remote_kwargs
                        else {}
                    ),
                },
            )
        )
        logger.info(
            f"RecoverEntities stage enabled: ({args.recover_entities_ground_truth_key} + "
            f"{args.recover_entities_normalized_key}) → {args.recover_entities_output_key}"
        )


    def _make_llm_language_id_stage(source_lang_allowlist: frozenset[str] | None = None) -> object:
        return text_stage_cls(
            name="LanguageID",
            prompt_file=language_id_prompt,
            text_key=langid_text_key,
            output_text_key=language_id_output_key,
            source_lang_allowlist=source_lang_allowlist,
            enable_validation=False,
            **_resolve_stage_sampling(args, "language_id"),
            **shared_model_kwargs,
        )

    def _append_language_id_stages() -> None:
        """Append one backend-neutral LanguageID + verification group."""
        if args.language_id_backend == "llm":
            stages.append(_make_llm_language_id_stage())
        else:
            if args.language_id_backend == "config":
                selected_backends = set(language_id_backend_config.values())
                llm_languages = frozenset(
                    language for language, backend in language_id_backend_config.items() if backend == "llm"
                )
            else:
                selected_backends = {args.language_id_backend}
                llm_languages = frozenset()

            if "llm" in selected_backends:
                stages.append(_make_llm_language_id_stage(llm_languages))

            common_cpu_lid_kwargs = {
                "name": "LanguageID",
                "text_key": langid_text_key,
                "output_text_key": language_id_output_key,
                "backend_by_language": language_id_backend_config,
            }
            if "fasttext" in selected_backends:
                stages.append(
                    FastTextLanguageIdentificationStage(
                        model_path=args.fasttext_lid_model_path,
                        **common_cpu_lid_kwargs,
                    )
                )
            if "indiclid" in selected_backends:
                stages.append(
                    IndicLIDLanguageIdentificationStage(
                        model_path=args.indiclid_lid_model_path,
                        **common_cpu_lid_kwargs,
                    )
                )

        # Every predictor writes the same field, and exactly one verifier owns
        # language-related _skipme and additional_notes mutations.
        stages.append(LLMLanguageVerificationStage())
        logger.info(
            "LanguageID + LLMLanguageVerification stages enabled (backend={}, position={}): {} → {} "
            "→ keep every code-switch; flag single-language mismatches",
            args.language_id_backend,
            "before PnC" if args.language_id_first else "after PnC",
            langid_text_key,
            language_id_output_key,
        )

    if args.enable_language_id and args.language_id_first:
        _append_language_id_stages()
    

    if args.enable_pnc:
        # When RecoverEntities ran, PnC punctuates the entity-recovered text;
        # otherwise it reads the raw normalized text (abbreviated_text by default).
        pnc_input_key = recovered_key or ("abbreviated_text" if args.text_key == "pnc_text" else args.text_key)
        stages.append(
            text_stage_cls(
                name="PnCRestoration",
                prompt_file=pnc_prompt,
                language_rules=pnc_language_rules,
                text_key=pnc_input_key,
                output_text_key=args.pnc_output_key,
                **_resolve_stage_sampling(args, "pnc"),
                **shared_model_kwargs,
            )
        )
        # Copy pnc_text → tn_raw so downstream stages that read tn_raw (ITN,
        # Captioning, CodeSwitching, SpeechQA when TN is disabled) receive the
        # punctuated text rather than the raw abbreviated_text.
        stages.append(
            FieldCopyStage(
                name="PnC→tn_raw",
                src_key=args.pnc_output_key,
                dst_key=args.tn_output_key,
                ignore_skip_me=args.ignore_skip_me,
            )
        )
        logger.info(
            "PnC stage enabled: %s → %s (also copied to %s)",
            pnc_input_key, args.pnc_output_key, args.tn_output_key,
        )

    if args.enable_language_id and not args.language_id_first:
        _append_language_id_stages()
    
    # Stages that consume outputs produced inside FusedRemoteTextLLMStage are
    # collected here and appended immediately after the fused actor.
    post_fused_stages: list = []

    if args.enable_tn:
        _tn_stage = text_stage_cls(
            name="TextNormalization",
            prompt_file=tn_prompt,
            text_key=base_text_key,
            output_text_key=args.tn_output_key,
            enable_validation=not args.disable_tn_validation,
            validation_mode="tn",
            **_resolve_stage_sampling(args, "tn"),
            **shared_model_kwargs,
        )
        # TN runs serially (before the fused stage) so tn_raw is available to the downstream
        # fused sub-stages (Captioning/CodeSwitching/SpeechQA) and to post-fused ITN.
        stages.append(_tn_stage)
        logger.info(
            "TN stage enabled: %s → %s (validation=%s, mode=tn)",
            base_text_key,
            args.tn_output_key,
            not args.disable_tn_validation,
        )

    if args.enable_itn:
        _itn_stage = text_stage_cls(
            name="ITNRestoration",
            prompt_file=itn_prompt,
            text_key=itn_input_key,
            output_text_key=args.itn_output_key,
            **_resolve_stage_sampling(args, "itn"),
            **shared_model_kwargs,
        )
        # Fuse only when reading pnc_text; tn_raw input runs post-fused.
        if use_fusing and not args.enable_tn:
            fuseable_sub_stages.append(_itn_stage)
        elif use_fusing:
            post_fused_stages.append(_itn_stage)
        else:
            stages.append(_itn_stage)
        logger.info(f"ITN stage enabled: {itn_input_key} → {args.itn_output_key}")

    if args.enable_itn_no_disfluencies:
        if not args.enable_itn:
            logger.warning(
                "--enable_itn_no-disfluencies requires --enable_itn (needs itn_raw as input). Enabling ITN automatically."
            )
            _itn_auto_stage = text_stage_cls(
                name="ITNRestoration",
                prompt_file=itn_prompt,
                text_key=itn_input_key,
                output_text_key=args.itn_output_key,
                **_resolve_stage_sampling(args, "itn"),
                **shared_model_kwargs,
            )
            # Same placement rule as the ITN block above.
            if use_fusing and not args.enable_tn:
                fuseable_sub_stages.append(_itn_auto_stage)
            elif use_fusing:
                post_fused_stages.append(_itn_auto_stage)
            else:
                stages.append(_itn_auto_stage)

        _disfl_stage = text_stage_cls(
            name="DisfluencyRemoval",
            prompt_file=itn_no_disfl_prompt,
            text_key=args.itn_output_key,
            output_text_key=args.itn_no_disfluencies_output_key,
            max_deletion_ratio=0.5,
            **_resolve_stage_sampling(args, "itn_no_disfluencies"),
            **shared_model_kwargs,
        )
        # DisfluencyRemoval reads itn_raw from the fused stage — must follow it.
        if use_fusing:
            post_fused_stages.append(_disfl_stage)
        else:
            stages.append(_disfl_stage)
        logger.info(f"DisfluencyRemoval stage enabled: {args.itn_output_key} → {args.itn_no_disfluencies_output_key}")

    if args.enable_captioning:
        _captioning_stage = text_stage_cls(
            name="Captioning",
            prompt_file=captioning_prompt,
            text_key=post_tn_text_key,
            output_text_key=args.captioning_output_key,
            enable_validation=False,
            **_resolve_stage_sampling(args, "captioning"),
            **shared_model_kwargs,
        )
        if use_fusing:
            fuseable_sub_stages.append(_captioning_stage)
        else:
            stages.append(_captioning_stage)
        logger.info(f"Captioning stage enabled: {post_tn_text_key} → {args.captioning_output_key}")

    if args.enable_context_asr:
        # Honor an explicit --context_asr_text_key. Only when it is left at its
        # "pnc_text" default AND recovery is feeding downstream directly (PnC off)
        # do we route it to the recovered text.
        context_asr_text_key = args.context_asr_text_key
        if downstream_recovered and args.context_asr_text_key == "pnc_text":
            context_asr_text_key = downstream_recovered
        # This stage runs as its own Ray Data actor with its own vLLM engine, so it
        # uses its own engine kwargs — note context_asr_max_model_len (8192), larger
        # than the global --max_model_len (2048) used by the lightweight stages.
        stages.append(
            ctx_stage_cls(
                model_id=args.model_id,
                prompt_file=context_asr_prompt,
                text_key=context_asr_text_key,
                source_lang_key="source_lang",
                high_quality_key=args.context_asr_high_quality_key,
                output_key=args.context_asr_output_key,
                tensor_parallel_size=args.tensor_parallel_size,
                max_output_tokens=args.context_asr_max_output_tokens,
                max_model_len=args.context_asr_max_model_len,
                max_num_seqs=args.max_num_seqs,
                gpu_memory_utilization=args.gpu_memory_utilization,
                kv_cache_dtype=args.kv_cache_dtype,
                num_workers_override=args.context_asr_num_workers
                if args.context_asr_num_workers is not None
                else args.num_workers,
                batch_size=args.batch_size,
                **_resolve_stage_sampling(
                    args,
                    "context_asr",
                    default_temperature=0.1,
                    default_top_p=0.95,
                ),
                **{
                    **remote_kwargs,
                    **(
                        {"max_concurrent_requests": args.context_asr_max_concurrent_requests}
                        if args.context_asr_max_concurrent_requests is not None and remote_kwargs
                        else {}
                    ),
                },
            )
        )
        if args.enable_acoustic_distractor:
            if not args.phoneme_vocab_path:
                msg = "--enable_acoustic_distractor requires --phoneme_vocab_path."
                raise ValueError(msg)
            stages.append(
                AcousticDistractorStage(
                    context_key=args.context_asr_output_key,
                    source_lang_key="source_lang",
                    language=args.phoneme_vocab_language,
                    phoneme_vocab_path=args.phoneme_vocab_path,
                    max_acoustic_distractors=args.max_acoustic_distractors,
                    max_total_distractors=args.max_total_distractors,
                    per_entity_top_k=args.acoustic_per_entity_top_k,
                    min_npd=args.min_npd,
                    max_npd=args.max_npd,
                    num_workers_override=args.acoustic_num_workers,
                )
            )
            logger.info(
                f"AcousticDistractor stage enabled: vocab={args.phoneme_vocab_path} "
                f"(language={args.phoneme_vocab_language or 'per-sample'}, "
                f"max_acoustic={args.max_acoustic_distractors}, total_cap={args.max_total_distractors})"
            )
        stages.append(
            ContextualASRPromptVariantStage(
                context_key=args.context_asr_output_key,
                source_lang_key="source_lang",
                seed=args.context_asr_seed,
                partial_keep_lo=args.context_asr_partial_keep_lo,
                partial_keep_hi=args.context_asr_partial_keep_hi,
            )
        )
        logger.info(
            f"Context ASR stages enabled: {context_asr_text_key} → {args.context_asr_output_key} "
            f"(extraction + 5 prompt variants)"
        )
        if args.context_asr_max_model_len < _CONTEXT_ASR_MIN_MAX_MODEL_LEN:
            logger.warning(
                f"--context_asr_max_model_len={args.context_asr_max_model_len} may be too small for context "
                f"ASR extraction. Consider 8192 or higher when --enable_context_asr is used."
            )
    if args.enable_code_switching:
        _code_switching_stage = text_stage_cls(
            name="CodeSwitching",
            prompt_file=code_switching_prompt,
            text_key=post_tn_text_key,
            output_text_key=args.code_switching_output_key,
            enable_validation=False,
            **_resolve_stage_sampling(args, "code_switching"),
            **shared_model_kwargs,
        )
        if use_fusing:
            fuseable_sub_stages.append(_code_switching_stage)
        else:
            stages.append(_code_switching_stage)
        logger.info(f"CodeSwitching stage enabled: {post_tn_text_key} → {args.code_switching_output_key}")

    if args.enable_speech_qa:
        _speech_qa_stage = text_stage_cls(
            name="SpeechQA",
            prompt_file=speech_qa_prompt,
            text_key=post_tn_text_key,
            output_text_key=args.speech_qa_output_key,
            enable_validation=False,
            **_resolve_stage_sampling(args, "speech_qa"),
            **shared_model_kwargs,
        )
        if use_fusing:
            fuseable_sub_stages.append(_speech_qa_stage)
        else:
            stages.append(_speech_qa_stage)
        logger.info(f"SpeechQA stage enabled: {post_tn_text_key} → {args.speech_qa_output_key}")

    # Fused stage assembly — one actor fires all collected sub-stage prompts in parallel.
    if use_fusing:
        if fuseable_sub_stages:
            stages.append(
                FusedRemoteTextLLMStage(
                    sub_stages=fuseable_sub_stages,
                    inference_base_url=remote_base_url,
                    inference_api_key=args.inference_api_key,
                    served_model_name=remote_model_name,
                    max_concurrent_requests=args.inference_max_concurrent_requests,
                    request_timeout=args.inference_request_timeout,
                    batch_size=args.batch_size,
                )
            )
            logger.info(
                "FusedRemoteTextLLMStage assembled: %s sub-stages firing in parallel",
                [s.name for s in fuseable_sub_stages],
            )
        # Post-fused stages depend on outputs written by the fused actor (for
        # example, itn_raw → DisfluencyRemoval). Append them even when there
        # are no fuseable sub-stages so TN → ITN does not disappear merely
        # because --fuse_stages was also requested.
        stages.extend(post_fused_stages)

    if args.enable_instruction_packer:
        stages.append(
            InstructionPackerStage(
                output_key=args.instructions_output_key,
                pnc_key=args.pnc_output_key,
                tn_key=args.tn_output_key,
                itn_key=args.itn_output_key,
                itn_no_disfluencies_key=args.itn_no_disfluencies_output_key,
                captioning_key=args.captioning_output_key,
                code_switched_key=args.code_switching_output_key,
                speech_qa_key=args.speech_qa_output_key,
                transcription_target_key=args.pnc_output_key,
                translations_key=args.instruction_packer_translations_key,
                min_translation_score=args.instruction_packer_min_translation_score,
                seed=args.instruction_packer_seed,
            )
        )
        logger.info(f"InstructionPacker stage enabled → {args.instructions_output_key}")

    stages.append(ShardedManifestWriterStage(output_dir=args.output_dir))

    from nemo_curator.backends.ray_data import RayDataExecutor

    executor = RayDataExecutor(ignore_head_node=bool(remote_base_url))

    # Process the input in shard batches: one pipeline.run() per batch, server reused.
    # stages[0] is the reader (rebuilt per batch); everything after it is the shared tail.
    shard_batches = _resolve_shard_batches(args.input_manifest, args.shards_per_batch)
    tail_stages = stages[1:]

    logger.info(
        f"Running text pipeline: {len(stages)} stages, mode={args.execution_mode}, "
        f"{len(shard_batches)} batch(es) of up to {args.shards_per_batch} shard(s)"
    )
    failed_batches = 0
    try:
        for _bi, _batch in enumerate(shard_batches):
            _reader = ALMManifestReader(manifest_path=_batch, output_dir=args.output_dir, fanout=False)
            _n = len(_batch) if isinstance(_batch, list) else "all"
            logger.info(f"--- Batch {_bi + 1}/{len(shard_batches)}: {_n} shard(s) ---")
            try:
                Pipeline(name="text_post_processing", stages=[_reader, *tail_stages]).run(executor=executor)
            except Exception:
                failed_batches += 1
                logger.exception(
                    f"Batch {_bi + 1}/{len(shard_batches)} failed; continuing (incomplete shards resume on re-run)."
                )
                continue
            # Batch drained cleanly → finalize .done deterministically for its shards.
            if isinstance(_batch, list):
                _finalize_done_markers(_batch, args.output_dir)
    finally:
        if inference_server is not None:
            inference_server.stop()
        if ray_client is not None:
            ray_client.stop()
    if failed_batches:
        logger.warning(
            f"Text pipeline finished with {failed_batches}/{len(shard_batches)} batch(es) failed "
            "(their incomplete shards will resume on re-run)."
        )
    logger.info("Text pipeline complete.")


if __name__ == "__main__":
    main()
