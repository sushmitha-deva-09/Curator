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

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from nemo_curator.stages.audio.inference.speaker_diarization.sortformer import (
    InferenceSortformerStage,
    _parse_sortformer_segments,
    _write_rttm,
)
from nemo_curator.tasks import AudioTask


class TestParseSortformerSegments:
    def test_parses_string_segments(self) -> None:
        raw = ["0.00 2.70 speaker_0", "0.80 13.60 speaker_1"]
        out = _parse_sortformer_segments(raw)
        assert len(out) == 2
        assert out[0] == {"start": 0.0, "end": 2.7, "speaker": "speaker_0"}
        assert out[1] == {"start": 0.8, "end": 13.6, "speaker": "speaker_1"}

    def test_parses_object_segments(self) -> None:
        seg1 = SimpleNamespace(start=1.0, end=3.5, speaker="speaker_0")
        seg2 = SimpleNamespace(start=4.0, end=7.2, speaker="speaker_1")
        out = _parse_sortformer_segments([seg1, seg2])
        assert out[0] == {"start": 1.0, "end": 3.5, "speaker": "speaker_0"}
        assert out[1] == {"start": 4.0, "end": 7.2, "speaker": "speaker_1"}

    def test_parses_object_with_label_attr(self) -> None:
        seg = SimpleNamespace(start=0.5, end=1.5, label="spk_2")
        out = _parse_sortformer_segments([seg])
        assert out[0]["speaker"] == "spk_2"

    def test_parses_tuple_segments(self) -> None:
        raw = [(0.0, 2.0, "speaker_0"), (3.0, 5.0, "speaker_1")]
        out = _parse_sortformer_segments(raw)
        assert len(out) == 2
        assert out[0] == {"start": 0.0, "end": 2.0, "speaker": "speaker_0"}

    def test_empty_list_returns_empty(self) -> None:
        assert _parse_sortformer_segments([]) == []

    def test_unrecognised_format_warns(self) -> None:
        out = _parse_sortformer_segments([42])
        assert out == []


class TestWriteRttm:
    def test_writes_rttm_file(self, tmp_path: Path) -> None:
        segments = [
            {"start": 0.0, "end": 2.5, "speaker": "speaker_0"},
            {"start": 3.0, "end": 5.0, "speaker": "speaker_1"},
        ]
        _write_rttm(segments, "test_session", str(tmp_path))
        rttm_path = tmp_path / "test_session.rttm"
        assert rttm_path.exists()
        lines = rttm_path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert lines[0].startswith("SPEAKER test_session 1 0.000 2.500")
        assert "speaker_0" in lines[0]
        assert lines[1].startswith("SPEAKER test_session 1 3.000 2.000")
        assert "speaker_1" in lines[1]


class TestInferenceSortformerStage:
    def test_setup_on_node_pre_caches_model(self) -> None:
        stage = InferenceSortformerStage(model_name="nvidia/diar_streaming_sortformer_4spk-v2")
        with patch("nemo_curator.stages.audio.inference.speaker_diarization.sortformer.snapshot_download") as mock_dl:
            stage.setup_on_node()
            mock_dl.assert_called_once_with(repo_id="nvidia/diar_streaming_sortformer_4spk-v2", cache_dir=None)

    def test_setup_on_node_skips_for_local_path(self) -> None:
        stage = InferenceSortformerStage(model_path="/local/model.nemo")
        with patch("nemo_curator.stages.audio.inference.speaker_diarization.sortformer.snapshot_download") as mock_dl:
            stage.setup_on_node()
            mock_dl.assert_not_called()

    def test_setup_skips_when_model_provided(self) -> None:
        mock_model = MagicMock()
        mock_model.sortformer_modules = MagicMock()
        stage = InferenceSortformerStage(diar_model=mock_model)
        stage.setup()
        assert mock_model.sortformer_modules.chunk_len == 340

    def test_streaming_config_applied(self) -> None:
        mock_model = MagicMock()
        mock_model.sortformer_modules = MagicMock()
        stage = InferenceSortformerStage(
            diar_model=mock_model,
            chunk_len=124,
            chunk_right_context=1,
            fifo_len=124,
            spkcache_update_period=124,
            spkcache_len=200,
        )
        stage.setup()
        sm = mock_model.sortformer_modules
        assert sm.chunk_len == 124
        assert sm.chunk_right_context == 1
        assert sm.fifo_len == 124
        assert sm.spkcache_update_period == 124
        assert sm.spkcache_len == 200

    def _make_mock_model(self, fake_segments_per_file: list[list[str]]) -> MagicMock:
        mock_model = MagicMock()
        mock_model.sortformer_modules = MagicMock()
        mock_model.diarize.return_value = fake_segments_per_file
        return mock_model

    def test_process_produces_tagging_compatible_output(self) -> None:
        """Segments written to 'segments' key with prefixed speaker IDs and non-speaker gaps."""
        fake_output = [
            ["0.00 2.70 speaker_0", "3.00 13.60 speaker_1"],
        ]
        mock_model = self._make_mock_model(fake_output)
        stage = InferenceSortformerStage(diar_model=mock_model)

        task = AudioTask(
            data={
                "resampled_audio_filepath": "/test/audio1.wav",
                "audio_item_id": "test_id",
                "duration": 20.0,
            },
        )
        result = stage.process(task)

        assert isinstance(result, AudioTask)
        assert "segments" in result.data
        assert "overlap_segments" in result.data
        assert result.data["overlap_segments"] == []

        segments = result.data["segments"]
        speaker_segments = [s for s in segments if s["speaker"] != "no-speaker"]
        no_speaker_segments = [s for s in segments if s["speaker"] == "no-speaker"]

        assert len(speaker_segments) == 2
        assert speaker_segments[0]["speaker"] == "test_id_speaker_0"
        assert speaker_segments[1]["speaker"] == "test_id_speaker_1"
        assert len(no_speaker_segments) > 0

        assert result.task_id.endswith("_sortformer")
        mock_model.diarize.assert_called_once_with(
            audio=["/test/audio1.wav"],
            batch_size=1,
        )

    def test_process_filters_short_segments(self) -> None:
        """Segments shorter than min_length are excluded."""
        fake_output = [["0.00 0.30 speaker_0", "1.00 5.00 speaker_1"]]
        mock_model = self._make_mock_model(fake_output)
        stage = InferenceSortformerStage(diar_model=mock_model, min_length=0.5)

        task = AudioTask(
            data={"resampled_audio_filepath": "/test/audio.wav", "audio_item_id": "x", "duration": 10.0},
        )
        result = stage.process(task)
        speaker_segments = [s for s in result.data["segments"] if s["speaker"] != "no-speaker"]
        assert len(speaker_segments) == 1
        assert speaker_segments[0]["start"] == 1.0

    def test_process_writes_rttm(self, tmp_path: Path) -> None:
        fake_output = [["0.00 2.50 speaker_0"]]
        mock_model = self._make_mock_model(fake_output)
        stage = InferenceSortformerStage(
            diar_model=mock_model,
            rttm_out_dir=str(tmp_path),
        )

        task = AudioTask(
            data={"resampled_audio_filepath": "/test/my_audio.wav", "duration": 5.0},
        )
        stage.process(task)

        rttm_file = tmp_path / "my_audio.rttm"
        assert rttm_file.exists()
        content = rttm_file.read_text()
        assert "SPEAKER my_audio" in content

    def test_process_preserves_existing_data(self) -> None:
        fake_output = [["0.00 1.00 speaker_0"]]
        mock_model = self._make_mock_model(fake_output)
        stage = InferenceSortformerStage(diar_model=mock_model)

        task = AudioTask(
            data={
                "resampled_audio_filepath": "/test/audio1.wav",
                "extra_key": "extra_value",
                "duration": 5.0,
            },
        )
        result = stage.process(task)
        assert result.data["extra_key"] == "extra_value"
        assert "segments" in result.data

    def test_process_uses_session_name_from_data(self, tmp_path: Path) -> None:
        fake_output = [["0.00 1.00 speaker_0"]]
        mock_model = self._make_mock_model(fake_output)
        stage = InferenceSortformerStage(
            diar_model=mock_model,
            rttm_out_dir=str(tmp_path),
        )

        task = AudioTask(
            data={
                "resampled_audio_filepath": "/test/audio1.wav",
                "session_name": "sess_42",
                "duration": 5.0,
            },
        )
        stage.process(task)
        assert (tmp_path / "sess_42.rttm").exists()

    def test_process_custom_filepath_key(self) -> None:
        """Stage can be configured to read from a different audio path key."""
        fake_output = [["0.00 3.00 speaker_0"]]
        mock_model = self._make_mock_model(fake_output)
        stage = InferenceSortformerStage(
            diar_model=mock_model,
            audio_filepath_key="audio_filepath",
        )

        task = AudioTask(
            data={"audio_filepath": "/test/raw.wav", "audio_item_id": "x", "duration": 5.0},
        )
        result = stage.process(task)
        assert "segments" in result.data
        mock_model.diarize.assert_called_once_with(audio=["/test/raw.wav"], batch_size=1)

    def test_non_speaker_gaps_cover_full_duration(self) -> None:
        """Non-speaker segments fill gaps between speaker turns up to audio duration."""
        fake_output = [["5.00 10.00 speaker_0", "15.00 20.00 speaker_1"]]
        mock_model = self._make_mock_model(fake_output)
        stage = InferenceSortformerStage(diar_model=mock_model)

        task = AudioTask(
            data={
                "resampled_audio_filepath": "/test/audio.wav",
                "audio_item_id": "x",
                "duration": 30.0,
            },
        )
        result = stage.process(task)
        segments = result.data["segments"]
        no_speaker = [s for s in segments if s["speaker"] == "no-speaker"]
        assert len(no_speaker) >= 2
        starts = sorted(s["start"] for s in no_speaker)
        assert starts[0] == 0.0
