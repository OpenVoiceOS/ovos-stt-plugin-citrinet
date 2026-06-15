"""End-to-end listener tests for ovos-stt-plugin-citrinet.

Tests the CitrinetSTT plugin both via direct transcription and through the
ovoscope MiniListener pipeline (audio → STT → recognizer_loop:utterance).

The Citrinet ONNX model is fetched once from HuggingFace Hub and cached; no
network access is required after the first run.

Fixture: test/fixtures/command.wav — 16 kHz mono, "what time is it in london"
"""

import os
from pathlib import Path

import pytest

import ovoscope
import ovos_stt_plugin_citrinet

FIXTURE = Path(__file__).parent / "fixtures" / "command.wav"


@pytest.fixture(scope="module")
def stt():
    """Real CitrinetSTT instance using the default English model."""
    from ovos_stt_plugin_citrinet import CitrinetSTT
    return CitrinetSTT({"lang": "en"})


@pytest.fixture(scope="module")
def audio_data():
    """AudioData loaded from the command.wav fixture."""
    from ovos_plugin_manager.utils.audio import AudioFile
    with AudioFile(str(FIXTURE)) as source:
        return source.read()


class TestDirectTranscription:
    """Direct STT.execute() call without the listener pipeline."""

    def test_transcript_non_empty(self, stt, audio_data):
        result = stt.execute(audio_data, language="en")
        assert result is not None
        assert isinstance(result, str)
        assert len(result.strip()) > 0

    def test_transcript_content(self, stt, audio_data):
        result = stt.execute(audio_data, language="en")
        lower = result.lower()
        # The model transcribes "what time is it in london" cleanly.
        # Accept either keyword as a loose pass criterion in case of minor
        # acoustic variation in CI.
        assert "time" in lower or "london" in lower, (
            f"Expected 'time' or 'london' in transcript, got: {result!r}"
        )


class TestListenerPipeline:
    """Full pipeline: audio file → MiniListener → recognizer_loop:utterance."""

    def test_utterance_emitted(self, stt):
        from ovoscope.listener import get_mini_listener

        listener = get_mini_listener(stt_instance=stt)
        try:
            msgs = listener.listen(str(FIXTURE), language="en-us")
        finally:
            listener.shutdown()

        utterance_msgs = [
            m for m in msgs if m.msg_type == "recognizer_loop:utterance"
        ]
        assert utterance_msgs, (
            f"No recognizer_loop:utterance emitted. All messages: "
            f"{[m.msg_type for m in msgs]}"
        )

    def test_utterance_non_empty(self, stt):
        from ovoscope.listener import get_mini_listener

        listener = get_mini_listener(stt_instance=stt)
        try:
            msgs = listener.listen(str(FIXTURE), language="en-us")
        finally:
            listener.shutdown()

        utterance_msgs = [
            m for m in msgs if m.msg_type == "recognizer_loop:utterance"
        ]
        assert utterance_msgs, "No utterance message emitted"

        utterances = utterance_msgs[0].data.get("utterances", [])
        assert utterances, "utterances list is empty"
        assert utterances[0].strip(), "utterance string is blank"

    def test_utterance_content(self, stt):
        from ovoscope.listener import get_mini_listener

        listener = get_mini_listener(stt_instance=stt)
        try:
            msgs = listener.listen(str(FIXTURE), language="en-us")
        finally:
            listener.shutdown()

        utterance_msgs = [
            m for m in msgs if m.msg_type == "recognizer_loop:utterance"
        ]
        assert utterance_msgs
        text = utterance_msgs[0].data["utterances"][0].lower()
        assert "time" in text or "london" in text, (
            f"Expected 'time' or 'london' in utterance, got: {text!r}"
        )
