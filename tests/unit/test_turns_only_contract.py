"""The contract the client's re-diarisation relies on (client T-55).

Re-diarisation must keep the user's transcript words. The client therefore
asks for ``transcribe=False, diarize=True`` and assigns the returned speaker
TURNS onto its own existing segments. This server stays dumb about that: it
does not learn about transcripts. What it must guarantee, and what these pin:

* the turns come back in ``diarization_segments``, with the same ``Person-N``
  labels the embeddings carry;
* Whisper is not run, loaded or asked for - no wasted GPU time, and no second
  rendering of the audio for anyone to adopt by mistake;
* the serialised result carries the turns (the client parses the JSON).

Fakes only - no GPU, no real models.
"""
import json
from types import SimpleNamespace

import pytest

from gpu_server.config import Config
from gpu_server.protocol import DiarizationSegment, SpeakerEmbedding
from gpu_server.worker import GPUWorker

from tests.unit.test_gpu_worker import FakeArbiter, FakeWS, _request


class _NoWhisper:
    def __getattr__(self, name):
        raise AssertionError(f"Whisper was touched ({name}) for a turns-only request")


class _Pyannote:
    turns = [DiarizationSegment(0.0, 8.5, "Person-1"), DiarizationSegment(8.5, 12.0, "Person-2")]

    async def diarize(self, **_k):
        return [DiarizationSegment(t.start, t.end, t.speaker) for t in self.turns]

    async def extract_embeddings(self, **_k):
        return [SpeakerEmbedding(speaker_label=t.speaker, meeting_id="r", segment_start=t.start,
                                 segment_duration=t.end - t.start, embedding=[0.1] * 4)
                for t in self.turns]

    def align_transcript_with_diarization(self, *_a):
        raise AssertionError("nothing to align: no transcript was made")


@pytest.fixture
def worker(monkeypatch):
    w = GPUWorker(Config(), queue=SimpleNamespace(), arbiter=FakeArbiter())
    real = (w._whisper_handle.processor, w._pyannote_handle.processor)
    monkeypatch.setattr(GPUWorker, "_whisper", property(lambda self: _NoWhisper()))
    monkeypatch.setattr(GPUWorker, "_pyannote", property(lambda self: _Pyannote()))
    yield w
    for p in real:
        p.shutdown()


async def test_turns_only_returns_turns_and_never_touches_whisper(worker):
    queued = SimpleNamespace(request=_request(diarize=True, extract_embeddings=True), websocket=FakeWS())

    result = await worker._process_request(queued)

    assert result.success, result.error_message
    assert result.transcript_segments == [] and result.full_text == ""
    assert [(t.start, t.end, t.speaker) for t in result.diarization_segments] == \
           [(0.0, 8.5, "Person-1"), (8.5, 12.0, "Person-2")]
    # The embeddings are keyed by the same labels as the turns.
    assert {e.speaker_label for e in result.speaker_embeddings} == {"Person-1", "Person-2"}
    assert result.warnings == []


async def test_the_serialised_result_carries_the_turns(worker):
    queued = SimpleNamespace(request=_request(diarize=True), websocket=FakeWS())
    data = json.loads((await worker._process_request(queued)).to_json())
    assert data["diarization_segments"] == [
        {"start": 0.0, "end": 8.5, "speaker": "Person-1"},
        {"start": 8.5, "end": 12.0, "speaker": "Person-2"},
    ]


def test_a_turns_only_request_pins_no_whisper(worker):
    assert "whisper" not in worker._workload_need(_request(diarize=True, extract_embeddings=True)).required_models
