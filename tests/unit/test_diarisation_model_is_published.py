"""The server ran a different diarisation model from the client, and said nothing.

The server pinned `speaker-diarization-3.1` while the client had moved to
`speaker-diarization-community-1`. The same recording diarised in the two places
therefore went through **two different models** while the user was told about
one — so every GPU-versus-local comparison was confounded. E3 saw 11–12 speakers
on the server against a constrained 5 locally, and we attributed the whole
difference to the speaker range when part of it may have been the model.

Two changes, and both are needed:

* the pin moves to `community-1`, so the two ends agree;
* the server **publishes** which model it is running in the auth handshake, so a
  client can check rather than assume. Aligning the pins today is a one-off;
  publishing is what stops the next divergence being silent.

The client half is already in place: it reads this field, compares it against
its own `PRIMARY_PIPELINE`, and on a mismatch proceeds with a loud warning and a
recorded provenance field rather than refusing — falling back to a slow CPU
costs the user more than the mismatch does.
"""
import inspect

import pytest

from gpu_server import server as server_module
from gpu_server.config import PyAnnoteConfig

#: What the client pins, in `teams_scanner/models/gated_models.py`
#: (`PRIMARY_PIPELINE`). Written out rather than imported because these are
#: separate repositories with no dependency between them — which is exactly why
#: they drifted, and why the handshake now carries the value instead of both
#: sides trusting a constant they cannot see.
CLIENT_PRIMARY_PIPELINE = "pyannote/speaker-diarization-community-1"


class TestThePinMatchesTheClient:

    def test_the_default_model_is_the_clients_pipeline(self):
        assert PyAnnoteConfig().model == CLIENT_PRIMARY_PIPELINE

    def test_the_retired_3_1_pin_is_gone(self):
        """The specific value that caused the divergence."""
        assert "3.1" not in PyAnnoteConfig().model

    def test_the_example_config_agrees_with_the_default(self):
        """A shipped example that disagrees with the code teaches the wrong
        value to everyone who copies it — which is most people."""
        from pathlib import Path

        example = (Path(__file__).resolve().parents[2] / "config.example.yaml")
        text = example.read_text(encoding="utf-8")
        assert f'model: "{CLIENT_PRIMARY_PIPELINE}"' in text


class TestTheHandshakeSaysWhatIsRunning:

    def _auth_payloads(self):
        """Every auth_ok dict the server can send."""
        source = inspect.getsource(server_module)
        return source

    def test_the_model_is_published(self):
        assert '"diarisation_model": self.config.pyannote.model' in self._auth_payloads()

    def test_it_is_published_on_BOTH_auth_paths(self):
        """There are two: auth-disabled accepts every connection and returns
        early, and the authenticated path returns after checking credentials.
        Publishing on only one would make the field appear or vanish depending
        on a server setting unrelated to models — and this server currently
        runs with auth disabled, so the early path is the one in use.
        """
        assert self._auth_payloads().count(
            '"diarisation_model": self.config.pyannote.model'
        ) == 2

    def test_it_reports_the_configured_value_not_a_constant(self):
        """A hardcoded string here would keep saying community-1 after someone
        changed the config, which is a worse lie than saying nothing."""
        source = self._auth_payloads()
        assert '"diarisation_model": "pyannote' not in source


class TestAGatedRefusalSaysWhatToDoAboutIt:
    """A licence that has not been accepted arrives as a generic exception
    mentioning 401 or 403. On the client side that exact shape once produced an
    error blaming HuggingFace while HuggingFace was working perfectly."""

    def _matcher(self):
        from gpu_server.processors.pyannote_processor import (
            _looks_like_gated_access,
        )
        return _looks_like_gated_access

    @pytest.mark.parametrize("message", [
        "401 Client Error. Cannot access gated repo for url ...",
        "403 Forbidden",
        "Access to model pyannote/speaker-diarization-community-1 is restricted",
        "Your request to access this repo is awaiting a review",
    ])
    def test_a_licence_problem_is_recognised(self, message):
        assert self._matcher()(Exception(message))

    @pytest.mark.parametrize("message", [
        "Connection reset by peer",
        "CUDA out of memory",
        "No space left on device",
    ])
    def test_a_real_fault_is_not_mistaken_for_one(self, message):
        """Negative control, and the reason the match is deliberately narrow: a
        false positive sends someone to accept a licence they already accepted,
        and away from the actual problem."""
        assert not self._matcher()(Exception(message))

    def test_the_advice_names_the_server_s_own_account(self):
        """The trap: the server uses its OWN token and never a client's, so
        'accept the licence' is useless without saying on which account."""
        source = inspect.getsource(
            __import__(
                "gpu_server.processors.pyannote_processor",
                fromlist=["pyannote_processor"],
            )
        )
        assert "THIS SERVER's token" in source
        assert "never uses a client's token" in source
