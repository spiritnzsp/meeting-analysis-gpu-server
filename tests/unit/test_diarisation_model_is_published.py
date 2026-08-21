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

    def test_the_advice_is_about_the_server_s_own_token(self, monkeypatch):
        """The trap: the server uses its OWN token and never a client's, so
        'accept the licence' is useless without saying on which account.

        Asserted on the MESSAGE rather than on the source text. An earlier
        version of this test matched literal strings in the module, and it broke
        the moment the message was improved — testing the wording of a comment
        rather than the behaviour it describes.
        """
        class _Api:
            def whoami(self, token=None):
                return {"name": "sean-server-bot"}

        monkeypatch.setattr("huggingface_hub.HfApi", _Api)

        from gpu_server.processors.pyannote_processor import (
            _diagnose_access_failure,
        )
        message = _diagnose_access_failure("hf_good", "pyannote/some-model")

        assert "own token" in message
        assert "any other account does nothing" in message


class TestItSaysWHICHAccessProblemItIs:
    """A revoked token and an unaccepted licence both arrive as
    `GatedRepoError: 401` — verified 2026-08-21 against a real revoked token.

    The first version of this message simply said "accept the licence", which
    would have sent someone to accept a licence they already held while the
    actual problem was a dead token. So the token is checked directly rather
    than inferred from the exception text.

    This one bit for real: Sean's server token had been removed from his
    HuggingFace account, and nothing noticed because the models already in the
    cache kept working. It would have surfaced as a "licence" error at the exact
    moment of switching to a new model — the least helpful possible time.
    """

    def _diagnose(self):
        from gpu_server.processors.pyannote_processor import (
            _diagnose_access_failure,
        )
        return _diagnose_access_failure

    def test_no_token_says_no_token(self):
        message = self._diagnose()("", "pyannote/speaker-diarization-community-1")
        assert "NO HuggingFace token" in message
        assert "config.yaml" in message

    def test_an_invalid_token_does_not_advise_accepting_a_licence(self, monkeypatch):
        """The whole point. Advising a licence here wastes the reader's time on
        the wrong account page while the real cause goes unmentioned."""
        import gpu_server.processors.pyannote_processor as mod

        class _Api:
            def whoami(self, token=None):
                raise RuntimeError("Invalid user token.")

        monkeypatch.setattr("huggingface_hub.HfApi", _Api)
        message = self._diagnose()("hf_dead", "pyannote/speaker-diarization-community-1")

        assert "NOT VALID" in message
        assert "will NOT help" in message
        assert "settings/tokens" in message

    def test_an_invalid_token_warns_that_the_cache_hides_it(self, monkeypatch):
        """Why it stayed invisible: already-downloaded models keep working, so
        the failure waits for the next fetch."""
        import gpu_server.processors.pyannote_processor as mod

        class _Api:
            def whoami(self, token=None):
                raise RuntimeError("Invalid user token.")

        monkeypatch.setattr("huggingface_hub.HfApi", _Api)
        message = self._diagnose()("hf_dead", "m")
        assert "cache" in message

    def test_a_valid_token_names_the_account_to_accept_on(self, monkeypatch):
        """Negative control, and the thing that makes the licence advice usable:
        'accept the licence' is useless without saying on which account."""
        class _Api:
            def whoami(self, token=None):
                return {"name": "sean-server-bot"}

        monkeypatch.setattr("huggingface_hub.HfApi", _Api)
        message = self._diagnose()("hf_good", "pyannote/speaker-diarization-community-1")

        assert "LICENCE problem" in message
        assert "sean-server-bot" in message
        assert "NOT VALID" not in message
