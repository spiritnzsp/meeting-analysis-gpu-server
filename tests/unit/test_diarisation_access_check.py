"""A revoked token was invisible for weeks. Now it is checked at startup.

Three things were true at once before this:

1. A **missing** token logged a config warning — fine as far as it went.
2. A token that was **present but revoked** logged nothing at all. This server
   ran in exactly that state, and nothing had to notice: models already in the
   HuggingFace cache load with no network call, so the credential was only ever
   exercised on a cache miss. It was found because a human suspected it.
3. The banner printed `PyAnnote model: <configured value>` regardless, stating
   a *setting* in the grammar of a *capability* — the same defect as logging a
   codec that will not be used.

The check **fails open by design**, and that is the part most likely to be
"fixed" into a bug later: if HuggingFace cannot be reached the server starts
anyway and says so. Cached models still load, whisper and video encoding are
unaffected, and turning a network blip into a dead server would be far worse
than the silence being removed.
"""
import pytest

from gpu_server.diarisation_access import (
    BAD_TOKEN,
    NO_ACCESS,
    NO_TOKEN,
    OK,
    UNKNOWN,
    AccessCheck,
    check_diarisation_access,
)

MODEL = "pyannote/speaker-diarization-community-1"


class _Api:
    """Stand-in for HfApi. `whoami_error`/`access_error` make it misbehave."""

    def __init__(self, whoami_error=None, access_error=None, name="sean"):
        self._whoami_error = whoami_error
        self._access_error = access_error
        self._name = name

    def whoami(self, token=None):
        if self._whoami_error:
            raise self._whoami_error
        return {"name": self._name}

    def auth_check(self, repo, token=None):
        if self._access_error:
            raise self._access_error


@pytest.fixture
def hub(monkeypatch):
    def _install(api):
        monkeypatch.setattr("huggingface_hub.HfApi", lambda: api)
    return _install


class TestTheThreeDefiniteAnswers:

    def test_no_token_says_so_and_says_where_to_put_one(self, hub):
        hub(_Api())
        result = check_diarisation_access("", MODEL)

        assert result.state == NO_TOKEN
        assert "config.yaml" in result.detail
        assert not result.diarisation_available

    def test_a_revoked_token_is_named_as_revoked(self, hub):
        """The case that was invisible. It must not be reported as a licence
        problem — a revoked token and an unaccepted licence both arrive as
        GatedRepoError 401, and advising the wrong one wastes the reader's
        time on the wrong page."""
        hub(_Api(whoami_error=RuntimeError("Invalid user token.")))
        result = check_diarisation_access("hf_dead", MODEL)

        assert result.state == BAD_TOKEN
        assert "NOT VALID" in result.detail
        assert "revoked" in result.detail
        assert "licence" not in result.detail.lower()

    def test_a_revoked_token_explains_why_nothing_noticed(self, hub):
        """The cache is the reason this hid for weeks, and the message has to
        say so or the next person draws the wrong conclusion from 'but it was
        working yesterday'."""
        hub(_Api(whoami_error=RuntimeError("Invalid user token.")))
        result = check_diarisation_access("hf_dead", MODEL)

        assert "cached" in result.detail

    def test_a_valid_token_without_the_licence_names_the_account(self, hub):
        """'Accept the licence' is useless without saying on which account —
        the server uses its own token, never a client's."""
        hub(_Api(access_error=RuntimeError("403 Forbidden"), name="server-bot"))
        result = check_diarisation_access("hf_good", MODEL)

        assert result.state == NO_ACCESS
        assert "server-bot" in result.detail
        assert MODEL in result.detail

    def test_all_three_are_reported_as_definite(self):
        for state in (NO_TOKEN, BAD_TOKEN, NO_ACCESS):
            assert AccessCheck(state, "").is_definite_failure


class TestItFailsOpen:
    """The design decision most at risk of being 'corrected' into a bug."""

    def test_an_unreachable_hub_is_unknown_not_unavailable(self, hub):
        hub(_Api(whoami_error=None, access_error=None))

        def _explode():
            raise OSError("Network is unreachable")

        import huggingface_hub
        original = huggingface_hub.HfApi
        huggingface_hub.HfApi = lambda: (_ for _ in ()).throw(
            OSError("Network is unreachable")
        )
        try:
            result = check_diarisation_access("hf_good", MODEL)
        finally:
            huggingface_hub.HfApi = original

        assert result.state == UNKNOWN
        assert not result.is_definite_failure, (
            "an unreachable Hub was reported as a definite failure — that would "
            "make the banner shout about a working server on a bad network"
        )

    def test_unknown_is_not_availability_either(self):
        """It must not resolve to 'fine' any more than to 'broken'. That is why
        the banner prints a state rather than a boolean."""
        assert not AccessCheck(UNKNOWN, "").diarisation_available

    def test_a_hanging_hub_does_not_hang_startup(self, hub, monkeypatch):
        """The check sits directly in the startup path. A wedged connection
        must abandon the probe, not hold the server down."""
        import time

        class _Slow:
            def whoami(self, token=None):
                time.sleep(30)

        hub(_Slow())
        monkeypatch.setattr(
            "gpu_server.diarisation_access._CHECK_TIMEOUT_SECONDS", 0.5
        )

        started = time.monotonic()
        result = check_diarisation_access("hf_good", MODEL)
        elapsed = time.monotonic() - started

        assert result.state == UNKNOWN
        assert elapsed < 10, f"startup was held for {elapsed:.0f}s"
        assert "did not respond" in result.detail


class TestTheHappyPath:
    """Negative controls. A check that never says OK would make the banner
    permanently alarming, which is the same as saying nothing."""

    def test_a_working_setup_is_reported_available(self, hub):
        hub(_Api(name="spiritnz"))
        result = check_diarisation_access("hf_good", MODEL)

        assert result.state == OK
        assert result.diarisation_available
        assert result.account == "spiritnz"

    def test_it_is_not_a_definite_failure(self, hub):
        hub(_Api())
        assert not check_diarisation_access("hf_good", MODEL).is_definite_failure


def test_the_banner_reports_the_check_not_the_config():
    """Guards the wiring. The whole point is that the banner stops asserting a
    capability it never verified."""
    import inspect

    from gpu_server import __main__ as main_module

    source = inspect.getsource(main_module)
    assert "check_diarisation_access" in source
    assert "Diarisation: AVAILABLE" in source
    assert "Diarisation: UNAVAILABLE" in source
    assert "Diarisation: UNVERIFIED" in source
