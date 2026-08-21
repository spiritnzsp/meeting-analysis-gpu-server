"""Can this server actually diarise? Asked at startup, not discovered mid-job.

Three things were true at once before this existed:

1. A **missing** token logged a config warning — fine as far as it went.
2. A token that was **present but revoked** logged nothing at all. Sean's server
   ran for weeks in exactly that state; models already in the HuggingFace cache
   load with no network call, so nothing had to notice.
3. The startup banner printed ``PyAnnote model: <configured value>`` regardless,
   which states the *configuration* in the grammar of a *capability*. That is
   the same defect as logging a codec that will not be used: an intention
   reported as an outcome.

So the banner now reports what was **checked**, not what was configured.

**It fails open, deliberately.** If HuggingFace cannot be reached, the server
starts and says the check was inconclusive — it does not refuse to run. The
models may well be cached, whisper and video encoding are unaffected, and
turning a network blip into a dead server would be a far worse failure than the
one being fixed. Only a definite answer is reported as definite.
"""
import logging
import threading
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

#: How long the check may take before the server gives up and starts anyway.
#: Two calls to the Hub on a healthy link are well under a second; this is the
#: bound on a wedged one, and it is deliberately short because it sits directly
#: in the startup path.
_CHECK_TIMEOUT_SECONDS = 8.0

OK = "ok"
NO_TOKEN = "no_token"
BAD_TOKEN = "bad_token"
NO_ACCESS = "no_access"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class AccessCheck:
    """What we know about this server's ability to load its diarisation model."""

    state: str
    detail: str
    account: Optional[str] = None

    @property
    def diarisation_available(self) -> bool:
        """True only when it was CHECKED and passed.

        ``UNKNOWN`` is not availability. It is also not unavailability — which
        is why the banner prints the state rather than a boolean.
        """
        return self.state == OK

    @property
    def is_definite_failure(self) -> bool:
        """A problem we are sure about, and that a human must act on."""
        return self.state in (NO_TOKEN, BAD_TOKEN, NO_ACCESS)


def check_diarisation_access(token: str, model: str) -> AccessCheck:
    """Verify the token and the model's accessibility, with a hard time bound.

    Runs the Hub calls on a daemon thread so a wedged connection cannot hold up
    startup: the thread is abandoned rather than waited on. That matters because
    this is the one check that must never be the reason a server fails to come
    up — the whole point is to make a *silent* problem visible, and a check that
    hangs the process would trade one bad outcome for a worse one.
    """
    if not token:
        return AccessCheck(
            NO_TOKEN,
            "no HuggingFace token configured — diarisation offload is "
            "unavailable. Set pyannote.huggingface_token in config.yaml.",
        )

    result = {}

    def _probe():
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            try:
                who = api.whoami(token=token)
            except Exception as token_error:      # noqa: BLE001
                result["value"] = AccessCheck(
                    BAD_TOKEN,
                    "the configured HuggingFace token is NOT VALID "
                    f"({type(token_error).__name__}) — it has most likely been "
                    "revoked or rotated. Diarisation will fail the moment a "
                    "model is not already cached. Issue a new token at "
                    "https://huggingface.co/settings/tokens.",
                )
                return

            account = who.get("name")
            try:
                api.auth_check(model, token=token)
            except Exception as access_error:     # noqa: BLE001
                result["value"] = AccessCheck(
                    NO_ACCESS,
                    f"the token is valid (account '{account}') but it cannot "
                    f"access {model} ({type(access_error).__name__}). Accept "
                    f"the licence at https://huggingface.co/{model} while "
                    f"signed in as '{account}'.",
                    account=account,
                )
                return

            result["value"] = AccessCheck(
                OK, f"verified as '{account}'", account=account
            )
        except Exception as e:                    # noqa: BLE001 - never fatal
            result["value"] = AccessCheck(
                UNKNOWN, f"could not be checked ({type(e).__name__}: {e})"
            )

    worker = threading.Thread(target=_probe, daemon=True,
                              name="diarisation-access-check")
    worker.start()
    worker.join(timeout=_CHECK_TIMEOUT_SECONDS)

    if worker.is_alive():
        return AccessCheck(
            UNKNOWN,
            f"could not be checked — HuggingFace did not respond within "
            f"{_CHECK_TIMEOUT_SECONDS:.0f}s. The server is starting anyway; "
            "cached models will still load.",
        )

    return result.get(
        "value", AccessCheck(UNKNOWN, "could not be checked (no result)")
    )
