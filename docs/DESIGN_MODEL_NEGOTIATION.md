# Model negotiation — negotiate and refuse, never mutate

**Status:** ✅ **IMPLEMENTED 2026-08-21**, both halves — see "What landed" at the
end. Decided in principle 2026-08-10 and banked during the build-F test pass so
the reasoning would survive to the triage; it did.

⚠️ **One step is NOT code and is outstanding:** the running server's
`config.yaml` still pins `speaker-diarization-3.1`, and it overrides the code
default. See "What is still to do".

**The one-line rule: the server never accepts executable content from a client.**

## What prompted it

Test F2 (2026-08-10, `req=1d4f8e00-b9ad-43a1-bcad-cb1c0b277cc7`) proved work
reaches the server, and the journal incidentally showed:

```
Loading PyAnnote pipeline: pyannote/speaker-diarization-3.1
```

The client has moved to `pyannote/speaker-diarization-community-1`. It is the
model the HF wizard verifies, the one `GATED_MODELS` lists, and the installer's
BEFORE YOU START page is *required* to name it and to **not** mention 3.1. So the
same recording diarised locally and diarised here goes through two different
models, and the user is told about one of them.

That is a real problem — it confounds every comparison between the two paths,
including E3's "11 speakers on the GPU versus a constrained 5 locally", which we
read as the range working and which may be partly the model.

## The proposal that was rejected, and why

> *"If a client connects and has a higher version model, the server should update
> its models to match — retrieve the newer models directly from the connected
> client."*

Rejected on security, and Sean reached the same conclusion independently
(ISO 27001 framing) before it was written down.

1. **Unsafe deserialisation → remote code execution.** A pyannote/torch
   checkpoint loads through `torch.load` → pickle, which *executes* code at load
   time. Accepting weights from a client means any client can run arbitrary code
   as the service user, on a box with a GPU, a HuggingFace token and configured
   shared filesystem paths. This is not a hardening gap to mitigate; it is an RCE
   primitive by design.
2. **It inverts the direction of trust.** Today the server pulls from one known
   registry using its own credentials — authenticated, pinned, auditable. Under
   client-push every client becomes a supplier of executable content. That is a
   supply-chain compromise vector, and "how do you establish provenance and
   integrity of the models running in production?" has no answer under it.
   Under registry-pull the answer is: pinned identifier, known registry, server's
   own credentials, recorded in config.
3. **Auth is currently off.** The service starts with `Auth enabled: False`, so
   anything that can reach the port is an authenticated client. Model upload on
   top of that is immediate exposure, not theoretical.
4. **It routes around the licence grant.** pyannote gated models are accepted
   **per HuggingFace account**. The server operator must accept the terms on the
   *server's* account. Weights arriving from a client bypass an acceptance the
   operator is required to hold.
5. **It does not even do what was wanted.** pyannote model ids are not ordered —
   `community-1` is not machine-comparably "newer" than `3.1`, so "has a higher
   version model" is not a computable predicate. And models here are resident and
   VRAM-arbitrated; swapping per client would thrash and make results
   irreproducible across concurrent clients.

## The design

1. **The client declares the model id it expects** in the request options.
2. **The server compares** it against its configured model and, on a mismatch,
   **says so explicitly** — same shape as `85a69c2`, which made an unhonourable
   codec request a loud event instead of a silent substitution. Loud disagreement
   is the whole pattern here.
3. **The client decides** what to do: proceed and record the difference, or fall
   back to local. The server never silently swaps and never mutates itself to
   satisfy a client.
4. **Record the model id in the result metadata**, so any transcript is
   attributable to the model that produced it. Nothing does this today, which is
   why the 3.1/community-1 divergence went unnoticed until a journal was read by
   hand.
5. **Updating the server is an operator action.** Change the pin, accept the
   terms on the server's HF account, restart. The server pulls from the registry,
   as it does now.

## Fixing the immediate mismatch

Small, and independent of the negotiation work: the server already runs
**pyannote.audio 4.0.1**, so community-1 is reachable.

- `gpu_server/config.py:243` — `model: str = "pyannote/speaker-diarization-3.1"`
- `config.example.yaml:54` — same value, plus the licence URL at `:51`
- Accept the community-1 terms on the **server's** HuggingFace account
- Restart the service

⚠️ **Not during a test pass.** Client results are bound to current server
behaviour and E3's GPU-half comparison would silently change meaning. Server
config changes wait for the same gap as a client build.


---

## What landed, 2026-08-21

**Server (this repo).**

* The pin moves to `pyannote/speaker-diarization-community-1` in both
  `PyAnnoteConfig` and `config.example.yaml`. community-1 is **self-contained**
  under pyannote 4.x — its segmentation, embedding and PLDA components live in
  subfolders of that one repo, unlike the 3.x pipelines which pulled a
  separately gated segmentation model — so this needs terms accepted on exactly
  one repo, not several.
* **The server publishes `diarisation_model` in the `auth_ok` handshake**, on
  both auth paths. Aligning the pins today is a one-off; publishing is what
  stops the *next* divergence being silent, which is the part worth having.
* A gated-model refusal now says what to do about it. It arrives as a generic
  exception mentioning 401/403, and on the client side that exact shape once
  produced an error blaming HuggingFace while HuggingFace was working perfectly.
  The message names the model, says the licence must be accepted **on the
  account that issued this server's token** — the server never uses a client's —
  and gives the URL. The matcher is deliberately narrow: a false positive sends
  someone to accept a licence they already hold and away from the real fault.

**Client (`Meeting_analysis`).** Reads the field, compares it against its own
`PRIMARY_PIPELINE`, and on a mismatch **proceeds with a loud warning** plus a
recorded `diarisation_model` provenance field in the transcript, rather than
refusing and falling back to local.

Sean's decision, and the reasoning is the part to keep: *"impact on the user who
is actually trying to get work done — diarisation on a slow CPU could cause a lot
of grief. Better to have this fast with the warning, then the user can redo
diarisation at a later time if necessary."* **Refusing is only "safe" if you
ignore what refusing costs**: on the no-GPU laptop this product targets, falling
back turns a two-minute job into a forty-five-minute one, to avoid a mismatch the
user can simply re-run.

Which is exactly why the provenance field is not optional. "Proceed anyway"
without recording *what produced the result* would recreate the confounded
evidence this whole item is about.

A server that omits the field reads as **"did not say"**, never as disagreement —
every server built before today omits it.

## What is still to do

1. **Accept the licence for `pyannote/speaker-diarization-community-1` on the
   HuggingFace account that issued the server's token.** Do this FIRST. The
   server uses its own token, so accepting it on the client's account achieves
   nothing.
2. **Then** change `pyannote.model` in the live `config.yaml` — it currently
   pins `speaker-diarization-3.1` and **overrides the code default**, so nothing
   above changes what the running server loads until it is edited.
3. `systemctl --user restart gpu-server`, and confirm the journal says
   `Loading PyAnnote pipeline: pyannote/speaker-diarization-community-1`.

Until then the client will report *"Diarised on the GPU server using
`pyannote/speaker-diarization-3.1`, but this project expects
`…community-1`"* — which is the honest state, and already better than the silence
it replaces.
