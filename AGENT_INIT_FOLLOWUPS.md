# Agent Initialization Follow-ups

The `AIAgent` initialization decomposition preserves the observed runtime phase
order, but its review on 2026-08-18 found three unfinished test tasks. These are
documented rather than repaired in the decomposition change so their current
failure evidence and intended scope remain explicit.

## Reset-aware fallback tests use the removed dict shape

`tests/run_agent/test_reset_aware_primary_restore.py` defines `FB` as one
provider dictionary and passes it directly as `fallback_providers` in eight
tests. The current constructor contract accepts only `list[dict] | None`, so
each test raises `TypeError` during initialization and never exercises runtime
restoration.

Reproduction:

```text
.venv/bin/pytest -q \
  tests/run_agent/test_reset_aware_primary_restore.py::TestResetAwareRestoreGate::test_stays_on_fallback_until_reset

TypeError: fallback_providers must be a list of provider entries or None
```

Repair boundary: make the shared `FB` fixture a list containing the existing
entry, then rerun all eight `TestResetAwareRestoreGate` tests. Do not restore
single-dict parsing in production.

## Context-length warning tests patch the old logger owner

`tests/run_agent/test_invalid_context_length_warning.py` patches
`run_agent.logger` at three sites. The extracted warning is emitted by
`agent.init_context.logger`, so the invalid-value warning occurs but the mock
records no call.

Reproduction:

```text
.venv/bin/pytest -q \
  tests/run_agent/test_invalid_context_length_warning.py::test_string_k_suffix_context_length_warns

assert 0 == 1
```

Repair boundary: patch `agent.init_context.logger` at all three sites and rerun
the file. The production warning path does not need a behavior change.

## Direct phase coverage is incomplete

`tests/agent/test_init_phases.py` currently proves only orchestration order and
rejection of a non-list `fallback_providers` value. It does not directly isolate
the phase contracts requested for:

- constructor and execution state;
- provider routing, client selection, credential-pool ordering, and final
  runtime snapshots;
- tool snapshot ordering and context-engine augmentation;
- session and persistence ownership;
- memory fail-open behavior, compression settings, context-engine fallback,
  Ollama sizing, and startup notices.

Repair boundary: add direct phase tests around the owning initialization
modules. Retain the existing end-to-end initialization suite as integration
coverage rather than using it as a substitute for phase isolation.

## Review evidence

The changed-test surface reported `1408 passed, 2 skipped, 37 failed`. Nine
failures are accounted for by the two stale test migrations above. The other
failures were not established as decomposition regressions; observed causes
included the absent optional `anthropic` package and gateway localization or
shared-state expectations. They require separate isolation before attribution.

A focused persistence and timeout probe passed 18 tests. Mechanical comparison
against the pre-decomposition `init_agent` found no concrete runtime
initialization regression, but the missing direct tests leave phase-local error
handling and snapshot behavior under-specified.
