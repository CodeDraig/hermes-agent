# Agent Runtime Decomposition Follow-ups

The field-only `AgentState` decomposition was reviewed on 2026-08-18. Its two
ACT-NOW lifecycle defects were repaired in the working tree. The defects below
remain deliberately deferred; this change documents their production behavior
and repair boundaries without restoring methods on `AgentState` or changing
tests.

## Explicit hard stops do not reach current agents

`agent.interrupt_compat.request_hard_interrupt()` still discovers
`hard_interrupt` or `interrupt` as methods. Field-only agents expose neither,
so Ctrl-C, API stop, gateway shutdown, timeout, and subagent-cancellation paths
that use this helper do not perform the responsibility-owned hard interrupt.

Containment: force-exit or restart a stuck owner process. Repair boundary:
route current-agent callers directly through `agent.interruption.hard_interrupt`
and retain compatibility discovery only at an explicitly identified external
agent boundary, if one still exists.

## Gateway and cron inactivity observation is disabled

Gateway and cron watchdog paths still guard
`agent.status_output.get_activity_summary(agent)` with
`hasattr(agent, "get_activity_summary")`. That guard is always false for
`AgentState`, so gateway stall scans return no activity and cron idle time stays
at zero.

Containment: restart a wedged gateway or cron worker. Repair boundary: remove
method feature detection and call the status owner directly at every current
agent watchdog and status site.

## Model switching and one-turn restoration retain method calls

Gateway cached-session model switches call `cached_agent.switch_model(...)`,
and CLI one-turn restoration requires `_restore_primary_runtime` or
`switch_model` methods before invoking the provider owner. Cached gateway
switches therefore fail, while a one-turn CLI runtime can remain active on
later turns.

Containment: reset or recreate the agent session before switching models, and
avoid one-turn selection. Repair boundary: use
`agent.provider_runtime.switch_model` and `restore_primary_runtime` directly.

## Some steer and redirect frontends reject field-only agents

CLI `/steer`, ACP active-turn redirect and `/steer`, the API run-steer endpoint,
and Kanban comment injection still require `steer` or `redirect` methods. Their
module functions are available, but the guards prevent reaching them. Depending
on the frontend, guidance is queued for the next turn, rejected, or never read.

Containment: send guidance as a later queued turn. Repair boundary: retain only
real state-capability checks, then call `agent.interruption.steer` or `redirect`
directly.

## Detached delegation progress is sampled through a removed method

`tools.delegate_tool` calls `child.get_activity_summary()` in the detached
batch progress callback. Every field-only child falls into the exception path
and contributes a constant `None` token, allowing the stale monitor to
force-finalize a healthy long-running batch.

Containment: use synchronous delegation for long-running work. Repair boundary:
sample each child with `agent.status_output.get_activity_summary(child)`.

## Responsibility modules still contain forwarding facades

The deleted `agent_runtime_helpers` surface has been replaced in part by thin
forwarders in `agent.lifecycle`, `agent.status_output`,
`agent.provider_runtime`, `agent.message_protocol`, and `agent.stream_runtime`.
Production callers still target several of those aliases instead of the module
that owns the behavior.

Containment: none is required for current runtime behavior. Repair boundary:
update production callers to import actual owners and delete forwarders that do
not enforce lifecycle, policy, translation, or observability semantics.

## Deferred verification boundary

Test correction and migration remain a separate phase. Do not restore runtime
methods, forwarding aliases, or compatibility shims merely to satisfy stale
patch or import locations. Future repairs should validate the current
state-first module interfaces directly.
