---
sidebar_position: 12
title: "Kanban (Multi-Agent Board)"
description: "Durable SQLite-backed task board for coordinating multiple Hermes profiles"
---

# Kanban — Multi-Agent Profile Collaboration

> **Want a walkthrough?** Read the [Kanban tutorial](./kanban-tutorial) — four user stories covering solo development, fleet farming, role pipelines, and circuit breakers. This page is the reference; the tutorial is the narrative.

Hermes Kanban is a durable task board, shared across all your Hermes profiles, that lets multiple named agents collaborate on work without fragile in-process subagent swarms. Every task is a row in `~/.hermes/kanban.db`; every handoff is a row anyone can read and write; every worker is a full OS process with its own identity.

### Two surfaces: the model talks through tools, you talk through the CLI

The board has two front doors, both backed by the same `~/.hermes/kanban.db`:

- **Agents drive the board through a dedicated `kanban_*` toolset** — `kanban_show`, `kanban_list`, `kanban_complete`, `kanban_request_review`, `kanban_request_changes`, `kanban_block`, `kanban_heartbeat`, `kanban_comment`, `kanban_attach`, `kanban_attach_url`, `kanban_attachments`, `kanban_create`, `kanban_link`, `kanban_unblock`. The dispatcher spawns each worker with these tools already in its schema; orchestrator profiles can also enable the `kanban` toolset explicitly. The model reads and routes tasks by calling tools directly, *not* by shelling out to `hermes kanban`. See [How workers interact with the board](#how-workers-interact-with-the-board) below.
- **You (and scripts, and cron) drive the board through `hermes kanban …`** on the CLI or `/kanban …` as a slash command. These are for humans and automation — the places without a tool-calling model behind them.

Both surfaces route through the same `kanban_db` layer, so reads see a consistent view and writes can't drift. The rest of this page shows CLI examples because they're easy to copy-paste, but every CLI verb has a tool-call equivalent the model uses.

This is the shape that covers the workloads `delegate_task` can't:

- **Research triage** — parallel researchers + analyst + writer, human-in-the-loop.
- **Scheduled ops** — recurring daily briefs that build a journal over weeks.
- **Digital twins** — persistent named assistants (`inbox-triage`, `ops-review`) that accumulate memory over time.
- **Engineering pipelines** — decompose → implement in parallel worktrees → review → iterate → PR.
- **Fleet work** — one specialist managing N subjects (50 social accounts, 12 monitored services).

For the full design rationale, comparative analysis against Cline Kanban / Paperclip / NanoClaw / Google Gemini Enterprise, and the eight canonical collaboration patterns, see `docs/hermes-kanban-v1-spec.pdf` in the repository.

## Kanban vs. `delegate_task`

They look similar; they are not the same primitive.

| | `delegate_task` | Kanban |
|---|---|---|
| Shape | RPC call (fork → join) | Durable message queue + state machine |
| Parent | Blocks until child returns | Fire-and-forget after `create` |
| Child identity | Anonymous subagent | Named profile with persistent memory |
| Resumability | None — failed = failed | Block → unblock → re-run; crash → reclaim |
| Human in the loop | Not supported | Comment / unblock at any point |
| Agents per task | One call = one subagent | N agents over task's life (retry, review, follow-up) |
| Audit trail | Lost on context compression | Durable rows in SQLite forever |
| Coordination | Hierarchical (caller → callee) | Peer — any profile reads/writes any task |

**One-sentence distinction:** `delegate_task` is a function call; Kanban is a work queue where every handoff is a row any profile (or human) can see and edit.

**Use `delegate_task` when** the parent agent needs a short reasoning answer before continuing, no humans involved, result goes back into the parent's context.

**Use Kanban when** work crosses agent boundaries, needs to survive restarts, might need human input, might be picked up by a different role, or needs to be discoverable after the fact.

They coexist: a kanban worker may call `delegate_task` internally during its run.

## Core concepts

- **Board** — a standalone queue of tasks with its own SQLite DB, workspaces
  directory, and dispatcher loop. A single install can have many boards
  (e.g. one per project, repo, or domain); see [Boards (multi-project)](#boards-multi-project)
  below. Single-project users stay on the `default` board and never see the
  word "board" outside this docs section.
- **Task** — a row with title, optional body, one assignee (a profile name), status (`triage | todo | ready | running | blocked | review | done | archived`), optional tenant namespace, optional idempotency key (dedup for retried automation).
- **Link** — `task_links` row recording a parent → child dependency. The dispatcher promotes `todo → ready` when all parents are `done`.
- **Comment** — the inter-agent protocol. Agents and humans append comments; when a worker is (re-)spawned it reads the full comment thread as part of its context.
- **Workspace** — the directory a worker operates in. Three kinds:
  - `scratch` (default) — fresh tmp dir under `~/.hermes/kanban/workspaces/<id>/` (or `~/.hermes/kanban/boards/<slug>/workspaces/<id>/` on non-default boards). **Deleted when the task completes** — scratch is ephemeral by design. Files explicitly declared through `kanban_complete(artifacts=[...])` are copied into durable per-task attachment storage before cleanup; existing deliverable paths in legacy completion summaries receive the same treatment. Other scratch files are removed. A missing declared scratch artifact keeps the task in-flight so the worker can correct the path and retry. Use `worktree:` or `dir:<path>` when the whole workspace should remain available. The first time a scratch workspace is created on an install, the dispatcher logs a warning and emits a `tip_scratch_workspace` event on the task (visible via `hermes kanban show <id>`).
  - `dir:<path>` — an existing shared directory (Obsidian vault, mail ops dir, per-account folder). **Must be an absolute path.** Relative paths like `dir:../tenants/foo/` are rejected at dispatch because they'd resolve against whatever CWD the dispatcher happens to be in, which is ambiguous and a confused-deputy escape vector. The path is otherwise trusted — it's your box, your filesystem, the worker runs with your uid. This is the trusted-local-user threat model; kanban is single-host by design. **Preserved on completion.**
  - `worktree` — a git worktree under `.worktrees/<id>/` for coding tasks. Use `worktree:<path>` to pin the exact target path. Worker-side `git worktree add` creates it, using `--branch` when provided. **Preserved on completion.**
- **Dispatcher** — a long-lived loop that, every N seconds (default 60): reclaims stale claims, reclaims crashed workers (PID gone but TTL not yet expired), promotes ready tasks, atomically claims, spawns assigned profiles. Runs **inside the gateway** by default (`kanban.dispatch_in_gateway: true`). One dispatcher sweeps all boards per tick; workers are spawned with `HERMES_KANBAN_BOARD` pinned so they can't see other boards. After `kanban.failure_limit` consecutive spawn failures on the same task (default: 2) the dispatcher auto-blocks it with the last error as the reason — prevents thrashing on tasks whose profile doesn't exist, workspace can't mount, etc.
- **Tenant** — optional string namespace *within* a board. One specialist fleet can serve multiple businesses (`--tenant business-a`) with data isolation by workspace path and memory key prefix. Tenants are a soft filter; boards are the hard isolation boundary.

## Boards (multi-project)

Boards let you separate unrelated streams of work — one per project, repo,
or domain — into isolated queues. A new install has exactly one board
called `default` (DB at `~/.hermes/kanban.db` for back-compat). Users who
only want one stream of work never need to know about boards; the feature
is opt-in.

Per-board isolation is absolute:

- Separate SQLite DB per board (`~/.hermes/kanban/boards/<slug>/kanban.db`).
- Separate `workspaces/` and `logs/` directories.
- Workers spawned for a task see **only** their board's tasks — the
  dispatcher sets `HERMES_KANBAN_BOARD` in the child env and every
  `kanban_*` tool the worker has access to reads it.
- Linking tasks across boards is not allowed (keeps the schema simple; if
  you really need cross-project refs, use free-text mentions and look
  them up by id manually).

### Managing boards from the CLI

```bash
# See what's on disk. Fresh installs show only "default".
hermes kanban boards list

# Create a new board.
hermes kanban boards create atm10-server \
    --name "ATM10 Server" \
    --description "Minecraft modded server ops" \
    --icon 🎮 \
    --switch                   # optional: make it the active board

# Operate on a specific board without switching.
hermes kanban --board atm10-server list
hermes kanban --board atm10-server create "Restart ATM server" --assignee ops

# Change which board is "current" for subsequent calls.
hermes kanban boards switch atm10-server
hermes kanban boards show             # who's active right now?

# Rename the display name (the slug is immutable — it's the directory name).
hermes kanban boards rename atm10-server "ATM10 (Prod)"

# Archive (default) — moves the board's dir to boards/_archived/<slug>-<ts>/.
# Recoverable by moving the dir back.
hermes kanban boards rm atm10-server

# Hard delete — `rm -rf` the board dir. No recovery.
hermes kanban boards rm atm10-server --delete
```

Board resolution order (highest precedence first):

1. Explicit `--board <slug>` on the CLI call.
2. `HERMES_KANBAN_BOARD` env var (set by the dispatcher when spawning a
   worker, so workers can't see other boards).
3. `~/.hermes/kanban/current` — the slug persisted by `hermes kanban
   boards switch`.
4. `default`.

Slugs are validated: lowercase alphanumerics + hyphens + underscores, 1-64
chars, must start with alphanumeric. Uppercase input is auto-downcased.
Anything else (slashes, spaces, dots, `..`) is rejected at the CLI layer
so path-traversal tricks can't name a board.


### Gateway-embedded dispatcher (default)

The dispatcher runs inside the gateway process. Nothing to install, no
separate service to manage — if the gateway is up, ready tasks get picked
up on the next tick (60s by default).

```yaml
# config.yaml
kanban:
  dispatch_in_gateway: true        # default
  dispatch_interval_seconds: 60    # default
  review_dispatch: true            # default: spawn the assigned profile with
                                   # the bundled sdlc-review skill. Set false
                                   # for human-only review boards.
```

Override the config flag at runtime via `HERMES_KANBAN_DISPATCH_IN_GATEWAY=0`
for debugging. Standard gateway supervision applies: run `hermes gateway
start` directly, or wire the gateway up as a systemd user unit (see the
gateway docs). Without a running gateway, `ready` tasks stay where they are
until one comes up — `hermes kanban create` warns about this at creation
time.

Running `hermes kanban daemon` as a separate process is **deprecated**;
use the gateway. If you truly cannot run the gateway (headless host
policy forbids long-lived services, etc.) a `--force` escape hatch keeps
the old standalone daemon alive for one release cycle, but running both
a gateway-embedded dispatcher AND a standalone daemon against the same
`kanban.db` causes claim races and is not supported.

### Idempotent create (for automation / webhooks)

```bash
# First call creates the task. Any subsequent call with the same key
# returns the existing task id instead of duplicating.
hermes kanban create "nightly ops review" \
    --assignee ops \
    --idempotency-key "nightly-ops-$(date -u +%Y-%m-%d)" \
    --json
```

### Bulk CLI verbs

All the lifecycle verbs accept multiple ids so you can clean up a batch
in one command:

```bash
hermes kanban complete t_abc t_def t_hij --result "batch wrap"
hermes kanban archive  t_abc t_def t_hij
hermes kanban unblock  t_abc t_def
hermes kanban block    t_abc "need input" --ids t_def t_hij
```

:::note Where an unblocked task lands
`unblock` restores the safe source phase: **`review`** for reviewer-origin work
whose parents are complete, **`ready`** for implementation work whose parents
are complete, or **`todo`** while any parent remains open. A `todo` task keeps
its source-phase provenance and returns to `review` or `ready` automatically
when the dependency gate clears. `unblock` never routes directly to `triage`.

If you unblock a task and it later shows up in **`triage`**, the unblock is not
what put it there. A subsequent *re-block for the same reason* did: after a task
is blocked → unblocked → re-blocked for the same cause `BLOCK_RECURRENCE_LIMIT`
times (default `2`), the unblock-loop breaker stops sending it back to `blocked`
— where a cron would just keep unblocking it — and routes it to `triage` for a
human decision. This is a deterministic DB guard, not an LLM judgment call, and
a task's body text cannot opt out of it: the recurrence counter deliberately
survives each unblock (it resets only on a successful `complete`). To keep an
unblocked task in the work pool, resolve *why it keeps re-blocking* (unfinished
parent, missing input, unmet capability) before unblocking, or raise
`BLOCK_RECURRENCE_LIMIT` if the loop is expected.
:::

## How workers interact with the board

**Workers do not shell out to `hermes kanban`.** When the dispatcher spawns a worker it sets `HERMES_KANBAN_TASK=t_abcd` in the child's env, and that env var flips on a dedicated **kanban toolset** in the model's schema. The same toolset is also available to orchestrator profiles that enable `kanban` in their toolsets config. These tools read and mutate the board directly via the Python `kanban_db` layer, same as the CLI does. A running worker calls these like any other tool; it never sees or needs the `hermes kanban` CLI.

| Tool | Purpose | Required params |
|---|---|---|
| `kanban_show` | Read the current task (title, body, prior attempts, parent handoffs, comments, full pre-formatted `worker_context`). Defaults to the env's task id. | — |
| `kanban_list` | List task summaries with filters for `assignee`, `status`, `tenant`, archived visibility, and limit. Intended for orchestrators discovering board work. | — |
| `kanban_complete` | Finish with `summary` + `metadata` structured handoff. | at least one of `summary` / `result` |
| `kanban_request_review` | Start same-card review with a durable `summary`, optional `metadata`, and optional reviewer profile. The task moves to `review`; this is not a block. | `summary` |
| `kanban_request_changes` | Reviewer verdict from an active review run. Closes that run, reapplies parent gating, and routes the task to its original implementer without block-loop accounting. | `reason` |
| `kanban_block` | Stop work and route by why: `kind=dependency` (waits in `todo`, auto-resumes), `needs_input`/`capability`/`transient` (surface to a human). Repeated same-kind re-blocks auto-escalate to `triage`. | `reason` |
| `kanban_heartbeat` | Signal liveness during long operations. Pure side-effect. | — |
| `kanban_comment` | Append a durable note to the task thread. | `task_id`, `body` |
| `kanban_attach` | Attach a file to a task by passing its bytes inline (base64); stored under the task's attachments dir (25 MB cap). | file bytes + name |
| `kanban_attach_url` | Attach a file to a task by URL. | `url` |
| `kanban_attachments` | List a task's attachments. | — |
| `kanban_create` | (Orchestrators) fan out into child tasks with an `assignee`, optional `parents`, `skills`, etc. | `title`, `assignee` |
| `kanban_link` | (Orchestrators) add a `parent_id → child_id` dependency edge after the fact. | `parent_id`, `child_id` |
| `kanban_unblock` | (Orchestrators) restore a blocked task to its source phase (`review` or `ready`), or `todo` while a parent remains open. | `task_id` |

A typical worker turn looks like:

```
# Model's tool calls, in order:
kanban_show()                                     # no args — uses HERMES_KANBAN_TASK
# (model reads the returned worker_context, does the work via terminal/file tools)
kanban_heartbeat(note="halfway through — 4 of 8 files transformed")
# (more work)
kanban_complete(
    summary="migrated limiter.py to token-bucket; added 14 tests, all pass",
    metadata={"changed_files": ["limiter.py", "tests/test_limiter.py"], "tests_run": 14},
)
```

An **orchestrator** worker fans out instead:

```
kanban_show()
kanban_create(
    title="research ICP funding 2024-2026",
    assignee="researcher-a",
    body="focus on seed + series A, North America, AI-adjacent",
)
# → returns {"task_id": "t_r1", ...}
kanban_create(title="research ICP funding — EU angle", assignee="researcher-b", body="…")
# → returns {"task_id": "t_r2", ...}
kanban_create(
    title="synthesize findings into launch brief",
    assignee="writer",
    parents=["t_r1", "t_r2"],                     # promotes to ready when both complete
    body="one-pager, 300 words, neutral tone",
)
kanban_complete(summary="decomposed into 2 research tasks + 1 writer; linked dependencies")
```

The "(Orchestrators)" tools — `kanban_list`, `kanban_create`, `kanban_link`, `kanban_unblock`, and `kanban_comment` on foreign tasks — are available through the same toolset; the convention (encoded in the auto-injected kanban guidance) is that worker profiles don't fan out or route unrelated work, and orchestrator profiles don't execute implementation work. Dispatcher-spawned workers are still task-scoped for destructive lifecycle operations and cannot mutate unrelated tasks.

### Why tools instead of shelling to `hermes kanban`

Three reasons:

1. **Backend portability.** Workers whose terminal tool points at a remote backend (Docker / Modal / Singularity / SSH) would run `hermes kanban complete` *inside* the container, where `hermes` isn't installed and `~/.hermes/kanban.db` isn't mounted. The kanban tools run in the agent's own Python process and always reach `~/.hermes/kanban.db` regardless of terminal backend.
2. **No shell-quoting fragility.** Passing `--metadata '{"files": [...]}'` through shlex + argparse is a latent footgun. Structured tool args skip it entirely.
3. **Better errors.** Tool results are structured JSON the model can reason about, not stderr strings it has to parse.

**Zero schema footprint on normal sessions.** A regular `hermes chat` session has zero `kanban_*` tools in its schema unless the active profile explicitly enables the `kanban` toolset for orchestrator work. Dispatcher-spawned task workers get task-scoped tools because `HERMES_KANBAN_TASK` is set; orchestrator profiles get the broader routing surface through config. No tool bloat for users who never touch kanban.

The auto-injected kanban guidance teaches the model which tool to call when and in what order.

### Recommended handoff evidence

`kanban_complete(summary=..., metadata={...})` is intentionally flexible:
the summary is the human-readable closeout, and `metadata` is the
machine-readable handoff that downstream agents, reviewers, or monitors can
reuse without scraping prose.

For engineering and review tasks, prefer this optional metadata shape:

```json
{
  "changed_files": ["path/to/file.py"],
  "verification": ["pytest tests/hermes_cli/test_kanban_db.py -q"],
  "dependencies": ["parent task id or external issue, if any"],
  "blocked_reason": null,
  "retry_notes": "what failed before, if this was a retry",
  "residual_risk": ["what was not tested or still needs human review"]
}
```

These keys are a convention, not a schema requirement. The useful property is
that every worker leaves enough evidence for the next reader to answer four
questions quickly:

1. What changed?
2. How was it verified?
3. What can unblock or retry this if it fails?
4. What risk is still deliberately left open?

Keep secrets, raw logs, tokens, OAuth material, and unrelated transcripts out of
`metadata`. Store pointers and summaries instead. If a task has no files or
tests, say so explicitly in `summary` and use `metadata` for the evidence that
does exist, such as source URLs, issue ids, or manual review steps.

### The worker lifecycle

Every profile that works kanban tasks automatically gets the worker lifecycle — it's injected into the worker's system prompt at spawn (the `KANBAN_GUIDANCE` block), so there is **nothing to install or configure**. It teaches the worker the full lifecycle in **tool calls**, not CLI commands:

1. On spawn, call `kanban_show()` to read title + body + parent handoffs + prior attempts + full comment thread.
2. `cd $HERMES_KANBAN_WORKSPACE` (via the terminal tool) and do the work there.
3. Call `kanban_heartbeat(note="...")` every few minutes during long operations. **If your work may run longer than 1 hour, call `kanban_heartbeat` at least once an hour** — the dispatcher reclaims tasks that have been running past `kanban.dispatch_stale_timeout_seconds` (default 4 h) with no heartbeat in the last hour, on the assumption the worker crashed without cleanup. A reclaim is benign (the task goes back to `ready` for re-dispatch without a failure-counter tick) but you lose your current run's progress.
4. Complete with `kanban_complete(summary="...", metadata={...})`, or `kanban_block(reason="...")` if stuck.

That final `kanban_complete` / `kanban_block` call is part of the worker
protocol. If the worker process exits with status 0 while the task is still
`running`, the dispatcher treats that as a protocol violation and emits a
`protocol_violation` event.

**Agent-side prevention:** Before the worker exits, Hermes injects up to two
synthetic nudges when it detects the model is about to stop without a terminal
board tool call. This catches the common case where the model narrates the next
step ("Let me write the report") and stops with `finish_reason=stop`. The nudge
reminds the model to call `kanban_complete` or `kanban_block` immediately. This
guard is active only for dispatcher-spawned workers (`HERMES_KANBAN_TASK` is
set) and can be disabled with `HERMES_KANBAN_STOP_NUDGE=0`.

**Dispatcher-side recovery:** If the nudges are exhausted or the worker crashes
before reaching the nudge, the dispatcher gives the violation a **bounded retry**
(up to `_PROTOCOL_VIOLATION_FAILURE_LIMIT` consecutive violations, default 3)
before auto-blocking the task instead of respawning it into the same loop. The
budget counts only *consecutive* clean-exit protocol violations — interleaved
rate-limited requeues are neutral, and any other failure kind resets the
streak — and a per-task `max_retries` overrides the bound. This usually means
the model wrote a plain-text answer and exited without using the Kanban tool
surface.

The lifecycle plus the load-bearing reference details (workspace kinds, deliverable `artifacts`, claiming created cards) ship in that system-prompt block, so every worker has them regardless of which profile it runs under — no per-profile skill setup required.

### Pinning extra skills to a specific task

Sometimes a single task needs specialist context the assignee profile doesn't carry by default — a translation job that needs the `translation` skill, a review task that needs `github-code-review`, a security audit that needs `security-pr-audit`. Rather than editing the assignee's profile every time, attach the skills directly to the task.

**From an orchestrator agent** (the usual case — one agent routing work to another), use the `kanban_create` tool's `skills` array:

```
kanban_create(
    title="translate README to Japanese",
    assignee="linguist",
    skills=["translation"],
)

kanban_create(
    title="audit auth flow",
    assignee="reviewer",
    skills=["security-pr-audit", "github-code-review"],
)
```

**From a human (CLI / slash command)**, repeat `--skill` for each one:

```bash
hermes kanban create "translate README to Japanese" \
    --assignee linguist \
    --skill translation

hermes kanban create "audit auth flow" \
    --assignee reviewer \
    --skill security-pr-audit \
    --skill github-code-review
```

The dispatcher emits one `--skills <name>` flag per skill listed, so the worker spawns with all of them loaded on top of the auto-injected kanban guidance. The skill names must match skills that are actually installed on the assignee's profile (run `hermes skills list` to see what's available); there's no runtime install.

### Per-task model override

Pin a task's worker to a specific model (and optionally provider), independent of the assignee profile's default:

```bash
# At creation
hermes kanban create "hard refactor" --assignee coder \
    --model claude-opus-4.6 --provider anthropic

# Or later — takes effect on the next dispatch
hermes kanban set-model t_abcd claude-opus-4.6 --provider anthropic
hermes kanban set-model t_abcd none    # clear the override
```

The dispatcher spawns the worker with the pinned model (`--provider <name>` is passed when set; `--provider` requires a model). With no override, the worker uses its profile's configured model.

### Cost strategy: frontier orchestrator, inexpensive workers

Kanban's per-profile configs make the planner/worker cost split natural. Decomposing a project into well-scoped cards takes frontier-level judgment; executing a card that already carries a clear goal, context, and handoff evidence usually doesn't — and the workers are where the vast majority of tokens are spent, so the worker model is where the cost lives. Run your orchestrator/dispatcher profile on a frontier model and point worker profiles at inexpensive models. Each profile has its own `config.yaml` under `~/.hermes/profiles/<name>/`, and the dispatcher injects the profile-scoped `HERMES_HOME` when it spawns `hermes -p <assignee>`, so each worker reads its own profile's model settings:

```yaml
# ~/.hermes/config.yaml (orchestrator / dispatcher profile)
model:
  default: "your-frontier-model"

# ~/.hermes/profiles/coder/config.yaml (worker profile)
model:
  default: "your-inexpensive-model"

# ~/.hermes/profiles/researcher/config.yaml (another worker profile)
model:
  default: "your-inexpensive-model"
```

For the occasional quality-sensitive card, pin just that task back to a stronger model with the [per-task model override](#per-task-model-override) (`--model`/`--provider` at create time or `hermes kanban set-model` later) — no profile edits needed.

### Lifecycle plugin hooks

Board transitions fire [plugin hooks](/user-guide/features/hooks#plugin-hooks): `kanban_task_claimed`, `kanban_task_completed`, and `kanban_task_blocked`, each carrying `task_id` and `profile_name`. Hooks fire **after** the board DB change commits, so callbacks always see durable state. Note the process split: `kanban_task_claimed` fires in the **dispatcher** process, while `kanban_task_completed`/`kanban_task_blocked` fire in the **worker** process — register the hook in the dispatcher profile to observe every transition centrally.

```python
def register(ctx):
    def on_blocked(task_id=None, profile_name=None, **kw):
        ctx.dispatch_tool("terminal", {"command": f"notify-send 'kanban blocked: {task_id}'"})
    ctx.register_hook("kanban_task_blocked", on_blocked)
```

### Goal-mode cards (`--goal`)

By default each worker gets **one shot** at its card — do the work, call `kanban_complete`/`kanban_block`, exit. Pass `--goal` (CLI) or `goal_mode=True` (the `kanban_create` tool) to instead run that worker in a **goal loop**, the same Ralph-style engine behind the `/goal` slash command: after every turn an auxiliary judge checks the worker's output against the card's title + body (treated as the acceptance criteria), and if the work isn't done — and the turn budget remains — the worker keeps going **in the same session** until the judge agrees, the worker terminates the task itself, or the budget runs out (which **blocks** the card for human review rather than exiting silently).

```bash
hermes kanban create "Translate the docs site to French" \
    --body "Acceptance: every page translated, no English left, links intact." \
    --assignee linguist \
    --goal \
    --goal-max-turns 15      # optional; default 20
```

Use it for open-ended, multi-step, or "keep going until X is true" cards. Skip it for cheap one-shot work — the per-turn judge overhead isn't worth it, and the dispatcher's existing retry/circuit-breaker already handles transient worker failures. The judge is only as good as your goal text, so write the body as **explicit acceptance criteria**.

:::note Goal-mode cards borrow the `/goal` engine — they don't connect to it
`--goal` runs the continuation loop *inside that one card's worker session*. It shares the engine with the [`/goal` slash command](./goals), not the state: setting a `/goal` in a chat session never creates, claims, or moves a kanban card, and a goal-mode card's loop is invisible to any chat session's `/goal status`. If you want this conversation to keep iterating, use [`/goal`](./goals); if you want work on the board, create a card.
:::

### How the orchestrator behaves

A **well-behaved orchestrator does not do the work itself.** It decomposes the user's goal into tasks, links them, assigns each to one of the profiles you've set up, and steps back. The orchestrator guidance — anti-temptation rules, a Step-0 profile-discovery prompt (the dispatcher silently fails on unknown assignee names, so the orchestrator must ground every card in profiles that actually exist on your machine), and a decomposition playbook keyed on `kanban_create` / `kanban_link` / `kanban_comment` — is injected into the worker's system prompt automatically; there is nothing to install.

A canonical orchestrator turn (two parallel researchers handing off to a writer):

```
# Goal from user: "draft a launch post on the ICP funding landscape"
kanban_create(title="research ICP funding, NA angle",  assignee="researcher-a", body="…")  # → t_r1
kanban_create(title="research ICP funding, EU angle",  assignee="researcher-b", body="…")  # → t_r2
kanban_create(
    title="synthesize ICP funding research into launch post draft",
    assignee="writer",
    parents=["t_r1", "t_r2"],        # promoted to 'ready' when both researchers complete
    body="one-pager, neutral tone, cite sources inline",
)                                     # → t_w1
# Optional: add cross-cutting deps discovered later without re-creating tasks
kanban_link(parent_id="t_r1", child_id="t_followup")
kanban_complete(
    summary="decomposed into 2 parallel research tasks → 1 synthesis task; writer starts when both researchers finish",
)
```

The orchestrator guidance ships in the worker's system prompt automatically — there is nothing to install or sync per profile.

**Decide before you fan out.** Design decisions belong to the orchestrator, not to the workers. If two parallel cards would each have to pick the same thing — a naming scheme, a schema, a file format, an API shape — the orchestrator decides it once and stamps the decision into **both** card bodies. Workers cannot see sibling cards, so every child card body must carry every decision it depends on. Example: for the parallel cards "build the exporter" and "build the importer", don't let each worker invent its own file format — pick one up front (say, newline-delimited JSON with a `version` field) and write it into both bodies, or the two halves will never round-trip.

For best results, pair it with a profile whose toolsets are restricted to board operations (`kanban`, `gateway`, `memory`) so the orchestrator literally cannot execute implementation tasks even if it tries.


## Auto vs Manual orchestration

The kanban board has two ways to handle a task you drop into the Triage column:

**Auto (default)** — `kanban.auto_decompose: true`. The gateway-embedded dispatcher runs the **decomposer** on each tick, capped by `kanban.auto_decompose_per_tick` (default 3 tasks per tick) so a bulk-load of triage tasks doesn't burst-spend the auxiliary LLM. The decomposer uses the built-in decomposition prompt plus the `auxiliary.kanban_decomposer` model path, reads your installed profiles + their descriptions, and asks the LLM to produce a JSON task graph: which tasks to spawn, who they go to, and which depend on which. The original triage task becomes the parent of every leaf in the graph, so it stays alive until the whole graph completes - and then promotes back to `ready` so its assignee (`kanban.orchestrator_profile`, or the active default profile when unset) can judge completion and add more tasks if the work isn't done. This is the "drop a one-liner, walk away" flow.

**Manual** — `kanban.auto_decompose: false`. Triage tasks stay in triage until you run `hermes kanban decompose <id>` (or `--all`) or use `/kanban decompose <id>` from a chat. This is useful when you want full control over what runs when.

**Important boundary:** Manual mode disables only the built-in Triage decomposer. It does not prevent a profile from calling `kanban_create`, and it does not disable creator-session wake-ups. With `kanban.auto_subscribe_on_create: true`, a task's terminal event resumes the originating agent with a synthetic status turn so it can inspect the handoff and decide whether genuinely new follow-up work is needed. Set `auto_subscribe_on_create: false` when task completion should remain passive. For provenance, built-in decomposer children use `created_by=auto-decomposer`; tasks created by a resumed profile carry that profile name instead.

Switch modes by editing `kanban.auto_decompose` in `config.yaml`. Both modes coexist with `hermes kanban specify` — that's still available as a single-task spec rewrite when you don't want fan-out.

The decomposer's routing decisions depend on profile descriptions, which is a per-profile labeling primitive you set with `hermes profile create --description "..."`, `hermes profile describe <name> --text "..."`, or `hermes profile describe <name> --auto` (LLM-generates from the profile's installed skills + model). Profiles without a description still appear in the roster — they're routable by name, just less precisely. The decomposer NEVER lands a child task with `assignee=None`: when the LLM picks an unknown profile, the child gets routed to `kanban.default_assignee` (or the active default profile if that's unset).

`kanban.orchestrator_profile` does not load that profile's prompt, skills, or custom logic into the decomposition call. It controls who owns the root/orchestration task after fan-out. To change the decomposer's model/provider, configure `auxiliary.kanban_decomposer`. To use a profile's custom task-splitting logic instead of the built-in decomposer, switch to Manual mode and have that profile create or decompose tasks explicitly.

Config knobs (all under `kanban:` in `~/.hermes/config.yaml`):

| Key | Default | Purpose |
|---|---|---|
| `auto_decompose` | `true` | Dispatcher auto-runs the built-in decomposer for Triage tasks every tick. It does not gate profile-driven `kanban_create` calls or creator wake turns. |
| `auto_decompose_per_tick` | `3` | Cap on decompositions per dispatcher tick. Excess defers to the next tick. |
| `orchestrator_profile` | `""` | Profile assigned to the root/orchestration task after decomposition. Empty = fall back to active default profile. |
| `default_assignee` | `""` | Where a child task lands when the LLM picks an unknown profile. Empty = fall back to active default. |
| `auto_subscribe_on_create` | `true` | When `kanban_create` runs inside a persistent gateway session, terminal events resume that originating agent with a synthetic status turn. Set to `false` for passive completion or to require explicit `kanban_notify-subscribe` calls. Independent of `auto_decompose`. |
| `done_sub_retention_days` | `30` | Notify subscriptions survive `done` (reopen-safe) and are removed on `archived`. The notifier GC purges subscriptions whose task has been `done` with no new events for this many days, bounding sub-table growth on boards that never archive. `0` disables the sweep. |

And the two auxiliary LLM slots:

| Key | Purpose |
|---|---|
| `auxiliary.kanban_decomposer` | Model that produces the task graph (called by Decompose). Set `provider`/`model` to override the main chat model. |
| `auxiliary.profile_describer` | Model that auto-generates profile descriptions (called by `hermes profile describe --auto`). |


## CLI command reference

This is the surface **you** (or scripts and cron) use to drive the board. Workers running inside the dispatcher use the `kanban_*` [tool surface](#how-workers-interact-with-the-board) for the same operations — the CLI here and the tools there both route through `kanban_db`, so the two surfaces agree by construction.

```
hermes kanban init                                     # create kanban.db + print daemon hint
hermes kanban create "<title>" [--body ...] [--assignee <profile>]
                                [--parent <id>]... [--tenant <name>]
                                [--workspace scratch|worktree|worktree:<path>|dir:<path>]
                                [--branch <name>]
                                [--priority N] [--triage] [--idempotency-key KEY]
                                [--max-runtime 30m|2h|1d|<seconds>]
                                [--max-retries N]
                                [--goal] [--goal-max-turns N]
                                [--skill <name>]...
                                [--json]
hermes kanban list [--mine] [--assignee P] [--status S] [--tenant T] [--archived]
        [--workflow-template-id <id>] [--current-step-key <key>]
        [--sort created|created-desc|priority|priority-desc|status|assignee|title|updated]
        [--json]
hermes kanban show <id> [--json]
hermes kanban assign <id> <profile>                    # or 'none' to unassign
hermes kanban reassign <id>... <profile>               # bulk re-assign tasks to a profile
hermes kanban edit <id> [--title ...] [--body ...]     # edit task title / body / priority in place
        [--priority N]
hermes kanban promote <id>...                          # move todo/blocked tasks to ready (recovery)
hermes kanban schedule <id> --at <ISO8601>             # set/clear a task's scheduled_at start time
hermes kanban diagnostics [--json]                     # board health snapshot (alias: diag)
hermes kanban link <parent_id> <child_id>
hermes kanban unlink <parent_id> <child_id>
hermes kanban claim <id> [--ttl SECONDS]
hermes kanban comment <id> "<text>" [--author NAME]

# Bulk verbs — accept multiple ids:
hermes kanban complete <id>... [--result "..."]
hermes kanban block <id> "<reason>" [--ids <id>...]
hermes kanban unblock <id>...
hermes kanban archive <id>...

hermes kanban request-review <id> [--summary "..."] [--metadata JSON] [--reviewer PROFILE]
hermes kanban request-changes <id> "<required changes>"               # active reviewer -> implementer
hermes kanban reopen-review  <id>... [--reason "..."]                 # changes requested: 'review' -> ready/todo

hermes kanban tail <id>                                # follow a single task's event stream
hermes kanban watch [--assignee P] [--tenant T]        # live stream ALL events to the terminal
        [--kinds completed,blocked,…] [--interval SECS]
hermes kanban heartbeat <id> [--note "..."]            # worker liveness signal for long ops
hermes kanban runs <id> [--json]                       # attempt history (one row per run)
hermes kanban assignees [--json]                       # profiles on disk + per-assignee task counts
hermes kanban dispatch [--dry-run] [--max N]           # one-shot pass
        [--failure-limit N] [--json]
hermes kanban daemon --force                           # DEPRECATED — standalone dispatcher (use `hermes gateway start` instead)
        [--failure-limit N] [--pidfile PATH] [-v]
hermes kanban stats [--json]                           # per-status + per-assignee counts
hermes kanban log <id> [--tail BYTES]                  # worker log from ~/.hermes/kanban/logs/
hermes kanban notify-subscribe <id>                    # gateway bridge hook (used by /kanban in the gateway)
        --platform <name> --chat-id <id> [--thread-id <id>] [--user-id <id>]
        [--chat-type dm|group|channel|thread] [--delivery-mode notify|notify+wake|wake]
hermes kanban notify-list [<id>] [--json]
hermes kanban notify-unsubscribe <id>
        --platform <name> --chat-id <id> [--thread-id <id>]
hermes kanban context <id>                             # what a worker sees
hermes kanban specify [<id> | --all] [--tenant T]      # flesh out a triage-column idea
        [--author NAME] [--json]                       #   into a full spec and promote to todo
hermes kanban gc [--event-retention-days N]            # workspaces + old events + old logs
        [--log-retention-days N]
```

All commands are also available as a slash command in the interactive CLI and in the messaging gateway (see [`/kanban` slash command](#kanban-slash-command) below).

`--max-retries` is a per-task circuit-breaker override for the dispatcher. `--max-retries 1` blocks the task on the first non-successful attempt, while `--max-retries 3` allows two retries and blocks on the third failure. Omit it to use `kanban.failure_limit` from `config.yaml`, then the built-in default.

### Concurrency, scheduling, and child promotion config

| Config key | Default | What it does |
|------------|---------|--------------|
| `kanban.max_in_progress` | unset (unlimited) | Caps the number of simultaneously running tasks. When the board already has N running, the dispatcher skips spawning more — useful for slow workers (local LLMs, resource-constrained hosts) so they finish what they have before more pile up and time out. Invalid or below-1 values log a warning and behave as unlimited. |
| `kanban.max_in_progress_per_profile` | unset (unlimited) | Per-profile variant of `max_in_progress` — caps how many tasks any single assignee profile may run concurrently. Useful when one profile is slow or rate-limited but others should keep flowing. Applies alongside the board-wide `max_in_progress`; both must allow a spawn for it to proceed. |
| `kanban.auto_promote_children` | `true` | After `decompose_triage_task()` produces children with no parent-blocker dependencies, they're automatically promoted to `ready` so the dispatcher can pick them up. Set to `false` to require manual review — children stay in `todo` until you promote them. |
| `kanban.default_workdir` | unset | Board-level default working directory applied to new tasks when neither `--workspace` nor the task itself overrides it. Per-task `workspace:` still wins. |

```yaml
kanban:
  max_in_progress: 2
  auto_promote_children: false
  default_workdir: ~/work/active-project
```

### Scheduled task starts (`scheduled_at`)

Set `scheduled_at` on a task to delay dispatch until a specific time. The dispatcher skips ready tasks whose `scheduled_at` is in the future and picks them up on the first tick after that timestamp.

```bash
hermes kanban create "nightly backup audit" \
  --assignee ops --scheduled-at "2026-06-01T03:00:00Z"
```

### Respawn guard

The dispatcher refuses to re-spawn a ready task when it hit a quota/auth/429 error on the previous run (`blocker_auth`), or completed a run successfully within the guard window (`recent_success`), or a recent task comment links to a GitHub PR (`active_pr`). This prevents repeat worker storms on the same bug or task while a human catches up. See the `respawn_guarded` row in the [event reference](#event-reference).


### Kanban Swarm topology helper

`hermes kanban swarm` creates a durable **Kanban Swarm v1** graph in one shot: a completed root/blackboard card, N parallel worker cards, a verifier card gated on all workers, and a synthesizer card gated on the verifier. Shared swarm context (the "blackboard") is stored as structured JSON comments on the root card so any worker can read it.

```bash
hermes kanban swarm "Design a multi-region failover plan" \
  --workers researcher,architect,sre \
  --verifier reviewer --synthesizer writer
```

The resulting graph is committed atomically: readers see either no new swarm or the complete topology, never a partially linked root/worker/verifier graph. It then dispatches normally — workers run in parallel, the verifier wakes after they all finish, and the synthesizer wakes after the verifier marks the work clean.

## `/kanban` slash command {#kanban-slash-command}

Every `hermes kanban <action>` verb is also reachable as `/kanban <action>` — from inside an interactive `hermes chat` session **and** from any gateway platform (Telegram, Discord, Slack, WhatsApp, Signal, Matrix, Mattermost, email, SMS). Both surfaces call the exact same `hermes_cli.kanban.run_slash()` entry point that reuses the `hermes kanban` argparse tree, so the argument surface, flags, and output format are identical across CLI, `/kanban`, and `hermes kanban`. You don't have to leave the chat to drive the board.

```
/kanban list
/kanban show t_abcd
/kanban create "write launch post" --assignee writer --parent t_research
/kanban comment t_abcd "looks good, ship it"
/kanban unblock t_abcd
/kanban dispatch --max 3
/kanban specify t_abcd                  # flesh out a triage one-liner into a real spec
/kanban specify --all --tenant engineering  # sweep every triage task in one tenant
```

Quote multi-word arguments the same way you would on a shell — `run_slash` parses the rest of the line with `shlex.split`, so `"..."` and `'...'` both work.

### Mid-run usage: `/kanban` bypasses the running-agent guard

The gateway normally queues slash commands and user messages while an agent is still thinking — that's what stops you from accidentally starting a second turn while the first is in flight. **`/kanban` is explicitly exempted from this guard.** The board lives in `~/.hermes/kanban.db`, not in the running agent's state, so reads (`list`, `show`, `context`, `tail`, `watch`, `stats`, `runs`) and writes (`comment`, `unblock`, `block`, `assign`, `archive`, `create`, `link`, …) all go through immediately, even mid-turn.

This is the whole point of the separation:

- A worker blocks waiting on a peer → you send `/kanban unblock t_abcd` from your phone and the dispatcher picks the peer up on its next tick. The blocked worker isn't interrupted — it just stops being blocked.
- You spot a card that needs human context → `/kanban comment t_xyz "use the 2026 schema, not 2025"` lands on the task thread and the *next* run of that task will read it in `kanban_show()`.
- You want to know what your fleet is doing without stopping the orchestrator → `/kanban list --mine` or `/kanban stats` inspects the board without touching your main conversation.

### Auto-subscribe on `/kanban create` (gateway only)

When you create a task from the gateway with `/kanban create "…"`, the originating chat (platform + chat id + thread id) is automatically subscribed to that task's terminal events (`completed`, `blocked`, `gave_up`, `crashed`, `timed_out`). You'll get one message back per terminal event — including the first line of the worker's result summary on `completed` — without having to poll or remember the task id.

```
you> /kanban create "transcribe today's podcast" --assignee transcriber
bot> Created t_9fc1a3  (ready, assignee=transcriber)
     (subscribed — you'll be notified when t_9fc1a3 completes or blocks)

… ~8 minutes later …

bot> ✓ t_9fc1a3 completed by transcriber
     transcribed 42 minutes, saved to podcast/2026-05-04.md
```

Subscriptions survive a task reaching `done` — completion is reversible (a reviewer or controller can reopen a done task), so the origin session keeps getting notified through reopen cycles. They auto-remove on `archived` (the irreversible end state). On boards that never archive, a GC sweep purges subscriptions for tasks that have sat in `done` with no new activity for `kanban.done_sub_retention_days` days (default 30; set 0 to disable), so stale rows don't accumulate forever. If you script a create with `--json` (machine output) the auto-subscribe is skipped — the assumption is that scripted callers want to manage subscriptions explicitly via `/kanban notify-subscribe`.

A chat-originated auto-subscribe is created in `notify+wake` mode: on a terminal event the destination agent both receives the passive message **and** takes a real turn, so it can read the board context and reply in its own voice. See [Delivery modes](#delivery-modes) below.

### Output truncation in messaging

Gateway platforms have practical message-length caps. If `/kanban list`, `/kanban show`, or `/kanban tail` produce more than ~3800 characters of output, the response is truncated with a `… (truncated; use \`hermes kanban …\` in your terminal for full output)` footer. The CLI surface has no such cap.

### Autocomplete

In the interactive CLI, typing `/kanban ` and hitting Tab cycles through the built-in subcommand list (`list`, `ls`, `show`, `create`, `assign`, `link`, `unlink`, `claim`, `comment`, `complete`, `block`, `unblock`, `archive`, `tail`, `dispatch`, `context`, `init`, `gc`). The remaining verbs listed in the CLI reference above (`watch`, `stats`, `runs`, `log`, `assignees`, `heartbeat`, `notify-subscribe`, `notify-list`, `notify-unsubscribe`, `daemon`) also work — they're just not in the autocomplete hint list yet.

## Collaboration patterns

The board supports these eight patterns without any new primitives:

| Pattern | Shape | Example |
|---|---|---|
| **P1 Fan-out** | N siblings, same role | "research 5 angles in parallel" |
| **P2 Pipeline** | role chain: scout → editor → writer | daily brief assembly |
| **P3 Voting / quorum** | N siblings + 1 aggregator | 3 researchers → 1 reviewer picks |
| **P4 Long-running journal** | same profile + shared dir + cron | Obsidian vault |
| **P5 Human-in-the-loop** | worker blocks → user comments → unblock | ambiguous decisions |
| **P6 `@mention`** | inline routing from prose | `@reviewer look at this` |
| **P7 Thread-scoped workspace** | `/kanban here` in a thread | per-project gateway threads |
| **P8 Fleet farming** | one profile, N subjects | 50 social accounts |
| **P9 Triage specifier** | rough idea → `triage` → `hermes kanban specify` expands body → `todo` | "turn this one-liner into a spec'd task" |

For worked examples of each, see `docs/hermes-kanban-v1-spec.pdf`.

## Handing context to follow-up cards (the parent link)

A parent link is not just a scheduling gate — it is the context handoff channel from a **completed** card to a new one. When you create a card with `--parent <done-card-id>`, two things happen:

1. **It's immediately eligible.** `create_task` sets status by parent state: a child whose parents are all `done` is created directly in `ready` — no waiting, no manual promotion. (Children of still-open parents sit in `todo` until `recompute_ready` promotes them when the last parent finishes.)
2. **The parent's handoff rides along.** The worker context assembled for the child (`build_worker_context`, what `kanban_show()` returns) contains a `## Parent task results` section with each parent's completion `summary` and `metadata`, verbatim:

```
## Parent task results
### t_77c26979 (completed just now)
Added exponential backoff with jitter to the retry helper.
_metadata_: `{"changed_files": ["hermes_cli/retry.py", "tests/test_retry.py"], "decisions": ["capped backoff at 60s", "jitter = full"]}`
```

This is why the pattern for follow-up work on a finished card is **a new child card, not reopening the done card**. Completed cards are immutable history — their context flows forward through the parent link. Same-card rework (retry loops on a failing card) is a different mechanism: prior attempts on the *same* card surface as "prior attempts" in that card's own context.

A worktree or branch alone is not a substitute: repo state tells the follow-up worker *what* the code looks like, but not *why* — the decisions, tests run, and files touched live in the parent's structured handoff, not in git. Evidence that didn't exist when the parent completed (e.g. a CI log that failed later) belongs in the new card's **body**.

```bash
# Implementation card t_impl is done. CI fails two hours later.
hermes kanban create "Fix CI failure from t_impl: test_retry flakes on 3.11" \
    --assignee coder \
    --parent t_impl \
    --body "$(cat <<'EOF'
CI run #4812 failed after t_impl merged.
Log excerpt: FAILED tests/test_retry.py::test_backoff_jitter - TimeoutError
Acceptance: tests/test_retry.py green on 3.11 and 3.12 in CI.
Use a fresh worktree/branch; do not force-push the original branch.
EOF
)"
```

The remediation worker spawns with the original card's summary and metadata (changed files, decisions) already in context, plus the fresh evidence you put in the body.

### Reconciling colliding worker branches

In engineering pipelines (P1/P2 with worktrees), two workers' branches can
conflict when merged. Don't let either worker self-adjudicate — the colliding
agent lacks its peer's context and reliably overwrites the other side or
abandons its own. Instead, create a reconciliation card assigned to a **third,
neutral profile** with **both** conflicted cards linked as parents: the parent
links carry both sides' completion summaries into the reconciler's context, so
it receives both diffs *and* both intents. The bundled
[`merge-reconciler` skill](https://github.com/NousResearch/hermes-agent/blob/main/skills/autonomous-ai-agents/merge-reconciler/SKILL.md)
gives that worker the full procedure: classify each conflicted hunk, resolve
impartially, verify, and hand back a summary naming every decision.

### Collision hotspots in parallel campaigns

In wide campaigns some files become collision magnets: many workers each add a
little to the same file, nobody owns keeping it small, and it turns into the
site of constant merge conflicts. The mitigation is a comment convention, not a
new primitive. A worker that notices its diff keeps colliding with siblings in
one file — or that a file it touches keeps appearing in other cards' recent
comments — should not silently pile on. Instead it leaves a comment on its own
card with a recognizable prefix:

```
hotspot: hermes_cli/kanban_db.py — third conflicting edit to the dispatch loop this wave
```

and repeats the flag in its completion `metadata`. Orchestrators (or humans
reviewing the board) who see **two or more `hotspot:` comments naming the same
path** should create a dedicated refactor/decomposition card for that file
**before** queuing more work that touches it — splitting the magnet file is
cheaper than reconciling every future collision it would cause. For conflicts
that have *already* happened, use the reconciliation-card pattern above with
the `merge-reconciler` skill; hotspot flagging is the upstream fix that keeps
the reconciler from becoming a standing lane.

## Multi-tenant usage

When one specialist fleet serves multiple businesses, tag each task with a tenant:

```bash
hermes kanban create "monthly report" \
    --assignee researcher \
    --tenant business-a \
    --workspace dir:~/tenants/business-a/data/
```

Workers receive `$HERMES_TENANT` and namespace their memory writes by prefix. The board, the dispatcher, and the profile definitions are all shared; only the data is scoped.

## Gateway notifications

When you run `/kanban create …` from the gateway (Telegram, Discord, Slack, etc.), the originating chat is automatically subscribed to the new task. The gateway's background notifier polls `task_events` every few seconds and delivers one message per terminal event (`completed`, `blocked`, `gave_up`, `crashed`, `timed_out`) to that chat. Completed tasks also send the first line of the worker's `--result` so you see the outcome without having to `/kanban show`.

You can manage subscriptions explicitly from the CLI — useful when a script / cron job wants to notify a chat it didn't originate from:

```bash
hermes kanban notify-subscribe t_abcd \
    --platform telegram --chat-id 12345678 --thread-id 7 \
    --chat-type group --delivery-mode notify+wake
hermes kanban notify-list
hermes kanban notify-unsubscribe t_abcd \
    --platform telegram --chat-id 12345678 --thread-id 7
```

A subscription removes itself automatically once the task reaches `done` or `archived`; no cleanup needed.

### Delivery modes

`--delivery-mode` controls **how** the notifier reacts to a terminal event. Every subscription is in one of three modes (`notify` is the default and the original behavior):

| Mode | Passive message | Wakes the agent | Use it when |
|------|-----------------|-----------------|-------------|
| `notify` | yes | no | You just want a heads-up message in the chat (default). |
| `notify+wake` | yes | yes | You also want the destination agent to take a real turn — read the board context and reply in its own voice. Chat-originated auto-subscribes use this. |
| `wake` | no | yes | You only want the agent to act on the event, with no separate ping. |

A "wake" forges a synthetic inbound message to the destination gateway agent so it takes a normal turn (reads the comment + result, reasons, replies) instead of getting a one-line passive notification. It only fires when the notifier runs inside a live gateway process; otherwise a `notify+wake` subscription still delivers its passive message, while a `wake`-only subscription does nothing in that process.

`--chat-type` (`dm` | `group` | `channel` | `thread`) records the originating chat's type so a woken turn resolves the operator's **real** session: `build_session_key` keys groups, channels, and threads differently from DMs, so an inaccurate `chat_type` would route the wake into a separate, context-less session. The `/kanban` auto-subscribe and slash-command paths capture this automatically — you only set it by hand when subscribing a chat from a script or cron. Omit it to leave an existing subscription unchanged (new subscriptions default to `dm`).

### Multi-profile setups: delivery is profile-owned

In a one-gateway-per-profile deployment (one dispatcher, separate gateway
processes for `writer`, `admin`, etc. — see the [multi-gateway
guide](https://github.com/NousResearch/hermes-agent/blob/main/docs/kanban/multi-gateway.md)),
dispatch and delivery have separate owners:

- **Dispatch stays single-owner.** Exactly one gateway keeps
  `kanban.dispatch_in_gateway: true` and runs the dispatcher; every other
  gateway sets it to `false`.
- **Notification delivery is profile-owned.** Every gateway — including
  non-dispatch ones — runs the notifier and polls only subscriptions stamped
  with a profile whose platform adapters it hosts. A task created from the
  `writer` profile's Telegram gets its `completed`/`blocked` message delivered
  by the `writer` gateway, even though the `default` gateway did the
  dispatching.
- **Legacy subscriptions** created before profile stamping (no
  `notifier_profile` on the row) are delivered only by the gateway that holds
  the actual dispatcher singleton lock, so two gateways never race for them.

Duplicate delivery across gateways is prevented by the atomic per-event claim
in the board DB. No relays, credential sharing, or extra dispatchers are
needed — each profile gateway simply delivers through its own adapters.

## Runs — one row per attempt

A task is a logical unit of work; a **run** is one attempt to execute it. When the dispatcher claims a ready task it creates a row in `task_runs` and points `tasks.current_run_id` at it. When that attempt ends — completed, blocked, crashed, timed out, spawn-failed, reclaimed — the run row closes with an `outcome` and the task's pointer clears. A task that's been attempted three times has three `task_runs` rows.

Why two tables instead of just mutating the task: you need **full attempt history** for real-world postmortems ("the second reviewer attempt got to approve, the third merged"), and you need a clean place to hang per-attempt metadata — which files changed, which tests ran, which findings a reviewer noted. Those are run facts, not task facts.

Runs are also where **structured handoff** lives. When a worker completes a task (via `kanban_complete(...)`) it can pass:

- `summary` (tool param) / `--summary` (CLI) — human handoff; goes on the run; downstream children see it in their `build_worker_context`.
- `metadata` (tool param) / `--metadata` (CLI) — free-form JSON dict on the run; children see it serialized alongside the summary.
- `result` (tool param) / `--result` (CLI) — short log line that goes on the task row (legacy field, kept for back-compat).

Downstream children read the most recent completed run's summary + metadata for each parent. Retrying workers read the prior attempts on their own task (outcome, summary, error) so they don't repeat a path that already failed.

```
# What a worker actually does — a tool call, from inside the agent loop:
kanban_complete(
    summary="implemented token bucket, keys on user_id with IP fallback, all tests pass",
    metadata={"changed_files": ["limiter.py", "tests/test_limiter.py"], "tests_run": 14},
    result="rate limiter shipped",
)
```

The same handoff is reachable from the CLI when you need to close out a task a worker can't — for example, a task that was abandoned:

```bash
hermes kanban complete t_abcd \
    --result "rate limiter shipped" \
    --summary "implemented token bucket, keys on user_id with IP fallback, all tests pass" \
    --metadata '{"changed_files": ["limiter.py", "tests/test_limiter.py"], "tests_run": 14}'

# Review the attempt history on a retried task:
hermes kanban runs t_abcd
#   #  OUTCOME       PROFILE           ELAPSED  STARTED
#   1  blocked       worker               12s  2026-04-27 14:02
#        → BLOCKED: need decision on rate-limit key
#   2  completed     worker                8m   2026-04-27 15:18
#        → implemented token bucket, keys on user_id with IP fallback
```

`hermes kanban runs <task-id>` exposes one row per attempt. `task_events` rows carry the `run_id` they belong to, and the `completed` event embeds the first-line summary in its payload (capped at 400 chars) so gateway notifiers can render structured handoffs without a second SQL round-trip.

**Bulk close caveat.** `hermes kanban complete a b c --summary X` is refused — structured handoff is per-run, so copy-pasting the same summary to N tasks is almost always wrong. Bulk close *without* `--summary` / `--metadata` still works for the common "I finished a pile of admin tasks" case.

**Reclaimed runs from status changes.** If a running task is moved back to `ready` or `todo`, or archived while still running, the in-flight run closes with `outcome='reclaimed'` rather than being orphaned. The `task_runs` row is always in a terminal state when `tasks.current_run_id` is `NULL`, and vice versa — that invariant holds across the CLI, dispatcher, and notifier.

**Synthetic runs for never-claimed completions.** Completing or blocking a task that was never claimed (for example, `hermes kanban complete <ready-task> --summary X`) would otherwise drop the handoff. Instead the kernel inserts a zero-duration run row (`started_at == ended_at`) carrying the summary / metadata / reason so attempt history stays complete. The `completed` / `blocked` event's `run_id` points at that row.

### Forward compatibility

Two nullable columns on `tasks` are reserved for v2 workflow routing: `workflow_template_id` (which template this task belongs to) and `current_step_key` (which step in that template is active). The v1 kernel ignores them for routing but lets clients write them, so a v2 release can add the routing machinery without another schema migration.

## Event reference

Every transition appends a row to `task_events`. Each row carries an optional `run_id` so UIs can group events by attempt. Kinds group into three clusters so filtering is easy (`hermes kanban watch --kinds completed,gave_up,timed_out`):

**Lifecycle** (what changed about the task as a logical unit):

| Kind | Payload | When |
|---|---|---|
| `created` | `{assignee, status, parents, tenant}` | Task inserted. `run_id` is `NULL`. |
| `promoted` | — | `todo → ready` because all parents hit `done`. `run_id` is `NULL`. |
| `claimed` | `{lock, expires, run_id}` | Dispatcher atomically claimed a `ready` task for spawn. |
| `completed` | `{result_len, summary?}` | Worker wrote `--result` / `--summary` and task hit `done`. `summary` is the first-line handoff (400-char cap); full version lives on the run row. If `complete_task` is called on a never-claimed task with handoff fields, a zero-duration run is synthesized so `run_id` still points at something. |
| `blocked` | `{reason, kind, recurrences}` | Worker or human flipped the task to `blocked`. `kind` is the typed block reason (`needs_input`, `capability`, `transient`, or `null` for a generic block); `recurrences` is the unblock-loop counter. Synthesizes a zero-duration run when called on a never-claimed task with `--reason`. |
| `dependency_wait` | `{reason, kind}` | Worker blocked with `kind=dependency` — the task is only waiting on another task, so it routes to `todo` (parent-gated, auto-promoted) instead of `blocked`. No human needed. |
| `block_loop_detected` | `{reason, kind, recurrences, limit}` | A task was unblocked and re-blocked for the same reason `BLOCK_RECURRENCE_LIMIT` times (default 2). Instead of landing in `blocked` again — where a cron would keep unblocking it — it routes to `triage` for a human decision, breaking the unblock↔re-block loop. |
| `unblocked` | — | `blocked → ready` (or `todo` if parents are still open), either manually or via `/unblock`. Resets the dispatcher's `consecutive_failures` but deliberately preserves `block_recurrences` so the loop breaker keeps its memory. `run_id` is `NULL`. |
| `archived` | — | Hidden from the default board. If the task was still running, carries the `run_id` of the run that was reclaimed as a side effect. |

**Edits** (human-driven changes that aren't transitions):

| Kind | Payload | When |
|---|---|---|
| `assigned` | `{assignee}` | Assignee changed (including unassignment). |
| `edited` | `{fields}` | Title or body updated. |
| `reprioritized` | `{priority}` | Priority changed. |
| `status` | `{status}` | A caller wrote a status directly (e.g. `todo → ready`). Carries the `run_id` of the run that was reclaimed when moving off `running`; otherwise `run_id` is NULL. |

**Worker telemetry** (about the execution process, not the logical task):

| Kind | Payload | When |
|---|---|---|
| `spawned` | `{pid}` | Dispatcher successfully started a worker process. |
| `heartbeat` | `{note?}` | Worker called `hermes kanban heartbeat $TASK` to signal liveness during long operations. |
| `reclaimed` | `{stale_lock}` | Claim TTL expired without a completion; task goes back to `ready`. |
| `crashed` | `{pid, claimer}` | Worker PID no longer alive but TTL hadn't expired yet. |
| `timed_out` | `{pid, elapsed_seconds, limit_seconds, sigkill}` | `max_runtime_seconds` exceeded; dispatcher SIGTERM'd (then SIGKILL'd after 5 s grace) and re-queued. |
| `stale` | `{elapsed_seconds, last_heartbeat_at, heartbeat_age_seconds, timeout_seconds, pid, terminated}` | Task ran longer than `kanban.dispatch_stale_timeout_seconds` (default 4 h) AND no `kanban_heartbeat` arrived in the last hour. Dispatcher SIGTERM'd the host-local worker (if any), reset the task to `ready` for re-dispatch. Does NOT tick the failure counter (stale is dispatcher-side absence detection, not a worker fault). Workers running long operations should call `kanban_heartbeat` at least once an hour to avoid this. |
| `reconciled` | `{reason, claim_lock, claim_expires, worker_pid}` | Orphaned-card reconciliation: the card was `running` with broken claim bookkeeping (`claim_lock` or `claim_expires` NULL — crash mid-claim, manual SQL, DB restore) and no live worker, so none of the TTL/crash/stale paths could ever recover it. The dispatcher requeued it to `ready` with an explanatory comment. Gated by `kanban.reconcile_orphans` in config.yaml (default `true`). |
| `respawn_guarded` | `{reason}` | Dispatcher refused to re-spawn this ready task this tick. Reasons: `blocker_auth` (last failure was a quota/auth/429 error — wait for the rate window to reset), `recent_success` (a completed run happened in the last hour — wait for review before re-running), `active_pr` (a GitHub PR URL appears in a recent comment — a prior worker already opened a PR). The task stays in `ready`; the next tick gets another chance to spawn. If the underlying condition persists, the normal `consecutive_failures` circuit breaker will auto-block via `gave_up` after `failure_limit` failures. |
| `spawn_failed` | `{error, failures}` | One spawn attempt failed (missing PATH, workspace unmountable, …). Counter increments; task returns to `ready` for retry. |
| `protocol_violation` | `{pid, claimer, exit_code, protocol_violation}` | Worker exited successfully while the task was still `running`, usually because it answered without calling `kanban_complete` or `kanban_block`. Emitted on every violation (the payload's `protocol_violation: true` marker is copied into the run metadata and feeds the violation-only retry budget). Below the budget — up to `_PROTOCOL_VIOLATION_FAILURE_LIMIT` (default 3) *consecutive* violations, per-task `max_retries` overriding — the task simply returns to `ready` for another attempt; when the streak reaches the bound the dispatcher also emits `gave_up` and auto-blocks. |
| `gave_up` | `{failures, effective_limit, limit_source, error}` | Circuit breaker fired after N consecutive non-successful attempts. Task auto-blocks with the last error. The effective limit resolves as task `max_retries`, then dispatcher `failure_limit` / `kanban.failure_limit`, then the built-in default. |

`hermes kanban tail <id>` shows these for a single task. `hermes kanban watch` streams them board-wide.

## Out of scope

Kanban is deliberately single-host. `~/.hermes/kanban.db` is a local SQLite file and the dispatcher spawns workers on the same machine. Running a shared board across two hosts is not supported — there's no coordination primitive for "worker X on host A, worker Y on host B," and the crash-detection path assumes PIDs are host-local. If you need multi-host, run an independent board per host and use `delegate_task` / a message queue to bridge them.

## Design spec

The complete design — architecture, concurrency correctness, comparison with other systems, implementation plan, risks, open questions — lives in `docs/hermes-kanban-v1-spec.pdf`. Read that before filing any behavior-change PR.
