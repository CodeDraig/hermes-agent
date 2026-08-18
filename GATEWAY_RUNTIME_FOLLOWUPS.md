# Gateway Runtime Decomposition Follow-ups

The `gateway/run.py` decomposition was reviewed on 2026-08-18. The two defects
below remain deliberately deferred. Both arise from using the active profile
home for files that coordinate the lifecycle of the single gateway process.
No runtime or test correction is included in this change.

## Secondary-profile restart markers use the wrong home

Secondary-profile messages run inside `_profile_runtime_scope`, so
`_gateway_config_home()` resolves to that profile. `/restart` consequently
writes `.restart_notify.json` and `.restart_last_processed.json` below the
secondary profile home, while gateway startup and Telegram redelivery
protection read those files from the fixed process home.

The gateway still restarts, but the initiating chat does not receive its
completion notification and a redelivered Telegram command can trigger another
restart. Until repaired, invoke `/restart` only from the primary profile or an
operator-controlled non-profile-scoped path. If the failure occurs, stop the
restart cycle, move or recreate both markers in the process home, acknowledge
or clear the stale platform update, and start the gateway once.

Repair boundary: give process-wide restart markers a lifecycle-owned path and
use it in the command writer, startup notification reader, and redelivery
reader. Do not derive these paths from the active profile scope.

## Secondary-profile update IPC uses the wrong home

`/update` also derives its pending, output, and exit-code paths from
`_gateway_config_home()`. Under a secondary profile, the detached updater
writes its IPC files below that profile home, while the current gateway, its
streaming watcher, and a replacement gateway look only in the fixed process
home.

The updater can start, but progress streaming, interactive prompt forwarding,
completion detection, and post-restart notification lose contact with it.
Until repaired, invoke `/update` only from the primary profile or directly from
the CLI. If the failure occurs, inspect the detached updater, move its complete
`.update_*` IPC set into the process home, and reconcile the checkout before
retrying if the updater already changed it.

Repair boundary: centralize update IPC paths under a process-lifecycle owner
and use that owner in the command handler, watcher, startup recovery, response
forwarding, and cleanup. Profile configuration should remain profile-scoped.

## Deferred verification boundary

Test correction remains a separate phase. Future validation should dispatch
both commands through a secondary profile and prove that restart markers and
the complete update IPC handshake remain visible to the process-wide readers
before and after gateway replacement.
