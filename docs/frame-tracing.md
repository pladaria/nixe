# Frame timeline capture

Build with `cargo build -p nixe-cli --profile profiling --features frame-trace`.
Default builds compile recording calls away and do not request GPU timestamp
features. A trace build still adds instrumentation overhead; compare FPS using
matched default builds, and use separate trace captures for diagnosis.

Set `NIXE_TRACE=/tmp/frame-timeline.json` to record a Chrome trace JSON file.
To exclude startup, also set `NIXE_TRACE_TRIGGER=/tmp/frame-trigger`: create that
file after entering gameplay and remove it to stop and export the capture.
Each invocation records one interval. Without a trigger, recording starts with
the command and exports when it finishes. Export runs on the trace control
thread, outside guest, display, audio and graphics execution.

Events share a monotonic host clock. Frame IDs link queueing, acquire fences,
latching, presentation and release. Submission IDs link frontend work, backend
segments, device completion and guest completion. IPC event names identify the
service; their `value` identifies the command. `ipc.host_enqueue`,
`ipc.host_ready` and `ipc.host_complete` link asynchronous service queueing,
worker completion and reply finalization using guest thread IDs and process IDs.
`ipc.host_retry` excludes rejected queue attempts from pending-service ages.
vCPU events identify the guest
thread. Audio events include append/nonzero sample counts, callback consumption
and underruns. CPU/shader compilation misses, boundary reasons and demanded
readback bytes are recorded as well.

Each thread retains at most 1,048,576 events. `dropped_events` reports overflow;
choose a common prefix without drops when comparing captures. GPU timestamp
readbacks have 128 reusable slots and never block to obtain a slot. Unsupported
query features leave host spans available. Vulkan device timestamps are
calibrated against the host clock when the device exposes calibrated timestamps;
`gpu.calibration_error_ns` bounds the observed calibration uncertainty. Otherwise
`gpu.device_duration_ns` records duration without inventing a host placement.
`gpu.timestamp_slots_full` and `gpu.timestamp_map_failed` identify missing device
intervals. Recording epochs exclude callbacks and spans from earlier captures.

`gpu.wait_completion` measures the blocking backend completion API; it includes
driver work and is not a hardware-only wait. `gpu.poll_completion` measures the
nonblocking API. `gpu.process_completion` covers handling every returned token,
including polled tokens and any nested preparation/submission of subsequent
segments. `gpu.owner_completion_cycle` encloses the explicit wait and processing;
do not add enclosing and nested durations together.

`gpu.inline_device` counts promoted four-byte words. The
`gpu.inline_rejected.*` events count words lacking a resident representation,
requiring unaligned writes, conflicting with a live/future physical alias, or
depending on another earlier GPU writer. Counts are aggregated per frontend
submission. `gpu.residency` and `gpu.inline_plan` isolate resource-version indexing
and ordered upload planning. Host definitions may be installed before their
first command use; device uploads remain in command order. CPU-only transfers
freeze earlier GPU read inputs at submission, and wait only for actual written
pages or guest completion dependencies. Resource retirement follows completion.

The display clock has a separate owner and progresses without coordinator
service. Surface acquisition and host encoding occur outside the shared queue
lock. Submission and presentation retain the lock required for native queue
external synchronization. The host mailbox remains bounded to one frame and
guest buffer ownership continues to follow acquire/release fences.

Incremental frontend diagnostics distinguish `gpu.resource_lookup` from
`gpu.resource_rebuild` and `gpu.prepared_draw_lookup` from `gpu.draw_rebuild`.
The rebuild spans include rebuilding components which could not be retained.
`gpu.resource_hit.current/indexed`, `gpu.resource_miss.*`, `gpu.draw_miss.*`,
`gpu.resource_component_reused`, `gpu.texture_component_reused`,
`gpu.draw_validation_hit` and `gpu.draw_fixed_components_reused` identify reuse
and its invalidation causes. A reusable component does not imply that its whole
operation or backend binding remains reusable.

`gpu.cpu_visibility` covers demanded canonical visibility, including real fence
waits. `gpu.visibility_buffer_demand_bytes` counts demanded dirty buffer bytes;
`gpu.visibility_known_bytes` counts the subset already present in the existing
page mirror from ordered uploads. These counters can count physically aliased
resource demands more than once; they are not unique physical byte totals.
`gpu.readback_bytes` continues to count actual aligned GPU downloads. Known bytes
avoid transfers only after actual producer completion; partial coverage never
makes an entire page clean. Image downloads merge around newer known physical
bytes. Upload knowledge has at most 64 intervals per existing mirrored page;
fragmentation exhaustion falls back to the ordinary dirty-range readback path.

Critical-path attribution uses `gpu.frontend_accepted` and `gpu.frontend_job`
with the same submission IDs as `gpu.frontend`. `vcpu.dispatched` to the start
of `vcpu.execute` measures worker queueing and host scheduling delay;
`vcpu.idle` to `vcpu.collected` measures result collection delay.
`coordinator.dispatch`, `coordinator.complete`, `coordinator.external_events`,
`horizon.svc`, and `host.input` expose work outside guest execution.
`scheduler.ready` and `scheduler.woken` identify readiness transitions.
Enclosing spans overlap nested service spans and concurrent workers; partition
interval unions rather than adding inclusive durations. `coordinator.reconcile`
includes waiting for worker completion and is not coordinator CPU time.
`coordinator.select` exposes deadline/selection work (including nested dispatch);
`coordinator.worker_wait` isolates the blocking receive from that selection.
`gpu.draw_fixed_components_reused` and `gpu.draw_validation_hit` count reuse of
bounded stable templates, independently of complete prepared-operation hits.
`gpu.fixed_state_write` reports the effective register writes changing fixed
configuration, including transient resets which can later restore the same key.
Linux per-thread schedstat deltas separately report CPU runtime and runnable
queue delay; they cannot assign kernel sleep time to a particular guest wait.

On Linux and Android, `vcpu.execution_cpu_ns` and `gpu.frontend_cpu_ns` record
thread CPU time for the corresponding elapsed spans. Other hosts omit these
events. Subtracting CPU from elapsed time estimates off-CPU time, including both
blocking and preemption; only per-thread scheduler counters distinguish runnable
queue delay in aggregate. CPU timing is erased in default builds. `vcpu.stop.*`
records each stop reason with retired guest instructions in `value`, allowing
budget-only handoffs to be separated from required service/event stops.

Realtime adaptive parallel execution targets 1 ms of measured worker elapsed
time. Adjustments are bounded to half/double the current instruction budget and
a ceiling of four million instructions (or the configured baseline if larger).
Only exhausted budgets update the estimate; SVCs do not reset it. This is a target,
not a hard deadline. Existing native control and interrupt polling remains
independent. Fixed-clock adaptive execution retains its instruction-based policy;
explicit budgets remain explicit and do not pay for elapsed-time clocks.
