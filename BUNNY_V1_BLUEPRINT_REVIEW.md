# Bunny Deluxe V1 MVP - Blueprint Review and Cline Handoff

## Verdict

Proceed, with the amendments below.

The blueprint has the right product instinct: trust the existing perception stack, enter conservatively, hold a stable clock, make one bounded musical addition, and lose cleverness before losing time. That is an excellent MVP.

It is not implementation-ready exactly as written. The missing pieces are mostly contracts at subsystem boundaries, not a change of direction. Cline should treat this handoff as an amendment to the PDF.

## Repository reality Cline must preserve

The repository already contains:

- `perception/event_listener.py`, `perception/pulse.py`, and `perception/bar.py`.
- `PulseState` and `BarState`, including competing hypotheses and timestamps.
- A legacy clock-driven `scheduler.py` and MIDI transport in `midi_out.py`.
- A newer performance pipeline in `drummer/pipeline.py` and `drummer/pipeline_midi.py`.
- Uncommitted `SimpleBrain` work in `drummer/simple_brain.py` plus related tests, demos, groove metadata, and documentation.
- Many existing behavioural, arrangement, feel, and humanisation modules that are deliberately beyond this MVP.

Do not delete, reset, rename, or broadly refactor the existing work. Do not replace `SimpleBrain`. The new V1 core loop should be a narrow live-play controller that can coexist with it. Reuse its conservative ideas where useful, but do not force the PDF's state machine into that beat-selection class.

## Required blueprint amendments

### 1. Define one clock domain

All live timestamps and deadlines crossing module boundaries must use seconds from the same monotonic clock. Prefer an injected callable defaulting to `time.perf_counter` because the existing live scheduling code already uses it.

If the audio callback provides sample-frame or stream-relative time, convert it once in the listener adapter. Never compare wall-clock time with audio-relative or synthetic timestamps.

Every state object consumed by the controller must distinguish:

- `observed_at`: when the underlying evidence occurred.
- `computed_at`: when the state was produced.
- `age_seconds(now)`: freshness derived in one place.

Tests must use a fake clock. Do not use real sleeps in unit tests.

### 2. Replace the ambiguous state list with explicit semantics

Use these persistent states:

- `LISTENING`: no groove scheduled; gather trustworthy evidence.
- `ARMED`: entry conditions passed and a future downbeat has been chosen; no MIDI before that deadline.
- `PLAYING`: locked BPM and bar epoch drive the anchor groove.
- `DEGRADED`: mirror disabled; locked clock is held during a bounded grace period.
- `STOPPED`: explicit stop or completed silence timeout; no pending ledger events.

`ENTER` should be a transition/event, not a long-lived state. `MAINTAIN` maps to `PLAYING`.

Every transition must state its guard, side effects, and cancellation behavior. In particular:

- `LISTENING -> ARMED`: tempo and downbeat are fresh, sufficiently confident, and sufficiently unambiguous.
- `ARMED -> PLAYING`: the chosen monotonic downbeat deadline is reached.
- `ARMED -> LISTENING`: evidence becomes stale/unsafe before entry; cancel the pending entry.
- `PLAYING -> DEGRADED`: confidence remains below its exit threshold for the configured dwell time.
- `DEGRADED -> PLAYING`: confidence exceeds the higher recovery threshold for the configured dwell time; recover on a bar boundary.
- `DEGRADED -> STOPPED`: silence/instability exceeds the grace timeout.
- Any state `-> STOPPED`: explicit stop; cancel future events and send MIDI panic/all-notes-off behavior supported by the transport.

### 3. Add hysteresis and duration-based thresholds

Avoid magic counts such as "four snapshots" unless snapshot cadence is fixed. Express entry, degradation, recovery, drift, and silence rules in beats/bars or elapsed seconds.

Create one immutable configuration object. Initial defaults are hypotheses to tune, not product truth:

- BPM range: use the existing tracker capability unless playtests justify narrowing it. The repository currently supports 40-250 BPM; the PDF's 60-200 rule must not silently change perception behavior.
- Entry confidence threshold and minimum evidence duration.
- Exit threshold lower than entry threshold.
- Recovery threshold higher than exit threshold.
- Tempo/bar freshness limits.
- Degradation grace and stop-silence timeout.
- Quantisation tolerance.
- Maximum lookahead.
- Tempo-drift percentage and required dwell.

Log the effective configuration at startup.

### 4. Gate entry on ambiguity, not just the winning confidence

The current pulse and bar trackers maintain competing hypotheses. A high top confidence is insufficient if the runner-up is nearly equal, especially for half/double tempo and half-bar phase ambiguity.

The adapter should expose at least:

- winning tempo and bar hypotheses;
- confidence of each winner;
- winner-to-runner-up margin or ratio;
- source event/support count;
- evidence age;
- predicted next downbeat in the shared monotonic clock domain.

Entry requires both adequate absolute confidence and adequate separation from alternatives. Add tests where confidence is high but ambiguity remains; Bunny must stay listening.

### 5. Define bar identity and quantised slots precisely

The locked playing grid is defined by:

- `locked_bpm`;
- `beat_period_seconds`;
- `bar_epoch` (a monotonic timestamp for beat 1);
- four beats per bar;
- 16 slots per bar, numbered 0-15.

Anchor slots are kick `{0, 8}`, snare `{4, 12}`, and closed hat `{0, 2, 4, 6, 8, 10, 12, 14}`.

For an observation at time `t`, quantise against the locked grid, not the latest fluttering tracker grid. Record `(absolute_bar_index, slot, offset_seconds, strength)`. Boundary wrapping must be deterministic: an event near slot 15/0 belongs to the nearest absolute slot, with its corresponding bar index.

### 6. Tighten the Anti-Rubbish Kick Mirror contract

For V1:

- Evaluate only while `PLAYING` and after at least two completed stable playing bars.
- Ignore anchor kick slots 0 and 8; they are already present.
- Ignore or explicitly blacklist snare backbeat slots 4 and 12 for the first version.
- Require the same non-anchor slot in two consecutive completed bars.
- Require both events to pass the strength and timing filters.
- Activate the mirror from the following bar boundary, never retroactively.
- Allow one active mirrored slot only.
- Use a moderate configurable velocity.
- Expire it after one unsupported bar (recommended) or another explicitly tested duration.
- Clear it immediately on degradation, stop, bar re-lock, or tempo re-lock.

Percentile strength filtering requires a minimum rolling sample count. Before that sample count exists, the mirror stays disabled. Keep observations bounded by time/count and never let the rolling history grow without limit.

### 7. Make the ledger an immutable, revisioned contract

Use an immutable scheduled-event record containing at least:

- unique event ID;
- absolute monotonic deadline;
- MIDI note, velocity, and channel;
- bar index and slot;
- priority/source (`anchor`, `hat`, `mirror`);
- controller generation/revision.

The scheduler owns a priority queue and emits due events. The controller owns musical decisions. On degrade, stop, or re-lock, increment the generation and invalidate stale future events. Deduplicate by event ID so repeated planning ticks cannot double-trigger a note.

Keep lookahead short and bounded (start around 50-100 ms, then measure). Do not pre-schedule a whole bar if that prevents rapid cancellation.

### 8. Specify scheduler behavior under lateness

The existing `scheduler.py` is useful reference code but is not yet the PDF's required ledger scheduler. The live scheduler must define:

- bounded sleep plus final deadline wait;
- deterministic ordering for simultaneous hits;
- late-event policy (emit within a small lateness budget; drop beyond it rather than burst old hits);
- queue cancellation/generation invalidation;
- graceful shutdown and port closure;
- timing diagnostics: target, actual, jitter, late/drop count, queue depth;
- no musical decisions and no access to raw listener/tracker state.

Add a fake MIDI sink and fake clock scheduler test. Keep a separate opt-in hardware integration test for loopMIDI/EZDrummer; CI must not require a physical/virtual port.

### 9. Define conservative tempo and bar re-lock protocols

Do not continuously mutate locked BPM or bar epoch.

For V1, repeated drift should first enter `DEGRADED`. If drift remains strong and unambiguous for the configured dwell:

1. Stop planning additions.
2. Choose a future bar boundary.
3. Invalidate future ledger generations beyond the handoff point.
4. Install the new BPM/bar epoch at that boundary.
5. Resume anchor-only playing, then require stable bars before mirror eligibility.

If phase evidence suggests the snare is displaced by a beat or half-bar, do not flip it mid-bar. Degrade and perform the same boundary-based re-arm, or stop/relisten if ambiguity remains.

### 10. Separate testable core from live wiring

Recommended minimal additions, following the repository's current layout:

- `drummer/live_controller.py`: state machine, locks, transitions, generation counter.
- `drummer/live_models.py`: immutable adapter inputs, controller snapshot, ledger events, configuration.
- `drummer/straight_pocket.py`: anchor ledger and kick-mirror observation/selection.
- `drummer/live_scheduler.py`: priority-queue MIDI scheduler.
- `perception/live_adapter.py`: adapt existing `PulseState`/`BarState` and timestamp domains without rebuilding trackers.
- Focused tests under `tests/` with fake clock, fake states, fake event streams, and fake MIDI.

Names may change to match local conventions. Responsibilities may not blur.

Do not create top-level `brain/`, `patterns/`, or `scheduler/` packages merely to imitate the PDF. Do not move current modules.

## Revised build order

1. Run the existing test suite and record the baseline. Do not "fix" unrelated failures without reporting them.
2. Inventory the exact live timestamp origins and current `PulseState`/`BarState` fields.
3. Add immutable live input adapters, configuration, fake clock, and fake MIDI sink.
4. Add the explicit controller state machine and transition tests without real audio or MIDI.
5. Add the locked grid and exact Straight Pocket ledger tests.
6. Add the revisioned priority-queue scheduler and deterministic timing/cancellation tests.
7. Connect controller ledger planning to the scheduler using the fake sink.
8. Add degradation, silence, stop, and re-lock behavior.
9. Add the tightly bounded one-slot kick mirror.
10. Wire the existing pulse/bar trackers through the adapter.
11. Add an opt-in live clap-to-loopMIDI integration path and diagnostic trace.
12. Playtest and tune configuration values; do not bury tuned numbers in logic.

## Acceptance criteria

Implementation is complete only when all of the following are demonstrated:

- Existing tests remain passing and new core tests are deterministic.
- Audio/listener code never emits MIDI or chooses patterns.
- Tracker code never changes the playing clock directly.
- High-confidence but ambiguous tempo/bar hypotheses do not trigger entry.
- Entry is armed for a future downbeat and is cancellable before it happens.
- Full pocket begins on the selected beat 1 with correct anchor slots.
- Playback BPM and bar epoch remain unchanged under small tracker flutter.
- Planning the same horizon twice produces no duplicate MIDI events.
- Degradation invalidates pending mirror events while preserving the bounded anchor grace period.
- Short silence keeps the anchor clock; prolonged silence produces a clean stop.
- Re-lock/re-alignment occurs only at a planned boundary, never as a mid-bar jump.
- Mirror cannot activate on weak, off-grid, one-bar, anchor, or backbeat evidence.
- One eligible repeated slot activates at most one mirrored kick from the following bar.
- Explicit stop prevents future emission and closes/clears MIDI safely.
- Timing diagnostics report jitter and late/dropped events.
- Hardware routing is documented and tested manually without making CI depend on loopMIDI.

## Non-goals to enforce

No GUI, genre inference, fills, crash logic, swing, odd meter, section memory, groove switching, advanced humanisation, ML, or broad architecture cleanup in this implementation. Do not connect the larger behaviour engine merely because it exists. This slice proves safe entry, clock stability, one conservative kick decision, and graceful uncertainty.

## Final instruction to Cline

Implement this as a small additive vertical slice on the current dirty worktree. Before editing, inspect `git status` and preserve every user change. Present the proposed file touch list before making a broad change. Work in small test-backed increments, and report any mismatch between repository reality and this handoff instead of silently inventing a second architecture.
