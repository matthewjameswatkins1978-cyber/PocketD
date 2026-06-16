"""Demo skeleton for a console playtest interview with real scenario runs.

Usage:
    python demo_playtest_interview.py --scenario enter --preset normal --no-play
    python demo_playtest_interview.py --scenario drop --preset normal
    python demo_playtest_interview.py --summarize-feedback --feedback-file playtest_feedback.jsonl

Update-summary mode:
    python demo_playtest_interview.py --scenario enter --preset normal --no-play \
        --feedback-file playtest_feedback.jsonl \
        --update-summary --summary-md playtest_learning_summary.md
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import tempfile
import threading
import time

from drummer.playtest_feedback import (
    PlaytestScenario,
    PlaytestQuestionnaire,
    PlaytestDiagnosticsSummary,
    PlaytestFeedbackEntry,
    PlaytestLearningSummary,
    list_playtest_scenarios,
    get_scenario_variations,
    append_feedback_entry,
    validate_questionnaire_answers,
    run_playtest_scenario,
    load_feedback_entries,
    summarize_feedback_entries,
    load_and_summarize_feedback,
    export_learning_summary_json,
    export_learning_summary_markdown,
    _print_console_summary,
    parse_key_choice,
    TIMING_KEY_MAP,
    AMOUNT_KEY_MAP,
    CONFIDENCE_KEY_MAP,
    UNDERSTOOD_KEY_MAP,
    SUGGESTED_CHANGE_KEY_MAP,
)
from drummer.golden_reference import (
    GoldenReference,
    save_golden_reference,
    save_golden_diagnostics,
)
from drummer.bar_transcript import (
    build_bar_transcript,
    save_bar_transcript,
)
from drummer.musical_doctor import (
    diagnose_bar_transcript,
    save_doctor_report,
    print_doctor_summary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prompt_key(prompt: str, mapping: dict[str, str], field_name: str) -> str:
    """Prompt until the user enters a valid key from *mapping*.

    Returns the canonical value.
    """
    while True:
        raw = input(prompt).strip()
        try:
            return parse_key_choice(raw, mapping, field_name)
        except ValueError as e:
            print(f"  {e}")


_SUGGESTED_HELP = (
    "  [0] none    [1] enter later  [2] enter sooner  [3] play less\n"
    "  [4] play more  [5] build less  [6] build more  [7] recover later\n"
    "  [8] recover sooner  [9] mark less  [10] mark more\n"
    "  [11] ending weaker  [12] ending stronger"
)


def _collect_answers() -> PlaytestQuestionnaire:
    """Ask the seven fixed questions using single-key input and return a Questionnaire.

    Only the Note field requires full typing.
    """
    print("\n--- Musical Questions ---")
    print("Overall feel:  [1] bad  [2] rough  [3] okay  [4] good  [5] excellent")
    overall = _prompt_key("  Overall (1-5): ", {
        "1": "1", "2": "2", "3": "3", "4": "4", "5": "5",
    }, "overall_rating")

    timing = _prompt_key(
        "Timing: [e]arly  [r]ight  [l]ate  [n/a]: ",
        TIMING_KEY_MAP, "timing_rating",
    )
    amount = _prompt_key(
        "Amount: [s]parse  [r]ight  [b]usy  [n/a]: ",
        AMOUNT_KEY_MAP, "amount_rating",
    )
    confidence = _prompt_key(
        "Confidence: [t]imid  [r]ight  bol[d]  [n/a]: ",
        CONFIDENCE_KEY_MAP, "confidence_rating",
    )
    understood = _prompt_key(
        "Understood: [y]es  [p]artly  [n]o: ",
        UNDERSTOOD_KEY_MAP, "understood_rating",
    )

    print(f"Suggested change:{_SUGGESTED_HELP}")
    suggested = _prompt_key(
        "  Suggested change [0-12]: ",
        SUGGESTED_CHANGE_KEY_MAP, "suggested_change",
    )

    note = input("Note (free text, Enter for empty): ").strip()

    return PlaytestQuestionnaire(
        overall_rating=int(overall),
        timing_rating=timing,
        amount_rating=amount,
        confidence_rating=confidence,
        understood_rating=understood,
        suggested_change=suggested,
        note=note,
    )


def _prompt_replay() -> str:
    """Prompt for replay/answer/skip/quit; return one of 'r','a','s','q'."""
    while True:
        raw = input("\n(r)eplay, (a)nswer now, (s)kip, (q)uit: ").strip().lower()
        if raw in ("r", "a", "s", "q"):
            return raw
        print("  Please enter r, a, s, or q.")


# ---------------------------------------------------------------------------
# Simple ear-test feedback — 2-question default mode
# ---------------------------------------------------------------------------

_Q1_OPTIONS = (
    "  [g] good   [b] bad   [m] mixed   [u] unsure   [r] replay   [q] quit"
)

_Q3_OPTIONS = (
    "  [t] timing   [s] samey   [d] decision felt wrong   [n] none   [Enter] skip"
)


def _prompt_q1() -> str:
    """Prompt for overall judgment; return one of 'g','b','m','u','r','q'."""
    while True:
        raw = input("  Your choice [g/b/m/u/r/q]: ").strip().lower()
        if raw in ("g", "b", "m", "u", "r", "q"):
            return raw
        print("  Please enter g, b, m, u, r, or q.")


def _prompt_q2() -> int:
    """Prompt for feel score 1-5; return integer 1-5."""
    while True:
        raw = input("  Feel score (1-5): ").strip()
        if raw in ("1", "2", "3", "4", "5"):
            return int(raw)
        print("  Please enter a number from 1 to 5.")


def _prompt_q3() -> str:
    """Prompt for optional obvious issue; return 't','s','d','n',''."""
    while True:
        raw = input("  Obvious issue [t/s/d/n/Enter]: ").strip().lower()
        if raw in ("t", "s", "d", "n", ""):
            return raw
        print("  Please enter t, s, d, n, or press Enter to skip.")


def _prompt_golden_yn() -> bool:
    """Prompt to save as golden reference; return True on 'y'."""
    raw = input("  Save this as golden / protect this feel? [y/N]: ").strip().lower()
    return raw == "y"


def _build_2q_answers(
    q1: str,
    score: int,
    q3: str = "",
) -> PlaytestQuestionnaire:
    """Build a PlaytestQuestionnaire from the 2-question flow.

    Q1 → overall judgment with understood_rating
    Q2 → feel score 1-5 as overall_rating
    Q3 → optional obvious issue appended to note
    """
    # Base note from Q1
    if q1 == "g":
        note = "Matthew judged the overall performance as good."
        understood = "yes"
    elif q1 == "b":
        note = "Matthew judged the overall performance as bad."
        understood = "partly"
    elif q1 == "m":
        note = "Matthew judged the overall performance as mixed."
        understood = "partly"
    else:  # q1 == "u"
        note = "Matthew was unsure how to judge the overall performance."
        understood = "partly"

    # Append Q3 issue if provided
    if q3 == "t":
        note += " Obvious issue: timing felt off."
    elif q3 == "s":
        note += " Obvious issue: too samey or boring."
    elif q3 == "d":
        note += " Obvious issue: drummer decision felt wrong for the input."
    elif q3 == "n":
        note += " Obvious issue: none."

    # Map Q1 to suggested_change (use valid canonical values only;
    # the note field captures the actual sentiment)
    if q1 == "b":
        suggested = "no_change"
    elif q1 == "m":
        suggested = "no_change"
    elif q1 == "u":
        suggested = "no_change"
    else:
        suggested = "no_change"

    return PlaytestQuestionnaire(
        overall_rating=score,
        timing_rating="about_right",
        amount_rating="about_right",
        confidence_rating="about_right",
        understood_rating=understood,
        suggested_change=suggested,
        note=note,
    )


def _save_golden_for_scenario(
    sc: PlaytestScenario,
    raw_diagnostics: list[dict],
    timestamp: str,
) -> None:
    """Create and save a GoldenReference plus diagnostics snapshot."""
    diag_dir = "data/golden_diagnostics"
    safe_variation = sc.variation_name.replace(" ", "_")
    safe_ts = timestamp.replace(":", "-").replace(" ", "_")
    diag_filename = f"{sc.name}_{sc.preset}_{safe_variation}_{safe_ts}.json"
    diag_path = f"{diag_dir}/{diag_filename}"

    save_golden_diagnostics(raw_diagnostics, diag_path)

    golden = GoldenReference(
        scenario=sc.name,
        preset=sc.preset,
        variation=sc.variation_name,
        command=f"python demo_playtest_interview.py --scenario {sc.name} --preset {sc.preset}",
        timestamp=timestamp,
        user_rating=5,
        user_note="Matthew liked this and marked it golden.",
        approval_status="approved",
        tag="protect_this_feel",
        diagnostics_path=diag_path,
        midi_path=None,
    )
    save_golden_reference(golden)
    print(f"  [Golden reference saved to data/golden_references.jsonl]")
    print(f"  [Diagnostics snapshot saved to {diag_path}]")


def _print_compact_diagnostics(
    summary: PlaytestDiagnosticsSummary,
    sanity_passed: bool,
) -> None:
    """Print a one-line diagnostics summary for simple feedback mode."""
    contracts = "passed" if summary.output_contracts_passed else "FAILED"
    sanity = "PASSED" if sanity_passed else "FAILED"
    print(f"  Events: {summary.total_events} | Contracts: {contracts} | Sanity: {sanity}")


def _print_diagnostics_summary(
    summary: PlaytestDiagnosticsSummary,
    raw_diagnostics: list[dict],
    scenario: PlaytestScenario,
) -> None:
    """Print a compact diagnostics summary for the user with feature values."""
    print(f"\n  --- Diagnostics ---")
    print(f"  Total events:        {summary.total_events}")
    print(f"  First enter bar:     {summary.first_enter_bar}")
    print(f"  First build bar:     {summary.first_build_bar}")
    print(f"  Confidence peak:     {summary.confidence_peak:.2f}")
    print(f"  Phrase markers:      {summary.phrase_marker_count}")
    print(f"  Drop events:         {summary.drop_event_count}")
    print(f"  Final bail events:   {summary.final_bail_event_count}")
    print(f"  Bail events:         {summary.bail_event_count}")
    contracts_pass = summary.output_contracts_passed
    print(f"  Contracts passed:    {contracts_pass}")
    if not contracts_pass:
        # Show which contract failed
        drop_ok = any(d.get("is_drop", False) and d.get("event_count", 0) > 0 for d in raw_diagnostics)
        bail_ok = any(d.get("is_bail", False) for d in raw_diagnostics)
        final_bail_ok = any(d.get("is_final_bail", False) for d in raw_diagnostics)
        failed = []
        if not drop_ok:
            failed.append("DROP (>0 events)")
        if not bail_ok:
            failed.append("BAIL (0 events)")
        if not final_bail_ok:
            failed.append("FINAL_BAIL (kick+crash on beat 1)")
        if failed:
            print(f"    (Failed contracts: {', '.join(failed)})")

    # Print inferred intents compactly
    intents = summary.inferred_intents
    if intents:
        sorted_intents = sorted(intents.items(), key=lambda x: x[1], reverse=True)
        parts = [f"{k}={v}" for k, v in sorted_intents]
        print(f"  Inferred intents:    {', '.join(parts)}")

    # Print section-focused diagnostic bars within listen range
    print(f"\n  --- Per-bar focus bars {scenario.listen_start_bar}-{scenario.listen_end_bar} ---")
    print(f"  {'Bar':>4s}  {'Section':>14s}  {'Dens':>5s}  {'Cert':>5s}  "
          f"{'Stab':>5s}  {'Phs':>4s}  {'Conf':>5s}  {'Inferred':>12s}  "
          f"{'Intent':>12s}  {'RendInt':>12s}  {'Events':>4s}  {'Notes'}")
    print(f"  {'-'*4}  {'-'*14}  {'-'*5}  {'-'*5}  {'-'*5}  "
          f"{'-'*4}  {'-'*5}  {'-'*12}  {'-'*12}  {'-'*8}  {'-'*4}  {'-'*30}")
    for d in raw_diagnostics:
        bar = d["bar"]
        if scenario.listen_start_bar <= bar <= scenario.listen_end_bar:
            intent = d.get("intent", "?")
            inferred = d.get("inferred_intent", intent)
            rendered = d.get("rendered_intent", intent)
            events = d.get("event_count", 0)
            dens = d.get("density", 0)
            cert = d.get("certainty", 0)
            stab = d.get("stability", 0)
            phs = d.get("phase", 0)
            conf = d.get("confidence", 0)
            notes = d.get("notes_summary", "")
            override = "*" if inferred != intent else " "
            print(f"  {bar:4d}  {d['section']:>14s}  {dens:5.2f}  {cert:5.2f}  "
                  f"{stab:5.2f}  {phs:4.2f}  {conf:5.2f}  "
                  f"{inferred:>12s}  {intent:>12s}{override}  {rendered:>12s}  {events:4d}  {notes}")


def _update_and_export_summary(
    feedback_path: str,
    summary_json_path: str | None,
    summary_md_path: str | None,
) -> None:
    """Reload feedback, regenerate summary, and export if requested."""
    summary = load_and_summarize_feedback(feedback_path)
    _print_console_summary(summary)
    if summary_json_path:
        export_learning_summary_json(summary, summary_json_path)
    if summary_md_path:
        export_learning_summary_markdown(summary, summary_md_path)
    print("\n  [Learning summary updated]")


def _write_bar_transcript_if_requested(
    args,
    sc: PlaytestScenario,
    global_events: list,
    raw_diags: list[dict],
):
    """Generate and save a bar transcript if --bar-transcript or --doctor was requested.

    Returns the BarTranscript if generated, None otherwise.
    """
    want_transcript = getattr(args, "bar_transcript", False) or getattr(args, "doctor", False)
    if not want_transcript:
        return None
    ts = datetime.datetime.now().isoformat()
    safe_variation = sc.variation_name.replace(" ", "_")
    safe_ts = ts.replace(":", "-").replace(" ", "_")
    base = f"{sc.name}_{sc.preset}_{safe_variation}_{safe_ts}"
    txt_path = f"artifacts/playtests/{base}_bar_transcript.txt"
    json_path = f"artifacts/playtests/{base}_bar_transcript.json"
    transcript = build_bar_transcript(
        global_events, raw_diags,
        scenario=sc.name, preset=sc.preset, variation=sc.variation_name,
    )
    save_bar_transcript(transcript, txt_path, json_path)
    print(f"  [Bar transcript: {txt_path}]")
    return transcript


def _write_doctor_report_if_requested(
    args,
    sc: PlaytestScenario,
    transcript: object,
) -> None:
    """Run Musical Doctor and save report if --doctor was requested."""
    if not getattr(args, "doctor", False) or transcript is None:
        return
    ts = datetime.datetime.now().isoformat()
    safe_variation = sc.variation_name.replace(" ", "_")
    safe_ts = ts.replace(":", "-").replace(" ", "_")
    base = f"{sc.name}_{sc.preset}_{safe_variation}_{safe_ts}"
    txt_path = f"artifacts/playtests/{base}_doctor.txt"
    json_path = f"artifacts/playtests/{base}_doctor.json"
    report = diagnose_bar_transcript(transcript)
    save_doctor_report(report, txt_path, json_path)
    print_doctor_summary(report)
    print(f"  [Doctor report: {txt_path}]")


def _bar_count_from_events(events: list) -> int:
    """Return the last event bar as a 1-based count, or 0 for no events."""
    if not events:
        return 0
    return max(getattr(evt, "bar_index", 0) for evt in events) + 1


def _print_bar_counter(
    total_bars: int,
    bpm: float,
    stop_event: threading.Event,
) -> None:
    """Print a lightweight bar counter while playback is running."""
    if total_bars <= 0:
        return

    bar_duration = (60.0 / bpm) * 4.0
    start = time.perf_counter()
    current_bar = 0

    while current_bar < total_bars and not stop_event.is_set():
        print(f"\r  Bar {current_bar + 1:02d}/{total_bars:02d}", end="", flush=True)
        current_bar += 1
        next_time = start + current_bar * bar_duration
        while not stop_event.is_set():
            remaining = next_time - time.perf_counter()
            if remaining <= 0:
                break
            time.sleep(min(remaining, 0.05))


def _try_play_midi(events, bpm: float, total_bars: int | None = None) -> bool:
    """Attempt MIDI playback of *events*.  Returns True if playback happened."""
    if not events:
        return False
    try:
        from midi_out import MidiOut
        from drummer.pipeline_midi import list_available_ports, find_or_none
        from demo_continuous_jam_midi import _play_global_schedule

        ports = list_available_ports()
        if not ports:
            print("  [No MIDI ports available — skipping playback]")
            return False

        port_name = find_or_none("PocketDrummer Out")
        if port_name is None:
            print(f"  [Default port not found.  Available: {ports}]")
            return False

        bars = total_bars if total_bars is not None else _bar_count_from_events(events)
        print(f"  Playing via {port_name} ...")
        midi = MidiOut(port_name)
        midi.open()
        stop_counter = threading.Event()
        counter_thread = threading.Thread(
            target=_print_bar_counter,
            args=(bars, bpm, stop_counter),
            daemon=True,
        )
        try:
            counter_thread.start()
            _play_global_schedule(midi, events, bpm)
        finally:
            stop_counter.set()
            counter_thread.join(timeout=0.2)
            if bars > 0:
                print()
            midi.close()
        return True
    except Exception as e:
        print(f"  [MIDI playback failed: {e}]")
        print("  Use --no-play to run diagnostics only.")
        return False


# ---------------------------------------------------------------------------
# Build parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pocket drummer playtest interview and feedback summariser"
    )
    parser.add_argument(
        "--scenario",
        default="enter",
        choices=list_playtest_scenarios() + ["all"],
        help="Scenario to play (default: enter). Use 'all' to list every scenario.",
    )
    parser.add_argument(
        "--preset",
        default="normal",
        choices=["cautious", "normal", "braver"],
        help="Drummer preset (default: normal).",
    )
    parser.add_argument(
        "--feedback-file",
        default=None,
        help="Path to JSONL feedback file (default: playtest_feedback.jsonl).",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Run diagnostics only (no MIDI playback).",
    )
    parser.add_argument(
        "--summarize-feedback",
        action="store_true",
        help="Analyse existing feedback and print a learning summary (skips interview).",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Export learning summary to JSON file.",
    )
    parser.add_argument(
        "--summary-md",
        default=None,
        help="Export learning summary to Markdown file.",
    )
    parser.add_argument(
        "--update-summary",
        action="store_true",
        help="After each feedback save, regenerate the learning summary and export.",
    )
    parser.add_argument(
        "--sanity-only",
        action="store_true",
        help="Run scenario, check musical sanity, print report, and exit (no interactive questions).",
    )
    parser.add_argument(
        "--detailed-feedback",
        action="store_true",
        help="Use the multi-question detailed questionnaire (default: simple liked/disliked/unsure/golden).",
    )
    parser.add_argument(
        "--bar-transcript",
        action="store_true",
        help="Write a compact per-bar transcript to artifacts/playtests/ for Lucy-readable diagnostics.",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Run Musical Doctor diagnosis after scenario (implies --bar-transcript).",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


_SEP = "=" * 60


# Default feedback file in project root
_DEFAULT_FEEDBACK_PATH = "playtest_feedback.jsonl"


def _resolve_feedback_path(args) -> str:
    if args.feedback_file is not None:
        return args.feedback_file
    return _DEFAULT_FEEDBACK_PATH


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Summarise-feedback mode (skip interview entirely)
    # -----------------------------------------------------------------------
    if args.summarize_feedback:
        feedback_file = _resolve_feedback_path(args)
        print(f"Reading feedback from: {feedback_file}")

        if not os.path.exists(feedback_file):
            print(f"  No feedback file found at {feedback_file}.")
            print(f"  Run a playtest interview first to generate feedback entries.")
            return

        summary = load_and_summarize_feedback(feedback_file)
        _print_console_summary(summary)

        if args.summary_json:
            export_learning_summary_json(summary, args.summary_json)
            print(f"JSON summary written to {args.summary_json}")

        if args.summary_md:
            export_learning_summary_markdown(summary, args.summary_md)
            print(f"Markdown summary written to {args.summary_md}")
        return

    # -----------------------------------------------------------------------
    # Interview mode
    # -----------------------------------------------------------------------
    # Determine scenarios to run
    if args.scenario == "all":
        scenarios: list[PlaytestScenario] = []
        for name in list_playtest_scenarios():
            scenarios.extend(get_scenario_variations(name, preset=args.preset))
    else:
        scenarios = get_scenario_variations(args.scenario, preset=args.preset)

    feedback_path = _resolve_feedback_path(args)

    print(f"\n{_SEP}")
    print(f"Playtest Interview  |  preset={args.preset}")
    print(f"Feedback: {feedback_path}")
    print(_SEP)

    no_play = args.no_play
    bpm = 120.0  # All scenarios currently use 120 BPM

    # Check MIDI availability if not in no-play mode
    if not no_play:
        print("\n[Play mode: will attempt MIDI playback]")
        try:
            from midi_out import MidiOut
            from drummer.pipeline_midi import list_available_ports, find_or_none
            from demo_continuous_jam_midi import _play_global_schedule
            ports = list_available_ports()
            if not ports:
                print("  WARNING: No MIDI ports available.")
                print("  Falling back to --no-play mode. Use --no-play to suppress this warning.")
                no_play = True
        except Exception as e:
            print(f"  WARNING: MIDI not available ({e}).")
            print("  Falling back to --no-play mode.")
            no_play = True
    else:
        print("\n[No-play mode: diagnostics only, no MIDI playback]")

    # Sanity-only: collect all reports, print summary at end
    if args.sanity_only:
        from drummer.musical_sanity import check_musical_sanity, MusicalSanityReport

        all_variation_reports: list[tuple[str, MusicalSanityReport]] = []
        dash = "-" * 60

        for i, sc in enumerate(scenarios, 1):
            print(f"\n{dash}")
            print(f"Variation {i}/{len(scenarios)}: {sc.variation_name}")
            print(f"  {sc.description}")
            print(dash)

            print("\n  Running scenario...")
            summary, raw_diags, global_events = run_playtest_scenario(sc, no_play=no_play)

            bar_events: dict[int, list] = {}
            for evt in global_events:
                bar_events.setdefault(evt.bar_index, []).append(evt)

            sanity_report = MusicalSanityReport()
            for diag in raw_diags:
                bar = diag.get("bar", 0)
                intent = diag.get("intent", "listen")
                events = bar_events.get(bar, [])
                br = check_musical_sanity(intent, events, bar_index=bar)
                sanity_report.issues.extend(br.issues)

            all_variation_reports.append((sc.variation_name, sanity_report))

            # Per-variation output
            if sanity_report.passed:
                print("  Sanity: PASSED")
            else:
                print(f"  Sanity: FAILED — {sanity_report.error_count} errors, "
                      f"{sanity_report.warning_count} warnings")
                for issue in sanity_report.issues:
                    prefix = "ERROR" if issue.severity == "error" else "WARN "
                    bar_label = f"bar {issue.bar_index}" if issue.bar_index is not None else ""
                    print(f"  [{prefix}] {issue.intent} {bar_label}: {issue.message}")

        # Final summary
        total_passed = sum(1 for _, r in all_variation_reports if r.passed)
        total_failed = sum(1 for _, r in all_variation_reports if not r.passed)
        total_errors = sum(r.error_count for _, r in all_variation_reports)
        total_warnings = sum(r.warning_count for _, r in all_variation_reports)

        sep = "=" * 60
        print(f"\n{sep}")
        print("  Sanity-only summary")
        print(f"  scenario: {args.scenario}")
        print(f"  preset:   {args.preset}")
        print(f"  variations checked: {len(scenarios)}")
        print(f"  passed:   {total_passed}")
        print(f"  failed:   {total_failed}")
        print(f"  errors:   {total_errors}")
        print(f"  warnings: {total_warnings}")
        if total_failed > 0:
            for name, r in all_variation_reports:
                if not r.passed:
                    print(f"\n  FAILED: {name}")
                    for issue in r.issues:
                        prefix = "ERROR" if issue.severity == "error" else "WARN "
                        bar_label = f"bar {issue.bar_index}" if issue.bar_index is not None else ""
                        print(f"    [{prefix}] {issue.intent} {bar_label}: {issue.message}")
        print(f"{sep}\n")
        return

    # Determine feedback mode
    use_detailed = getattr(args, "detailed_feedback", False)

    # Process each variation (interactive mode)
    dash = "-" * 60
    for i, sc in enumerate(scenarios, 1):
        saved = False
        # Cache scenario run results so replay doesn't re-run
        last_summary: PlaytestDiagnosticsSummary | None = None
        last_raw_diags: list[dict] | None = None
        last_events: list | None = None

        while True:
            if use_detailed:
                # --- Detailed feedback: old verbose output ---
                print(f"\n{dash}")
                print(f"Variation {i}/{len(scenarios)}: {sc.variation_name}")
                print(f"  Preset: {sc.preset}")
                print(f"  {sc.description}")
                print(f"  Listen: {sc.what_to_listen_for}")
                print(f"  Focus bars: {sc.listen_start_bar}-{sc.listen_end_bar}")
                print(dash)
            else:
                # --- Simple feedback: compact output ---
                listen_sentence = sc.what_to_listen_for.split(".")[0] + "." if sc.what_to_listen_for else ""
                print(f"\n  {sc.name.upper()} | preset={sc.preset} | {sc.variation_name}")
                if listen_sentence:
                    print(f"  {listen_sentence.strip()}")
                print(f"  Focus bars: {sc.listen_start_bar}-{sc.listen_end_bar}")

            # Run the scenario (or re-run if we need fresh data)
            if use_detailed:
                print("\n  Running scenario...")
            summary, raw_diags, global_events = run_playtest_scenario(sc, no_play=no_play)
            last_summary = summary
            last_raw_diags = raw_diags
            last_events = global_events

            # --- Musical sanity check ---
            from drummer.musical_sanity import check_musical_sanity, MusicalSanityReport

            bar_events_var: dict[int, list] = {}
            for evt in global_events:
                bar_events_var.setdefault(evt.bar_index, []).append(evt)

            sanity_report = MusicalSanityReport()
            for diag in raw_diags:
                bar = diag.get("bar", 0)
                intent = diag.get("intent", "listen")
                events = bar_events_var.get(bar, [])
                br = check_musical_sanity(intent, events, bar_index=bar)
                sanity_report.issues.extend(br.issues)

            # Store sanity in summary
            summary.musical_sanity_passed = sanity_report.passed
            summary.musical_sanity_errors = sanity_report.error_count
            summary.musical_sanity_warnings = sanity_report.warning_count
            summary.musical_sanity_issues = [i.to_dict() for i in sanity_report.issues]

            if use_detailed:
                # --- Detailed: full output ---
                print(f"\n  --- Musical Sanity ---")
                if sanity_report.passed:
                    print("  Sanity: PASSED")
                else:
                    print(f"  Sanity: FAILED — {sanity_report.error_count} errors, "
                          f"{sanity_report.warning_count} warnings")
                    for issue in sanity_report.issues:
                        prefix = "ERROR" if issue.severity == "error" else "WARN "
                        bar_label = f"bar {issue.bar_index}" if issue.bar_index is not None else ""
                        print(f"  [{prefix}] {issue.intent} {bar_label}: {issue.message}")
                _print_diagnostics_summary(summary, raw_diags, sc)
            else:
                # --- Simple: compact diagnostics ---
                _print_compact_diagnostics(summary, sanity_report.passed)

            # Bar transcript + doctor (if --bar-transcript or --doctor)
            transcript = _write_bar_transcript_if_requested(args, sc, global_events, raw_diags)
            _write_doctor_report_if_requested(args, sc, transcript)

            # MIDI playback if not in no-play mode
            if not no_play and global_events:
                _try_play_midi(global_events, bpm, total_bars=sc.bars)

            if use_detailed:
                # --- Detailed: old replay/answer/skip/quit flow ---
                action = _prompt_replay()

                if action == "r":
                    print("\n  [Replaying...]")
                    if not no_play and last_events:
                        _try_play_midi(last_events, bpm, total_bars=sc.bars)
                    continue  # replay without saving

                if action == "s":
                    print("\n  [Skipped — no feedback saved for this variation]")
                    break  # skip without saving

                if action == "q":
                    print("\n  [Quit requested]")
                    print("\n  Exiting.")
                    return

                # action == "a" — collect detailed answers
                answers = _collect_answers()

                validate_questionnaire_answers(
                    overall_rating=answers.overall_rating,
                    timing_rating=answers.timing_rating,
                    amount_rating=answers.amount_rating,
                    confidence_rating=answers.confidence_rating,
                    understood_rating=answers.understood_rating,
                    suggested_change=answers.suggested_change,
                )

                entry = PlaytestFeedbackEntry(
                    scenario=sc, diagnostics=summary, answers=answers
                )

                append_feedback_entry(feedback_path, entry)
                saved = True
                print(f"\n  Feedback saved to {feedback_path}")

                # Optionally update / export the learning summary
                if args.update_summary:
                    print("\n  Regenerating learning summary...")
                    _update_and_export_summary(
                        feedback_path,
                        args.summary_json,
                        args.summary_md,
                    )

                break  # move to next variation

            else:
                # --- Simple: 2-question feedback ---
                print("\n  Q1 — Overall?")
                print(_Q1_OPTIONS)
                q1 = _prompt_q1()

                if q1 == "r":
                    print("\n  [Replaying...]")
                    if not no_play and last_events:
                        _try_play_midi(last_events, bpm, total_bars=sc.bars)
                    continue  # replay without saving

                if q1 == "q":
                    print("\n  [Quit requested]")
                    print("\n  Exiting.")
                    return

                print("\n  Q2 — Feel score?  1-5  (1 worst, 5 best)")
                score = _prompt_q2()

                print("\n  Q3 (optional) — Obvious issue?")
                print(_Q3_OPTIONS)
                q3 = _prompt_q3()

                # Build answers
                answers = _build_2q_answers(q1, score, q3)

                entry = PlaytestFeedbackEntry(
                    scenario=sc, diagnostics=summary, answers=answers
                )

                append_feedback_entry(feedback_path, entry)
                saved = True

                q1_labels = {"g": "Good", "b": "Bad", "m": "Mixed", "u": "Unsure"}
                label = q1_labels.get(q1, q1)
                print(f"\n  [Saved — {label}, score {score}/5]")

                # Golden reference: if Q1=good and score=5
                if q1 == "g" and score == 5:
                    if _prompt_golden_yn():
                        ts = datetime.datetime.now().isoformat()
                        _save_golden_for_scenario(sc, raw_diags, ts)

                # Optionally update / export the learning summary
                if args.update_summary:
                    print("\n  Regenerating learning summary...")
                    _update_and_export_summary(
                        feedback_path,
                        args.summary_json,
                        args.summary_md,
                    )

                break  # move to next variation

    print(f"\n{_SEP}")
    print("Done.  Thank you!")
    print(f"{_SEP}\n")


if __name__ == "__main__":
    main()
