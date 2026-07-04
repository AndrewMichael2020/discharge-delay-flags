#!/usr/bin/env python3
"""Build static GitHub Pages demo data for Discharge Delay Flags."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

SCENARIOS = ["balanced", "diagnostics_heavy", "weekend_flow_gap"]
SCENARIO_LABELS = {
    "balanced": "Balanced operating day",
    "diagnostics_heavy": "Diagnostics pressure",
    "weekend_flow_gap": "Weekend flow gap",
}
SCENARIO_DESCRIPTIONS = {
    "balanced": "Mixed operating pressure across discharge logistics, community capacity, diagnostics, and weekend flow.",
    "diagnostics_heavy": "A stress view where imaging, lab, ECG, and diagnostic sign-off signals carry more recoverable bed-hours.",
    "weekend_flow_gap": "A stress view where Friday/weekend service timing dominates recoverable bed-hour impact.",
}
BUILD_ID = "20260704-ods-v3"
ROOT_FILES = ["docs/scenario_run_summary.csv", "docs/model_metrics_summary.csv"]
SCENARIO_FILES = [
    "management_signal_groups.csv",
    "management_signal_kpis.csv",
    "ranked_actionable_signals.csv",
    "delay_resolution_actions.csv",
    "control_chart_points.csv",
    "scenario_detection_summary.csv",
]


def title_case(value: Any) -> str:
    return str(value or "").replace("_", " ").replace("/", " / ").title()


def clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        return round(value, 4)
    return value


def records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [{k: clean_value(v) for k, v in row.items()} for row in df.to_dict(orient="records")]


def require_files(root: Path) -> None:
    missing = [f for f in ROOT_FILES if not (root / f).exists()]
    for scenario in SCENARIOS:
        for name in SCENARIO_FILES:
            rel = f"sample_data/{scenario}/{name}"
            if not (root / rel).exists():
                missing.append(rel)
    if missing:
        raise SystemExit("Curated demo source data is missing:\n" + "\n".join(f"- {m}" for m in missing))


def split_multi(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def display_row(row: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in row.items():
        new_key = {
            "primary_blocker": "delay_reason",
            "top_blocker": "top_delay_reason",
            "recommended_management_action": "recommended_action",
        }.get(key, key)
        out[new_key] = clean_value(value)
    for raw_key, display_key in [
        ("signal_family", "signal_family_display"),
        ("recommended_owner", "recommended_owner_display"),
        ("unit_id", "unit_display"),
        ("service_line", "service_line_display"),
        ("shift_name", "shift_display"),
        ("priority", "priority_display"),
    ]:
        if raw_key in out:
            out[display_key] = title_case(out[raw_key])
    if "delay_reason" in out:
        out["delay_reason_display"] = str(out["delay_reason"] or "").capitalize()

    # Canonical filter arrays make mixed grains behave consistently:
    # management groups use affected_* fields, action rows use unit_id/service_line.
    out["filter_signal_family"] = split_multi(out.get("signal_family"))
    out["filter_delay_reason"] = split_multi(out.get("delay_reason"))
    out["filter_unit"] = split_multi(out.get("unit_id")) or split_multi(out.get("affected_units"))
    out["filter_service_line"] = split_multi(out.get("service_line")) or split_multi(out.get("affected_service_lines"))
    out["filter_shift"] = split_multi(out.get("shift_name"))
    if out.get("priority"):
        priority = str(out["priority"])
    else:
        priority = "high" if float(out.get("median_priority_score") or 0) >= 25 else "routine"
        out["priority"] = priority
        out["priority_display"] = title_case(priority)
    out["filter_priority"] = [priority]
    out["filter_owner"] = split_multi(out.get("recommended_owner"))
    return out


def display_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [display_row(row) for row in df.to_dict(orient="records")]


def filter_options(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, list[dict[str, str]]]:
    options: dict[str, list[dict[str, str]]] = {}
    for key in keys:
        seen: dict[str, str] = {}
        for row in rows:
            raw_value = row.get(key)
            values = raw_value if isinstance(raw_value, list) else [raw_value]
            for value in values:
                if value is None or value == "":
                    continue
                seen[str(value)] = title_case(value)
        options[key] = [{"value": value, "label": seen[value]} for value in sorted(seen, key=lambda x: seen[x].lower())]
    return options


def summarize_control(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw = pd.read_csv(path, parse_dates=["date"])
    raw["date"] = raw["date"].dt.strftime("%Y-%m-%d")
    daily = (
        raw.groupby("date", as_index=False)
        .agg(
            discharges=("discharges", "sum"),
            oob_cases=("oob_cases", "sum"),
            post_ready_hard_cap_cases=("post_ready_hard_cap_cases", "sum"),
            centerline_oob_rate=("centerline_oob_rate", "mean"),
            upper_control_limit=("upper_control_limit", "mean"),
            control_chart_signal_flag=("control_chart_signal_flag", "max"),
            estimated_bed_days_recoverable=("estimated_bed_days_recoverable", "sum"),
        )
        .sort_values("date")
    )
    daily["oob_rate"] = daily.apply(lambda r: r.oob_cases / r.discharges if r.discharges else 0, axis=1)
    return records(daily), display_records(raw)


def metric_docs() -> list[dict[str, str]]:
    return [
        {"metric": "Patient encounters", "formula": "count(encounter_id)", "meaning": "Synthetic admissions in the selected case window."},
        {"metric": "OOB signal rate", "formula": "raw_oob_delay_signals / patient_encounters", "meaning": "Share of encounters with at least one out-of-bounds operational delay signal."},
        {"metric": "Management signals per shift", "formula": "management_signal_groups / total_shift_windows", "meaning": "Number of grouped action signals a shift huddle needs to review."},
        {"metric": "Recoverable bed-hours", "formula": "sum(estimated_recoverable_bed_hours)", "meaning": "Estimated excess bed-hours above expected operating thresholds."},
        {"metric": "Compression ratio", "formula": "raw_oob_delay_signals / management_signal_groups", "meaning": "How much patient-level evidence is compressed into shift-level signals."},
        {"metric": "Control-chart signal", "formula": "daily_oob_rate > centerline + 3 * sigma", "meaning": "A unit/day pattern above expected operating variation."},
        {"metric": "Management score", "formula": "0.40*bed_hours + 10*cases + 0.30*priority + 20*confidence", "meaning": "Sorting score for the shift huddle worklist."},
        {"metric": "Priority score", "formula": "0.65*hours_above_limit + 0.55*post_ready_excess + flags", "meaning": "Patient-level sorting aid, not a clinical severity score."},
    ]


def signal_dictionary() -> list[dict[str, str]]:
    return [
        {"reason": "ALC placement wait", "family": "Community capacity", "owner": "Transition services lead", "meaning": "Patient appears medically stable but is waiting for LTC, rehab, or alternate-level placement."},
        {"reason": "Home care confirmation wait", "family": "Community capacity", "owner": "Home-care liaison", "meaning": "Discharge depends on home-care confirmation or community support setup."},
        {"reason": "Friday/weekend discharge gap", "family": "Weekend flow", "owner": "Site operations director", "meaning": "Readiness crosses a period with reduced service availability."},
        {"reason": "Therapy assessment stall", "family": "Care-team assessment", "owner": "Therapy services manager", "meaning": "PT/OT or therapy assessment takes longer than expected for a discharge-dependent case."},
        {"reason": "Transport delay", "family": "Transport logistics", "owner": "Patient transport coordinator", "meaning": "Discharge or transfer waits on transport coordination."},
        {"reason": "Vulnerable patient porter wait", "family": "Transport logistics", "owner": "Patient transport coordinator", "meaning": "Mobility-limited or vulnerable patients wait for safe movement support."},
        {"reason": "Pharmacy discharge delay", "family": "Pharmacy readiness", "owner": "Pharmacy operations lead", "meaning": "Medication reconciliation or discharge medication readiness delays discharge."},
        {"reason": "Radiology CT turnaround delay", "family": "Diagnostics access", "owner": "Radiology operations lead", "meaning": "CT completion or reporting appears discharge-dependent and delayed."},
        {"reason": "Radiology MRI access delay", "family": "Diagnostics access", "owner": "Radiology operations lead", "meaning": "MRI access or reporting appears discharge-dependent and delayed."},
        {"reason": "Blood testing turnaround delay", "family": "Diagnostics access", "owner": "Laboratory operations lead", "meaning": "Blood collection, processing, or result release appears discharge-dependent and delayed."},
        {"reason": "ECG availability delay", "family": "Diagnostics access", "owner": "Cardiology diagnostics lead", "meaning": "ECG completion or interpretation appears discharge-dependent and delayed."},
        {"reason": "Diagnostic sign-off stall", "family": "Diagnostics access", "owner": "Diagnostics operations lead", "meaning": "Final diagnostic completion, sign-off, or consultant interpretation appears to hold discharge."},
        {"reason": "Nurse discharge screening wait", "family": "Nursing readiness", "owner": "Nursing flow lead", "meaning": "Nursing readiness checklist or discharge screening is delayed."},
        {"reason": "Social work assessment wait", "family": "Social support", "owner": "Social work lead", "meaning": "Social work assessment is delayed for a discharge-dependent case."},
        {"reason": "Unattributed operational delay", "family": "Unattributed", "owner": "Patient flow coordinator", "meaning": "A delay signal exists but no single dependency explains it."},
    ]


def build(root: Path, out_dir: Path) -> dict[str, Any]:
    require_files(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_summary = pd.read_csv(root / "docs/scenario_run_summary.csv")
    model_summary = pd.read_csv(root / "docs/model_metrics_summary.csv")
    model_by_scenario = model_summary.set_index("scenario_mode").to_dict(orient="index")

    scenarios = []
    management_rows: dict[str, list[dict[str, Any]]] = {}
    action_rows: dict[str, list[dict[str, Any]]] = {}
    delay_reasons: dict[str, list[dict[str, Any]]] = {}
    control_daily: dict[str, list[dict[str, Any]]] = {}
    control_unit: dict[str, list[dict[str, Any]]] = {}
    filters: dict[str, dict[str, list[dict[str, str]]]] = {}

    for scenario in SCENARIOS:
        summary = run_summary.loc[run_summary["scenario_mode"] == scenario].iloc[0].to_dict()
        model = model_by_scenario.get(scenario, {})
        scenarios.append({
            "scenario_mode": scenario,
            "label": SCENARIO_LABELS[scenario],
            "description": SCENARIO_DESCRIPTIONS[scenario],
            "admissions": int(summary["admissions"]),
            "journey_events": int(summary["journey_events"]),
            "raw_oob_delay_signals": int(summary["raw_oob_delay_signals"]),
            "oob_signal_rate": clean_value(summary["oob_signal_rate"]),
            "management_signal_groups": int(summary["management_signal_groups"]),
            "avg_signal_groups_per_shift": clean_value(summary["avg_signal_groups_per_shift"]),
            "compression_ratio": clean_value(summary["compression_ratio"]),
            "detected_scenario": summary["detected_scenario"],
            "top_signal_family": summary["top_signal_family"],
            "top_family_share": clean_value(summary["top_family_share"]),
            "top_management_delay_reason": summary["top_management_delay_reason"],
            "top_raw_delay_reason": summary["top_raw_delay_reason"],
            "roc_auc": clean_value(model.get("roc_auc")),
            "pr_auc": clean_value(model.get("pr_auc")),
            "recall_at_top_5pct_risk": clean_value(model.get("recall_at_top_5pct_risk")),
            "los_mae_hours": clean_value(model.get("los_mae_hours")),
            "los_rmse_hours": clean_value(model.get("los_rmse_hours")),
        })

        scenario_dir = root / "sample_data" / scenario
        groups = pd.read_csv(scenario_dir / "management_signal_groups.csv").sort_values(["signal_group_rank", "management_score"], ascending=[True, False])
        actions = pd.read_csv(scenario_dir / "delay_resolution_actions.csv").sort_values("executive_rank")
        ranked = pd.read_csv(scenario_dir / "ranked_actionable_signals.csv")
        daily, unit = summarize_control(scenario_dir / "control_chart_points.csv")

        management_rows[scenario] = display_records(groups)
        action_rows[scenario] = display_records(actions)
        control_daily[scenario] = daily
        control_unit[scenario] = unit

        reason_summary = (
            ranked.groupby(["primary_blocker", "signal_family"], as_index=False)
            .agg(
                flagged_cases=("flagged_cases", "sum"),
                recoverable_bed_hours=("recoverable_bed_hours", "sum"),
                median_priority_score=("median_priority_score", "median"),
                mean_confidence=("mean_confidence", "mean"),
                actionability_score=("actionability_score", "sum"),
            )
            .sort_values(["recoverable_bed_hours", "actionability_score"], ascending=False)
        )
        delay_reasons[scenario] = display_records(reason_summary.rename(columns={"primary_blocker": "delay_reason"}))

        combined = management_rows[scenario] + action_rows[scenario]
        filters[scenario] = filter_options(combined, [
            "filter_signal_family",
            "filter_delay_reason",
            "filter_unit",
            "filter_service_line",
            "filter_shift",
            "filter_priority",
            "filter_owner",
        ])
        dates = sorted({str(row.get("shift_date")) for row in combined if row.get("shift_date")})
        filters[scenario]["date"] = [{"value": d, "label": d} for d in dates]

    payloads = {
        "demo_summary": {
            "title": "Discharge Delay Flags Demo",
            "build_id": BUILD_ID,
            "default_scenario": "balanced",
            "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "source": "Curated synthetic sample data in sample_data/. No PHI or real patient data.",
            "metric_glossary": metric_docs()[:5],
        },
        "scenarios": scenarios,
        "filters": filters,
        "delay_reasons": delay_reasons,
        "management_worklist": management_rows,
        "control_chart": control_daily,
        "control_chart_unit": control_unit,
        "action_examples": action_rows,
        "docs": {
            "metrics": metric_docs(),
            "signals": signal_dictionary(),
            "datasets": [
                {"name": "patient_admission_events.parquet", "type": "Parquet", "grain": "Encounter", "description": "Synthetic admission-level case facts."},
                {"name": "patient_journey_events.parquet", "type": "Parquet", "grain": "Event", "description": "Journey timestamps, transitions, and dependency events."},
                {"name": "management_signal_groups.csv", "type": "CSV", "grain": "Shift signal", "description": "Grouped huddle-ready management signals."},
                {"name": "delay_resolution_actions.csv", "type": "CSV", "grain": "Action row", "description": "Owner, evidence, priority, and suggested next step."},
                {"name": "control_chart_points.csv", "type": "CSV", "grain": "Facility/unit/day", "description": "Daily signal-rate points and review thresholds."},
                {"name": "ranked_actionable_signals.csv", "type": "CSV", "grain": "Delay reason/unit", "description": "Impact-ranked delay reasons by recoverable bed-hours."},
                {"name": "prediction_metrics.json", "type": "JSON", "grain": "Scenario", "description": "Local statistical model quality summary."},
            ],
        },
    }

    for name, payload in payloads.items():
        (out_dir / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "demo_bundle.js").write_text("window.DEMO_DATA = " + json.dumps(payloads, indent=2) + ";\n", encoding="utf-8")
    return payloads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build static JSON payloads for the GitHub Pages demo.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out = args.out.resolve() if args.out else root / "demo" / "data"
    payloads = build(root, out)
    print(f"Demo data written to {out}")
    print(f"Scenarios: {len(payloads['scenarios'])}")
    print(f"Build ID: {payloads['demo_summary']['build_id']}")


if __name__ == "__main__":
    main()
