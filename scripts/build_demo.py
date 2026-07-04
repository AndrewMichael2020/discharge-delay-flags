#!/usr/bin/env python3
"""Build static GitHub Pages demo data for Operational Delay Sentinel.

The demo intentionally uses compact JSON extracts from curated sample CSVs so
GitHub Pages can render a clear product showcase without Python, Parquet, or a
backend service.
"""
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
    "balanced": "Mixed operational pressure in one large synthetic hospital over 90 days.",
    "diagnostics_heavy": "A stress case where imaging, lab, ECG, and sign-off timing drive more recoverable bed-hours.",
    "weekend_flow_gap": "A stress case where Friday and weekend service gaps dominate the action story.",
}
REQUIRED_ROOT_FILES = [
    "docs/scenario_run_summary.csv",
    "docs/model_metrics_summary.csv",
]
REQUIRED_SCENARIO_FILES = [
    "management_signal_groups.csv",
    "management_signal_kpis.csv",
    "ranked_actionable_signals.csv",
    "delay_resolution_actions.csv",
    "control_chart_points.csv",
    "scenario_detection_summary.csv",
]
DISPLAY_FIELD_NAMES = {
    "primary_blocker": "delay_reason",
    "top_blocker": "top_delay_reason",
}


def clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        return round(value, 4)
    return value


def records(df: pd.DataFrame) -> list[dict[str, Any]]:
    renamed = df.rename(columns=DISPLAY_FIELD_NAMES)
    return [{k: clean_value(v) for k, v in row.items()} for row in renamed.to_dict(orient="records")]


def require_files(root: Path) -> None:
    missing: list[str] = []
    for rel in REQUIRED_ROOT_FILES:
        if not (root / rel).exists():
            missing.append(rel)
    for scenario in SCENARIOS:
        for name in REQUIRED_SCENARIO_FILES:
            rel = f"sample_data/{scenario}/{name}"
            if not (root / rel).exists():
                missing.append(rel)
    if missing:
        formatted = "\n".join(f"- {m}" for m in missing)
        raise SystemExit(f"Curated demo source data is missing:\n{formatted}")


def summarize_control_chart(path: Path) -> list[dict[str, Any]]:
    df = pd.read_csv(path, parse_dates=["date"])
    day = (
        df.groupby("date", as_index=False)
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
    day["oob_rate"] = day.apply(lambda r: r.oob_cases / r.discharges if r.discharges else 0, axis=1)
    day["date"] = day["date"].dt.strftime("%Y-%m-%d")
    return records(day.tail(90))


def build(root: Path, out_dir: Path, top_n: int) -> dict[str, Any]:
    require_files(root)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_summary = pd.read_csv(root / "docs/scenario_run_summary.csv")
    model_summary = pd.read_csv(root / "docs/model_metrics_summary.csv")
    model_by_scenario = model_summary.set_index("scenario_mode").to_dict(orient="index")

    scenario_cards = []
    delay_reasons: dict[str, list[dict[str, Any]]] = {}
    management_worklist: dict[str, list[dict[str, Any]]] = {}
    control_chart: dict[str, list[dict[str, Any]]] = {}
    action_examples: dict[str, list[dict[str, Any]]] = {}

    for scenario in SCENARIOS:
        row = run_summary.loc[run_summary["scenario_mode"] == scenario]
        if row.empty:
            raise SystemExit(f"Scenario {scenario} is missing from docs/scenario_run_summary.csv")
        summary = row.iloc[0].to_dict()
        model = model_by_scenario.get(scenario, {})
        scenario_cards.append(
            {
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
            }
        )

        scenario_dir = root / "sample_data" / scenario
        ranked = pd.read_csv(scenario_dir / "ranked_actionable_signals.csv")
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
            .head(12)
        )
        delay_reasons[scenario] = records(reason_summary)

        groups = pd.read_csv(scenario_dir / "management_signal_groups.csv")
        groups = groups.sort_values(["signal_group_rank", "management_score"], ascending=[True, False]).head(top_n)
        management_worklist[scenario] = records(groups)

        actions = pd.read_csv(scenario_dir / "delay_resolution_actions.csv")
        action_cols = [
            "executive_rank",
            "action_id",
            "facility_id",
            "unit_id",
            "service_line",
            "priority",
            "primary_blocker",
            "evidence_summary",
            "recommended_owner",
            "recommended_action",
            "target_resolution_hours",
            "estimated_recoverable_bed_hours",
            "shift_date",
            "shift_name",
            "signal_family",
        ]
        actions = actions[action_cols].sort_values("executive_rank").head(top_n)
        action_examples[scenario] = records(actions)

        control_chart[scenario] = summarize_control_chart(scenario_dir / "control_chart_points.csv")

    demo_summary = {
        "title": "Discharge Delay Flags Demo",
        "subtitle": "A GitHub Pages case study for finding the few discharge-delay signals worth acting on today.",
        "default_scenario": "balanced",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source": "Curated synthetic sample data in sample_data/. No PHI or real patient data.",
        "method_note": "Raw encounter-level signals are compressed into a small shift-level action agenda so operational teams see a few meaningful signals, not a wall of normal friction.",
        "metric_glossary": [
            {"metric": "Out-of-bounds signal rate", "meaning": "Share of encounters that crossed at least one robust delay threshold."},
            {"metric": "Management signals per shift", "meaning": "Average number of grouped signals a shift team would review."},
            {"metric": "Recoverable bed-hours", "meaning": "Estimated bed-hours above expected operational thresholds, used to rank impact."},
            {"metric": "Compression ratio", "meaning": "How many raw patient-level signals are summarized into each management signal."},
            {"metric": "Control-chart signal", "meaning": "A day where unit-level delay activity is above expected variation."},
        ],
    }

    payloads = {
        "demo_summary": demo_summary,
        "scenarios": scenario_cards,
        "delay_reasons": delay_reasons,
        "management_worklist": management_worklist,
        "control_chart": control_chart,
        "action_examples": action_examples,
    }
    for name, payload in payloads.items():
        (out_dir / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    bundle = "window.DEMO_DATA = " + json.dumps(payloads, indent=2) + ";\n"
    (out_dir / "demo_bundle.js").write_text(bundle, encoding="utf-8")
    return payloads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build static JSON payloads for the GitHub Pages demo.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out = args.out.resolve() if args.out else root / "demo" / "data"
    payloads = build(root, out, args.top_n)
    print(f"Demo data written to {out}")
    print(f"Scenarios: {len(payloads['scenarios'])}")
    print(f"Default scenario: {payloads['demo_summary']['default_scenario']}")


if __name__ == "__main__":
    main()
