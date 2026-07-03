# Operational Delay Sentinel

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="Synthetic Data" src="https://img.shields.io/badge/Synthetic%20Data-No%20PHI-0F766E?style=for-the-badge">
  <img alt="Dashboard First" src="https://img.shields.io/badge/Dashboard--first-CSV%20workflow-F59E0B?style=for-the-badge">
  <img alt="Scikit Learn" src="https://img.shields.io/badge/Models-scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white">
  <img alt="License MIT" src="https://img.shields.io/badge/License-MIT-111827?style=for-the-badge">
</p>

<p align="center"><strong>A synthetic hospital-flow app for finding out-of-bounds discharge delays, ranking operational blockers, and turning them into dashboard-ready action lists.</strong></p>

Operational Delay Sentinel is a small, self-contained prototype for hospital operations teams who need to answer a deceptively hard question:

> Which discharge delays are no longer normal clinical variation, and which operational blocker should we look at first?

It generates synthetic hospital-flow data, flags out-of-bounds operational delay signals, attributes likely operational constraints, trains local statistical baselines, and produces an interactive HTML dashboard plus CSV worklists. It is intentionally **dashboard-first and CSV-first**: no EHR write-back, no PHI, no punitive language, and no dependency on a live hospital system.

The important design choice is signal compression: thousands of patient-level rows are rolled into **3-5 management signal groups per shift**, while case counts and recoverable bed-hours are reported separately. A blocker can be high-impact without being high-volume, and common service throughput is not treated as a crisis just because it is frequent.

## Why this exists

Hospital discharge delays are not just statistical outliers. A five-day miss on predicted discharge timing may be caused by an ALC/community placement wait, Friday service closure, delayed imaging, late blood results, ECG access, transport coordination, or home-care confirmation. Treating all of that as model error hides the operational story.

This app reframes those misses as **reviewable delay signals**:

- soft operational language,
- ranked recoverable bed-hours,
- clear recommended owners,
- auditable CSV action lists,
- dashboard filters for facility, unit, signal family, and exact blocker.

## Screenshots

### Balanced one-hospital run, mixed signals with weekend/community pressure

![Balanced dashboard](docs/screenshots/balanced-dashboard.png)

### Diagnostics-heavy run, detected as diagnostics-access dominant

![Diagnostics dashboard](docs/screenshots/diagnostics-dashboard.png)

### Weekend-flow run, detected as Friday/weekend gap dominant

![Weekend dashboard](docs/screenshots/weekend-dashboard.png)

## What it detects

The current synthetic workflow ranks blockers such as:

- ALC placement wait,
- Friday/weekend discharge gap,
- home-care confirmation wait,
- therapy assessment stall,
- transport delay,
- pharmacy discharge delay,
- radiology CT turnaround delay,
- radiology MRI access delay,
- radiology ultrasound turnaround delay,
- blood testing turnaround delay,
- ECG availability delay,
- diagnostic sign-off stall,
- unit-level bed-flow bottleneck,
- triage screening backlog,
- nurse discharge screening wait,
- vulnerable patient porter wait,
- social work assessment wait,
- interpreter availability wait,
- discharge documentation readiness wait.

## How the sentinel works

```mermaid
flowchart LR
  A["Synthetic admissions"] --> B["Journey events"]
  B --> C["OOB delay detection"]
  C --> D["Blocker attribution"]
  D --> E["Ranked signals"]
  E --> F["Shift-level management signal groups"]
  E --> G["Dashboard + CSV exports"]
  C --> H["Model baselines"]
```

The detection layer uses three complementary signals:

- robust length-of-stay control limits by facility, service line, case mix group, and frailty band,
- post-medically-ready hard caps,
- daily control-chart signals for systemic unit/facility patterns.

The action layer keeps three views:

- `delay_blocker_attribution.parquet`: raw patient-level evidence,
- `management_signal_groups.csv`: capped shift-level management agenda, default 5 groups per shift,
- `delay_resolution_actions.csv`: supporting patient examples for the selected management groups.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run the one-large-hospital balanced synthetic case:

```bash
python3 run_discharge_delay_workflow.py \
  --facilities 1 \
  --days 90 \
  --encounters-per-day 110 \
  --oob-rate-target 0.05 \
  --post-ready-hard-cap-hours 48 \
  --weekend-service-reduction 0.35 \
  --alc-pressure-multiplier 1.25 \
  --scenario-mode balanced \
  --out outputs/synthetic_90d_large_hospital_balanced_v1 \
  --print-top-n 5
```

Run the diagnostics-heavy and weekend-flow scenarios:

```bash
python3 run_discharge_delay_workflow.py --scenario-mode diagnostics_heavy --out outputs/synthetic_90d_large_hospital_diagnostics_v1
python3 run_discharge_delay_workflow.py --scenario-mode weekend_flow_gap --out outputs/synthetic_90d_large_hospital_weekend_v1
```

## Scenario modes

| Mode | Purpose |
|---|---|
| `balanced` | Mixed operational delay pressure. Useful default demo. |
| `alc_heavy` | ALC, LTC, rehab, home care, and community-capacity pressures dominate. |
| `diagnostics_heavy` | Radiology, blood testing, ECG, and diagnostic sign-off delays dominate. |
| `weekend_flow_gap` | Friday/weekend service gaps dominate recoverable bed-hours. |

The app also detects the scenario actually produced by the data in `scenario_detection_summary.csv`. That matters because the requested scenario and the observed signal mix can differ.

## Latest synthetic run results

All three latest runs use one large synthetic hospital, 90 days, and 9,900 admissions. Earlier health-authority-scale stress runs are intentionally not used as the public demo baseline.

| Scenario mode | Detected scenario | Raw OOB signals | Management groups | Avg groups / shift | Compression | ALC cases | ALC case rate | Top signal family | Family share |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| `balanced` | `weekend_flow_gap_detected` | 811 | 543 | 2.25 | 1.49x | 43 | 0.43% | `weekend_flow_gap` | 34.19% |
| `diagnostics_heavy` | `diagnostics_heavy_detected` | 1,467 | 745 | 2.79 | 1.97x | 7 | 0.07% | `diagnostics_access` | 54.95% |
| `weekend_flow_gap` | `weekend_flow_gap_detected` | 952 | 373 | 1.55 | 2.55x | 14 | 0.14% | `weekend_flow_gap` | 94.78% |

The latest runs intentionally separate **raw evidence volume**, **case rates**, and **recoverable excess bed-hours**. Raw OOB rows remain available for audit, but the dashboard caps the shift-level agenda to a manageable number of signal groups. This prevents normal large-system friction, such as radiology throughput at scale, from being mislabeled as a crisis simply because it is frequent. ALC/community-capacity rows are bounded as a small high-impact subset rather than a structurally dominant source of synthetic delay.

Full run summary:

- `docs/scenario_run_summary.csv`
- `docs/model_metrics_summary.csv`

### One-hospital interpretation

The public demo is intentionally sized like a single large hospital, not a provincial or health-authority-wide extract. The current default produces:

- `9,900` admissions over 90 days,
- `811` to `1,467` raw OOB delay signals depending on scenario,
- `373` to `745` shift-level management groups,
- about `1.55` to `2.79` management groups per shift,
- ALC case rates between `0.07%` and `0.43%`.

That distinction matters: ALC is represented as a small, high-impact subset. Recoverable excess bed-hours are not case counts.

## Dashboard features

The generated dashboard includes:

- KPI cards,
- recoverable excess bed-hours bar chart,
- OOB trend SVG chart,
- scenario detection mix,
- ranked actionable signals,
- executive worklist,
- control-chart daily metrics,
- filters for facility, unit, signal family, and exact operational signal,
- clickable bars that filter the tables,
- visible-row CSV export buttons,
- print/save-PDF support.

## Dashboard screenshot/export

Install screenshot support:

```bash
pip install '.[screenshot]'
python3 -m playwright install chromium
```

Export dashboard HTML and PNG:

```bash
python3 scripts/export_dashboard.py \
  --dashboard outputs/synthetic_90d_large_hospital_weekend_v1/operational_delay_dashboard.html \
  --out exports \
  --png
```


## GitHub repository structure

```text
.
├── README.md
├── IMPLEMENTATION_PLAN.md
├── pyproject.toml
├── requirements.txt
├── run_discharge_delay_workflow.py
├── scripts/
│   └── export_dashboard.py
├── src/discharge_delays/
│   └── workflow.py
├── docs/
│   ├── scenario_run_summary.csv
│   ├── model_metrics_summary.csv
│   └── screenshots/
│       ├── balanced-dashboard.png
│       ├── diagnostics-dashboard.png
│       └── weekend-dashboard.png
└── .github/workflows/ci.yml
```

Generated `outputs/`, `exports/`, local data, virtual environments, and timestamped screenshot exports are ignored. The repository keeps only curated screenshots and summary CSVs for the README.

## Main outputs

| File | Purpose |
|---|---|
| `patient_admission_events.parquet` | Synthetic admission-level table. |
| `patient_journey_events.parquet` | Synthetic event-level journey table. |
| `bed_resource_daily.parquet` | Bed occupancy and resource context. |
| `service_availability.parquet` | PT, OT, imaging, pharmacy, home care, transport, LTC, rehab availability. |
| `out_of_bounds_delay_flags.parquet` | OOB delay signals and priority scores. |
| `delay_blocker_attribution.parquet` | Likely blocker attribution and evidence. |
| `ranked_actionable_signals.csv` | Raw ranked facility/unit/service/signal pattern table. |
| `management_signal_groups.csv` | Capped 3-5-per-shift management agenda. |
| `management_signal_kpis.csv` | Signal compression and manageability KPIs. |
| `delay_resolution_actions.csv` | Supporting patient-level examples for management groups. |
| `delay_resolution_actions_all.csv` | Full raw action inventory. |
| `scenario_detection_summary.csv` | Detected scenario mix by signal family. |
| `operational_delay_dashboard.html` | Interactive local dashboard. |
| `discharge_delay_sentinel_report.md` | Markdown run report. |
| `discharge_delay_sentinel_report.html` | HTML run report. |

## Model baselines

The workflow trains local statistical baselines:

- `HistGradientBoostingRegressor`,
- `ExtraTreesRegressor`,
- `HistGradientBoostingClassifier`,
- `ExtraTreesClassifier`.

The models are not the whole product. They are used to estimate expected LOS and OOB risk, while the operational layer converts delay signals into explainable blockers and worklists.

## Language and governance

This project deliberately avoids punitive terminology. It uses terms like:

- delay signal,
- operational blocker,
- capacity constraint,
- unresolved discharge dependency,
- recoverable excess bed-hours.

The intended first deployment pattern is shadow mode:

1. generate dashboard and CSV outputs,
2. review with discharge huddles and patient-flow teams,
3. record adoption and reasons-not-actioned,
4. only later consider idempotent integration with operational systems.

## Synthetic data only

This repository uses fully synthetic data. It contains no PHI and makes no claim about a specific real hospital, health authority, or provincial program.

## Cute but serious roadmap

- Add a proper web front end around the generated dashboard.
- Add adoption simulation: accepted, deferred, already resolved, not actionable.
- Add idempotent recommendation IDs for repeated daily runs.
- Add scenario comparison pages.
- Add optional integration adapters for CSV/SFTP/data-warehouse handoff.

## License

MIT.
