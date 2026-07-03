# Metrics and Signal Dictionary

This document defines the metrics, formulas, and operational delay signals used by **Discharge Delay Sentinel**.

The dashboard is intentionally built around two layers:

1. **Raw patient-level evidence**: encounter-level delay flags used for audit and traceability.
2. **Management signal groups**: shift-level grouped signals intended for huddles and operational review.

These are KPIs, but with an important distinction: most are **operational diagnostic KPIs**, not final performance outcomes by themselves. They tell teams where to look, what kind of delay pattern is emerging, and whether the signal is large enough to act on.

## KPI framework

### Primary operating KPI

| KPI | Formula | Interpretation | Good direction |
|---|---|---|---|
| OOB signal rate | `raw_oob_delay_signals / patient_encounters` | Share of encounters with at least one out-of-bounds operational delay signal. | Lower, if coding and detection are stable. |
| Management signal groups per shift | `management_signal_groups / total_shift_windows` | Number of grouped action signals a shift huddle needs to review. | Usually 3-5 or fewer. |
| Recoverable excess bed-hours | `sum(estimated_recoverable_bed_hours)` | Estimated bed-hours above expected operating thresholds. This is not a case count. | Lower. |

### Driver KPIs

| KPI | Formula | Interpretation | Good direction |
|---|---|---|---|
| Delay-reason case rate | `flagged_cases_for_delay_reason / patient_encounters` | How often a specific delay reason appears. | Usually below 1-5% per reason in this synthetic demo. |
| Delay-reason bed-hour share | `recoverable_bed_hours_for_delay_reason / total_recoverable_bed_hours` | How much of the estimated recoverable excess is concentrated in one signal family. | Lower concentration unless running a known stress scenario. |
| Average recoverable hours per case | `recoverable_bed_hours_for_delay_reason / flagged_cases_for_delay_reason` | Whether a delay reason is rare but high-impact, or common but low-impact. | Context-dependent. |
| Compression ratio | `raw_oob_delay_signals / management_signal_groups` | How much patient-level evidence is compressed into huddle-level signals. | High enough to reduce noise, but not so high that context is lost. |

### Guardrail KPIs

| Guardrail | Formula / rule | Why it matters |
|---|---|---|
| Signal-volume guardrail | No delay-reason group should be more than about 5% of all encounters in the public one-hospital demo. | Prevents synthetic data from making normal operating friction look like a crisis. |
| Shift workload guardrail | `avg_signal_groups_per_shift <= 5` by default. | Keeps the worklist manageable for real huddles. |
| ALC realism guardrail | ALC should be a small high-impact subset, not a dominant volume. | ALC waits matter, but they should not exceed plausible case rates in a general hospital demo. |
| Mixed-pressure label | If the largest signal family is below 40% of recoverable excess bed-hours, label the run as mixed. | Avoids overclaiming a single dominant cause. |

## Core formulas

### Patient encounters

```text
patient_encounters = count(patient_admission_events.encounter_id)
```

In the public demo this is one large synthetic hospital:

```text
1 facility * 90 days * 110 encounters/day = 9,900 encounters
```

### Robust LOS threshold

The model builds segment-specific LOS limits by:

```text
segment = facility_id + service_line + case_mix_group + frailty_band
```

For each segment:

```text
median_los = median(actual_los_hours)
mad_los = median(abs(actual_los_hours - median_los))
robust_sigma = 1.4826 * mad_los
robust_threshold_width = max(2 * robust_sigma, 3 * mad_los)
oob_limit_hours = median_los + robust_threshold_width
```

An encounter receives a robust LOS signal when:

```text
actual_los_hours > oob_limit_hours
```

### Post-medically-ready hard-cap signal

Default hard cap:

```text
post_ready_hard_cap_hours = 48
```

Formula:

```text
hours_after_medically_ready = discharge_timestamp - medically_ready_timestamp
post_ready_excess_hours = max(hours_after_medically_ready - post_ready_hard_cap_hours, 0)
post_ready_hard_cap_flag = hours_after_medically_ready > post_ready_hard_cap_hours
```

Interpretation:

This signal focuses on patients who remain in hospital after they are medically ready. It should be interpreted as a discharge-dependency signal, not as a clinical judgment.

### Control-chart signal

For each facility/unit/day:

```text
daily_oob_rate = daily_oob_cases / daily_discharges
centerline = mean(daily_oob_rate for facility/unit)
sigma = stddev(daily_oob_rate for facility/unit)
upper_control_limit = centerline + 3 * sigma
control_chart_signal_flag = daily_oob_rate > upper_control_limit
```

Interpretation:

This catches unusual day-level patterns that may be operationally meaningful even when individual cases are borderline.

### Overall OOB flag

```text
oob_flag = robust_los_oob_flag OR post_ready_hard_cap_flag OR control_chart_signal_flag
```

### Priority score

```text
priority_score =
  0.65 * hours_above_limit
+ 0.55 * post_ready_excess_hours
+ 18   * post_ready_hard_cap_flag
+ 8    * control_chart_signal_flag
```

Interpretation:

The score is a sorting aid. It is not a clinical severity score. It prioritizes operational excess time, post-ready delay, and systemic unit/day signals.

### Estimated recoverable excess bed-hours

For each attributed OOB row:

```text
los_excess = max(actual_los_hours - oob_limit_hours, 0)
post_ready_excess = max(hours_after_medically_ready - post_ready_hard_cap_hours, 0)

if post_ready_excess > 0:
    estimated_recoverable_bed_hours = min(los_excess, post_ready_excess)
else:
    estimated_recoverable_bed_hours = 0.35 * los_excess
```

Interpretation:

This is a conservative operational estimate of excess bed-hours that may be worth review. It is not a promise that all hours are recoverable.

## Management signal grouping

Raw patient-level evidence is grouped into shift-level huddle signals.

Shift assignment:

```text
shift_date = date(discharge_timestamp)
shift_name =
  day      if discharge hour is 07:00-14:59
  evening  if discharge hour is 15:00-22:59
  night    otherwise
```

Grouping key:

```text
shift_date + shift_name + signal_family + delay_reason
```

Management score:

```text
management_score =
  0.40 * recoverable_bed_hours
+ 10.0 * flagged_cases
+ 0.30 * median_priority_score
+ 20.0 * mean_confidence
```

Default cap:

```text
max_signal_groups_per_shift = 5
```

Interpretation:

This design keeps the dashboard useful for operational huddles. The patient-level rows remain available for audit, but the huddle sees a short ranked agenda.

## Signal families

| Signal family | Plain meaning | Examples |
|---|---|---|
| `alc_community_capacity` | Discharge depends on community, LTC, rehab, or home-care capacity. | ALC placement wait, home-care confirmation wait. |
| `weekend_flow_gap` | Discharge readiness crosses a period with reduced service availability. | Friday/weekend discharge gap. |
| `diagnostics_access` | Discharge depends on diagnostic completion, reporting, or sign-off. | CT, MRI, ultrasound, blood testing, ECG, diagnostic sign-off. |
| `care_team_assessment` | Discharge depends on therapy or care-team assessment. | Therapy assessment stall. |
| `transport_discharge_logistics` | Discharge depends on transport or porter coordination. | Transport delay, vulnerable patient porter wait. |
| `pharmacy_discharge_readiness` | Discharge depends on medication reconciliation or pharmacy readiness. | Pharmacy discharge delay. |
| `front_door_screening` | Early screening creates downstream flow pressure. | Triage screening backlog. |
| `nursing_discharge_readiness` | Discharge depends on nursing readiness review or checklist completion. | Nurse discharge screening wait. |
| `social_support_readiness` | Discharge depends on social supports or communication services. | Social work assessment wait, interpreter availability wait. |
| `documentation_readiness` | Discharge depends on paperwork or documentation readiness. | Discharge documentation readiness wait. |
| `unit_flow_capacity` | Unit-level throughput pattern crosses a control-chart threshold. | Unit-level bed-flow bottleneck. |
| `unattributed` | Delay signal exists but no single dependency explains it. | Unattributed operational delay. |

## Delay reason dictionary

| Delay reason | Signal family | What it means | Typical evidence | Default owner | Caveat |
|---|---|---|---|---|---|
| ALC placement wait | `alc_community_capacity` | Patient appears medically stable but is waiting for LTC, rehab, or alternate-level placement. | ALC status, LTC/rehab referral timing, discharge disposition. | Transition services lead | ALC is high-impact but should remain a small subset in the synthetic demo. |
| Home care confirmation wait | `alc_community_capacity` | Discharge depends on home-care confirmation or community support setup. | Home-care referral and confirmation timestamps. | Home-care liaison | May overlap with frailty or social-support needs. |
| Friday/weekend discharge gap | `weekend_flow_gap` | Patient becomes ready near a period with reduced service availability. | Medically ready timing near Friday/weekend and delayed discharge. | Site operations director | Not every weekend discharge is a problem; signal rate is capped for realism. |
| Therapy assessment stall | `care_team_assessment` | PT/OT or therapy assessment takes longer than expected for a discharge-dependent case. | Assessment requested/completed timestamps. | Therapy services manager | Some delays may be clinically appropriate. |
| Transport delay | `transport_discharge_logistics` | Discharge or transfer waits on transport coordination. | Transport requested/completed timestamps. | Patient transport coordinator | May reflect external ambulance or family pickup dependencies. |
| Vulnerable patient porter wait | `vulnerable_patient_flow` | Mobility-limited or vulnerable patients wait for safe movement support. | Porter request/completion timestamps and frailty context. | Patient transport coordinator | Should be interpreted with accessibility and safety context. |
| Pharmacy discharge delay | `pharmacy_discharge_readiness` | Medication reconciliation or discharge medication readiness delays discharge. | Discharge order and pharmacy dependency timing. | Pharmacy operations lead | Some complex medication reviews are appropriate. |
| Radiology CT turnaround delay | `diagnostics_access` | CT completion or reporting appears discharge-dependent and delayed. | CT ordered/completed timestamps. | Radiology operations lead | Frequent imaging is normal; only threshold-crossing cases should signal. |
| Radiology MRI access delay | `diagnostics_access` | MRI access or reporting appears discharge-dependent and delayed. | MRI ordered/completed timestamps. | Radiology operations lead | MRI can be scarce; signal should guide review, not blame. |
| Radiology ultrasound turnaround delay | `diagnostics_access` | Ultrasound completion or reporting appears discharge-dependent and delayed. | Ultrasound ordered/completed timestamps. | Radiology operations lead | Some waits are due to appropriate prioritization. |
| Blood testing turnaround delay | `diagnostics_access` | Blood collection, processing, or result release appears discharge-dependent and delayed. | Blood test ordered/available timestamps. | Laboratory operations lead | Not all abnormal or repeat tests are operational delay. |
| ECG availability delay | `diagnostics_access` | ECG completion or interpretation appears discharge-dependent and delayed. | ECG ordered/completed timestamps. | Cardiology diagnostics lead | Urgency and clinical context matter. |
| Diagnostic sign-off stall | `diagnostics_access` | Diagnostic completion or final sign-off is the likely delay reason. | Diagnostic event completion/sign-off timing. | Diagnostics operations lead | May reflect consultant or reporting dependency. |
| Triage screening backlog | `front_door_screening` | Front-door screening delay may contribute to downstream flow pressure. | Triage screening requested/completed timestamps. | Triage operations lead | This is a flow signal, not a discharge-only signal. |
| Nurse discharge screening wait | `nursing_discharge_readiness` | Nursing checklist/readiness review is delayed for discharge-dependent case. | Nurse discharge screening requested/completed timestamps. | Nursing flow lead | Staffing and acuity context should be reviewed. |
| Social work assessment wait | `social_support_readiness` | Social work assessment is delayed for discharge-dependent case. | Social work assessment requested/completed timestamps. | Social work lead | May indicate complexity rather than avoidable delay. |
| Interpreter availability wait | `social_support_readiness` | Interpreter support is delayed for discharge-dependent communication. | Interpreter requested/completed timestamps. | Language services lead | Must be handled as access and equity support, not waste. |
| Discharge documentation readiness wait | `documentation_readiness` | Required discharge documentation is not ready in expected window. | Documentation started/ready timestamps. | Unit clerk lead | Documentation quality should not be sacrificed for speed. |
| Unit-level bed-flow bottleneck | `unit_flow_capacity` | A facility/unit/day crosses the control-chart threshold. | Daily OOB rate above upper control limit. | Unit operations manager | This is a pattern signal, not an individual-case conclusion. |
| Unattributed operational delay | `unattributed` | Delay is detected but no single dependency explains the episode. | OOB flag without clear event-level attribution. | Patient flow coordinator | Useful for data-quality review and missing workflow instrumentation. |

## How to read the dashboard responsibly

1. Start with `management_signal_groups.csv`, not raw patient rows.
2. Use `case_rate` to avoid overreacting to normal high-volume workflows.
3. Use `recoverable_bed_hours` to identify high-impact rare signals.
4. Use patient-level examples for audit, not as the whole executive worklist.
5. Treat all signals as review prompts, not punitive findings.

## What these KPIs are not

These KPIs are not:

- clinical quality indicators,
- patient-safety event determinations,
- staff performance ratings,
- proof that a delay was avoidable,
- a replacement for chart review or operational context.

They are designed to help patient-flow teams ask better questions faster.
