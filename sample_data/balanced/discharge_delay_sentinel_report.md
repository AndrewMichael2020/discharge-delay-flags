# Operational Delay Sentinel Run Report

## Executive summary

This synthetic run evaluated **9,900** patient encounters and flagged **811** out-of-bounds operational delay signals.

- OOB signal rate: **8.2%**
- Post-medically-ready hard-cap signals: **428**
- Estimated recoverable bed-hours: **7,013.5**
- Estimated recoverable bed-days: **292.2**
- Top delay reason: **Friday/weekend discharge gap**

Flags are operational delay signals for review, not punitive findings.

## Model quality

```json
{
  "regression": {
    "HistGradientBoostingRegressor": {
      "mae_hours": 12.352,
      "rmse_hours": 16.941,
      "median_absolute_error_hours": 9.978,
      "prediction_std_ratio": 0.781
    },
    "ExtraTreesRegressor": {
      "mae_hours": 12.441,
      "rmse_hours": 16.984,
      "median_absolute_error_hours": 10.032,
      "prediction_std_ratio": 0.784
    },
    "statistical_ensemble": {
      "mae_hours": 12.316,
      "rmse_hours": 16.853,
      "median_absolute_error_hours": 10.021,
      "prediction_std_ratio": 0.779
    }
  },
  "classification": {
    "HistGradientBoostingClassifier": {
      "roc_auc": 0.566,
      "pr_auc": 0.111,
      "recall_at_top_5pct_risk": 0.103,
      "score_std": 0.06561
    },
    "ExtraTreesClassifier": {
      "roc_auc": 0.55,
      "pr_auc": 0.104,
      "recall_at_top_5pct_risk": 0.089,
      "score_std": 0.11597
    },
    "statistical_ensemble": {
      "roc_auc": 0.567,
      "pr_auc": 0.112,
      "recall_at_top_5pct_risk": 0.069,
      "score_std": 0.07955
    }
  }
}
```

## Delay reason impact breakdown

```text
               primary_blocker  flagged_cases  recoverable_bed_hours  median_priority_score  case_rate  avg_recoverable_bed_hours_per_case
  Friday/weekend discharge gap            119                2397.62                 45.840     0.0120                               20.15
            ALC placement wait             43                1060.25                 44.800     0.0043                               24.66
   home care confirmation wait            142                1004.88                 19.225     0.0143                                7.08
unattributed operational delay            198                 859.39                  6.075     0.0200                                4.34
               transport delay             94                 495.95                 19.345     0.0095                                5.28
    radiology MRI access delay             24                 239.89                 26.185     0.0024                               10.00
      therapy assessment stall             28                 183.23                 23.735     0.0028                                6.54
nurse discharge screening wait             19                 160.12                 28.230     0.0019                                8.43
     diagnostic sign-off stall             19                 158.10                 23.590     0.0019                                8.32
   social work assessment wait             12                 125.48                 39.285     0.0012                               10.46
```

## Recommended action preview

```text
 executive_rank        action_id encounter_id facility_id           unit_id service_line priority                primary_blocker                                                         evidence_summary             recommended_owner                                                                                  recommended_action  target_resolution_hours  estimated_recoverable_bed_hours status  reviewed_by  reviewed_timestamp  action_taken  reason_not_actioned           discharge_timestamp shift_date shift_name                 signal_family
              1 f97a31dc380a64a7 ENC_00000013     FAC_000    acute_medicine     medicine     high                transport delay                  Transport completion lag observed after discharge order patient_transport_coordinator                                   Review and address transport delay for this discharge dependency.                       24                             6.70    new          NaN                 NaN           NaN                  NaN 2026-01-08 12:41:23.315594758 2026-01-08        day transport_discharge_logistics
              2 1bcf02e54560e187 ENC_00000036     FAC_000          surgical      surgery  routine unattributed operational delay         Delay signal found but no single dependency explains the episode      patient_flow_coordinator                    Review and address unattributed operational delay for this discharge dependency.                       48                             1.10    new          NaN                 NaN           NaN                  NaN 2026-01-08 10:43:09.235392836 2026-01-08        day                  unattributed
              3 c0520b5c53f28130 ENC_00000071     FAC_000    acute_medicine     medicine     high vulnerable patient porter wait Porter support for vulnerable patient exceeded expected operating window patient_transport_coordinator       Review porter support timing for vulnerable or mobility-limited discharge-dependent patients.                       24                            13.38    new          NaN                 NaN           NaN                  NaN 2026-01-08 16:20:40.829977412 2026-01-08    evening       vulnerable_patient_flow
              4 c97b5eba73183b33 ENC_00000074     FAC_000       respiratory  respiratory     high  radiology CT turnaround delay           CT completion lag exceeded expected discharge-dependent window     radiology_operations_lead Review CT queue priority, protocol readiness, and report finalization for discharge-dependent case.                       24                             6.52    new          NaN                 NaN           NaN                  NaN 2026-01-08 19:22:19.389261284 2026-01-08    evening            diagnostics_access
              5 183b7617ceee3b93 ENC_00000002     FAC_000       respiratory  respiratory     high    home care confirmation wait                Home-care confirmation exceeded expected operating window             home_care_liaison                       Review and address home care confirmation wait for this discharge dependency.                       24                             4.98    new          NaN                 NaN           NaN                  NaN 2026-01-08 21:11:05.605996284 2026-01-08    evening        alc_community_capacity
              6 d806b27d72c71ec0 ENC_00000004     FAC_000       respiratory  respiratory  routine blood testing turnaround delay     Blood testing result availability exceeded expected operating window    laboratory_operations_lead         Review collection, processing, and result-release timing for discharge-dependent bloodwork.                       48                             0.49    new          NaN                 NaN           NaN                  NaN 2026-01-08 05:01:57.906841910 2026-01-08      night            diagnostics_access
              7 e4f61bceaf391238 ENC_00000139     FAC_000           cardiac   cardiology  routine nurse discharge screening wait         Nurse discharge screening lag exceeded expected operating window             nursing_flow_lead                             Review nurse discharge screening queue and readiness checklist handoff.                       48                             0.31    new          NaN                 NaN           NaN                  NaN 2026-01-08 03:22:47.763788798 2026-01-08      night   nursing_discharge_readiness
              8 26f340b9d60e8ede ENC_00000007     FAC_000       orthopedics  orthopedics  routine                transport delay                  Transport completion lag observed after discharge order patient_transport_coordinator                                   Review and address transport delay for this discharge dependency.                       48                             7.14    new          NaN                 NaN           NaN                  NaN 2026-01-09 14:52:12.926801779 2026-01-09        day transport_discharge_logistics
              9 df7cf1c33a4d50ef ENC_00000196     FAC_000    acute_medicine     medicine     high unattributed operational delay         Delay signal found but no single dependency explains the episode      patient_flow_coordinator                    Review and address unattributed operational delay for this discharge dependency.                       24                             2.99    new          NaN                 NaN           NaN                  NaN 2026-01-09 13:03:55.844782699 2026-01-09        day                  unattributed
             10 2f483118e2494042 ENC_00000009     FAC_000 transitional_care   geriatrics  routine unattributed operational delay         Delay signal found but no single dependency explains the episode      patient_flow_coordinator                    Review and address unattributed operational delay for this discharge dependency.                       48                             2.12    new          NaN                 NaN           NaN                  NaN 2026-01-09 09:44:20.418752208 2026-01-09        day                  unattributed
```

## Data quality summary

```text
                data_quality_note  rows
                               ok  9524
missing_medically_ready_timestamp   341
          timestamp_inconsistency    35
```

## Top-N table previews

```text

=== patient_admission_events.parquet ===
shape: (9900, 24)
columns: ['patient_id_synthetic', 'encounter_id', 'facility_id', 'unit_id', 'admission_timestamp', 'discharge_timestamp', 'service_line', 'diagnosis_group', 'case_mix_group', 'triage_level', 'admission_source', 'age_band', 'frailty_band', 'complexity_score', 'expected_los_hours', 'actual_los_hours', 'medically_ready_timestamp', 'discharge_order_timestamp', 'alc_status', 'alc_status_late_coded', 'discharge_disposition', 'readmission_risk_band', 'synthetic_primary_blocker', 'synthetic_data_quality_note']
patient_id_synthetic encounter_id facility_id        unit_id admission_timestamp           discharge_timestamp service_line diagnosis_group  case_mix_group  triage_level admission_source age_band frailty_band  complexity_score  expected_los_hours  actual_los_hours     medically_ready_timestamp     discharge_order_timestamp  alc_status  alc_status_late_coded discharge_disposition readmission_risk_band synthetic_primary_blocker synthetic_data_quality_note
          PT_0000000 ENC_00000000     FAC_000 acute_medicine 2026-01-01 02:34:00 2026-01-06 20:23:14.860024950     medicine CMG_MED_COMPLEX CMG_MED_COMPLEX             3               ED      80+     moderate             0.639              132.59            137.82 2026-01-06 05:28:44.307583671 2026-01-06 07:18:18.546196171       False                  False              transfer              moderate                      none                          ok

=== patient_journey_events.parquet ===
shape: (74180, 15)
columns: ['event_id', 'encounter_id', 'facility_id', 'unit_id', 'event_timestamp', 'event_type', 'event_status', 'event_owner_service', 'requested_timestamp', 'completed_timestamp', 'event_duration_hours', 'is_weekend', 'is_after_hours', 'delay_reason_observed', 'dependency_event_id']
      event_id encounter_id facility_id        unit_id     event_timestamp event_type event_status event_owner_service requested_timestamp completed_timestamp  event_duration_hours  is_weekend  is_after_hours delay_reason_observed dependency_event_id
EVT_0000000001 ENC_00000000     FAC_000 acute_medicine 2026-01-01 02:34:00   admitted    completed           admitting 2026-01-01 02:34:00 2026-01-01 02:34:00                   0.0       False            True                  none                None

=== bed_resource_daily.parquet ===
shape: (540, 8)
columns: ['date', 'facility_id', 'unit_id', 'staffed_beds', 'occupied_beds', 'blocked_beds', 'boarding_patients', 'occupancy_pct']
      date facility_id        unit_id  staffed_beds  occupied_beds  blocked_beds  boarding_patients  occupancy_pct
2026-01-01     FAC_000 acute_medicine            28             19             0                  5          0.675

=== service_availability.parquet ===
shape: (720, 7)
columns: ['date', 'facility_id', 'service_name', 'is_weekend', 'availability_index', 'open_hours', 'capacity_constraint_note']
      date facility_id service_name  is_weekend  availability_index  open_hours capacity_constraint_note
2026-01-01     FAC_000           PT       False               0.997          16                   normal

=== discharge_delay_reference.parquet ===
shape: (18, 10)
columns: ['facility_id', 'service_line', 'case_mix_group', 'frailty_band', 'segment_count', 'median_los_hours', 'mad_los_hours', 'robust_sigma_hours', 'p95_los_hours', 'oob_limit_hours']
facility_id service_line case_mix_group frailty_band  segment_count  median_los_hours  mad_los_hours  robust_sigma_hours  p95_los_hours  oob_limit_hours
    FAC_000   cardiology    CMG_CARDIAC         high            231            117.35          11.07               16.41         153.71           150.56

=== out_of_bounds_delay_flags.parquet ===
shape: (9900, 25)
columns: ['encounter_id', 'patient_id_synthetic', 'facility_id', 'unit_id', 'service_line', 'case_mix_group', 'frailty_band', 'admission_timestamp', 'discharge_timestamp', 'medically_ready_timestamp', 'actual_los_hours', 'expected_los_hours', 'median_los_hours', 'oob_limit_hours', 'hours_above_limit', 'hours_after_medically_ready', 'post_ready_excess_hours', 'robust_los_oob_flag', 'post_ready_hard_cap_flag', 'control_chart_signal_flag', 'oob_flag', 'priority_score', 'alc_status', 'synthetic_primary_blocker', 'synthetic_data_quality_note']
encounter_id patient_id_synthetic facility_id        unit_id service_line  case_mix_group frailty_band admission_timestamp           discharge_timestamp     medically_ready_timestamp  actual_los_hours  expected_los_hours  median_los_hours  oob_limit_hours  hours_above_limit  hours_after_medically_ready  post_ready_excess_hours  robust_los_oob_flag  post_ready_hard_cap_flag  control_chart_signal_flag  oob_flag  priority_score  alc_status synthetic_primary_blocker synthetic_data_quality_note
ENC_00000000           PT_0000000     FAC_000 acute_medicine     medicine CMG_MED_COMPLEX     moderate 2026-01-01 02:34:00 2026-01-06 20:23:14.860024950 2026-01-06 05:28:44.307583671            137.82              132.59            125.75           158.79                0.0                        14.91                      0.0                False                     False                      False     False             0.0       False                      none                          ok

=== delay_blocker_attribution.parquet ===
shape: (811, 14)
columns: ['encounter_id', 'facility_id', 'unit_id', 'service_line', 'discharge_timestamp', 'primary_blocker', 'secondary_blocker', 'blocker_confidence', 'evidence_summary', 'hours_above_limit', 'hours_after_medically_ready', 'post_ready_excess_hours', 'estimated_recoverable_bed_hours', 'priority_score']
encounter_id facility_id     unit_id service_line           discharge_timestamp             primary_blocker secondary_blocker  blocker_confidence                                          evidence_summary  hours_above_limit  hours_after_medically_ready  post_ready_excess_hours  estimated_recoverable_bed_hours  priority_score
ENC_00000002     FAC_000 respiratory  respiratory 2026-01-08 21:11:05.605996284 home care confirmation wait              none                0.86 Home-care confirmation exceeded expected operating window               9.87                        52.98                     4.98                             4.98           27.15

=== ranked_actionable_signals.csv ===
shape: (100, 11)
columns: ['signal_rank', 'primary_blocker', 'facility_id', 'unit_id', 'service_line', 'flagged_cases', 'recoverable_bed_hours', 'median_priority_score', 'mean_confidence', 'signal_family', 'actionability_score']
 signal_rank              primary_blocker facility_id        unit_id service_line  flagged_cases  recoverable_bed_hours  median_priority_score  mean_confidence    signal_family  actionability_score
           1 Friday/weekend discharge gap     FAC_000 acute_medicine     medicine             33                 793.74                  48.94             0.88 weekend_flow_gap               503.88

=== management_signal_groups.csv ===
shape: (543, 15)
columns: ['shift_date', 'shift_name', 'signal_family', 'primary_blocker', 'flagged_cases', 'affected_facilities', 'affected_units', 'affected_service_lines', 'recoverable_bed_hours', 'median_priority_score', 'mean_confidence', 'management_score', 'recommended_owner', 'recommended_management_action', 'signal_group_rank']
shift_date shift_name                 signal_family primary_blocker  flagged_cases affected_facilities              affected_units affected_service_lines  recoverable_bed_hours  median_priority_score  mean_confidence  management_score             recommended_owner                                     recommended_management_action  signal_group_rank
2026-01-08        day transport_discharge_logistics transport delay              2             FAC_000 acute_medicine, respiratory  medicine, respiratory                    6.7                  24.07             0.78              45.5 patient_transport_coordinator Review and address transport delay for this discharge dependency.                  1

=== management_signal_kpis.csv ===
shape: (1, 10)
columns: ['max_signal_groups_per_shift', 'total_shift_windows', 'management_signal_groups', 'avg_signal_groups_per_shift', 'raw_oob_delay_signals', 'raw_to_management_compression_ratio', 'management_recoverable_bed_hours', 'alc_wait_cases', 'alc_wait_case_rate', 'alc_recoverable_bed_hours']
 max_signal_groups_per_shift  total_shift_windows  management_signal_groups  avg_signal_groups_per_shift  raw_oob_delay_signals  raw_to_management_compression_ratio  management_recoverable_bed_hours  alc_wait_cases  alc_wait_case_rate  alc_recoverable_bed_hours
                           5                  241                       543                         2.25                    811                                 1.49                           7011.21              43              0.0043                    1060.25

=== scenario_detection_summary.csv ===
shape: (13, 6)
columns: ['requested_scenario_mode', 'scenario_label', 'signal_family', 'recoverable_bed_hours', 'flagged_cases', 'share_of_recoverable_hours']
requested_scenario_mode                      scenario_label    signal_family  recoverable_bed_hours  flagged_cases  share_of_recoverable_hours
               balanced mixed_operational_pressure_detected weekend_flow_gap                2397.62            119                      0.3419

=== delay_resolution_actions.csv ===
shape: (645, 22)
columns: ['executive_rank', 'action_id', 'encounter_id', 'facility_id', 'unit_id', 'service_line', 'priority', 'primary_blocker', 'evidence_summary', 'recommended_owner', 'recommended_action', 'target_resolution_hours', 'estimated_recoverable_bed_hours', 'status', 'reviewed_by', 'reviewed_timestamp', 'action_taken', 'reason_not_actioned', 'discharge_timestamp', 'shift_date', 'shift_name', 'signal_family']
 executive_rank        action_id encounter_id facility_id        unit_id service_line priority primary_blocker                                        evidence_summary             recommended_owner                                                recommended_action  target_resolution_hours  estimated_recoverable_bed_hours status  reviewed_by  reviewed_timestamp  action_taken  reason_not_actioned           discharge_timestamp shift_date shift_name                 signal_family
              1 f97a31dc380a64a7 ENC_00000013     FAC_000 acute_medicine     medicine     high transport delay Transport completion lag observed after discharge order patient_transport_coordinator Review and address transport delay for this discharge dependency.                       24                              6.7    new          NaN                 NaN           NaN                  NaN 2026-01-08 12:41:23.315594758 2026-01-08        day transport_discharge_logistics

=== admin_delay_dashboard_metrics.csv ===
shape: (555, 15)
columns: ['discharge_date', 'facility_id', 'unit_id', 'discharges', 'oob_cases', 'post_ready_hard_cap_cases', 'median_los_hours', 'median_post_ready_delay_hours', 'oob_rate', 'centerline_oob_rate', 'sigma_oob_rate', 'upper_control_limit', 'control_chart_signal_flag', 'top_blocker', 'estimated_bed_days_recoverable']
discharge_date facility_id unit_id  discharges  oob_cases  post_ready_hard_cap_cases  median_los_hours  median_post_ready_delay_hours  oob_rate  centerline_oob_rate  sigma_oob_rate  upper_control_limit  control_chart_signal_flag                top_blocker  estimated_bed_days_recoverable
    2026-01-04     FAC_000 cardiac           1          0                          0             72.83                            NaN       0.0             0.069244        0.096592             0.359021                      False computed_after_attribution                             NaN
```

## Recommended next operational actions

1. Review urgent and high-priority rows in `delay_resolution_actions.csv`.
2. Compare the delay-reason impact breakdown with discharge huddle experience.
3. Use the control chart to distinguish day-to-day noise from systemic delay signals.
4. Record adoption outcomes in `action_adoption_audit.csv` during shadow-mode review.
