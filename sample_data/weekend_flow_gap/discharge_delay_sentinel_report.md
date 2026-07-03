# Operational Delay Sentinel Run Report

## Executive summary

This synthetic run evaluated **9,900** patient encounters and flagged **910** out-of-bounds operational delay signals.

- OOB signal rate: **9.2%**
- Post-medically-ready hard-cap signals: **712**
- Estimated recoverable bed-hours: **27,768.3**
- Estimated recoverable bed-days: **1,157.0**
- Top delay reason: **Friday/weekend discharge gap**

Flags are operational delay signals for review, not punitive findings.

## Model quality

```json
{
  "regression": {
    "HistGradientBoostingRegressor": {
      "mae_hours": 16.429,
      "rmse_hours": 26.044,
      "median_absolute_error_hours": 11.412,
      "prediction_std_ratio": 0.639
    },
    "ExtraTreesRegressor": {
      "mae_hours": 16.595,
      "rmse_hours": 26.036,
      "median_absolute_error_hours": 11.802,
      "prediction_std_ratio": 0.641
    },
    "statistical_ensemble": {
      "mae_hours": 16.355,
      "rmse_hours": 25.866,
      "median_absolute_error_hours": 11.422,
      "prediction_std_ratio": 0.633
    }
  },
  "classification": {
    "HistGradientBoostingClassifier": {
      "roc_auc": 0.591,
      "pr_auc": 0.129,
      "recall_at_top_5pct_risk": 0.097,
      "score_std": 0.08097
    },
    "ExtraTreesClassifier": {
      "roc_auc": 0.614,
      "pr_auc": 0.141,
      "recall_at_top_5pct_risk": 0.101,
      "score_std": 0.13555
    },
    "statistical_ensemble": {
      "roc_auc": 0.613,
      "pr_auc": 0.142,
      "recall_at_top_5pct_risk": 0.101,
      "score_std": 0.09781
    }
  }
}
```

## Delay reason impact breakdown

```text
               primary_blocker  flagged_cases  recoverable_bed_hours  median_priority_score  case_rate  avg_recoverable_bed_hours_per_case
  Friday/weekend discharge gap            466               25385.53                 86.090     0.0471                               54.48
   home care confirmation wait            114                 704.80                 20.805     0.0115                                6.18
unattributed operational delay            121                 365.70                  8.530     0.0122                                3.02
               transport delay             61                 311.56                 22.210     0.0062                                5.11
            ALC placement wait             15                 277.90                 42.740     0.0015                               18.53
nurse discharge screening wait             19                 191.72                 26.720     0.0019                               10.09
      therapy assessment stall             23                 125.48                 20.340     0.0023                                5.46
    radiology MRI access delay             13                 102.11                 23.490     0.0013                                7.85
   social work assessment wait             10                  79.26                 27.755     0.0010                                7.93
vulnerable patient porter wait             15                  70.21                 21.650     0.0015                                4.68
```

## Recommended action preview

```text
 executive_rank        action_id encounter_id facility_id        unit_id service_line priority                primary_blocker                                                             evidence_summary             recommended_owner                                                                            recommended_action  target_resolution_hours  estimated_recoverable_bed_hours status  reviewed_by  reviewed_timestamp  action_taken  reason_not_actioned           discharge_timestamp shift_date shift_name                 signal_family
              1 ef404b796f0a2a8f ENC_00000044     FAC_000       surgical      surgery     high   Friday/weekend discharge gap Medically ready near weekend with discharge delayed into next service window      site_operations_director                Review and address friday/weekend discharge gap for this discharge dependency.                       24                             9.28    new          NaN                 NaN           NaN                  NaN 2026-01-07 23:45:41.385700906 2026-01-07      night              weekend_flow_gap
              2 1bcf02e54560e187 ENC_00000036     FAC_000       surgical      surgery  routine unattributed operational delay             Delay signal found but no single dependency explains the episode      patient_flow_coordinator              Review and address unattributed operational delay for this discharge dependency.                       48                             0.29    new          NaN                 NaN           NaN                  NaN 2026-01-08 10:43:09.235392836 2026-01-08        day                  unattributed
              3 c0520b5c53f28130 ENC_00000071     FAC_000 acute_medicine     medicine     high vulnerable patient porter wait     Porter support for vulnerable patient exceeded expected operating window patient_transport_coordinator Review porter support timing for vulnerable or mobility-limited discharge-dependent patients.                       24                            13.38    new          NaN                 NaN           NaN                  NaN 2026-01-08 16:20:40.829977412 2026-01-08    evening       vulnerable_patient_flow
              4 9e906c8592cc204b ENC_00000074     FAC_000    respiratory  respiratory     high                transport delay                      Transport completion lag observed after discharge order patient_transport_coordinator                             Review and address transport delay for this discharge dependency.                       24                             6.52    new          NaN                 NaN           NaN                  NaN 2026-01-08 19:22:19.389261284 2026-01-08    evening transport_discharge_logistics
              5 183b7617ceee3b93 ENC_00000002     FAC_000    respiratory  respiratory     high    home care confirmation wait                    Home-care confirmation exceeded expected operating window             home_care_liaison                 Review and address home care confirmation wait for this discharge dependency.                       24                             4.98    new          NaN                 NaN           NaN                  NaN 2026-01-08 21:11:05.605996284 2026-01-08    evening        alc_community_capacity
              6 cbfa1010aad502b1 ENC_00000004     FAC_000    respiratory  respiratory  routine unattributed operational delay             Delay signal found but no single dependency explains the episode      patient_flow_coordinator              Review and address unattributed operational delay for this discharge dependency.                       48                             0.22    new          NaN                 NaN           NaN                  NaN 2026-01-08 05:01:57.906841910 2026-01-08      night                  unattributed
              7 a2000a8bfe86f6fa ENC_00000007     FAC_000    orthopedics  orthopedics  routine unattributed operational delay             Delay signal found but no single dependency explains the episode      patient_flow_coordinator              Review and address unattributed operational delay for this discharge dependency.                       48                             6.67    new          NaN                 NaN           NaN                  NaN 2026-01-09 14:52:12.926801779 2026-01-09        day                  unattributed
              8 df7cf1c33a4d50ef ENC_00000196     FAC_000 acute_medicine     medicine     high unattributed operational delay             Delay signal found but no single dependency explains the episode      patient_flow_coordinator              Review and address unattributed operational delay for this discharge dependency.                       24                             2.99    new          NaN                 NaN           NaN                  NaN 2026-01-09 13:03:55.844782699 2026-01-09        day                  unattributed
              9 a1bbfafd00a8302f ENC_00000116     FAC_000    respiratory  respiratory     high                transport delay                      Transport completion lag observed after discharge order patient_transport_coordinator                             Review and address transport delay for this discharge dependency.                       24                             1.81    new          NaN                 NaN           NaN                  NaN 2026-01-09 09:35:44.730645544 2026-01-09        day transport_discharge_logistics
             10 579081b2d39a98f7 ENC_00000062     FAC_000 acute_medicine     medicine  routine             ALC placement wait                         ALC status present with delayed discharge dependency      transition_services_lead                          Review and address alc placement wait for this discharge dependency.                       48                             1.50    new          NaN                 NaN           NaN                  NaN 2026-01-09 03:16:52.462637017 2026-01-09      night        alc_community_capacity
```

## Data quality summary

```text
                data_quality_note  rows
                               ok  9523
missing_medically_ready_timestamp   337
          timestamp_inconsistency    40
```

## Top-N table previews

```text

=== patient_admission_events.parquet ===
shape: (9900, 24)
columns: ['patient_id_synthetic', 'encounter_id', 'facility_id', 'unit_id', 'admission_timestamp', 'discharge_timestamp', 'service_line', 'diagnosis_group', 'case_mix_group', 'triage_level', 'admission_source', 'age_band', 'frailty_band', 'complexity_score', 'expected_los_hours', 'actual_los_hours', 'medically_ready_timestamp', 'discharge_order_timestamp', 'alc_status', 'alc_status_late_coded', 'discharge_disposition', 'readmission_risk_band', 'synthetic_primary_blocker', 'synthetic_data_quality_note']
patient_id_synthetic encounter_id facility_id        unit_id admission_timestamp           discharge_timestamp service_line diagnosis_group  case_mix_group  triage_level admission_source age_band frailty_band  complexity_score  expected_los_hours  actual_los_hours     medically_ready_timestamp     discharge_order_timestamp  alc_status  alc_status_late_coded discharge_disposition readmission_risk_band synthetic_primary_blocker synthetic_data_quality_note
          PT_0000000 ENC_00000000     FAC_000 acute_medicine 2026-01-01 02:34:00 2026-01-06 20:23:14.860024950     medicine CMG_MED_COMPLEX CMG_MED_COMPLEX             3               ED      80+     moderate             0.639              132.59            137.82 2026-01-06 05:28:44.307583671 2026-01-06 05:28:44.307583671       False                  False                 rehab                   low                      none                          ok

=== patient_journey_events.parquet ===
shape: (73812, 15)
columns: ['event_id', 'encounter_id', 'facility_id', 'unit_id', 'event_timestamp', 'event_type', 'event_status', 'event_owner_service', 'requested_timestamp', 'completed_timestamp', 'event_duration_hours', 'is_weekend', 'is_after_hours', 'delay_reason_observed', 'dependency_event_id']
      event_id encounter_id facility_id        unit_id     event_timestamp event_type event_status event_owner_service requested_timestamp completed_timestamp  event_duration_hours  is_weekend  is_after_hours delay_reason_observed dependency_event_id
EVT_0000000001 ENC_00000000     FAC_000 acute_medicine 2026-01-01 02:34:00   admitted    completed           admitting 2026-01-01 02:34:00 2026-01-01 02:34:00                   0.0       False            True                  none                None

=== bed_resource_daily.parquet ===
shape: (540, 8)
columns: ['date', 'facility_id', 'unit_id', 'staffed_beds', 'occupied_beds', 'blocked_beds', 'boarding_patients', 'occupancy_pct']
      date facility_id        unit_id  staffed_beds  occupied_beds  blocked_beds  boarding_patients  occupancy_pct
2026-01-01     FAC_000 acute_medicine            35             26             0                  7          0.733

=== service_availability.parquet ===
shape: (720, 7)
columns: ['date', 'facility_id', 'service_name', 'is_weekend', 'availability_index', 'open_hours', 'capacity_constraint_note']
      date facility_id service_name  is_weekend  availability_index  open_hours capacity_constraint_note
2026-01-01     FAC_000           PT       False               1.021          16                   normal

=== discharge_delay_reference.parquet ===
shape: (18, 10)
columns: ['facility_id', 'service_line', 'case_mix_group', 'frailty_band', 'segment_count', 'median_los_hours', 'mad_los_hours', 'robust_sigma_hours', 'p95_los_hours', 'oob_limit_hours']
facility_id service_line case_mix_group frailty_band  segment_count  median_los_hours  mad_los_hours  robust_sigma_hours  p95_los_hours  oob_limit_hours
    FAC_000   cardiology    CMG_CARDIAC         high            231            116.81          11.07               16.41         160.04           150.02

=== out_of_bounds_delay_flags.parquet ===
shape: (9900, 25)
columns: ['encounter_id', 'patient_id_synthetic', 'facility_id', 'unit_id', 'service_line', 'case_mix_group', 'frailty_band', 'admission_timestamp', 'discharge_timestamp', 'medically_ready_timestamp', 'actual_los_hours', 'expected_los_hours', 'median_los_hours', 'oob_limit_hours', 'hours_above_limit', 'hours_after_medically_ready', 'post_ready_excess_hours', 'robust_los_oob_flag', 'post_ready_hard_cap_flag', 'control_chart_signal_flag', 'oob_flag', 'priority_score', 'alc_status', 'synthetic_primary_blocker', 'synthetic_data_quality_note']
encounter_id patient_id_synthetic facility_id        unit_id service_line  case_mix_group frailty_band admission_timestamp           discharge_timestamp     medically_ready_timestamp  actual_los_hours  expected_los_hours  median_los_hours  oob_limit_hours  hours_above_limit  hours_after_medically_ready  post_ready_excess_hours  robust_los_oob_flag  post_ready_hard_cap_flag  control_chart_signal_flag  oob_flag  priority_score  alc_status synthetic_primary_blocker synthetic_data_quality_note
ENC_00000000           PT_0000000     FAC_000 acute_medicine     medicine CMG_MED_COMPLEX     moderate 2026-01-01 02:34:00 2026-01-06 20:23:14.860024950 2026-01-06 05:28:44.307583671            137.82              132.59            126.44           160.88                0.0                        14.91                      0.0                False                     False                      False     False             0.0       False                      none                          ok

=== delay_blocker_attribution.parquet ===
shape: (910, 14)
columns: ['encounter_id', 'facility_id', 'unit_id', 'service_line', 'discharge_timestamp', 'primary_blocker', 'secondary_blocker', 'blocker_confidence', 'evidence_summary', 'hours_above_limit', 'hours_after_medically_ready', 'post_ready_excess_hours', 'estimated_recoverable_bed_hours', 'priority_score']
encounter_id facility_id     unit_id service_line           discharge_timestamp             primary_blocker secondary_blocker  blocker_confidence                                          evidence_summary  hours_above_limit  hours_after_medically_ready  post_ready_excess_hours  estimated_recoverable_bed_hours  priority_score
ENC_00000002     FAC_000 respiratory  respiratory 2026-01-08 21:11:05.605996284 home care confirmation wait              none                0.86 Home-care confirmation exceeded expected operating window               9.68                        52.98                     4.98                             4.98           27.03

=== ranked_actionable_signals.csv ===
shape: (86, 11)
columns: ['signal_rank', 'primary_blocker', 'facility_id', 'unit_id', 'service_line', 'flagged_cases', 'recoverable_bed_hours', 'median_priority_score', 'mean_confidence', 'signal_family', 'actionability_score']
 signal_rank              primary_blocker facility_id        unit_id service_line  flagged_cases  recoverable_bed_hours  median_priority_score  mean_confidence    signal_family  actionability_score
           1 Friday/weekend discharge gap     FAC_000 acute_medicine     medicine            139                6691.96                  79.33             0.88 weekend_flow_gap              3211.86

=== management_signal_groups.csv ===
shape: (456, 15)
columns: ['shift_date', 'shift_name', 'signal_family', 'primary_blocker', 'flagged_cases', 'affected_facilities', 'affected_units', 'affected_service_lines', 'recoverable_bed_hours', 'median_priority_score', 'mean_confidence', 'management_score', 'recommended_owner', 'recommended_management_action', 'signal_group_rank']
shift_date shift_name    signal_family              primary_blocker  flagged_cases affected_facilities affected_units affected_service_lines  recoverable_bed_hours  median_priority_score  mean_confidence  management_score        recommended_owner                                                  recommended_management_action  signal_group_rank
2026-01-07      night weekend_flow_gap Friday/weekend discharge gap              1             FAC_000       surgical                surgery                   9.28                  45.15             0.88             44.86 site_operations_director Review and address friday/weekend discharge gap for this discharge dependency.                  1

=== management_signal_kpis.csv ===
shape: (1, 10)
columns: ['max_signal_groups_per_shift', 'total_shift_windows', 'management_signal_groups', 'avg_signal_groups_per_shift', 'raw_oob_delay_signals', 'raw_to_management_compression_ratio', 'management_recoverable_bed_hours', 'alc_wait_cases', 'alc_wait_case_rate', 'alc_recoverable_bed_hours']
 max_signal_groups_per_shift  total_shift_windows  management_signal_groups  avg_signal_groups_per_shift  raw_oob_delay_signals  raw_to_management_compression_ratio  management_recoverable_bed_hours  alc_wait_cases  alc_wait_case_rate  alc_recoverable_bed_hours
                           5                  247                       456                         1.85                    910                                  2.0                          27767.63              15              0.0015                      277.9

=== scenario_detection_summary.csv ===
shape: (13, 6)
columns: ['requested_scenario_mode', 'scenario_label', 'signal_family', 'recoverable_bed_hours', 'flagged_cases', 'share_of_recoverable_hours']
requested_scenario_mode            scenario_label    signal_family  recoverable_bed_hours  flagged_cases  share_of_recoverable_hours
       weekend_flow_gap weekend_flow_gap_detected weekend_flow_gap               25385.53            466                      0.9142

=== delay_resolution_actions.csv ===
shape: (599, 22)
columns: ['executive_rank', 'action_id', 'encounter_id', 'facility_id', 'unit_id', 'service_line', 'priority', 'primary_blocker', 'evidence_summary', 'recommended_owner', 'recommended_action', 'target_resolution_hours', 'estimated_recoverable_bed_hours', 'status', 'reviewed_by', 'reviewed_timestamp', 'action_taken', 'reason_not_actioned', 'discharge_timestamp', 'shift_date', 'shift_name', 'signal_family']
 executive_rank        action_id encounter_id facility_id  unit_id service_line priority              primary_blocker                                                             evidence_summary        recommended_owner                                                             recommended_action  target_resolution_hours  estimated_recoverable_bed_hours status  reviewed_by  reviewed_timestamp  action_taken  reason_not_actioned           discharge_timestamp shift_date shift_name    signal_family
              1 ef404b796f0a2a8f ENC_00000044     FAC_000 surgical      surgery     high Friday/weekend discharge gap Medically ready near weekend with discharge delayed into next service window site_operations_director Review and address friday/weekend discharge gap for this discharge dependency.                       24                             9.28    new          NaN                 NaN           NaN                  NaN 2026-01-07 23:45:41.385700906 2026-01-07      night weekend_flow_gap

=== admin_delay_dashboard_metrics.csv ===
shape: (565, 15)
columns: ['discharge_date', 'facility_id', 'unit_id', 'discharges', 'oob_cases', 'post_ready_hard_cap_cases', 'median_los_hours', 'median_post_ready_delay_hours', 'oob_rate', 'centerline_oob_rate', 'sigma_oob_rate', 'upper_control_limit', 'control_chart_signal_flag', 'top_blocker', 'estimated_bed_days_recoverable']
discharge_date facility_id unit_id  discharges  oob_cases  post_ready_hard_cap_cases  median_los_hours  median_post_ready_delay_hours  oob_rate  centerline_oob_rate  sigma_oob_rate  upper_control_limit  control_chart_signal_flag                top_blocker  estimated_bed_days_recoverable
    2026-01-04     FAC_000 cardiac           1          0                          0             72.83                          13.85       0.0             0.106511        0.165065             0.601707                      False computed_after_attribution                             NaN
```

## Recommended next operational actions

1. Review urgent and high-priority rows in `delay_resolution_actions.csv`.
2. Compare the delay-reason impact breakdown with discharge huddle experience.
3. Use the control chart to distinguish day-to-day noise from systemic delay signals.
4. Record adoption outcomes in `action_adoption_audit.csv` during shadow-mode review.
