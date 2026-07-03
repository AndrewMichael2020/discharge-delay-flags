# Discharge Delay Sentinel Implementation Plan

## Summary

Discharge Delay Sentinel is a standalone synthetic-data app for detecting out-of-bounds operational discharge delays, explaining likely blockers, producing dashboard-ready CSVs, and simulating audit/adoption outcomes.

It is intentionally separate from Hospital Schedule Optimizer. It uses different synthetic data, different operational objects, and a different decision workflow.

## Implementation Steps

1. Generate synthetic hospital-flow data at admission, journey-event, bed-resource, and service-availability grain.
2. Build robust out-of-bounds delay thresholds by facility, service line, case mix group, and frailty band.
3. Flag delay signals using robust LOS thresholds, medically-ready hard caps, and control-chart signals.
4. Attribute likely blockers with transparent deterministic rules.
5. Train local statistical models for expected LOS and OOB risk.
6. Produce dashboard-ready CSVs and action worklists.
7. Generate Markdown and HTML reports with Top-N table previews.
8. Keep the first deployment model as dashboard plus CSV, not EHR write-back.

## Defaults

- 4 facilities
- 90 days
- 180 encounters per day
- 48-hour post-medically-ready hard cap
- 35% weekend support-service reduction
- 1.25 ALC pressure multiplier
- Top-N output of 5 rows per table

## Language Guardrails

Use operational terms such as delay signal, blocker, capacity constraint, unresolved dependency, and recoverable bed-hours. Avoid punitive wording.
