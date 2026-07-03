from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, median_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

SERVICE_LINES = ["medicine", "surgery", "cardiology", "respiratory", "orthopedics", "geriatrics"]
UNITS = ["acute_medicine", "surgical", "cardiac", "respiratory", "orthopedics", "transitional_care"]
SEGMENT_COLS = ["facility_id", "service_line", "case_mix_group", "frailty_band"]

@dataclass
class SyntheticFlowConfig:
    facilities: int = 4
    days: int = 90
    encounters_per_day: int = 180
    seed: int = 42
    oob_rate_target: float = 0.05
    weekend_service_reduction: float = 0.35
    alc_pressure_multiplier: float = 1.25
    scenario_mode: str = "balanced"
    start_date: str = "2026-01-01"


def _choice(rng, values, size, probs=None):
    return rng.choice(np.array(values, dtype=object), size=size, p=probs)


def generate_synthetic_flow(config: SyntheticFlowConfig, out_dir: Path) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(config.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = config.facilities * config.days * config.encounters_per_day
    start = pd.Timestamp(config.start_date)
    facility_ids = np.array([f"FAC_{i:03d}" for i in range(config.facilities)], dtype=object)
    service_probs = np.array([0.27, 0.16, 0.12, 0.18, 0.12, 0.15])
    case_map = {"medicine":"CMG_MED_COMPLEX", "surgery":"CMG_SURG_ROUTINE", "cardiology":"CMG_CARDIAC", "respiratory":"CMG_RESP", "orthopedics":"CMG_ORTHO", "geriatrics":"CMG_FRAIL_ELDER"}
    unit_map = {"medicine":"acute_medicine", "surgery":"surgical", "cardiology":"cardiac", "respiratory":"respiratory", "orthopedics":"orthopedics", "geriatrics":"transitional_care"}
    base_los = {"medicine":82, "surgery":70, "cardiology":62, "respiratory":88, "orthopedics":96, "geriatrics":118}

    idx = np.arange(n)
    day_index = idx // (config.facilities * config.encounters_per_day)
    within_day = idx % (config.facilities * config.encounters_per_day)
    facility = facility_ids[within_day // config.encounters_per_day]
    admission_ts = start + pd.to_timedelta(day_index, unit="D") + pd.to_timedelta(rng.integers(0,24,n), unit="h") + pd.to_timedelta(rng.integers(0,60,n), unit="m")
    service_line = _choice(rng, SERVICE_LINES, n, service_probs)
    case_mix_group = np.array([case_map[x] for x in service_line], dtype=object)
    unit_id = np.array([unit_map[x] for x in service_line], dtype=object)
    frailty_band = _choice(rng, ["low", "moderate", "high"], n, [0.42,0.38,0.20])
    age_band = _choice(rng, ["18-44", "45-64", "65-79", "80+"], n, [0.18,0.30,0.30,0.22])
    admission_source = _choice(rng, ["ED", "clinic", "transfer", "direct_admit"], n, [0.62,0.16,0.15,0.07])
    triage_level = rng.choice([1,2,3,4,5], n, p=[0.05,0.19,0.42,0.25,0.09])
    complexity = np.clip(rng.normal(0.45,0.17,n) + (frailty_band=="high")*0.22 + (age_band=="80+")*0.14 + (triage_level<=2)*0.12, 0, 1.5)
    expected_los = np.array([base_los[x] for x in service_line], dtype=float) + complexity*42 + rng.gamma(2.0,7.0,n)
    clinical_ready_hours = np.maximum(12, expected_los - rng.normal(10,4,n))
    ready_ts = admission_ts + pd.to_timedelta(clinical_ready_hours, unit="h")
    ready_dow = pd.Series(ready_ts).dt.dayofweek.to_numpy()
    ready_hour = pd.Series(ready_ts).dt.hour.to_numpy()
    is_friday_late = (ready_dow == 4) & (ready_hour >= 12)
    is_weekend_ready = ready_dow >= 5

    target_scale = float(np.clip(config.oob_rate_target / 0.10, 0.25, 1.5))
    p_alc = np.clip((0.025 + (service_line=="geriatrics")*0.060 + (frailty_band=="high")*0.035) * config.alc_pressure_multiplier * target_scale, 0, 0.40)
    p_home = (0.028 + (age_band=="80+")*0.020 + (frailty_band!="low")*0.015) * target_scale
    p_therapy = (0.035 + (service_line=="orthopedics")*0.050 + (frailty_band=="high")*0.020) * target_scale
    p_weekend = (is_friday_late*0.42 + is_weekend_ready*config.weekend_service_reduction*0.30) * target_scale
    p_transport = (0.030 + (admission_source=="transfer")*0.020) * target_scale
    p_pharm = (0.025 + (service_line=="cardiology")*0.015) * target_scale
    p_radiology_ct = (0.018 + (triage_level<=2)*0.025 + (service_line=="respiratory")*0.012) * target_scale
    p_radiology_mri = (0.010 + (service_line=="orthopedics")*0.024 + (service_line=="geriatrics")*0.010) * target_scale
    p_radiology_ultrasound = (0.012 + (service_line=="surgery")*0.018) * target_scale
    p_blood_testing = (0.018 + (service_line=="medicine")*0.012 + (triage_level<=2)*0.010) * target_scale
    p_ecg = (0.012 + (service_line=="cardiology")*0.035 + (age_band=="80+")*0.010) * target_scale
    p_diag = (0.018 + (triage_level<=2)*0.012) * target_scale
    if config.scenario_mode == "alc_heavy":
        p_alc *= 1.75; p_home *= 0.85; p_therapy *= 0.90; p_weekend *= 0.90
        p_radiology_ct *= 0.75; p_radiology_mri *= 0.75; p_radiology_ultrasound *= 0.75; p_blood_testing *= 0.80; p_ecg *= 0.80; p_diag *= 0.80
    elif config.scenario_mode == "diagnostics_heavy":
        p_alc *= 0.15; p_home *= 0.45; p_therapy *= 0.55; p_weekend *= 0.45
        p_radiology_ct *= 5.20; p_radiology_mri *= 6.00; p_radiology_ultrasound *= 4.80; p_blood_testing *= 4.60; p_ecg *= 4.20; p_diag *= 3.20
    elif config.scenario_mode == "weekend_flow_gap":
        p_alc *= 0.35; p_home *= 0.90; p_therapy *= 0.90; p_weekend *= 5.20
        p_radiology_ct *= 0.55; p_radiology_mri *= 0.50; p_radiology_ultrasound *= 0.55; p_blood_testing *= 0.55; p_ecg *= 0.55; p_diag *= 0.55

    blocker = np.array(["none"]*n, dtype=object)
    alc = rng.random(n) < p_alc
    home = (~alc) & (rng.random(n) < p_home)
    therapy = (~alc) & (~home) & (rng.random(n) < p_therapy)
    friday = (~alc) & (~home) & (~therapy) & (rng.random(n) < p_weekend)
    transport = (~alc) & (~home) & (~therapy) & (~friday) & (rng.random(n) < p_transport)
    pharm = (~alc) & (~home) & (~therapy) & (~friday) & (~transport) & (rng.random(n) < p_pharm)
    rad_ct = (~alc) & (~home) & (~therapy) & (~friday) & (~transport) & (~pharm) & (rng.random(n) < p_radiology_ct)
    rad_mri = (~alc) & (~home) & (~therapy) & (~friday) & (~transport) & (~pharm) & (~rad_ct) & (rng.random(n) < p_radiology_mri)
    rad_us = (~alc) & (~home) & (~therapy) & (~friday) & (~transport) & (~pharm) & (~rad_ct) & (~rad_mri) & (rng.random(n) < p_radiology_ultrasound)
    blood = (~alc) & (~home) & (~therapy) & (~friday) & (~transport) & (~pharm) & (~rad_ct) & (~rad_mri) & (~rad_us) & (rng.random(n) < p_blood_testing)
    ecg = (~alc) & (~home) & (~therapy) & (~friday) & (~transport) & (~pharm) & (~rad_ct) & (~rad_mri) & (~rad_us) & (~blood) & (rng.random(n) < p_ecg)
    diag = (~alc) & (~home) & (~therapy) & (~friday) & (~transport) & (~pharm) & (~rad_ct) & (~rad_mri) & (~rad_us) & (~blood) & (~ecg) & (rng.random(n) < p_diag)
    blocker[alc] = "ALC placement wait"
    blocker[home] = "home care confirmation wait"
    blocker[therapy] = "therapy assessment stall"
    blocker[friday] = "Friday/weekend discharge gap"
    blocker[transport] = "transport delay"
    blocker[pharm] = "pharmacy discharge delay"
    blocker[rad_ct] = "radiology CT turnaround delay"
    blocker[rad_mri] = "radiology MRI access delay"
    blocker[rad_us] = "radiology ultrasound turnaround delay"
    blocker[blood] = "blood testing turnaround delay"
    blocker[ecg] = "ECG availability delay"
    blocker[diag] = "diagnostic sign-off stall"

    op_delay = rng.gamma(1.4,5.0,n)
    op_delay += alc*rng.gamma(3.0,28.0,n) + home*rng.gamma(2.6,18.0,n) + therapy*rng.gamma(2.2,14.0,n)
    op_delay += friday*rng.uniform(42,72,n) + transport*rng.gamma(1.8,10.0,n) + pharm*rng.gamma(1.7,8.0,n)
    op_delay += rad_ct*rng.gamma(1.9,9.0,n) + rad_mri*rng.gamma(2.4,14.0,n) + rad_us*rng.gamma(1.8,8.5,n)
    op_delay += blood*rng.gamma(1.7,6.5,n) + ecg*rng.gamma(1.6,5.5,n) + diag*rng.gamma(1.9,9.0,n)
    if config.scenario_mode == "diagnostics_heavy":
        diagnostic_mask = rad_ct | rad_mri | rad_us | blood | ecg | diag
        op_delay += diagnostic_mask * rng.gamma(2.4, 18.0, n)
    if config.scenario_mode == "weekend_flow_gap":
        op_delay += friday * rng.gamma(2.2, 18.0, n)
    op_delay *= 1 + 0.20*np.sin(2*np.pi*day_index/max(config.days,1))
    actual_los = expected_los + op_delay
    discharge_ts = admission_ts + pd.to_timedelta(actual_los, unit="h")
    order_ts = admission_ts + pd.to_timedelta(clinical_ready_hours + np.maximum(0, op_delay - rng.gamma(1.6,4.5,n)), unit="h")
    ready_series = pd.Series(ready_ts)
    missing_ready = rng.random(n) < 0.035
    inconsistent = rng.random(n) < 0.004
    ready_series[missing_ready] = pd.NaT
    ready_series[inconsistent] = pd.Series(discharge_ts)[inconsistent] + pd.to_timedelta(rng.uniform(1,12,inconsistent.sum()), unit="h")
    disp = _choice(rng, ["home", "home_with_support", "rehab", "long_term_care", "transfer", "deceased"], n, [0.50,0.22,0.12,0.08,0.06,0.02])
    alc_disp_mask = (blocker == "ALC placement wait") & (rng.random(n) < 0.65)
    disp[alc_disp_mask] = _choice(rng, ["rehab", "long_term_care"], int(alc_disp_mask.sum()), [0.45, 0.55])
    pae = pd.DataFrame({
        "patient_id_synthetic":[f"PT_{i:07d}" for i in range(n)], "encounter_id":[f"ENC_{i:08d}" for i in range(n)],
        "facility_id":facility, "unit_id":unit_id, "admission_timestamp":admission_ts, "discharge_timestamp":discharge_ts,
        "service_line":service_line, "diagnosis_group":case_mix_group, "case_mix_group":case_mix_group, "triage_level":triage_level,
        "admission_source":admission_source, "age_band":age_band, "frailty_band":frailty_band, "complexity_score":complexity.round(3),
        "expected_los_hours":expected_los.round(2), "actual_los_hours":actual_los.round(2), "medically_ready_timestamp":ready_series,
        "discharge_order_timestamp":order_ts, "alc_status":blocker=="ALC placement wait", "alc_status_late_coded":(blocker=="ALC placement wait") & (rng.random(n)<0.18),
        "discharge_disposition":disp, "readmission_risk_band":_choice(rng, ["low","moderate","high"], n, [0.55,0.33,0.12]),
        "synthetic_primary_blocker":blocker, "synthetic_data_quality_note":np.where(missing_ready,"missing_medically_ready_timestamp",np.where(inconsistent,"timestamp_inconsistency","ok")),
    })
    pae.loc[rng.random(n)<0.015, "discharge_disposition"] = None

    events = []
    event_no = 0
    for r in pae.itertuples(index=False):
        base_events = [("admitted", r.admission_timestamp, r.admission_timestamp, "admitting", "none"), ("medically_ready", r.medically_ready_timestamp, r.medically_ready_timestamp, "most_responsible_provider", r.synthetic_primary_blocker), ("discharge_order_written", r.discharge_order_timestamp, r.discharge_order_timestamp, "most_responsible_provider", r.synthetic_primary_blocker), ("discharged", r.discharge_timestamp, r.discharge_timestamp, "unit_clerk", "none")]
        if r.synthetic_primary_blocker in ("therapy assessment stall", "ALC placement wait") or rng.random() < 0.25:
            req = r.admission_timestamp + pd.Timedelta(hours=float(rng.uniform(12,36))); comp = req + pd.Timedelta(hours=float(rng.uniform(8,96 if r.synthetic_primary_blocker=="therapy assessment stall" else 30)))
            base_events += [("pt_assessment_requested", req, req, "therapy", "none"), ("pt_assessment_completed", req, comp, "therapy", r.synthetic_primary_blocker)]
        if r.synthetic_primary_blocker == "home care confirmation wait" or rng.random() < 0.18:
            req = r.medically_ready_timestamp if pd.notna(r.medically_ready_timestamp) else r.discharge_order_timestamp; comp = req + pd.Timedelta(hours=float(rng.uniform(12,96)))
            base_events += [("home_care_referral_sent", req, req, "transition_services", "none"), ("home_care_confirmed", req, comp, "home_care", r.synthetic_primary_blocker)]
        if r.synthetic_primary_blocker == "ALC placement wait":
            req = r.medically_ready_timestamp if pd.notna(r.medically_ready_timestamp) else r.discharge_order_timestamp; comp = r.discharge_timestamp - pd.Timedelta(hours=float(rng.uniform(1,10)))
            base_events += [("ltc_referral_sent", req, req, "transition_services", "none"), ("ltc_bed_available", req, comp, "community_capacity", r.synthetic_primary_blocker)]
        if r.synthetic_primary_blocker in ("radiology CT turnaround delay", "radiology MRI access delay", "radiology ultrasound turnaround delay", "diagnostic sign-off stall") or rng.random() < 0.22:
            req = r.admission_timestamp + pd.Timedelta(hours=float(rng.uniform(4,30)))
            if r.synthetic_primary_blocker == "radiology MRI access delay":
                modality = "mri"; comp = req + pd.Timedelta(hours=float(rng.uniform(18,96)))
            elif r.synthetic_primary_blocker == "radiology CT turnaround delay":
                modality = "ct"; comp = req + pd.Timedelta(hours=float(rng.uniform(4,36)))
            elif r.synthetic_primary_blocker == "radiology ultrasound turnaround delay":
                modality = "ultrasound"; comp = req + pd.Timedelta(hours=float(rng.uniform(6,42)))
            else:
                modality = "imaging"; comp = req + pd.Timedelta(hours=float(rng.uniform(4,48)))
            base_events += [(f"radiology_{modality}_ordered", req, req, "radiology", "none"), (f"radiology_{modality}_completed", req, comp, "radiology", r.synthetic_primary_blocker)]
        if r.synthetic_primary_blocker == "blood testing turnaround delay" or rng.random() < 0.28:
            req = r.admission_timestamp + pd.Timedelta(hours=float(rng.uniform(1,18))); comp = req + pd.Timedelta(hours=float(rng.uniform(2,30 if r.synthetic_primary_blocker == "blood testing turnaround delay" else 10)))
            base_events += [("blood_test_ordered", req, req, "laboratory", "none"), ("blood_test_available", req, comp, "laboratory", r.synthetic_primary_blocker)]
        if r.synthetic_primary_blocker == "ECG availability delay" or (r.service_line == "cardiology" and rng.random() < 0.35):
            req = r.admission_timestamp + pd.Timedelta(hours=float(rng.uniform(1,12))); comp = req + pd.Timedelta(hours=float(rng.uniform(1,24 if r.synthetic_primary_blocker == "ECG availability delay" else 5)))
            base_events += [("ecg_ordered", req, req, "cardiology_diagnostics", "none"), ("ecg_completed", req, comp, "cardiology_diagnostics", r.synthetic_primary_blocker)]
        if r.synthetic_primary_blocker == "transport delay" or rng.random() < 0.16:
            base_events += [("transport_requested", r.discharge_order_timestamp, r.discharge_order_timestamp, "transport", "none"), ("transport_completed", r.discharge_order_timestamp, r.discharge_timestamp, "transport", r.synthetic_primary_blocker)]
        for etype, req, comp, owner, reason in base_events:
            event_no += 1
            if pd.isna(req): req = r.admission_timestamp + pd.Timedelta(hours=12)
            if pd.isna(comp): comp = req
            events.append({"event_id":f"EVT_{event_no:010d}", "encounter_id":r.encounter_id, "facility_id":r.facility_id, "unit_id":r.unit_id, "event_timestamp":comp, "event_type":etype, "event_status":"completed", "event_owner_service":owner, "requested_timestamp":req, "completed_timestamp":comp, "event_duration_hours":max(0.0,(comp-req).total_seconds()/3600), "is_weekend":comp.dayofweek>=5, "is_after_hours":comp.hour<7 or comp.hour>=17, "delay_reason_observed":reason, "dependency_event_id":None})
    pje = pd.DataFrame(events)

    days = pd.date_range(start, periods=config.days, freq="D")
    brd_rows, svc_rows = [], []
    for fac in facility_ids:
        for unit in UNITS:
            beds = int(rng.integers(24,64))
            for d in days:
                pressure = float(np.clip(0.74 + 0.12*np.sin(2*np.pi*(d.dayofyear%365)/90) + rng.normal(0,0.05), 0.50, 1.05))
                brd_rows.append({"date":d.date().isoformat(), "facility_id":fac, "unit_id":unit, "staffed_beds":beds, "occupied_beds":int(np.clip(round(beds*pressure),0,beds)), "blocked_beds":max(0,int(rng.normal(1.5,1.2)+(unit=="transitional_care")*2)), "boarding_patients":max(0,int(rng.normal(3,2)+pressure*4)), "occupancy_pct":round(pressure,3)})
        for svc in ["PT","OT","imaging","pharmacy","home_care","transport","LTC_placement","rehab_placement"]:
            for d in days:
                weekend = d.dayofweek >= 5; cap = 1.0 - (config.weekend_service_reduction if weekend and svc in ["PT","OT","home_care","LTC_placement","rehab_placement"] else 0.0) + rng.normal(0,0.08)
                svc_rows.append({"date":d.date().isoformat(), "facility_id":fac, "service_name":svc, "is_weekend":weekend, "availability_index":round(float(np.clip(cap,0.20,1.20)),3), "open_hours":8 if weekend and svc in ["PT","OT","home_care"] else 16, "capacity_constraint_note":"weekend_reduced" if weekend and cap<0.8 else "normal"})
    brd, svc = pd.DataFrame(brd_rows), pd.DataFrame(svc_rows)
    for name, df in [("patient_admission_events",pae),("patient_journey_events",pje),("bed_resource_daily",brd),("service_availability",svc)]:
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
    return {"patient_admission_events":pae, "patient_journey_events":pje, "bed_resource_daily":brd, "service_availability":svc}


def build_delay_reference(pae: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    for keys, g in pae.groupby(SEGMENT_COLS, dropna=False):
        los = g["actual_los_hours"].astype(float); med = float(los.median()); mad = float((los-med).abs().median()); sigma = 1.4826*mad; q95 = float(los.quantile(0.95)); limit = max(med + max(2*sigma, 3*mad), q95*0.90)
        rows.append(dict(zip(SEGMENT_COLS, keys), segment_count=len(g), median_los_hours=round(med,2), mad_los_hours=round(mad,2), robust_sigma_hours=round(sigma,2), p95_los_hours=round(q95,2), oob_limit_hours=round(limit,2)))
    ref = pd.DataFrame(rows); ref.to_parquet(out_dir / "discharge_delay_reference.parquet", index=False); return ref


def detect_oob_delays(pae: pd.DataFrame, brd: pd.DataFrame, out_dir: Path, post_ready_hard_cap_hours: float = 48.0):
    ref = build_delay_reference(pae, out_dir)
    f = pae.merge(ref[SEGMENT_COLS + ["median_los_hours", "oob_limit_hours"]], on=SEGMENT_COLS, how="left")
    med_ready = pd.to_datetime(f["medically_ready_timestamp"]); discharge = pd.to_datetime(f["discharge_timestamp"])
    f["hours_after_medically_ready"] = ((discharge-med_ready).dt.total_seconds()/3600).round(2); f.loc[med_ready.isna(), "hours_after_medically_ready"] = np.nan
    f["discharge_date"] = discharge.dt.date.astype(str)
    f["robust_los_oob_flag"] = f["actual_los_hours"] > f["oob_limit_hours"]
    f["post_ready_hard_cap_flag"] = f["hours_after_medically_ready"] > post_ready_hard_cap_hours
    f["hours_above_limit"] = (f["actual_los_hours"] - f["oob_limit_hours"]).clip(lower=0).round(2)
    daily = f.groupby(["discharge_date","facility_id","unit_id"], as_index=False).agg(discharges=("encounter_id","count"), oob_cases=("robust_los_oob_flag","sum"), post_ready_hard_cap_cases=("post_ready_hard_cap_flag","sum"), median_los_hours=("actual_los_hours","median"), median_post_ready_delay_hours=("hours_after_medically_ready","median"))
    daily["oob_rate"] = daily["oob_cases"] / daily["discharges"].clip(lower=1)
    stats = daily.groupby(["facility_id","unit_id"])["oob_rate"].agg(["mean","std"]).reset_index().rename(columns={"mean":"centerline_oob_rate","std":"sigma_oob_rate"})
    daily = daily.merge(stats, on=["facility_id","unit_id"], how="left"); daily["upper_control_limit"] = (daily["centerline_oob_rate"] + 3*daily["sigma_oob_rate"].fillna(0)).clip(upper=1)
    daily["control_chart_signal_flag"] = daily["oob_rate"] > daily["upper_control_limit"]
    sig = daily.loc[daily["control_chart_signal_flag"], ["discharge_date","facility_id","unit_id"]].copy(); sig["control_chart_signal_flag"] = True
    f = f.merge(sig, on=["discharge_date","facility_id","unit_id"], how="left"); f["control_chart_signal_flag"] = f["control_chart_signal_flag"].fillna(False)
    f["oob_flag"] = f["robust_los_oob_flag"] | f["post_ready_hard_cap_flag"] | f["control_chart_signal_flag"]
    f["priority_score"] = (f["hours_above_limit"].fillna(0)*0.6 + f["hours_after_medically_ready"].fillna(0).clip(lower=0)*0.4 + f["post_ready_hard_cap_flag"].astype(int)*24).round(2)
    cols = ["encounter_id","patient_id_synthetic","facility_id","unit_id","service_line","case_mix_group","frailty_band","admission_timestamp","discharge_timestamp","medically_ready_timestamp","actual_los_hours","expected_los_hours","median_los_hours","oob_limit_hours","hours_above_limit","hours_after_medically_ready","robust_los_oob_flag","post_ready_hard_cap_flag","control_chart_signal_flag","oob_flag","priority_score","alc_status","synthetic_primary_blocker","synthetic_data_quality_note"]
    out = f[cols].copy(); out.to_parquet(out_dir / "out_of_bounds_delay_flags.parquet", index=False)
    daily["top_blocker"] = "computed_after_attribution"; daily["estimated_bed_days_recoverable"] = np.nan
    daily.to_csv(out_dir / "admin_delay_dashboard_metrics.csv", index=False); daily.rename(columns={"discharge_date":"date"}).to_csv(out_dir / "control_chart_points.csv", index=False)
    return out, daily, ref


def attribute_blockers(flags: pd.DataFrame, pje: pd.DataFrame, out_dir: Path):
    pivot = pje.pivot_table(index="encounter_id", columns="event_type", values="event_duration_hours", aggfunc="max").reset_index()
    df = flags.merge(pivot, on="encounter_id", how="left")
    rows = []
    for r in df.itertuples(index=False):
        d = r._asdict(); b = "unattributed operational delay"; conf = 0.45; ev = "Delay signal found but no single dependency explains the episode"
        if bool(d.get("alc_status", False)): b, conf, ev = "ALC placement wait", 0.94, "ALC status present with delayed discharge dependency"
        elif d.get("synthetic_primary_blocker") == "Friday/weekend discharge gap": b, conf, ev = "Friday/weekend discharge gap", 0.88, "Medically ready near weekend with discharge delayed into next service window"
        elif d.get("home_care_confirmed", 0) and d.get("home_care_confirmed", 0) > 24: b, conf, ev = "home care confirmation wait", 0.86, "Home-care confirmation exceeded expected operating window"
        elif d.get("pt_assessment_completed", 0) and d.get("pt_assessment_completed", 0) > 36: b, conf, ev = "therapy assessment stall", 0.82, "Therapy assessment completion exceeded expected operating window"
        elif d.get("transport_completed", 0) and d.get("transport_completed", 0) > 10: b, conf, ev = "transport delay", 0.78, "Transport completion lag observed after discharge order"
        elif d.get("synthetic_primary_blocker") == "pharmacy discharge delay": b, conf, ev = "pharmacy discharge delay", 0.74, "Medication reconciliation or discharge medication delay signal"
        elif d.get("radiology_mri_completed", 0) and d.get("radiology_mri_completed", 0) > 24: b, conf, ev = "radiology MRI access delay", 0.84, "MRI completion lag exceeded expected discharge-dependent window"
        elif d.get("radiology_ct_completed", 0) and d.get("radiology_ct_completed", 0) > 16: b, conf, ev = "radiology CT turnaround delay", 0.82, "CT completion lag exceeded expected discharge-dependent window"
        elif d.get("radiology_ultrasound_completed", 0) and d.get("radiology_ultrasound_completed", 0) > 18: b, conf, ev = "radiology ultrasound turnaround delay", 0.80, "Ultrasound completion lag exceeded expected discharge-dependent window"
        elif d.get("blood_test_available", 0) and d.get("blood_test_available", 0) > 8: b, conf, ev = "blood testing turnaround delay", 0.79, "Blood testing result availability exceeded expected operating window"
        elif d.get("ecg_completed", 0) and d.get("ecg_completed", 0) > 6: b, conf, ev = "ECG availability delay", 0.78, "ECG completion lag exceeded expected operating window"
        elif d.get("synthetic_primary_blocker") == "diagnostic sign-off stall": b, conf, ev = "diagnostic sign-off stall", 0.72, "Diagnostic completion or sign-off timing exceeded expected operating window"
        elif bool(d.get("control_chart_signal_flag", False)): b, conf, ev = "unit-level bed-flow bottleneck", 0.62, "Unit-day OOB rate crossed control-chart signal threshold"
        rows.append({"encounter_id":r.encounter_id,"facility_id":r.facility_id,"unit_id":r.unit_id,"service_line":r.service_line,"primary_blocker":b,"secondary_blocker":r.synthetic_primary_blocker if r.synthetic_primary_blocker != b else "none","blocker_confidence":conf,"evidence_summary":ev,"hours_above_limit":r.hours_above_limit,"hours_after_medically_ready":r.hours_after_medically_ready,"estimated_recoverable_bed_hours":round(float(max(0, min(r.hours_above_limit or 0, r.hours_after_medically_ready if pd.notna(r.hours_after_medically_ready) else r.hours_above_limit or 0))),2),"priority_score":r.priority_score})
    attr = pd.DataFrame(rows); attr = attr[flags["oob_flag"].to_numpy()].copy(); attr.to_parquet(out_dir / "delay_blocker_attribution.parquet", index=False)
    owner = {"ALC placement wait":"transition_services_lead","Friday/weekend discharge gap":"site_operations_director","home care confirmation wait":"home_care_liaison","therapy assessment stall":"therapy_services_manager","transport delay":"patient_transport_coordinator","pharmacy discharge delay":"pharmacy_operations_lead","radiology CT turnaround delay":"radiology_operations_lead","radiology MRI access delay":"radiology_operations_lead","radiology ultrasound turnaround delay":"radiology_operations_lead","blood testing turnaround delay":"laboratory_operations_lead","ECG availability delay":"cardiology_diagnostics_lead","diagnostic sign-off stall":"diagnostics_operations_lead","unit-level bed-flow bottleneck":"unit_operations_manager","unattributed operational delay":"patient_flow_coordinator"}
    action = {k:f"Review and address {k.lower()} for this discharge dependency." for k in owner}
    action.update({
        "radiology CT turnaround delay":"Review CT queue priority, protocol readiness, and report finalization for discharge-dependent case.",
        "radiology MRI access delay":"Review MRI access constraints and consider alternate imaging or escalation for discharge-dependent case.",
        "radiology ultrasound turnaround delay":"Review ultrasound slot availability and report finalization for discharge-dependent case.",
        "blood testing turnaround delay":"Review collection, processing, and result-release timing for discharge-dependent bloodwork.",
        "ECG availability delay":"Review ECG completion queue and interpretation handoff for discharge-dependent case.",
    })
    actions = attr.copy(); run_date = pd.Timestamp.utcnow().date().isoformat()
    actions["action_id"] = actions.apply(lambda x: hashlib.sha1(f"{x['encounter_id']}|{x['primary_blocker']}|{run_date}".encode()).hexdigest()[:16], axis=1)
    actions["priority"] = pd.cut(actions["priority_score"], bins=[-1,24,72,999999], labels=["routine","high","urgent"])
    actions["recommended_owner"] = actions["primary_blocker"].map(owner); actions["recommended_action"] = actions["primary_blocker"].map(action)
    actions["target_resolution_hours"] = np.where(actions["priority"].astype(str)=="urgent", 12, np.where(actions["priority"].astype(str)=="high", 24, 48))
    for c in ["status","reviewed_by","reviewed_timestamp","action_taken","reason_not_actioned"]: actions[c] = "new" if c == "status" else ""
    action_cols = ["action_id","encounter_id","facility_id","unit_id","service_line","priority","primary_blocker","evidence_summary","recommended_owner","recommended_action","target_resolution_hours","estimated_recoverable_bed_hours","status","reviewed_by","reviewed_timestamp","action_taken","reason_not_actioned"]
    all_actions = actions[action_cols].sort_values(["estimated_recoverable_bed_hours", "priority"], ascending=[False, False]).copy()
    all_actions.to_csv(out_dir / "delay_resolution_actions_all.csv", index=False)
    executive_count = max(1, int(np.ceil(len(all_actions) * 0.05)))
    executive_actions = all_actions.head(executive_count).copy()
    executive_actions.insert(0, "executive_rank", np.arange(1, len(executive_actions) + 1))
    executive_actions.to_csv(out_dir / "delay_resolution_actions.csv", index=False)
    pareto = attr.groupby("primary_blocker", as_index=False).agg(flagged_cases=("encounter_id","count"), recoverable_bed_hours=("estimated_recoverable_bed_hours","sum"), median_priority_score=("priority_score","median")).sort_values("recoverable_bed_hours", ascending=False); pareto.to_csv(out_dir / "blocker_pareto.csv", index=False)
    heatmap = attr.groupby(["facility_id","unit_id"], as_index=False).agg(flagged_cases=("encounter_id","count"), recoverable_bed_hours=("estimated_recoverable_bed_hours","sum"), median_priority_score=("priority_score","median")).sort_values("recoverable_bed_hours", ascending=False); heatmap.to_csv(out_dir / "unit_delay_heatmap.csv", index=False)
    ranked = attr.groupby(["primary_blocker", "facility_id", "unit_id", "service_line"], as_index=False).agg(flagged_cases=("encounter_id","count"), recoverable_bed_hours=("estimated_recoverable_bed_hours","sum"), median_priority_score=("priority_score","median"), mean_confidence=("blocker_confidence","mean"))
    ranked["actionability_score"] = (ranked["recoverable_bed_hours"] * 0.55 + ranked["flagged_cases"] * 4.0 + ranked["median_priority_score"] * 0.35 + ranked["mean_confidence"] * 20).round(2)
    ranked = ranked.sort_values("actionability_score", ascending=False)
    ranked.insert(0, "signal_rank", np.arange(1, len(ranked) + 1))
    ranked.to_csv(out_dir / "ranked_actionable_signals.csv", index=False)
    audit = all_actions.copy(); audit["adoption_status"] = "pending_review"; audit["simulated_review_outcome"] = np.where(audit["priority"].astype(str).isin(["urgent","high"]), "likely_reviewed", "queue"); audit.to_csv(out_dir / "action_adoption_audit.csv", index=False)
    return attr, executive_actions, pareto, heatmap


def train_prediction_baselines(pae: pd.DataFrame, flags: pd.DataFrame, out_dir: Path, seed: int = 42):
    df = pae.merge(flags[["encounter_id","oob_flag","priority_score"]], on="encounter_id", how="left"); df["oob_flag"] = df["oob_flag"].fillna(False).astype(int)
    df["admission_dow"] = pd.to_datetime(df["admission_timestamp"]).dt.dayofweek; df["admission_hour"] = pd.to_datetime(df["admission_timestamp"]).dt.hour
    cols = ["facility_id","unit_id","service_line","case_mix_group","triage_level","admission_source","age_band","frailty_band","complexity_score","readmission_risk_band","admission_dow","admission_hour"]
    X, yr, yc = df[cols], df["actual_los_hours"].astype(float), df["oob_flag"].astype(int)
    cat = ["facility_id", "unit_id", "service_line", "case_mix_group", "admission_source", "age_band", "frailty_band", "readmission_risk_band"]
    num = [c for c in cols if c not in cat]
    pre = ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat), ("num", "passthrough", num)])
    Xtr, Xte, yrtr, yrte, yctr, ycte, idtr, idte = train_test_split(X, yr, yc, df["encounter_id"], test_size=0.25, random_state=seed, stratify=yc if yc.nunique()>1 else None)
    metrics = {"regression":{}, "classification":{}}; reg_preds = []
    for name, model in {"HistGradientBoostingRegressor":HistGradientBoostingRegressor(random_state=seed, max_iter=120), "ExtraTreesRegressor":ExtraTreesRegressor(n_estimators=60, random_state=seed, min_samples_leaf=4, n_jobs=-1)}.items():
        pipe = Pipeline([("pre",pre),("model",model)]); pipe.fit(Xtr, yrtr); pred = pipe.predict(Xte); reg_preds.append(pred)
        metrics["regression"][name] = {"mae_hours":round(float(mean_absolute_error(yrte,pred)),3), "rmse_hours":round(float(np.sqrt(mean_squared_error(yrte,pred))),3), "median_absolute_error_hours":round(float(median_absolute_error(yrte,pred)),3), "prediction_std_ratio":round(float(np.std(pred)/max(np.std(yrte),1e-9)),3)}
    ensemble_reg = np.mean(np.vstack(reg_preds), axis=0); metrics["regression"]["statistical_ensemble"] = {"mae_hours":round(float(mean_absolute_error(yrte,ensemble_reg)),3), "rmse_hours":round(float(np.sqrt(mean_squared_error(yrte,ensemble_reg))),3), "median_absolute_error_hours":round(float(median_absolute_error(yrte,ensemble_reg)),3), "prediction_std_ratio":round(float(np.std(ensemble_reg)/max(np.std(yrte),1e-9)),3)}
    cls_scores = []
    for name, model in {"HistGradientBoostingClassifier":HistGradientBoostingClassifier(random_state=seed, max_iter=120), "ExtraTreesClassifier":ExtraTreesClassifier(n_estimators=80, random_state=seed, min_samples_leaf=4, n_jobs=-1, class_weight="balanced")}.items():
        pipe = Pipeline([("pre",pre),("model",model)]); pipe.fit(Xtr, yctr); score = pipe.predict_proba(Xte)[:,1]; cls_scores.append(score); top = np.argsort(score)[-max(1,int(0.05*len(score))):]
        metrics["classification"][name] = {"roc_auc":round(float(roc_auc_score(ycte,score)) if ycte.nunique()>1 else 0.0,3), "pr_auc":round(float(average_precision_score(ycte,score)) if ycte.nunique()>1 else 0.0,3), "recall_at_top_5pct_risk":round(float(ycte.to_numpy()[top].sum()/max(ycte.sum(),1)),3), "score_std":round(float(np.std(score)),5)}
    ens = np.mean(np.vstack(cls_scores), axis=0); top = np.argsort(ens)[-max(1,int(0.05*len(ens))):]
    metrics["classification"]["statistical_ensemble"] = {"roc_auc":round(float(roc_auc_score(ycte,ens)) if ycte.nunique()>1 else 0.0,3), "pr_auc":round(float(average_precision_score(ycte,ens)) if ycte.nunique()>1 else 0.0,3), "recall_at_top_5pct_risk":round(float(ycte.to_numpy()[top].sum()/max(ycte.sum(),1)),3), "score_std":round(float(np.std(ens)),5)}
    risk = pd.DataFrame({"encounter_id":idte.to_numpy(), "actual_los_hours":yrte.to_numpy(), "actual_oob_flag":ycte.to_numpy().astype(bool), "predicted_los_hours":ensemble_reg.round(2), "oob_risk_score":ens.round(5), "risk_percentile":pd.Series(ens).rank(pct=True).round(4).to_numpy(), "lead_time_proxy_hours":np.maximum(0,ensemble_reg-48).round(2)}).sort_values("oob_risk_score", ascending=False)
    (out_dir / "prediction_metrics.json").write_text(json.dumps(metrics, indent=2)); risk.to_parquet(out_dir / "patient_oob_risk_scores.parquet", index=False)
    return metrics, risk


def read_table(path: Path):
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def show_top_tables(paths: list[Path], n: int = 5) -> str:
    chunks = []
    for p in paths:
        df = read_table(p); chunks += [f"\n=== {p.name} ===", f"shape: {df.shape}", f"columns: {list(df.columns)}", df.head(n).to_string(index=False)]
    text = "\n".join(chunks); print(text); return text


def build_scenario_detection_summary(out_dir: Path, requested_mode: str) -> pd.DataFrame:
    pareto = pd.read_csv(out_dir / "blocker_pareto.csv")
    family_map = {
        "ALC placement wait": "alc_community_capacity",
        "home care confirmation wait": "alc_community_capacity",
        "Friday/weekend discharge gap": "weekend_flow_gap",
        "therapy assessment stall": "care_team_assessment",
        "transport delay": "transport_discharge_logistics",
        "pharmacy discharge delay": "pharmacy_discharge_readiness",
        "radiology CT turnaround delay": "diagnostics_access",
        "radiology MRI access delay": "diagnostics_access",
        "radiology ultrasound turnaround delay": "diagnostics_access",
        "blood testing turnaround delay": "diagnostics_access",
        "ECG availability delay": "diagnostics_access",
        "diagnostic sign-off stall": "diagnostics_access",
        "unit-level bed-flow bottleneck": "unit_flow_capacity",
        "unattributed operational delay": "unattributed",
    }
    summary = pareto.copy()
    summary["signal_family"] = summary["primary_blocker"].map(family_map).fillna("other")
    summary = summary.groupby("signal_family", as_index=False).agg(
        recoverable_bed_hours=("recoverable_bed_hours", "sum"),
        flagged_cases=("flagged_cases", "sum"),
    ).sort_values("recoverable_bed_hours", ascending=False)
    total = max(float(summary["recoverable_bed_hours"].sum()), 1.0)
    summary["share_of_recoverable_hours"] = (summary["recoverable_bed_hours"] / total).round(4)
    dominant = summary.iloc[0]["signal_family"] if len(summary) else "none"
    scenario_label = {
        "alc_community_capacity": "alc_heavy_detected",
        "diagnostics_access": "diagnostics_heavy_detected",
        "weekend_flow_gap": "weekend_flow_gap_detected",
    }.get(dominant, "balanced_or_mixed_detected")
    summary.insert(0, "requested_scenario_mode", requested_mode)
    summary.insert(1, "scenario_label", scenario_label)
    summary.to_csv(out_dir / "scenario_detection_summary.csv", index=False)
    return summary


def build_dashboard_html(out_dir: Path) -> Path:
    pareto = pd.read_csv(out_dir / "blocker_pareto.csv")
    signals = pd.read_csv(out_dir / "ranked_actionable_signals.csv")
    scenario = pd.read_csv(out_dir / "scenario_detection_summary.csv") if (out_dir / "scenario_detection_summary.csv").exists() else pd.DataFrame()
    actions = pd.read_csv(out_dir / "delay_resolution_actions.csv")
    metrics = pd.read_csv(out_dir / "admin_delay_dashboard_metrics.csv")
    flags = pd.read_parquet(out_dir / "out_of_bounds_delay_flags.parquet")
    total = len(flags)
    oob = int(flags["oob_flag"].sum())
    recoverable = float(pareto["recoverable_bed_hours"].sum()) if len(pareto) else 0.0
    top_signal = signals.iloc[0]["primary_blocker"] if len(signals) else "none"
    css = """
    <style>
      body{font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin:0; background:#f5f2ec; color:#1f2933;}
      header{background:linear-gradient(135deg,#17324d,#0f766e); color:white; padding:34px 46px;}
      main{padding:28px 46px 60px;}
      .grid{display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:16px; margin:22px 0;}
      .card{background:white; border:1px solid #e5dfd3; border-radius:18px; padding:18px; box-shadow:0 12px 30px rgba(31,41,51,.08);}
      .metric{font-size:30px; font-weight:800; margin-top:6px;}
      h1{margin:0; font-size:34px;} h2{margin-top:30px;}
      table{width:100%; border-collapse:collapse; background:white; border-radius:14px; overflow:hidden; box-shadow:0 8px 22px rgba(31,41,51,.06);}
      th,td{padding:10px 12px; border-bottom:1px solid #eee7dc; text-align:left; font-size:13px;}
      th{background:#e8f2ef; color:#17324d;}
      .note{max-width:900px; color:#4b5563; line-height:1.55;}
      .pill{display:inline-block; padding:4px 10px; border-radius:999px; background:#d9f99d; color:#365314; font-weight:700; font-size:12px;}
      .barrow{display:grid; grid-template-columns:260px 1fr 96px; gap:12px; align-items:center; margin:10px 0;}
      .bartrack{height:18px; background:#edf2f7; border-radius:999px; overflow:hidden;}
      .bar{height:18px; background:linear-gradient(90deg,#0f766e,#f59e0b); border-radius:999px;}
      .small{font-size:12px;color:#64748b}.chartwrap{display:grid;grid-template-columns:1fr 1fr;gap:18px}.charttitle{font-weight:800;margin-bottom:8px}
      .filters{display:grid;grid-template-columns:repeat(4,minmax(160px,1fr));gap:12px;margin:18px 0}.filters label{font-size:12px;font-weight:800;color:#475569}.filters select{width:100%;padding:10px;border:1px solid #cbd5e1;border-radius:12px;background:white}.muted{color:#64748b;font-size:13px}
      .barbutton{width:100%;border:0;background:transparent;text-align:left;cursor:pointer;font:inherit;color:inherit}.barbutton:hover .bar{filter:brightness(1.1)}.barbutton:focus{outline:3px solid #f59e0b;outline-offset:3px;border-radius:12px}.toolbar{display:flex;gap:10px;flex-wrap:wrap;margin-top:12px}.toolbar button{border:1px solid #0f766e;background:#0f766e;color:white;border-radius:12px;padding:9px 12px;font-weight:800;cursor:pointer}.toolbar button:hover{background:#115e59}@media print{.toolbar,.filters,#filterStatus{display:none}.card{box-shadow:none}.chartwrap{grid-template-columns:1fr}}
    </style>
    """
    def table_html(df, cols, n=10, table_id=None):
        attrs = f" id='{table_id}'" if table_id else ""
        return df[cols].head(n).to_html(index=False, escape=False).replace("<table", f"<table{attrs}", 1)
    facilities = sorted(signals['facility_id'].dropna().unique().tolist()) if 'facility_id' in signals else []
    units = sorted(signals['unit_id'].dropna().unique().tolist()) if 'unit_id' in signals else []
    families = sorted(scenario['signal_family'].dropna().unique().tolist()) if not scenario.empty and 'signal_family' in scenario else []
    blockers = sorted(signals['primary_blocker'].dropna().unique().tolist()) if 'primary_blocker' in signals else []
    family_for_blocker = {
        "ALC placement wait":"alc_community_capacity", "home care confirmation wait":"alc_community_capacity",
        "Friday/weekend discharge gap":"weekend_flow_gap", "therapy assessment stall":"care_team_assessment",
        "transport delay":"transport_discharge_logistics", "pharmacy discharge delay":"pharmacy_discharge_readiness",
        "radiology CT turnaround delay":"diagnostics_access", "radiology MRI access delay":"diagnostics_access", "radiology ultrasound turnaround delay":"diagnostics_access",
        "blood testing turnaround delay":"diagnostics_access", "ECG availability delay":"diagnostics_access", "diagnostic sign-off stall":"diagnostics_access",
        "unit-level bed-flow bottleneck":"unit_flow_capacity", "unattributed operational delay":"unattributed"
    }
    signals = signals.copy(); signals['signal_family'] = signals['primary_blocker'].map(family_for_blocker).fillna('other')
    actions = actions.copy(); actions['signal_family'] = actions['primary_blocker'].map(family_for_blocker).fillna('other')
    def options(vals):
        return "<option value='all'>All</option>" + "".join([f"<option value='{v}'>{v}</option>" for v in vals])
    def bar_chart(df, label_col, value_col, n=8):
        if df.empty:
            return "<p>No rows.</p>"
        d = df[[label_col, value_col]].head(n).copy()
        max_v = max(float(d[value_col].max()), 1.0)
        rows = []
        for _, row in d.iterrows():
            label = str(row[label_col])
            family = family_for_blocker.get(label, 'other')
            width = max(2, min(100, float(row[value_col]) / max_v * 100))
            rows.append(f"<button class='barrow barbutton' type='button' data-family='{family}' data-blocker='{label}' title='Filter worklist to {family}'><div>{label}</div><div class='bartrack'><div class='bar' style='width:{width:.1f}%'></div></div><div class='small'>{float(row[value_col]):,.1f}</div></button>")
        return "".join(rows)
    trend = metrics.groupby('discharge_date', as_index=False).agg(oob_cases=('oob_cases','sum'), discharges=('discharges','sum'))
    trend['oob_rate_pct'] = trend['oob_cases'] / trend['discharges'].clip(lower=1) * 100
    top_trend = trend.tail(30)
    trend_svg = ""
    if len(top_trend) > 1:
        vals = top_trend['oob_rate_pct'].to_numpy(); max_v = max(float(vals.max()), 1.0); pts=[]
        for i, v in enumerate(vals):
            x = 20 + i * (560 / max(len(vals)-1,1)); y = 170 - (float(v)/max_v*130); pts.append(f"{x:.1f},{y:.1f}")
        trend_svg = f"<svg viewBox='0 0 620 200' width='100%' height='220' role='img'><line x1='20' y1='170' x2='600' y2='170' stroke='#94a3b8'/><line x1='20' y1='30' x2='20' y2='170' stroke='#94a3b8'/><polyline fill='none' stroke='#0f766e' stroke-width='4' points='{' '.join(pts)}'/><text x='20' y='18' fill='#475569' font-size='13'>Daily OOB signal rate, last 30 discharge dates</text></svg>"
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>Operational Delay Sentinel Dashboard</title>{css}</head><body>
<header><span class='pill'>Synthetic dashboard preview</span><h1>Operational Delay Sentinel</h1><p>Ranked operational discharge-delay signals for executive review and action huddles.</p></header>
<main>
<p class='note'>This dashboard highlights the most actionable delay signals, not every delay. The executive worklist is constrained to the top 5% of flagged delay episodes by estimated recoverable bed-hours and priority.</p>
<section class='grid'>
  <div class='card'><div>Patient encounters</div><div class='metric'>{total:,}</div></div>
  <div class='card'><div>OOB delay signals</div><div class='metric'>{oob:,}</div></div>
  <div class='card'><div>Executive worklist rows</div><div class='metric'>{len(actions):,}</div></div>
  <div class='card'><div>Recoverable bed-days</div><div class='metric'>{recoverable/24:,.1f}</div></div>
</section>
<section class='card'><strong>Top signal:</strong> {top_signal}</section>
<section class='card'><div class='charttitle'>Dashboard filters</div><p class='muted'>Filters apply to the ranked signal and executive worklist tables. Click any blocker bar to filter by that signal family.</p><div class='filters'>
<label>Facility<select id='facilityFilter'>{options(facilities)}</select></label>
<label>Unit<select id='unitFilter'>{options(units)}</select></label>
<label>Signal family<select id='familyFilter'>{options(families)}</select></label>
<label>Exact blocker<select id='blockerFilter'>{options(blockers)}</select></label>
</div><div class='toolbar'><button type='button' onclick='resetFilters()'>Reset filters</button><button type='button' onclick='exportVisibleRows("signalsTable","visible_ranked_signals.csv")'>Export visible signals CSV</button><button type='button' onclick='exportVisibleRows("actionsTable","visible_executive_worklist.csv")'>Export visible worklist CSV</button><button type='button' onclick='window.print()'>Print / save PDF</button></div><p id='filterStatus' class='muted'>Showing all rows.</p></section>
<div class='chartwrap'>
<section class='card'><div class='charttitle'>Recoverable bed-hours by blocker</div>{bar_chart(pareto, 'primary_blocker', 'recoverable_bed_hours', 8)}</section>
<section class='card'><div class='charttitle'>OOB trend</div>{trend_svg}</section>
</div>
<h2>Detected scenario mix</h2>{table_html(scenario, ['scenario_label','signal_family','recoverable_bed_hours','share_of_recoverable_hours','flagged_cases'], 12) if not scenario.empty else '<p>No scenario summary.</p>'}
<h2>Ranked actionable signals</h2>{table_html(signals, ['signal_rank','primary_blocker','signal_family','facility_id','unit_id','service_line','flagged_cases','recoverable_bed_hours','actionability_score'], 20, 'signalsTable')}
<h2>Blocker Pareto</h2>{table_html(pareto, ['primary_blocker','flagged_cases','recoverable_bed_hours','median_priority_score'], 12)}
<h2>Executive action worklist</h2>{table_html(actions, ['executive_rank','encounter_id','facility_id','unit_id','signal_family','priority','primary_blocker','recommended_owner','estimated_recoverable_bed_hours'], 20, 'actionsTable')}
<h2>Control-chart daily metrics</h2>{table_html(metrics, ['discharge_date','facility_id','unit_id','discharges','oob_cases','oob_rate','control_chart_signal_flag'], 12)}
<script>
function currentFilters(){{
  return {{
    facility: document.getElementById('facilityFilter').value,
    unit: document.getElementById('unitFilter').value,
    family: document.getElementById('familyFilter').value,
    blocker: document.getElementById('blockerFilter').value
  }};
}}
function updateFilterStatus(visible, total){{
  const f=currentFilters();
  const parts=[];
  if(f.facility!=='all') parts.push('facility '+f.facility);
  if(f.unit!=='all') parts.push('unit '+f.unit);
  if(f.family!=='all') parts.push('family '+f.family);
  if(f.blocker!=='all') parts.push('blocker '+f.blocker);
  document.getElementById('filterStatus').textContent = `${{visible}} of ${{total}} table rows visible${{parts.length ? ' for ' + parts.join(', ') : ''}}.`;
}}
function filterTables(){{
  const f=currentFilters();
  let visible=0,total=0;
  ['signalsTable','actionsTable'].forEach(id=>{{
    const table=document.getElementById(id); if(!table) return;
    const headers=[...table.querySelectorAll('th')].map(th=>th.textContent.trim());
    const fi=headers.indexOf('facility_id'), ui=headers.indexOf('unit_id'), si=headers.indexOf('signal_family'), bi=headers.indexOf('primary_blocker');
    table.querySelectorAll('tbody tr').forEach(row=>{{
      total += 1;
      const cells=row.querySelectorAll('td');
      const okFacility=f.facility==='all'||(fi>=0&&cells[fi].textContent.trim()===f.facility);
      const okUnit=f.unit==='all'||(ui>=0&&cells[ui].textContent.trim()===f.unit);
      const okFamily=f.family==='all'||(si>=0&&cells[si].textContent.trim()===f.family);
      const okBlocker=f.blocker==='all'||(bi>=0&&cells[bi].textContent.trim()===f.blocker);
      const ok = okFacility&&okUnit&&okFamily&&okBlocker;
      row.style.display=ok?'':'none';
      if(ok) visible += 1;
    }});
  }});
  updateFilterStatus(visible,total);
}}
function resetFilters(){{
  ['facilityFilter','unitFilter','familyFilter','blockerFilter'].forEach(id=>document.getElementById(id).value='all');
  filterTables();
}}
function exportVisibleRows(tableId, filename){{
  const table=document.getElementById(tableId); if(!table) return;
  const rows=[...table.querySelectorAll('tr')].filter((row,idx)=>idx===0||row.style.display!=='none');
  const csv=rows.map(row=>[...row.children].map(cell=>'"'+cell.textContent.replaceAll('"','""').trim()+'"').join(',')).join('\n');
  const blob=new Blob([csv],{{type:'text/csv'}});
  const url=URL.createObjectURL(blob);
  const a=document.createElement('a'); a.href=url; a.download=filename; a.click(); URL.revokeObjectURL(url);
}}
document.querySelectorAll('.barbutton').forEach(btn=>btn.addEventListener('click',()=>{{
  document.getElementById('familyFilter').value = btn.dataset.family || 'all';
  document.getElementById('blockerFilter').value = btn.dataset.blocker || 'all';
  filterTables();
  document.getElementById('signalsTable').scrollIntoView({{behavior:'smooth', block:'start'}});
}}));
['facilityFilter','unitFilter','familyFilter','blockerFilter'].forEach(id=>document.getElementById(id).addEventListener('change', filterTables));
filterTables();
</script>
</main></body></html>"""
    path = out_dir / "operational_delay_dashboard.html"
    path.write_text(html)
    return path


def build_reports(out_dir: Path, top_n_text: str = ""):
    flags = pd.read_parquet(out_dir / "out_of_bounds_delay_flags.parquet"); attr = pd.read_parquet(out_dir / "delay_blocker_attribution.parquet"); actions = pd.read_csv(out_dir / "delay_resolution_actions.csv"); pareto = pd.read_csv(out_dir / "blocker_pareto.csv"); signals = pd.read_csv(out_dir / "ranked_actionable_signals.csv"); metrics = json.loads((out_dir / "prediction_metrics.json").read_text()); adm = pd.read_parquet(out_dir / "patient_admission_events.parquet")
    total = len(flags); oob = int(flags["oob_flag"].sum()); hard = int(flags["post_ready_hard_cap_flag"].sum()); rec = float(attr["estimated_recoverable_bed_hours"].sum()) if len(attr) else 0.0; top = pareto.iloc[0]["primary_blocker"] if len(pareto) else "none"
    dq = adm["synthetic_data_quality_note"].value_counts(dropna=False).reset_index(); dq.columns = ["data_quality_note","rows"]
    md = f"""# Operational Delay Sentinel Run Report

## Executive summary

This synthetic run evaluated **{total:,}** patient encounters and flagged **{oob:,}** out-of-bounds operational delay signals.

- OOB signal rate: **{oob / max(total, 1):.1%}**
- Post-medically-ready hard-cap signals: **{hard:,}**
- Estimated recoverable bed-hours: **{rec:,.1f}**
- Estimated recoverable bed-days: **{rec / 24:,.1f}**
- Top blocker: **{top}**

Flags are operational delay signals for review, not punitive findings.

## Model quality

```json
{json.dumps(metrics, indent=2)}
```

## Blocker Pareto

```text\n{pareto.head(10).to_string(index=False)}\n```

## Recommended action preview

```text\n{actions.head(10).to_string(index=False)}\n```

## Data quality summary

```text\n{dq.to_string(index=False)}\n```

## Top-N table previews

```text
{top_n_text}
```

## Recommended next operational actions

1. Review urgent and high-priority rows in `delay_resolution_actions.csv`.
2. Compare blocker Pareto with discharge huddle experience.
3. Use the control chart to distinguish day-to-day noise from systemic delay signals.
4. Record adoption outcomes in `action_adoption_audit.csv` during shadow-mode review.
"""
    md_path = out_dir / "discharge_delay_sentinel_report.md"; html_path = out_dir / "discharge_delay_sentinel_report.html"
    md_path.write_text(md); html_path.write_text("<html><body style='font-family:Arial;max-width:1100px;margin:40px auto;line-height:1.5'><pre>" + md.replace("&","&amp;").replace("<","&lt;") + "</pre></body></html>")
    return md_path, html_path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run Operational Delay Sentinel synthetic workflow.")
    p.add_argument("--facilities", type=int, default=4); p.add_argument("--days", type=int, default=90); p.add_argument("--encounters-per-day", type=int, default=180); p.add_argument("--seed", type=int, default=42); p.add_argument("--oob-rate-target", type=float, default=0.05); p.add_argument("--post-ready-hard-cap-hours", type=float, default=48.0); p.add_argument("--weekend-service-reduction", type=float, default=0.35); p.add_argument("--alc-pressure-multiplier", type=float, default=1.25); p.add_argument("--scenario-mode", choices=["balanced", "alc_heavy", "diagnostics_heavy", "weekend_flow_gap"], default="balanced"); p.add_argument("--out", type=Path, default=Path("outputs/synthetic_90d_v1")); p.add_argument("--print-top-n", type=int, default=5)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv); root = Path(__file__).resolve().parents[2]; out = args.out if args.out.is_absolute() else root / args.out; out.mkdir(parents=True, exist_ok=True)
    cfg = SyntheticFlowConfig(args.facilities, args.days, args.encounters_per_day, args.seed, args.oob_rate_target, args.weekend_service_reduction, args.alc_pressure_multiplier, args.scenario_mode)
    tables = generate_synthetic_flow(cfg, out); flags, daily, ref = detect_oob_delays(tables["patient_admission_events"], tables["bed_resource_daily"], out, args.post_ready_hard_cap_hours); attr, actions, pareto, heatmap = attribute_blockers(flags, tables["patient_journey_events"], out); scenario_summary = build_scenario_detection_summary(out, args.scenario_mode); metrics, risk = train_prediction_baselines(tables["patient_admission_events"], flags, out, args.seed)
    top_paths = [out / x for x in ["patient_admission_events.parquet","patient_journey_events.parquet","bed_resource_daily.parquet","service_availability.parquet","discharge_delay_reference.parquet","out_of_bounds_delay_flags.parquet","delay_blocker_attribution.parquet","ranked_actionable_signals.csv","scenario_detection_summary.csv","delay_resolution_actions.csv","admin_delay_dashboard_metrics.csv"]]
    top = show_top_tables(top_paths, args.print_top_n); dashboard = build_dashboard_html(out); md, html = build_reports(out, top)
    print("\n=== Run summary ==="); print(f"Output directory: {out}"); print(f"Admissions: {len(tables['patient_admission_events']):,}"); print(f"Journey events: {len(tables['patient_journey_events']):,}"); print(f"OOB delay signals: {int(flags['oob_flag'].sum()):,}"); print(f"OOB signal rate: {flags['oob_flag'].mean():.2%}"); print(f"Attributed delay signals: {len(attr):,}"); print(f"Executive worklist actions: {len(actions):,}"); print(f"Requested scenario mode: {args.scenario_mode}"); print(f"Detected scenario label: {scenario_summary.iloc[0]['scenario_label'] if len(scenario_summary) else 'none'}"); print(f"Dashboard HTML: {dashboard}"); print(f"Report markdown: {md}"); print(f"Report HTML: {html}")

if __name__ == "__main__":
    main()
