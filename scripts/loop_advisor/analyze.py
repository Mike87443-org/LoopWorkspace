#!/usr/bin/env python3
"""Loop Advisor: fetches Nightscout data, analyzes with Claude, posts GitHub Issue."""

import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import anthropic
import requests

NIGHTSCOUT_URL = os.environ["NIGHTSCOUT_URL"].rstrip("/")
NIGHTSCOUT_SECRET = os.environ["NIGHTSCOUT_SECRET"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
GITHUB_REPOSITORY = os.environ["GITHUB_REPOSITORY"]

DAYS = 30  # maximum lookback window to fetch
LOW = 70
HIGH = 180


def ns_headers():
    return {
        "api-secret": hashlib.sha1(NIGHTSCOUT_SECRET.encode()).hexdigest(),
        "Content-Type": "application/json",
    }


def ns_fetch(path, params):
    resp = requests.get(
        f"{NIGHTSCOUT_URL}{path}", headers=ns_headers(), params=params, timeout=30
    )
    resp.raise_for_status()
    return resp.json()


def fetch_all():
    since = (datetime.now(timezone.utc) - timedelta(days=DAYS)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    since_ms = int((datetime.now(timezone.utc) - timedelta(days=DAYS)).timestamp() * 1000)

    print("Fetching CGM entries...")
    entries = [
        e for e in ns_fetch("/api/v1/entries.json", {"count": 10000})
        if e.get("date", 0) >= since_ms
    ]

    print("Fetching treatments...")
    # Fetch without server-side date filter (server silently drops records when
    # find[created_at] is used). Filter in Python instead.
    all_treatments = ns_fetch("/api/v1/treatments.json", {"count": 5000})
    treatments = [
        t for t in all_treatments
        if (t.get("created_at") or "") >= since or (t.get("timestamp") or "") >= since
    ]

    print("Fetching device status...")
    device_status = ns_fetch("/api/v1/devicestatus.json", {
        "count": 10000,
        "find[created_at][$gte]": since,
    })

    print("Fetching profile...")
    profile = ns_fetch("/api/v1/profile.json", {})

    print(
        f"  CGM: {len(entries)}, Treatments: {len(treatments)}, "
        f"DeviceStatus: {len(device_status)}"
    )
    return entries, treatments, device_status, profile


def get_local_tz(profile):
    """Extract IANA timezone: env var → Nightscout profile → UTC."""
    env_tz = os.environ.get("NIGHTSCOUT_TZ", "").strip()
    if env_tz:
        try:
            return ZoneInfo(env_tz), env_tz
        except (ZoneInfoNotFoundError, KeyError):
            pass

    if not profile:
        return timezone.utc, "UTC"
    p = profile[0] if isinstance(profile, list) else profile
    tz_name = p.get("timezone")
    if not tz_name:
        store = p.get("store", {})
        key = p.get("defaultProfile", "") or next(iter(store), "")
        tz_name = store.get(key, {}).get("timezone")
    if tz_name:
        try:
            return ZoneInfo(tz_name), tz_name
        except (ZoneInfoNotFoundError, KeyError):
            pass
    return timezone.utc, "UTC"


def _parse_ts(ts_str):
    """Parse an ISO-8601 timestamp string to a timezone-aware datetime."""
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except ValueError:
        return None


def compute_tir(entries):
    sgvs = [e["sgv"] for e in entries if "sgv" in e and e["sgv"] > 0]
    if not sgvs:
        return {"tir": 0, "low": 0, "high": 0, "avg": 0, "std": 0, "cv_pct": 0, "n": 0}
    n = len(sgvs)
    mean = sum(sgvs) / n
    variance = sum((x - mean) ** 2 for x in sgvs) / (n - 1) if n > 1 else 0
    std = math.sqrt(variance)
    return {
        "tir": round(sum(1 for g in sgvs if LOW <= g <= HIGH) / n * 100, 1),
        "low": round(sum(1 for g in sgvs if g < LOW) / n * 100, 1),
        "high": round(sum(1 for g in sgvs if g > HIGH) / n * 100, 1),
        "avg": round(mean, 1),
        "std": round(std, 1),
        "cv_pct": round(std / mean * 100, 1) if mean > 0 else 0,
        "n": n,
    }


def tir_by_hour(entries, local_tz):
    """Hour-of-day TIR breakdown in LOCAL time."""
    buckets = defaultdict(list)
    for e in entries:
        if "sgv" not in e or e["sgv"] <= 0:
            continue
        h = datetime.fromtimestamp(e["date"] / 1000, tz=local_tz).hour
        buckets[h].append(e["sgv"])
    out = {}
    for h in range(24):
        g = buckets[h]
        if not g:
            continue
        n = len(g)
        out[h] = {
            "tir": round(sum(1 for v in g if LOW <= v <= HIGH) / n * 100, 1),
            "low_pct": round(sum(1 for v in g if v < LOW) / n * 100, 1),
            "avg": round(sum(g) / n, 1),
            "n": n,
        }
    return out


def tir_weekday_weekend(entries, local_tz):
    """Split TIR into weekday (Mon–Fri) vs weekend (Sat–Sun)."""
    weekday, weekend = [], []
    for e in entries:
        if "sgv" not in e or e["sgv"] <= 0:
            continue
        dt = datetime.fromtimestamp(e["date"] / 1000, tz=local_tz)
        (weekday if dt.weekday() < 5 else weekend).append(e["sgv"])

    def _tir(sgvs):
        if not sgvs:
            return None
        n = len(sgvs)
        return {
            "tir": round(sum(1 for v in sgvs if LOW <= v <= HIGH) / n * 100, 1),
            "low_pct": round(sum(1 for v in sgvs if v < LOW) / n * 100, 1),
            "avg": round(sum(sgvs) / n, 1),
            "n": n,
        }

    return {"weekday": _tir(weekday), "weekend": _tir(weekend)}


def summarize_treatments(treatments, local_tz, actual_days):
    manual_bolus_types = {"Bolus", "Meal Bolus", "Snack Bolus", "Correction Bolus"}
    manual_boluses = [
        t for t in treatments
        if t.get("eventType") in manual_bolus_types and float(t.get("insulin") or 0) > 0
    ]
    carb_events = [t for t in treatments if float(t.get("carbs") or 0) > 0]
    temp_basals = [t for t in treatments if t.get("eventType") == "Temp Basal"]

    event_types = {}
    for t in treatments:
        et = t.get("eventType", "<none>")
        event_types[et] = event_types.get(et, 0) + 1
    print(f"  Treatment event types: {event_types}")

    # Manual bolus units and carb grams by local hour
    bolus_by_hour = defaultdict(float)
    carbs_by_hour = defaultdict(float)
    for b in manual_boluses:
        dt = _parse_ts(b.get("created_at") or b.get("timestamp", ""))
        if dt:
            bolus_by_hour[dt.astimezone(local_tz).hour] += float(b.get("insulin", 0))
    for c in carb_events:
        dt = _parse_ts(c.get("created_at") or c.get("timestamp", ""))
        if dt:
            carbs_by_hour[dt.astimezone(local_tz).hour] += float(c.get("carbs", 0))

    total_manual_insulin = sum(float(b.get("insulin", 0)) for b in manual_boluses)
    total_carbs = sum(float(c.get("carbs", 0)) for c in carb_events)
    return {
        "manual_bolus_count": len(manual_boluses),
        "avg_manual_bolus_units": (
            round(total_manual_insulin / len(manual_boluses), 2) if manual_boluses else 0
        ),
        "avg_daily_manual_bolus_units": round(total_manual_insulin / actual_days, 2),
        "carb_events_count": len(carb_events),
        "total_carbs_g": round(total_carbs, 1),
        "avg_carbs_per_event": (
            round(total_carbs / len(carb_events), 1) if carb_events else 0
        ),
        "temp_basal_count": len(temp_basals),
        "manual_bolus_insulin_by_local_hour": {
            str(h): round(v, 2) for h, v in sorted(bolus_by_hour.items())
        },
        "carb_grams_by_local_hour": {
            str(h): round(v, 1) for h, v in sorted(carbs_by_hour.items())
        },
    }


def summarize_loop(device_status, local_tz, actual_days):
    records = [d for d in device_status if "loop" in d]
    if not records:
        return {"loop_records": 0}

    failures = sum(1 for d in records if d["loop"].get("failureReason"))
    has_enacted = sum(1 for d in records if d["loop"].get("enacted"))

    # Auto-boluses are in enacted.bolusVolume (not in treatments).
    # Guard: enacted is sometimes a bool rather than a dict.
    auto_bolus_records = [
        d for d in records
        if isinstance(d["loop"].get("enacted"), dict)
        and float(d["loop"]["enacted"].get("bolusVolume") or 0) > 0
    ]
    auto_bolus_total = sum(
        float(d["loop"]["enacted"].get("bolusVolume", 0)) for d in auto_bolus_records
    )

    auto_bolus_by_hour = defaultdict(float)
    for d in auto_bolus_records:
        dt = _parse_ts(d.get("created_at") or d.get("dateString", ""))
        if dt:
            auto_bolus_by_hour[dt.astimezone(local_tz).hour] += float(
                d["loop"]["enacted"].get("bolusVolume", 0)
            )

    # Safe IOB/COB extraction
    iob_vals = [
        d["loop"]["iob"]["iob"]
        for d in records
        if isinstance(d["loop"].get("iob"), dict) and "iob" in d["loop"]["iob"]
    ]

    def _cob_num(c):
        return float(c["cob"]) if isinstance(c, dict) else float(c)

    cob_vals = [
        _cob_num(d["loop"]["cob"])
        for d in records
        if d["loop"].get("cob") is not None
    ]

    print(
        f"  Loop debug: records={len(records)}, enacted={has_enacted}, "
        f"failures={failures}, auto_bolus_events={len(auto_bolus_records)}, "
        f"auto_bolus_total={round(auto_bolus_total, 1)}U"
    )

    return {
        "loop_records": len(records),
        "intervention_rate_pct": round(has_enacted / len(records) * 100, 1),
        "failure_pct": round(failures / len(records) * 100, 1),
        "auto_bolus_events": len(auto_bolus_records),
        "auto_bolus_total_units": round(auto_bolus_total, 2),
        "avg_daily_auto_bolus_units": round(auto_bolus_total / actual_days, 2),
        "auto_bolus_insulin_by_local_hour": {
            str(h): round(v, 2) for h, v in sorted(auto_bolus_by_hour.items())
        },
        "avg_iob": round(sum(iob_vals) / len(iob_vals), 2) if iob_vals else 0,
        "avg_cob": round(sum(cob_vals) / len(cob_vals), 1) if cob_vals else 0,
    }


def analyze_post_meal_windows(entries, treatments, local_tz):
    """For each meal (carb entry >=10g), compute 3-hour post-meal glucose metrics."""
    cgm = sorted(
        [(e["date"] / 1000, e["sgv"]) for e in entries if "sgv" in e and e["sgv"] > 0]
    )

    def nearest_glucose(target_ts, tolerance=360):
        if not cgm:
            return None
        best = min(cgm, key=lambda x: abs(x[0] - target_ts))
        return best[1] if abs(best[0] - target_ts) <= tolerance else None

    meals = []
    for t in treatments:
        carbs = float(t.get("carbs") or 0)
        if carbs < 10:
            continue
        dt = _parse_ts(t.get("created_at") or t.get("timestamp", ""))
        if not dt:
            continue
        t0 = dt.timestamp()

        pre = nearest_glucose(t0)
        g1h = nearest_glucose(t0 + 3600)
        g2h = nearest_glucose(t0 + 7200)
        g3h = nearest_glucose(t0 + 10800)
        window = [v for ts, v in cgm if 0 <= ts - t0 <= 7200]
        peak = max(window) if window else None

        if pre is None or peak is None:
            continue

        meals.append({
            "local_hour": dt.astimezone(local_tz).hour,
            "carbs_g": carbs,
            "pre_meal_glucose": pre,
            "peak_glucose": peak,
            "peak_rise_mg_dl": round(peak - pre, 1),
            "glucose_at_1h": g1h,
            "glucose_at_2h": g2h,
            "glucose_at_3h": g3h,
            "in_range_at_3h": (LOW <= g3h <= HIGH) if g3h is not None else None,
        })

    if not meals:
        return {"note": "No qualifying meal entries (>=10 g carbs) found"}

    rises = [m["peak_rise_mg_dl"] for m in meals]
    in_range_3h = sum(1 for m in meals if m["in_range_at_3h"] is True)
    return {
        "meals_analyzed": len(meals),
        "avg_peak_rise_mg_dl": round(sum(rises) / len(rises), 1),
        "max_peak_rise_mg_dl": round(max(rises), 1),
        "pct_in_range_at_3h": round(in_range_3h / len(meals) * 100, 1),
        "meal_details": meals,
    }


def analyze_isf_effectiveness(entries, treatments, profile_summary):
    """For isolated correction boluses (no food +/-2h), compare actual vs ISF-predicted drop."""
    avg_isf = profile_summary.get("weighted_avg_isf_mg_dl_per_u")
    if not avg_isf:
        return {"note": "No ISF data in profile"}

    cgm = sorted(
        [(e["date"] / 1000, e["sgv"]) for e in entries if "sgv" in e and e["sgv"] > 0]
    )

    def nearest_glucose(target_ts, tolerance=360):
        if not cgm:
            return None
        best = min(cgm, key=lambda x: abs(x[0] - target_ts))
        return best[1] if abs(best[0] - target_ts) <= tolerance else None

    carb_times = [
        _parse_ts(t.get("created_at") or t.get("timestamp", "")).timestamp()
        for t in treatments
        if float(t.get("carbs") or 0) >= 5
        and _parse_ts(t.get("created_at") or t.get("timestamp", ""))
    ]

    results = []
    for t in treatments:
        if t.get("eventType") != "Correction Bolus":
            continue
        units = float(t.get("insulin") or 0)
        if units <= 0:
            continue
        dt = _parse_ts(t.get("created_at") or t.get("timestamp", ""))
        if not dt:
            continue
        t0 = dt.timestamp()
        if any(abs(ct - t0) <= 7200 for ct in carb_times):
            continue  # not isolated — food within +/-2h

        pre = nearest_glucose(t0)
        post_2h = nearest_glucose(t0 + 7200)
        if pre is None or post_2h is None:
            continue

        expected = round(units * avg_isf, 1)
        actual = round(pre - post_2h, 1)
        results.append({
            "units": units,
            "pre_glucose": pre,
            "post_2h_glucose": post_2h,
            "expected_drop_mg_dl": expected,
            "actual_drop_mg_dl": actual,
            "isf_ratio": round(actual / expected, 2) if expected else None,
        })

    if not results:
        return {"note": "No isolated correction boluses found (all occurred within 2h of carb entries)"}

    ratios = [r["isf_ratio"] for r in results if r["isf_ratio"] is not None]
    avg_ratio = round(sum(ratios) / len(ratios), 2) if ratios else None
    return {
        "correction_boluses_analyzed": len(results),
        "avg_isf_mg_dl_per_u": avg_isf,
        "avg_actual_vs_expected_ratio": avg_ratio,
        "interpretation": (
            "ISF well-calibrated (ratio 0.85-1.15)" if avg_ratio and 0.85 <= avg_ratio <= 1.15
            else "Glucose drops MORE than ISF predicts — ISF may be too aggressive" if avg_ratio and avg_ratio > 1.15
            else "Glucose drops LESS than ISF predicts — ISF may be too weak" if avg_ratio and avg_ratio < 0.85
            else "insufficient data"
        ),
        "detail": results[:10],
    }


def summarize_profile(profile):
    if not profile:
        return {}
    p = profile[0] if isinstance(profile, list) else profile
    store = p.get("store", {})
    default = store.get(
        p.get("defaultProfile", ""), store.get(next(iter(store), ""), {})
    )

    def fmt_schedule(schedule):
        """Return time->value dict. Times are LOCAL (Nightscout stores in local time)."""
        if not schedule:
            return None
        return {
            f"{int(s['timeAsSeconds'] // 3600):02d}:{int((s['timeAsSeconds'] % 3600) // 60):02d}":
            round(float(s.get("value", 0)), 3)
            for s in sorted(schedule, key=lambda x: x["timeAsSeconds"])
        }

    def wavg(schedule):
        """Time-weighted average across schedule blocks."""
        if not schedule:
            return None
        sched = sorted(schedule, key=lambda x: x["timeAsSeconds"])
        total_weight = 0.0
        total_val = 0.0
        for i, s in enumerate(sched):
            start = s["timeAsSeconds"]
            end = sched[i + 1]["timeAsSeconds"] if i + 1 < len(sched) else 86400
            weight = end - start
            total_val += float(s.get("value", 0)) * weight
            total_weight += weight
        return round(total_val / total_weight, 2) if total_weight else None

    tz_name = default.get("timezone") or p.get("timezone", "UTC")
    return {
        "timezone": tz_name,
        "basal_schedule_u_hr": fmt_schedule(default.get("basal")),
        "isf_schedule_mg_dl_per_u": fmt_schedule(default.get("sens")),
        "icr_schedule_g_per_u": fmt_schedule(default.get("carbratio")),
        "target_low_schedule": fmt_schedule(default.get("target_low")),
        "target_high_schedule": fmt_schedule(default.get("target_high")),
        "weighted_avg_basal_u_hr": wavg(default.get("basal")),
        "weighted_avg_isf_mg_dl_per_u": wavg(default.get("sens")),
        "weighted_avg_icr_g_per_u": wavg(default.get("carbratio")),
    }


SYSTEM_PROMPT = """\
You are a diabetes technology specialist analyzing Loop closed-loop insulin delivery data.

Produce a thorough report for the patient/caregiver from their Loop data.

Data notes:
- "period_days" is the ACTUAL number of days of Nightscout data available — use this in your report title and throughout. Do NOT assume 30 days.
- If period_days < 14, add a prominent caveat that hourly patterns and trend conclusions are based on limited data and may not yet be statistically reliable.
- All hours are in the patient's LOCAL timezone (from their Nightscout profile).
- "intervention_rate_pct" = % of Loop cycles where delivery was actively changed. This is NOT a closed-loop uptime metric — cycles where Loop ran and kept current delivery unchanged are excluded.
- Auto-boluses are Loop's automatic insulin deliveries (from loop_performance). Manual boluses are corrections the patient entered manually (from treatment_summary).
- "carb_grams_by_local_hour" shows total carbohydrates logged per hour. Cross-reference this directly against "manual_bolus_insulin_by_local_hour" to identify meal windows — do NOT ask whether a bolus was a meal; look at whether carbs were logged in the same hour.
- "cv_pct" = coefficient of variation (std/mean x 100). Target <36% indicates stable glucose control.
- Profile schedules show the actual time-block settings in local time. Use these for specific recommendations.
- "tir_by_local_hour" includes "n" = number of CGM readings in that hour bucket. With 5-min CGM, 1 night = ~12 readings/hour. When n < 24 (fewer than 2 nights of data), the low_pct for that hour is statistically unreliable — a single low event can inflate it dramatically. Always note the sample count (e.g. "32% lows at 2 AM, n=14 — only 1 night of data") and qualify any hour with n < 24 as insufficient to draw conclusions from.
- "tir_weekday_vs_weekend" compares weekday (Mon-Fri) vs weekend (Sat-Sun) glucose. If one is significantly worse, discuss likely lifestyle causes (meal timing, sleep, activity).
- "post_meal_analysis" tracks glucose in the 3 hours after each logged carb entry (>=10g). "avg_peak_rise_mg_dl" is the mean glucose rise from pre-meal baseline to peak; target <80 mg/dL rise for good ICR. "pct_in_range_at_3h" shows how often glucose returned to 70-180 by 3h post-meal; target >80%. Use meal_details (indexed by local_hour) to identify which meal windows have the worst post-meal excursions and relate them to the current ICR schedule.
- "isf_effectiveness" analyzes isolated correction boluses where no food was logged within +/-2h. "avg_actual_vs_expected_ratio" near 1.0 means ISF is well-calibrated; >1.15 means glucose dropped more than predicted (ISF too aggressive); <0.85 means it dropped less (ISF too weak). If no isolated boluses exist, state that ISF cannot be assessed from this period's data.

Time format rules — STRICTLY ENFORCED throughout the entire report:
- ALWAYS use 12-hour AM/PM format for clock times in all narrative, tables, section headers, and recommendations. Examples: "2 AM", "6 PM", "10:30 PM", "7 AM-9 AM".
- NEVER write 24-hour time (e.g. never write 02:00, 18:00, 22:00, 14:00).
- Profile schedule keys (from current_settings) are stored as "HH:MM" internally — when citing them in your output, convert to AM/PM (e.g. "00:00" -> "12 AM", "14:00" -> "2 PM").

Rules:
- Be specific and quantitative — cite actual numbers.
- For every setting change recommendation state: which schedule block to change, current value, suggested new value, and rationale from the data.
- Distinguish settings changes from behavioral adjustments.
- Flag safety concerns prominently (especially recurring lows), but always note sample size when n is small.
- Skip generic diabetes advice; focus on what the Loop data shows.
- Format output as GitHub-flavored Markdown.

Use these exact section headers:
## Summary
## Time in Range & Variability
(include weekday vs weekend comparison if there is a meaningful difference)
## Overnight Performance (10 PM-6 AM)
## Daytime & Post-Meal Performance
(draw on post_meal_analysis meal_details — cite specific meal windows, their pre-meal glucose, peak rise, and whether glucose was in range at 3h)
## Insulin Delivery Analysis
## ISF & ICR Effectiveness
(use isf_effectiveness to assess ISF calibration; use post_meal_analysis to assess ICR; cite specific numbers)
## Setting Change Recommendations
## Customization Opportunities
(Customization Opportunities = findings requiring a Loop fork code change, not just settings. Write "None identified this month." if nothing qualifies.)"""


def run_analysis(payload):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=20000,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        system=[{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{
            "role": "user",
            "content": (
                "Analyze this Loop data and produce the full report:\n\n"
                f"```json\n{json.dumps(payload, indent=2)}\n```"
            ),
        }],
    )
    # Extended thinking returns thinking blocks + text blocks; extract only text
    return "\n".join(b.text for b in msg.content if b.type == "text")


def post_issue(title, body):
    owner, repo = GITHUB_REPOSITORY.split("/", 1)
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    label = "loop-advisor"

    requests.post(
        f"https://api.github.com/repos/{owner}/{repo}/labels",
        headers=headers,
        json={"name": label, "color": "0075ca", "description": "Automated Loop analysis"},
        timeout=30,
    )

    existing = [
        i for i in requests.get(
            f"https://api.github.com/repos/{owner}/{repo}/issues",
            headers=headers,
            params={"state": "open", "labels": label, "per_page": 20},
            timeout=30,
        ).json()
        if i.get("title") == title
    ]

    if existing:
        resp = requests.patch(
            f"https://api.github.com/repos/{owner}/{repo}/issues/{existing[0]['number']}",
            headers=headers,
            json={"body": body},
            timeout=30,
        )
    else:
        resp = requests.post(
            f"https://api.github.com/repos/{owner}/{repo}/issues",
            headers=headers,
            json={"title": title, "body": body, "labels": [label]},
            timeout=30,
        )

    resp.raise_for_status()
    url = resp.json()["html_url"]
    print(f"Issue: {url}")
    return url


def main():
    entries, treatments, device_status, profile = fetch_all()
    local_tz, tz_name = get_local_tz(profile)
    print(f"  Timezone: {tz_name}")

    # Detect actual data span from earliest CGM reading
    now_utc = datetime.now(timezone.utc)
    if entries:
        earliest_ms = min(e["date"] for e in entries if "date" in e)
        earliest_dt = datetime.fromtimestamp(earliest_ms / 1000, tz=timezone.utc)
        actual_days = max(1, min(DAYS, round((now_utc - earliest_dt).total_seconds() / 86400)))
    else:
        actual_days = DAYS
    print(f"  Data span: {actual_days} days (requested {DAYS})")

    profile_summary = summarize_profile(profile)
    payload = {
        "period_days": actual_days,
        "analysis_date": now_utc.strftime("%Y-%m-%d"),
        "local_timezone": tz_name,
        "overall_tir": compute_tir(entries),
        "tir_by_local_hour": tir_by_hour(entries, local_tz),
        "tir_weekday_vs_weekend": tir_weekday_weekend(entries, local_tz),
        "treatment_summary": summarize_treatments(treatments, local_tz, actual_days),
        "loop_performance": summarize_loop(device_status, local_tz, actual_days),
        "post_meal_analysis": analyze_post_meal_windows(entries, treatments, local_tz),
        "isf_effectiveness": analyze_isf_effectiveness(entries, treatments, profile_summary),
        "current_settings": profile_summary,
    }
    print(f"Overall TIR: {payload['overall_tir']['tir']}%, CV: {payload['overall_tir']['cv_pct']}%")

    print("Running Claude analysis...")
    report = run_analysis(payload)

    if actual_days >= 25:
        period_label = "Monthly"
    else:
        period_label = f"{actual_days}-Day"

    date_str = now_utc.strftime("%Y-%m-%d")
    post_issue(f"Loop Advisor — {period_label} Report ({date_str})", report)


if __name__ == "__main__":
    main()
