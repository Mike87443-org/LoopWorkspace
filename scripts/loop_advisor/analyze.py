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

DAYS = 30
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
    treatments = ns_fetch("/api/v1/treatments.json", {
        "count": 5000,
        "find[created_at][$gte]": since,
    })

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
    """Extract IANA timezone from Nightscout profile. Falls back to UTC."""
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


def summarize_treatments(treatments, local_tz):
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

    # Manual bolus units by local hour (to correlate with meal times)
    bolus_by_hour = defaultdict(float)
    for b in manual_boluses:
        dt = _parse_ts(b.get("created_at") or b.get("timestamp", ""))
        if dt:
            bolus_by_hour[dt.astimezone(local_tz).hour] += float(b.get("insulin", 0))

    total_manual_insulin = sum(float(b.get("insulin", 0)) for b in manual_boluses)
    return {
        "manual_bolus_count": len(manual_boluses),
        "avg_manual_bolus_units": (
            round(total_manual_insulin / len(manual_boluses), 2) if manual_boluses else 0
        ),
        "avg_daily_manual_bolus_units": round(total_manual_insulin / DAYS, 2),
        "carb_events_count": len(carb_events),
        "avg_carbs_per_event": (
            round(
                sum(float(c.get("carbs", 0)) for c in carb_events) / len(carb_events), 1
            )
            if carb_events else 0
        ),
        "temp_basal_count": len(temp_basals),
        "manual_bolus_insulin_by_local_hour": {
            str(h): round(v, 2) for h, v in sorted(bolus_by_hour.items())
        },
    }


def summarize_loop(device_status, local_tz):
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

    # Auto-bolus distribution by local hour
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
        "avg_daily_auto_bolus_units": round(auto_bolus_total / DAYS, 2),
        "auto_bolus_insulin_by_local_hour": {
            str(h): round(v, 2) for h, v in sorted(auto_bolus_by_hour.items())
        },
        "avg_iob": round(sum(iob_vals) / len(iob_vals), 2) if iob_vals else 0,
        "avg_cob": round(sum(cob_vals) / len(cob_vals), 1) if cob_vals else 0,
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
        """Return time→value dict. Times are LOCAL (Nightscout stores in local time)."""
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

Produce a thorough monthly report for the patient/caregiver from 30 days of Loop data.

Data notes:
- All hours are in the patient's LOCAL timezone (from their Nightscout profile).
- "intervention_rate_pct" = % of Loop cycles where delivery was actively changed. This is NOT a closed-loop uptime metric — cycles where Loop ran and kept current delivery unchanged are excluded.
- Auto-boluses are Loop's automatic insulin deliveries (from loop_performance). Manual boluses are corrections the patient entered manually (from treatment_summary).
- "cv_pct" = coefficient of variation (std/mean × 100). Target <36% indicates stable glucose control.
- Profile schedules show the actual time-block settings in local time. Use these for specific recommendations.

Rules:
- Be specific and quantitative — cite actual numbers.
- For every setting change recommendation state: which schedule block to change, current value, suggested new value, and rationale from the data.
- Distinguish settings changes from behavioral adjustments.
- Flag safety concerns prominently (especially recurring lows).
- Skip generic diabetes advice; focus on what the Loop data shows.
- Format output as GitHub-flavored Markdown.

Use these exact section headers:
## Summary
## Time in Range & Variability
## Overnight Performance (10pm–6am)
## Daytime & Post-Meal Performance
## Insulin Delivery Analysis
## Setting Change Recommendations
## Customization Opportunities
(Customization Opportunities = findings requiring a Loop fork code change, not just settings. Write "None identified this week." if nothing qualifies.)"""


def run_analysis(payload):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=8192,
        system=[{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{
            "role": "user",
            "content": (
                "Analyze this Loop data and produce the monthly report:\n\n"
                f"```json\n{json.dumps(payload, indent=2)}\n```"
            ),
        }],
    )
    return msg.content[0].text


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

    payload = {
        "period_days": DAYS,
        "analysis_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "local_timezone": tz_name,
        "overall_tir": compute_tir(entries),
        "tir_by_local_hour": tir_by_hour(entries, local_tz),
        "treatment_summary": summarize_treatments(treatments, local_tz),
        "loop_performance": summarize_loop(device_status, local_tz),
        "current_settings": summarize_profile(profile),
    }
    print(f"Overall TIR: {payload['overall_tir']['tir']}%, CV: {payload['overall_tir']['cv_pct']}%")

    print("Running Claude analysis...")
    report = run_analysis(payload)

    week = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    post_issue(f"Loop Advisor — Monthly Report ({week})", report)


if __name__ == "__main__":
    main()
