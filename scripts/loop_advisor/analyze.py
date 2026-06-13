#!/usr/bin/env python3
"""Loop Advisor: fetches Nightscout data, analyzes with Claude, posts GitHub Issue."""

import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone

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


def compute_tir(entries):
    sgvs = [e["sgv"] for e in entries if "sgv" in e and e["sgv"] > 0]
    if not sgvs:
        return {"tir": 0, "low": 0, "high": 0, "avg": 0, "n": 0}
    n = len(sgvs)
    return {
        "tir": round(sum(1 for g in sgvs if LOW <= g <= HIGH) / n * 100, 1),
        "low": round(sum(1 for g in sgvs if g < LOW) / n * 100, 1),
        "high": round(sum(1 for g in sgvs if g > HIGH) / n * 100, 1),
        "avg": round(sum(sgvs) / n, 1),
        "n": n,
    }


def tir_by_hour(entries):
    buckets = defaultdict(list)
    for e in entries:
        if "sgv" not in e or e["sgv"] <= 0:
            continue
        h = datetime.fromtimestamp(e["date"] / 1000, tz=timezone.utc).hour
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


def summarize_treatments(treatments):
    # Catch all insulin-delivering treatments regardless of eventType label
    boluses = [t for t in treatments if float(t.get("insulin") or 0) > 0]

    # Log the event types actually seen for debugging
    event_types = {}
    for t in treatments:
        et = t.get("eventType", "<none>")
        event_types[et] = event_types.get(et, 0) + 1
    print(f"  Treatment event types: {event_types}")

    manual_bolus_types = {"Bolus", "Meal Bolus", "Snack Bolus", "Correction Bolus"}
    manual_boluses = [t for t in boluses if t.get("eventType") in manual_bolus_types]
    auto_boluses = [t for t in boluses if t.get("eventType") not in manual_bolus_types]

    carb_events = [t for t in treatments if float(t.get("carbs") or 0) > 0]
    temp_basals = [t for t in treatments if t.get("eventType") == "Temp Basal"]
    return {
        "bolus_count": len(boluses),
        "manual_bolus_count": len(manual_boluses),
        "auto_bolus_count": len(auto_boluses),
        "avg_bolus_units": (
            round(sum(float(b.get("insulin", 0)) for b in boluses) / len(boluses), 2)
            if boluses else 0
        ),
        "total_daily_bolus_avg": round(
            sum(float(b.get("insulin", 0)) for b in boluses) / DAYS, 2
        ),
        "carb_events_count": len(carb_events),
        "avg_carbs_per_meal": (
            round(
                sum(float(c.get("carbs", 0)) for c in carb_events) / len(carb_events),
                1,
            )
            if carb_events else 0
        ),
        "temp_basal_count": len(temp_basals),
    }


def summarize_loop(device_status):
    records = [d for d in device_status if "loop" in d]
    if not records:
        return {"loop_records": 0}

    failures = sum(1 for d in records if d["loop"].get("failureReason"))

    # "Closed loop" = Loop made a recommendation (auto or manual mode active).
    # Records with neither enacted nor recommended indicate open-loop/no-data cycles.
    has_enacted = sum(1 for d in records if d["loop"].get("enacted"))
    has_recommended = sum(1 for d in records if d["loop"].get("recommended"))
    # A cycle is active if Loop produced any recommendation, enacted or not
    active = sum(
        1 for d in records
        if d["loop"].get("enacted") or d["loop"].get("recommended")
    )

    # Auto-boluses: stored in enacted.bolusVolume, not in treatments
    auto_bolus_records = [
        d for d in records
        if d["loop"].get("enacted", {}).get("bolusVolume", 0) > 0
    ]
    auto_bolus_total = sum(
        float(d["loop"]["enacted"].get("bolusVolume", 0))
        for d in auto_bolus_records
    )

    iob_vals = [d["loop"]["iob"]["iob"] for d in records if d["loop"].get("iob")]
    def _cob_num(c):
        return float(c["cob"]) if isinstance(c, dict) else float(c)
    cob_vals = [_cob_num(d["loop"]["cob"]) for d in records if d["loop"].get("cob") is not None]

    print(
        f"  Loop uptime debug: records={len(records)}, enacted={has_enacted}, "
        f"recommended={has_recommended}, active={active}, failures={failures}, "
        f"auto_bolus_events={len(auto_bolus_records)}"
    )

    return {
        "loop_records": len(records),
        "closed_loop_pct": round(active / len(records) * 100, 1),
        "enacted_pct": round(has_enacted / len(records) * 100, 1),
        "failure_pct": round(failures / len(records) * 100, 1),
        "auto_bolus_events": len(auto_bolus_records),
        "auto_bolus_total_units": round(auto_bolus_total, 2),
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

    def avg(schedule):
        if not schedule:
            return None
        return round(
            sum(float(s.get("value", 0)) for s in schedule) / len(schedule), 2
        )

    return {
        "avg_basal_u_hr": avg(default.get("basal")),
        "avg_isf_mg_dl_per_u": avg(default.get("sens")),
        "avg_icr_g_per_u": avg(default.get("carbratio")),
        "target_low": (default.get("target_low") or [{}])[0].get("value"),
        "target_high": (default.get("target_high") or [{}])[0].get("value"),
    }


SYSTEM_PROMPT = """\
You are a diabetes technology specialist analyzing Loop closed-loop insulin delivery data.

Produce a practical weekly report for the patient/caregiver from 30 days of Loop data.

Rules:
- Be specific and quantitative — cite the actual numbers from the data.
- Distinguish settings changes from behavioral adjustments.
- For any setting change recommendation, include direction and magnitude (e.g. "increase ISF from ~X to ~Y mg/dL per unit during 6am–10am").
- Flag safety concerns prominently.
- Stay focused on what the Loop data reveals; skip generic diabetes advice.
- Format output as GitHub-flavored Markdown.

Use these exact section headers:
## Summary
## Time in Range
## Overnight Performance (10pm–6am)
## Post-Meal Performance
## Loop System Performance
## Setting Change Recommendations
## Customization Opportunities
(Customization Opportunities = findings requiring a Loop fork code change, not just settings. Write "None identified this week." if nothing qualifies.)"""


def run_analysis(payload):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=4096,
        system=[{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{
            "role": "user",
            "content": (
                "Analyze this Loop data and produce the weekly report:\n\n"
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

    payload = {
        "period_days": DAYS,
        "analysis_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "overall_tir": compute_tir(entries),
        "tir_by_hour": tir_by_hour(entries),
        "treatment_summary": summarize_treatments(treatments),
        "loop_performance": summarize_loop(device_status),
        "current_settings": summarize_profile(profile),
    }
    print(f"Overall TIR: {payload['overall_tir']['tir']}%")

    print("Running Claude analysis...")
    report = run_analysis(payload)

    week = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    post_issue(f"Loop Advisor — Weekly Report ({week})", report)


if __name__ == "__main__":
    main()
