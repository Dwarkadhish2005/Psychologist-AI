"""
Analytics router — serves historical data from JSON memory files.
"""

import csv
import io
import json
import datetime as dt_module
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse, HTMLResponse

router = APIRouter()

MEMORY_DIR = Path("data/user_memory")
MAX_ALERTS_STORED = 500


def _load_memory(user_id: str) -> dict:
    f = MEMORY_DIR / f"{user_id}_longterm_memory.json"
    if not f.exists():
        raise HTTPException(status_code=404, detail="No history for this user")
    with open(f, encoding="utf-8") as fh:
        return json.load(fh)


@router.get("/{user_id}/history")
def get_daily_history(
    user_id: str,
    days: int = Query(default=30, ge=1, le=365),
):
    """Return the last N daily profiles for a user."""
    data = _load_memory(user_id)
    daily: dict = data.get("daily_profiles", {})
    sorted_dates = sorted(daily.keys())[-days:]
    return {date: daily[date] for date in sorted_dates}


@router.get("/{user_id}/psv")
def get_psv(user_id: str):
    """Return the Personality State Vector for a user."""
    f = MEMORY_DIR / f"{user_id}_psv.json"
    if not f.exists():
        raise HTTPException(
            status_code=404, detail="No PSV data yet — complete more sessions"
        )
    with open(f, encoding="utf-8") as fh:
        return json.load(fh)


@router.get("/{user_id}/sessions")
def get_sessions(
    user_id: str,
    limit: int = Query(default=20, ge=1, le=200),
):
    """Return recent session summaries from long-term memory."""
    data = _load_memory(user_id)
    sessions = data.get("sessions", [])
    # newest first
    return sessions[-limit:][::-1]


@router.get("/{user_id}/summary")
def get_summary(user_id: str):
    """High-level summary: total sessions, days tracked, latest risk."""
    data = _load_memory(user_id)
    daily = data.get("daily_profiles", {})
    sessions = data.get("sessions", [])

    latest_risk = None
    if daily:
        latest_day = daily[sorted(daily.keys())[-1]]
        latest_risk = latest_day.get("avg_risk_level")

    return {
        "total_sessions": len(sessions),
        "days_tracked": len(daily),
        "latest_risk_avg": latest_risk,
        "last_updated": data.get("last_updated"),
    }


# ─── Alerts ──────────────────────────────────────────────────────────────────

@router.get("/{user_id}/alerts")
def get_alerts(
    user_id: str,
    limit: int = Query(default=50, ge=1, le=500),
):
    """Return recent deviation alerts for a user (newest first)."""
    f = MEMORY_DIR / f"{user_id}_alerts.json"
    if not f.exists():
        return {"alerts": [], "total": 0}
    with open(f, encoding="utf-8") as fh:
        data = json.load(fh)
    alerts = data.get("alerts", [])
    return {"alerts": alerts[-limit:][::-1], "total": len(alerts)}


@router.delete("/{user_id}/alerts")
def clear_alerts(user_id: str):
    """Clear all stored alerts for a user."""
    f = MEMORY_DIR / f"{user_id}_alerts.json"
    if f.exists():
        with open(f, "w", encoding="utf-8") as fh:
            json.dump({"alerts": []}, fh)
    return {"message": "Alerts cleared"}


# ─── Export ───────────────────────────────────────────────────────────────────

@router.get("/{user_id}/export/json")
def export_json(user_id: str):
    """Download full memory + PSV + alerts as a single JSON file."""
    data = _load_memory(user_id)

    psv_f = MEMORY_DIR / f"{user_id}_psv.json"
    if psv_f.exists():
        with open(psv_f, encoding="utf-8") as fh:
            data["psv"] = json.load(fh)

    alerts_f = MEMORY_DIR / f"{user_id}_alerts.json"
    if alerts_f.exists():
        with open(alerts_f, encoding="utf-8") as fh:
            data["alerts"] = json.load(fh).get("alerts", [])

    content = json.dumps(data, indent=2, default=str)
    filename = f"psychologist_ai_{user_id}_export.json"
    return StreamingResponse(
        io.BytesIO(content.encode()),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{user_id}/export/csv")
def export_csv(user_id: str):
    """Download daily profiles as CSV."""
    data = _load_memory(user_id)
    daily: dict = data.get("daily_profiles", {})

    fieldnames = [
        "date", "total_sessions", "total_duration_minutes",
        "avg_stress_ratio", "avg_high_stress_ratio", "avg_stress_intensity",
        "avg_confidence", "avg_stability", "avg_risk_level",
        "high_risk_duration_ratio", "total_risk_escalations",
        "positive_ratio", "negative_ratio", "neutral_ratio",
        "emotional_volatility", "total_masking_events", "masking_duration_ratio",
        "top_mental_state",
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()

    for date_str in sorted(daily.keys()):
        row = dict(daily[date_str])
        states = row.get("dominant_mental_states", [])
        row["top_mental_state"] = states[0][0] if states else ""
        for k, v in row.items():
            if isinstance(v, float):
                row[k] = round(v, 4)
        writer.writerow(row)

    filename = f"psychologist_ai_{user_id}_daily.csv"
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ─── HTML Report ─────────────────────────────────────────────────────────────

@router.get("/{user_id}/report", response_class=HTMLResponse)
def generate_report(user_id: str):
    """Generate a self-contained HTML psychological profile report."""
    data = _load_memory(user_id)
    daily = data.get("daily_profiles", {})

    psv = None
    psv_f = MEMORY_DIR / f"{user_id}_psv.json"
    if psv_f.exists():
        with open(psv_f, encoding="utf-8") as fh:
            psv = json.load(fh)

    alerts: list = []
    alerts_f = MEMORY_DIR / f"{user_id}_alerts.json"
    if alerts_f.exists():
        with open(alerts_f, encoding="utf-8") as fh:
            alerts = json.load(fh).get("alerts", [])

    user_name = "_".join(user_id.split("_")[:-2]).replace("_", " ").title() or user_id
    total_sessions = sum(d.get("total_sessions", 0) for d in daily.values())
    total_minutes = round(sum(d.get("total_duration_minutes", 0) for d in daily.values()))
    days_tracked = len(daily)
    avg_stress_str = "—"
    if daily:
        avg_v = sum(d.get("avg_stress_ratio", 0) for d in daily.values()) / len(daily)
        avg_stress_str = f"{round(avg_v * 100)}%"

    def pct(v: float) -> str:
        return f"{round(v * 100)}%"

    # Daily table rows (last 30, newest first)
    daily_rows = ""
    for d_str in sorted(daily.keys(), reverse=True)[:30]:
        d = daily[d_str]
        risk_val = d.get("avg_risk_level", 0)
        risk_label = "Low" if risk_val < 1 else "Moderate" if risk_val < 2 else "High" if risk_val < 3 else "Critical"
        risk_color = "#22c55e" if risk_val < 1 else "#f59e0b" if risk_val < 2 else "#ef4444" if risk_val < 3 else "#dc2626"
        top_state = (d.get("dominant_mental_states") or [["—"]])[0][0].replace("_", " ").title()
        daily_rows += (
            f"<tr><td>{d_str}</td><td>{d.get('total_sessions',0)}</td>"
            f"<td>{round(d.get('total_duration_minutes',0),1)} min</td>"
            f"<td>{round(d.get('avg_stress_ratio',0)*100)}%</td>"
            f"<td><span style='color:{risk_color}'>{risk_label}</span></td>"
            f"<td>{top_state}</td>"
            f"<td>{round(d.get('avg_stability',0)*100)}%</td></tr>"
        )

    # PSV section
    psv_section = ""
    if psv:
        def trait_bar(label: str, val: float, color: str) -> str:
            return (
                f"<div class='trait'><span>{label}</span>"
                f"<div class='bar'><div style='width:{pct(val)};background:{color}'></div></div>"
                f"<b>{pct(val)}</b></div>"
            )
        psv_section = (
            f"<div class='card'><h2>Personality State Vector</h2>"
            f"<p class='subtitle'>Based on {psv.get('total_sessions_processed',0)} sessions"
            f" · Confidence: {pct(psv.get('confidence',0))}</p>"
            f"<div class='traits'>"
            + trait_bar("Emotional Stability", psv.get('emotional_stability', 0), "#6366f1")
            + trait_bar("Stress Sensitivity",  psv.get('stress_sensitivity', 0),  "#f97316")
            + trait_bar("Recovery Speed",       psv.get('recovery_speed', 0),       "#22c55e")
            + trait_bar("Positivity Bias",      psv.get('positivity_bias', 0),      "#eab308")
            + trait_bar("Consistency", 1 - psv.get('volatility', 0),                "#0ea5e9")
            + "</div></div>"
        )

    # Alerts section
    alerts_section = ""
    if alerts:
        rows = ""
        for a in alerts[-10:][::-1]:
            ts = a.get("timestamp", 0)
            dt_str = dt_module.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "—"
            sev = round(a.get("severity", 0) * 100)
            sev_color = "#22c55e" if sev < 50 else "#f59e0b" if sev < 75 else "#ef4444"
            rows += (
                f"<tr><td>{dt_str}</td>"
                f"<td>{a.get('type','').replace('_',' ').title()}</td>"
                f"<td style='color:{sev_color}'>{sev}%</td>"
                f"<td>{a.get('description','')}</td></tr>"
            )
        alerts_section = (
            "<div class='card'><h2>Recent Deviation Alerts</h2>"
            "<table><thead><tr><th>Time</th><th>Type</th><th>Severity</th><th>Description</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
        )

    report_date = dt_module.date.today().isoformat()
    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Psychologist AI Report — {user_name}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f1117;color:#e2e8f0;padding:2rem}}
h1{{color:#fff;font-size:1.8rem}}h2{{font-size:1.1rem;color:#e2e8f0;margin-bottom:.75rem}}
.subtitle{{color:#64748b;font-size:.85rem;margin-bottom:1rem}}
.header{{border-bottom:1px solid #334155;padding-bottom:1.5rem;margin-bottom:2rem}}
.header p{{color:#64748b;margin-top:.25rem}}
.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:2rem}}
.stat{{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:1.25rem}}
.stat .val{{font-size:1.8rem;font-weight:700;color:#818cf8}}
.stat .lbl{{font-size:.8rem;color:#64748b;margin-top:.25rem}}
.card{{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:1.5rem;margin-bottom:1.5rem}}
table{{width:100%;border-collapse:collapse;font-size:.85rem}}
th{{text-align:left;color:#64748b;font-weight:600;padding:.5rem .75rem;border-bottom:1px solid #334155}}
td{{padding:.5rem .75rem;border-bottom:1px solid #1e293b}}
.traits{{display:flex;flex-direction:column;gap:.75rem}}
.trait{{display:flex;align-items:center;gap:1rem}}
.trait span{{width:180px;font-size:.85rem;color:#94a3b8;flex-shrink:0}}
.trait b{{width:40px;text-align:right;font-size:.85rem}}
.bar{{flex:1;height:8px;background:#334155;border-radius:4px;overflow:hidden}}
.bar div{{height:100%;border-radius:4px}}
.footer{{color:#475569;font-size:.75rem;margin-top:2rem;border-top:1px solid #334155;padding-top:1rem}}
</style></head>
<body>
<div class="header">
  <h1>🧠 Psychologist AI Report</h1>
  <p>User: <strong style="color:#e2e8f0">{user_name}</strong> &nbsp;·&nbsp; Generated: {report_date}</p>
</div>
<div class="stats">
  <div class="stat"><div class="val">{days_tracked}</div><div class="lbl">Days Tracked</div></div>
  <div class="stat"><div class="val">{total_sessions}</div><div class="lbl">Total Sessions</div></div>
  <div class="stat"><div class="val">{total_minutes} min</div><div class="lbl">Analysis Time</div></div>
  <div class="stat"><div class="val">{avg_stress_str}</div><div class="lbl">Avg Stress</div></div>
</div>
{psv_section}
<div class="card">
  <h2>Daily Session History (last 30 days)</h2>
  <table>
    <thead><tr><th>Date</th><th>Sessions</th><th>Duration</th><th>Stress</th><th>Risk</th><th>Top State</th><th>Stability</th></tr></thead>
    <tbody>{daily_rows}</tbody>
  </table>
</div>
{alerts_section}
<div class="footer">
  Generated by Psychologist AI · Phase 6.0 · Personal behavioral analysis — not a medical diagnosis.
</div>
</body></html>"""
    filename = f"psychologist_ai_report_{user_id}_{report_date}.html"
    return HTMLResponse(
        content=html,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
