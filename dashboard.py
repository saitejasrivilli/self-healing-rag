"""
apps/admin_dashboard/dashboard.py
===================================
Streamlit admin dashboard for platform monitoring.
Shows real-time metrics, token usage, traces, and safety flags.

Run: streamlit run apps/admin_dashboard/dashboard.py
"""

import json
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Admin Dashboard", page_icon="📊",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@800&display=swap');
html,body,[class*="css"]{font-family:'JetBrains Mono',monospace;}
.stApp{background:#0a0e1a;color:#e2e8f0;}
.card{background:#1a2035;border:1px solid #334155;border-radius:12px;padding:1rem 1.5rem;margin-bottom:.75rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style="font-family:'Syne',sans-serif;font-size:2rem;margin:0;
  background:linear-gradient(135deg,#60a5fa,#a78bfa);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
  📊 Platform Admin Dashboard
</h1>
<p style="color:#64748b;margin-top:.3rem">Agentic AI Platform — Real-time Monitoring</p>
""", unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────────────────────
def load_jsonl(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    records = []
    with p.open() as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    return records

obs      = load_jsonl("./data/logs/pipeline_obs.jsonl")
traces   = load_jsonl("./data/logs/traces.jsonl")
episodes = load_jsonl("./data/logs/episodes.jsonl")
metrics_raw = load_jsonl("./data/logs/metrics.jsonl")

# ── KPI Row ────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
n = len(obs)
k1.metric("Total Queries", n)
if n:
    k2.metric("Avg Confidence", f"{sum(r.get('confidence',0) for r in obs)/n:.0%}")
    k3.metric("Verified Rate",  f"{sum(1 for r in obs if r.get('verified'))/n:.0%}")
    k4.metric("Self-Healed",    sum(1 for r in obs if r.get('attempts',1)>1))
    k5.metric("Avg Latency",    f"{sum(r.get('latency_ms',0) for r in obs)/n:.0f}ms")
else:
    for col in [k2, k3, k4, k5]:
        col.metric("—", "N/A")

st.divider()

# ── Charts ────────────────────────────────────────────────────────────────────
if obs:
    import plotly.graph_objects as go
    confs = [r.get("confidence", 0) for r in obs]
    left, right = st.columns(2)
    with left:
        fig = go.Figure(go.Scatter(
            y=confs, mode="lines+markers",
            line=dict(color="#3b82f6", width=2),
            marker=dict(color=["#34d399" if c>=0.65 else "#f87171" for c in confs], size=7),
        ))
        fig.add_hline(y=0.65, line_dash="dash", line_color="#f59e0b",
                      annotation_text="Heal Threshold")
        fig.update_layout(title="Confidence Over Time", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,14,26,.5)",
            font=dict(family="JetBrains Mono"), height=280,
            yaxis=dict(range=[0, 1.05]))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        healed = sum(1 for r in obs if r.get("attempts", 1) > 1)
        verified = sum(1 for r in obs if r.get("verified"))
        fig2 = go.Figure(go.Pie(
            labels=["Verified", "Unverified", "Self-Healed"],
            values=[max(verified - healed, 0), n - verified, healed],
            hole=0.6, marker=dict(colors=["#34d399", "#f87171", "#a78bfa"]),
        ))
        fig2.update_layout(title="Answer Quality Breakdown", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", font=dict(family="JetBrains Mono"), height=280)
        st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No query data yet. Run some queries via the Chatbot UI or API.")

# ── Traces ────────────────────────────────────────────────────────────────────
st.markdown("### 🔍 Recent Traces")
if traces:
    recent = sorted(traces, key=lambda t: t.get("start_time", 0), reverse=True)[:10]
    for t in recent:
        status_icon = "✅" if t.get("status") == "ok" else "❌"
        st.markdown(f"""
        <div class="card">
          {status_icon} <strong>{t.get('name','unknown')}</strong>
          &nbsp;|&nbsp; trace: <code>{t.get('trace_id','')}</code>
          &nbsp;|&nbsp; {t.get('duration_ms',0):.0f}ms
        </div>
        """, unsafe_allow_html=True)
else:
    st.caption("No traces logged yet.")

# ── Safety flags ──────────────────────────────────────────────────────────────
st.markdown("### 🛡️ Safety Events")
flagged = [r for r in obs if r.get("flags") or r.get("safety_flags")]
if flagged:
    for r in flagged[-5:]:
        flags = r.get("flags") or r.get("safety_flags") or []
        st.markdown(f"""
        <div class="card" style="border-color:#f59e0b">
          ⚠️ <strong>{r.get('query','')[:80]}</strong><br>
          <span style="color:#f59e0b">{', '.join(flags)}</span>
        </div>
        """, unsafe_allow_html=True)
else:
    st.success("✅ No safety flags in recent queries.")

if st.button("🔄 Refresh"):
    st.rerun()
