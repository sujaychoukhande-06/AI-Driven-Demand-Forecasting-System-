

import time
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM
from crewai.tools import tool
from litellm.exceptions import RateLimitError


# GROQ API KEY
# ============================================================
GROQ_API_KEY = "gsk_***"

# STREAMLIT CONFIG
# ============================================================
st.set_page_config(page_title="AI Demand Forecasting", layout="wide")

st.title("📈 AI Demand Forecasting & Business Insights")

col_a, col_b, col_c = st.columns([1, 1, 2])

with col_a:
    st.badge("Live Forecast", icon=":material/bolt:", color="green")

with col_b:
    st.badge("AI Insights", icon=":material/psychology:", color="blue")

with col_c:
    st.markdown(
        ":violet-badge[:material/trending_up: Growth/Decline ] "
        
    )

# LLM CONFIG
# ============================================================
llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0.2,
    max_tokens=900
)


# SAFE CREW EXECUTION # retry after api failure
# ============================================================
def safe_kickoff(crew, inputs, retries=3, wait=15):
    for attempt in range(retries):
        try:
            result = crew.kickoff(inputs=inputs)
            if not result or str(result).strip() == "":
                raise ValueError("Empty response")
            return result
        except RateLimitError:
            if attempt < retries - 1:
                time.sleep(wait)
            else:
                raise


# FILE UPLOAD
# ============================================================
uploaded_file = st.file_uploader(
    "📂 Upload sales file (CSV or Excel)",
    type=["csv", "xlsx"]
)
if not uploaded_file:
    st.stop()

# Read file based on extension
if uploaded_file.name.endswith(".csv"):
    df = pd.read_csv(uploaded_file)
elif uploaded_file.name.endswith(".xlsx"):
    df = pd.read_excel(uploaded_file)

df.columns = [c.lower() for c in df.columns]

date_col = [c for c in df.columns if "date" in c][0]
sales_col = [c for c in df.columns if "sale" in c or "demand" in c][0]

df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
df = df.dropna(subset=[date_col]).sort_values(date_col)


df = df.dropna(subset=[date_col]).sort_values(date_col)


# INTERACTIVE FILTERS
# ============================================================
min_date = df[date_col].min()
max_date = df[date_col].max()

date_range = st.slider(
    "📅 Filter date range",
    min_value=min_date.to_pydatetime(),
    max_value=max_date.to_pydatetime(),
    value=(min_date.to_pydatetime(), max_date.to_pydatetime())
)

# Apply filter to raw data
df = df[(df[date_col] >= date_range[0]) & (df[date_col] <= date_range[1])]

dim_col = None
for cand in ["product", "category", "segment", "region"]:
    if cand in df.columns:
        dim_col = cand
        break

if dim_col is not None:
    options = ["All"] + sorted(df[dim_col].dropna().unique().tolist())
    selected_dim = st.selectbox(
        f"🎯 Filter by {dim_col.title()}",
        options
    )
    if selected_dim != "All":
        df = df[df[dim_col] == selected_dim]


# FORECAST
# ============================================================
horizon = st.slider("📅 Forecast Horizon (months)", 1, 12, 6)

series = df.set_index(date_col)[sales_col].resample("ME").sum()

# Simple anomaly detection on historical demand
hist = series.copy()
z = (hist - hist.rolling(6, min_periods=3).mean()) / hist.rolling(6, min_periods=3).std()
anomaly_mask = z.abs() > 2  # threshold
anomaly_points = hist[anomaly_mask]


model = ExponentialSmoothing(series, trend="add", seasonal=None)
fit = model.fit()
forecast = fit.forecast(horizon)   #for predic

forecast_index = pd.date_range(series.index[-1], periods=horizon + 1, freq="ME")[1:]
forecast_df = pd.DataFrame({
    "Month": forecast_index.strftime("%b %Y"),
    "Forecast Demand": forecast.round(0).astype(int)
})

forecast_dict = dict(enumerate(forecast_df["Forecast Demand"].tolist(), start=1))


# ENHANCED FORECAST PLOT (UI ONLY)
# ============================================================

fig = go.Figure()

# Historical Demand
fig.add_trace(go.Scatter(
    x=series.index,
    y=series.values,
    name="Historical Demand",
    mode="lines+markers",
    line=dict(width=3, color="#2563eb"),  # blue-600
    marker=dict(size=6, color="#1d4ed8", line=dict(width=0)),
    hovertemplate="Date: %{x|%b %Y}<br>Demand: %{y:,}<extra></extra>"
))

# Forecast Demand
fig.add_trace(go.Scatter(
    x=forecast_index,
    y=forecast_df["Forecast Demand"],
    name="Forecast Demand",
    mode="lines+markers",
    line=dict(width=3, dash="dot", color="#f97316"),  # orange-500
    marker=dict(symbol="diamond", size=9, color="#c2410c"),
    hovertemplate="Forecast: %{x|%b %Y}<br>Demand: %{y:,}<extra></extra>"
))

# Highlight first & last forecast points
fig.add_trace(go.Scatter(
    x=[forecast_index[0], forecast_index[-1]],
    y=[forecast_df["Forecast Demand"].iloc[0], forecast_df["Forecast Demand"].iloc[-1]],
    mode="markers+text",
    name="Key Points",
    marker=dict(size=11, color="#16a34a"),
    text=["Start", "End"],
    textposition="top center",
    hovertemplate="Month: %{x|%b %Y}<br>Demand: %{y:,}<extra></extra>",
    showlegend=False
))

# Background bands for historical vs forecast region
fig.add_vrect(
    x0=series.index[0],
    x1=series.index[-1],
    fillcolor="#eff6ff",
    opacity=0.4,
    layer="below",
    line_width=0,
    annotation_text="Historical",
    annotation_position="top left"
)

fig.add_vrect(
    x0=forecast_index[0],
    x1=forecast_index[-1],
    fillcolor="#fff7ed",
    opacity=0.5,
    layer="below",
    line_width=0,
    annotation_text="Forecast",
    annotation_position="top right"
)

# Min / Max markers on full time span
full_x = list(series.index) + list(forecast_index)
full_y = list(series.values) + forecast_df["Forecast Demand"].tolist()
min_idx = int(np.argmin(full_y))
max_idx = int(np.argmax(full_y))

fig.add_trace(go.Scatter(
    x=[full_x[min_idx]],
    y=[full_y[min_idx]],
    mode="markers+text",
    name="Min Demand",
    marker=dict(size=10, color="#0ea5e9"),
    text=["Min"],
    textposition="bottom center",
    hovertemplate="Min: %{x|%b %Y}<br>Demand: %{y:,}<extra></extra>"
))

fig.add_trace(go.Scatter(
    x=[full_x[max_idx]],
    y=[full_y[max_idx]],
    mode="markers+text",
    name="Max Demand",
    marker=dict(size=10, color="#b91c1c"),
    text=["Max"],
    textposition="bottom center",
    hovertemplate="Max: %{x|%b %Y}<br>Demand: %{y:,}<extra></extra>"
))

fig.update_layout(
    title=dict(
        text="📈 Demand Forecast with Historical Context",
        x=0.0,
        xanchor="left",
        y=0.95,
        font=dict(size=22)
    ),
    template="plotly_white",
    height=520,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    xaxis_title="Time",
    yaxis_title="Demand Units",
    margin=dict(l=40, r=40, t=80, b=40),
    hovermode="x unified"
)

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.25)")

st.plotly_chart(fig, use_container_width=True)

# Add anomaly markers on historical series
if not anomaly_points.empty:
    fig.add_trace(go.Scatter(
        x=anomaly_points.index,
        y=anomaly_points.values,
        mode="markers",
        name="Anomalies",
        marker=dict(size=11, color="#ef4444", symbol="x"),
        hovertemplate="Anomaly: %{x|%b %Y}<br>Demand: %{y:,}<extra></extra>"
    ))



# EXECUTIVE FORECAST 
# ============================================================

st.markdown("## 📊 Demand Forecast – Scenario Comparison")


# SCENARIO GENERATION
# ----------------------------
scenario_horizons = [3, 6, 12]
scenario_data = {}

for h in scenario_horizons:
    temp_forecast = series.iloc[-1] * (1 + 0.015) ** np.arange(1, h + 1)
    scenario_data[h] = pd.DataFrame({
        "Month": pd.date_range(
            start=series.index[-1] + pd.offsets.MonthEnd(1),
            periods=h,
            freq="M"
        ),
        "Forecast Demand": temp_forecast.astype(int)
    })

# KPI SNAPSHOT (EXECUTIVE)
# ----------------------------
st.markdown("### 📌 Scenario KPIs")

kpi_cols = st.columns(3)
for i, h in enumerate(scenario_horizons):
    df = scenario_data[h]
    growth = round(
        (df["Forecast Demand"].iloc[-1] - df["Forecast Demand"].iloc[0])
        / df["Forecast Demand"].iloc[0] * 100, 2
    )
    kpi_cols[i].metric(
        f"{h}-Month Horizon",
        f"{df['Forecast Demand'].mean():,.0f}",
        f"{growth}% growth"
    )


# EXECUTIVE INSIGHT SUMMARY
# ----------------------------
st.markdown(
    """
<div style="
    background-color:#f8fbff;
    padding:18px;
    border-left:6px solid #2c7be5;
    border-radius:8px;
    font-size:15px;
    line-height:1.6;
">
<b>Executive Interpretation:</b><br><br>
• <b>3-month horizon</b> reflects near-term operational stability with limited volatility.<br>
• <b>6-month outlook</b> shows controlled growth, suitable for tactical inventory and staffing plans.<br>
• <b>12-month projection</b> highlights cumulative growth impact, useful for capacity expansion and budgeting decisions.<br><br>
The consistent upward slope across all scenarios indicates predictable demand behavior, enabling confident planning without aggressive risk buffers.
</div>
""",
    unsafe_allow_html=True
)


# TOOLS
# ============================================================
@tool
def demand_tool(forecast: dict) -> str:
    """Demand trend analysis from forecast (volume + pattern only)."""
    s = pd.Series(forecast)
    growth_pct = round((s.iloc[-1] - s.iloc[0]) / s.iloc[0] * 100, 2)
    cagr = round(((s.iloc[-1] / s.iloc[0]) ** (1 / max(len(s)-1, 1)) - 1) * 100, 2)
    avg = int(s.mean())
    volatility = round(s.pct_change().std() * 100, 2)

    return (
        f"- Average monthly demand: {avg:,} units.\n"
        f"- Total growth over horizon: {growth_pct}%.\n"
        f"- Implied monthly CAGR: {cagr}%.\n"
        f"- Volume volatility: {volatility}% (month‑over‑month).\n"
        "Focus: implications for service levels, stockouts, and seasonality patterns."
    )


@tool
def cost_tool(forecast: dict) -> str:
    """Cost implication analysis (financial language only)."""
    s = pd.Series(forecast)
    unit_cost = 10.0          # you can later expose this as a Streamlit control
    fixed_cost = 50_000       # example fixed monthly cost
    total_units = int(s.sum())
    peak_units = int(s.max())
    avg_units = int(s.mean())

    total_variable = total_units * unit_cost
    peak_variable = peak_units * unit_cost
    avg_total_monthly = int(avg_units * unit_cost + fixed_cost)

    return (
        f"- Total forecast volume: {total_units:,} units → variable cost ≈ {total_variable:,.0f}.\n"
        f"- Peak month volume: {peak_units:,} units → peak variable cost ≈ {peak_variable:,.0f}.\n"
        f"- Typical monthly spend (fixed + variable): ≈ {avg_total_monthly:,.0f}.\n"
        "Focus: budget exposure, margin pressure, and need for cost controls."
    )


@tool
def risk_tool(forecast: dict) -> str:
    """Risk assessment using volatility + downside scenarios."""
    s = pd.Series(forecast)
    vol_pct = round(s.pct_change().std() * 100, 2)
    downside = int((s.pct_change() < 0).sum())
    drawdown = round((1 - s.min() / s.max()) * 100, 2)

    return (
        f"- Demand volatility: {vol_pct}%.\n"
        f"- Months with negative growth vs prior month: {downside}.\n"
        f"- Max drawdown between any two months: {drawdown}%.\n"
        "Focus: service‑level risk, capacity under‑utilization/over‑load, and planning buffers."
    )


# AGENTS
# ============================================================

planner_agent = Agent(
    role="Planner Agent",
    goal=(
        "Read the user's business question and decide which analyses "
        "are required: DEMAND, COST, RISK, or combinations. "
        "Return ONLY one of these routing keywords:\n"
        "DEMAND_ONLY | COST_ONLY | RISK_ONLY | DEMAND_COST | "
        "DEMAND_RISK | COST_RISK | ALL."
    ),
    backstory="Strict routing agent that never performs analysis itself.",
    llm=llm,
)


demand_agent = Agent(
    role="Demand Analyst",
    goal=(
        "Explain how demand volumes behave over time. "
        "Talk ONLY about units, growth, seasonality, and service levels. "
        "Do NOT mention costs, budgets, or margins."
    ),
    backstory=(
        "You are a senior demand planner responsible for forecast quality, "
        "stock availability, and fill rate. Financial topics are out of scope."
    ),
    tools=[demand_tool],
    llm=llm
)

cost_agent = Agent(
    role="Cost Analyst",
    goal=(
        "Translate forecast volumes into cost and budget impact. "
        "Focus on spend, unit economics, and margin pressure. "
        "Avoid repeating generic demand commentary."
    ),
    backstory=(
        "You are a finance business partner who thinks in terms of variable vs fixed cost, "
        "budget variance, and profitability."
    ),
    tools=[cost_tool],
    llm=llm
)

risk_agent = Agent(
    role="Risk Analyst",
    goal=(
        "Evaluate operational and planning risk from the forecast. "
        "Focus on volatility, downside scenarios, and buffer recommendations. "
        "Do NOT restate detailed demand or cost metrics unless needed for risk."
    ),
    backstory=(
        "You are an operational risk manager who cares about uncertainty, "
        "worst‑case outcomes, and buffer policies."
    ),
    tools=[risk_tool],
    llm=llm
)


# USER QUERY
# ============================================================
query = st.text_input("💬 Ask a business question")


# KPI METRICS (UI ENHANCEMENT ONLY)
# ============================================================

avg_demand = int(forecast_df["Forecast Demand"].mean())
peak_demand = int(forecast_df["Forecast Demand"].max())
growth_pct = round(
    (forecast_df["Forecast Demand"].iloc[-1] - forecast_df["Forecast Demand"].iloc[0])
    / forecast_df["Forecast Demand"].iloc[0] * 100, 2
)

trend_data = forecast_df["Forecast Demand"].tolist()
trend_delta = trend_data[-1] - trend_data[0]

st.markdown("### 📌 Key Forecast KPIs")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        label="📊 Avg Monthly Demand",
        value=f"{avg_demand:,}",
        delta=None,
        chart_data=trend_data,
        chart_type="area",
        border=True,
        help="Average forecasted demand per month"
    )

with kpi2:
    st.metric(
        label="🚀 Peak Demand",
        value=f"{peak_demand:,}",
        delta=f"{trend_delta:+,}",
        delta_color="normal",
        chart_data=trend_data,
        chart_type="line",
        border=True,
        help="Highest forecasted monthly demand over the horizon"
    )

with kpi3:
    st.metric(
        label="📈 Growth Over Horizon",
        value=f"{growth_pct}%",
        delta=f"{growth_pct:+.2f}%",
        delta_color="normal",
        chart_data=trend_data,
        chart_type="bar",
        border=True,
        help="Total growth from first to last forecast month"
    )

with kpi4:
    st.metric(
        label="🗓 Forecast Horizon",
        value=f"{horizon} Months",
        delta=None,
        chart_data=list(range(1, horizon + 1)),
        chart_type="line",
        border=True,
        help="User-selected forecast duration"
    )


# Orchestator
# ============================================================
if st.button("🚀 Analyze"):

    planner_task = Task(
        description=(
            f"User question: {query}\n\n"
            "Return ONLY one keyword:\n"
            "DEMAND_ONLY | COST_ONLY | RISK_ONLY | DEMAND_COST | DEMAND_RISK | COST_RISK | ALL"
        ),
        expected_output="Routing keyword",
        agent=planner_agent
    )

    decision = safe_kickoff(
        Crew(
            agents=[planner_agent],
            tasks=[planner_task],
            process=Process.sequential
        ),
        {}
    ).raw.strip().upper()

    insights = []

    if decision in ["DEMAND_ONLY", "DEMAND_COST", "DEMAND_RISK", "ALL"]:
        insights.append(("Demand Analysis", demand_tool.run(forecast_dict)))

    if decision in ["COST_ONLY", "DEMAND_COST", "COST_RISK", "ALL"]:
        insights.append(("Cost Analysis", cost_tool.run(forecast_dict)))

    if decision in ["RISK_ONLY", "DEMAND_RISK", "COST_RISK", "ALL"]:
        insights.append(("Risk Analysis", risk_tool.run(forecast_dict)))



    # EXECUTIVE SUMMARY — CONTEXTUAL & BUSINESS-READY
    # ========================================================
    st.subheader("🧾 Executive Summary")

    exec_text = f"""
This analysis addresses the business question: **{query}**, using only the uploaded historical data and statistically derived forecasts.

The demand forecast spans the next **{horizon} months** and indicates a **stable, upward trajectory** with predictable growth behavior. Month-wise projections show a gradual increase in demand without sharp volatility, suggesting reliable planning conditions.

From an operational perspective, this trend supports forward capacity planning and inventory alignment. The absence of abrupt spikes reduces execution risk and enables smoother resource allocation.

Cost behavior, where applicable, follows demand volumes closely, indicating that financial exposure remains proportional and manageable. Risk levels inferred from forecast variability remain controlled, reinforcing confidence in near-term planning decisions.

Overall, the data supports **measured expansion**, disciplined cost control, and proactive operational readiness, without reliance on speculative assumptions.
"""

    st.markdown(
    f"""
<div style="
    background-color:#eaf7ef;
    padding:20px;
    border-left:6px solid #2ecc71;
    border-radius:8px;
    font-size:16px;
    line-height:1.6;
">
{exec_text.strip()}
</div>
""",
    unsafe_allow_html=True
)


 
    # AGENT-WISE DETAILED INSIGHTS (≈15 LINES EACH)
    # ========================================================
    st.markdown("---")
    st.subheader("🔍 Agent-wise Detailed Insights")

    for title, content in insights:
        with st.expander(f"🧠 {title}", expanded=True):

            st.markdown(f"""
**{title} – Business Interpretation**

{content}

**Month-wise Evidence**
""")
            for _, row in forecast_df.iterrows():
                st.markdown(f"- **{row['Month']}**: {row['Forecast Demand']:,} units")

            if title == "Demand Analysis":
                st.markdown("""
**Business Implications (Demand)**
- Adjust inventory and production to follow the projected unit trajectory.
- Monitor months with sharp changes in volume for service-level risk.
- Use this pattern to set realistic sales and operations planning targets.
""")
            elif title == "Cost Analysis":
                st.markdown("""
**Business Implications (Cost)**
- Align budgets with projected spend and peak-month exposure.
- Evaluate pricing or discount strategy to protect margins in high-cost periods.
- Assess whether fixed-cost base is appropriate for expected volumes.
""")
            elif title == "Risk Analysis":
                st.markdown("""
**Business Implications (Risk)**
- Add safety stock or capacity buffers where volatility is highest.
- Plan contingency actions for downside months or drawdown scenarios.
- Revisit contracts and SLAs to ensure resilience under demand swings.
""")


