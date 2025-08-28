#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 11:28:04 2025

@author: user
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
from PIL import Image
import io
import glob

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Crime Outcomes Dashboard", layout="wide")

st.title("📊 Crime Outcomes Dashboard (Apr 2023 – Apr 2025)")
st.write("Explore trends, outcomes, and model outputs from the cleaned dataset.")

# -----------------------------
# Resolve paths safely
# -----------------------------
def project_root() -> Path:
    try:
        return Path(__file__).parent
    except NameError:
        # When run interactively (e.g., Spyder)
        return Path.cwd()

ROOT = project_root()
DATA_DEFAULT = ROOT / "data" / "cleaned_data.csv"
FIG_DIR = ROOT / "figures"

# -----------------------------
# Load data (with caching)
# -----------------------------
@st.cache_data
def load_data():
    url = st.secrets.get("DATA_URL", "").strip()
    if url:
        return pd.read_csv(url)
    # fallback to local file if you also keep a copy in the repo
    import pathlib
    p = pathlib.Path(__file__).parent / "data" / "cleaned_data.csv"
    return pd.read_csv(p)

df = load_data()

# -----------------------------
# Light cleaning / parsing
# -----------------------------
# Normalise column names for safety
df.columns = [c.strip() for c in df.columns]

# Parse Month if present
if "Month" in df.columns:
    with st.spinner("Parsing Month..."):
        try:
            df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
        except Exception:
            pass

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("🔎 Filters")

# Crime type filter
if "Crime type" in df.columns:
    crime_values = sorted([c for c in df["Crime type"].dropna().unique().tolist()])
    sel_crimes = st.sidebar.multiselect("Crime type", crime_values, default=crime_values[:5] if crime_values else [])
else:
    sel_crimes = []

# Month range filter
if "Month" in df.columns and pd.api.types.is_datetime64_any_dtype(df["Month"]):
    min_m, max_m = df["Month"].min(), df["Month"].max()
    start, end = st.sidebar.slider(
        "Month range",
        min_value=min_m.to_pydatetime() if pd.notna(min_m) else None,
        max_value=max_m.to_pydatetime() if pd.notna(max_m) else None,
        value=(min_m.to_pydatetime(), max_m.to_pydatetime()) if pd.notna(min_m) and pd.notna(max_m) else None,
        format="YYYY-MM"
    )
else:
    start = end = None

# Apply filters
fdf = df.copy()
if sel_crimes and "Crime type" in fdf.columns:
    fdf = fdf[fdf["Crime type"].isin(sel_crimes)]

if start and end and "Month" in fdf.columns and pd.api.types.is_datetime64_any_dtype(fdf["Month"]):
    mask = (fdf["Month"] >= pd.to_datetime(start)) & (fdf["Month"] <= pd.to_datetime(end))
    fdf = fdf[mask]

# -----------------------------
# KPIs
# -----------------------------
k1, k2, k3 = st.columns(3)
k1.metric("Total records", f"{len(fdf):,}")
k2.metric("Unique crime types", fdf["Crime type"].nunique() if "Crime type" in fdf.columns else 0)
if "Month" in fdf.columns and pd.api.types.is_datetime64_any_dtype(fdf["Month"]):
    k3.metric("Month span", f"{fdf['Month'].min():%Y-%m} → {fdf['Month'].max():%Y-%m}")
else:
    k3.metric("Month span", "N/A")

# -----------------------------
# Data preview
# -----------------------------
with st.expander("👀 Preview data", expanded=False):
    st.dataframe(fdf.head(20), use_container_width=True)

# -----------------------------
# Charts
# -----------------------------
# Monthly trend
if "Month" in fdf.columns and pd.api.types.is_datetime64_any_dtype(fdf["Month"]):
    monthly = (
        fdf.dropna(subset=["Month"])
           .groupby(pd.Grouper(key="Month", freq="MS"))
           .size()
           .reset_index(name="Count")
           .sort_values("Month")
    )
    if not monthly.empty:
        fig1 = px.line(monthly, x="Month", y="Count", markers=True, title="Monthly Crime Trend")
        st.plotly_chart(fig1, use_container_width=True)



# -------------------
# Outcome distribution (robust)
# -------------------
# -------------------
# Outcome distribution (robust + % + Top N + download)
# -------------------
import re, unicodedata
import plotly.express as px

def _clean(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Resolve the outcome column robustly (handles spacing/case)
candidate_names = ["Last outcome category", "Last outcome", "Outcome"]
col_map = {_clean(c): c for c in df.columns}
outcome_col = None
for nm in candidate_names:
    key = _clean(nm)
    # exact clean match
    if key in {_clean(c) for c in df.columns}:
        # pick the original-cased name
        for c in df.columns:
            if _clean(c) == key:
                outcome_col = c
                break
        break

if outcome_col:
    # Sidebar controls
    st.sidebar.markdown("### Outcome chart options")
    top_n = st.sidebar.slider("Show Top N outcomes", min_value=5, max_value=30, value=15, step=1)

    # Build summary
    outc = (df[outcome_col]
            .astype("string")
            .fillna("Missing/Unknown")
            .value_counts(dropna=False)
            .reset_index()
            .rename(columns={"index": "Outcome", outcome_col: "Count"}))
    outc["Percent"] = (outc["Count"] / outc["Count"].sum() * 100).round(2)

    # Plot (horizontal for readability)
    show = outc.head(top_n).iloc[::-1]  # reverse for ascending display
    fig2 = px.bar(
        show,
        x="Count",
        y="Outcome",
        orientation="h",
        text="Count",
        title=f"Outcome Distribution (Top {top_n})",
    )
    fig2.update_layout(
        margin=dict(l=10, r=10, t=50, b=10),
        yaxis=dict(tickfont=dict(size=11)),
        xaxis_title="Count",
        yaxis_title="Outcome",
    )
    # Add percentage in hover
    fig2.update_traces(
        hovertemplate="<b>%{y}</b><br>Count: %{x}<br>Share: %{customdata:.2f}%<extra></extra>",
        customdata=show["Percent"].values
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Show table & download
    with st.expander("See outcome table / download"):
        st.dataframe(outc, use_container_width=True)
        csv = outc.to_csv(index=False).encode("utf-8")
        st.download_button("Download outcome summary (CSV)", csv, file_name="outcome_summary.csv")
else:
    st.info(f"Outcome column not found. Looked for: {candidate_names}")
    st.write("Columns detected:", df.columns.tolist())




# -----------------------------
# Model outputs: images from /figures
# -----------------------------
st.subheader("🧪 Model Outputs (Images)")
if FIG_DIR.exists():
    # Confusion matrix first if present
    cm_path = FIG_DIR / "confusion_matrix.png"
    if cm_path.exists():
        st.image(str(cm_path), caption="Confusion Matrix", use_container_width=True)

    # Any other PNGs in figures/
    other_pngs = [p for p in sorted(FIG_DIR.glob("*.png")) if p.name != "confusion_matrix.png"]
    if other_pngs:
        st.write("Additional figures:")
        cols = st.columns(2)
        for i, p in enumerate(other_pngs):
            with cols[i % 2]:
                st.image(str(p), caption=p.name, use_container_width=True)
else:
    st.info("No `figures/` directory found. Add charts (PNG) to display model outputs.")

# -----------------------------
# Footer
# -----------------------------
st.caption("✅ Tip: Use the sidebar to filter by crime type and month. Add your trained-model charts to `figures/`.")
