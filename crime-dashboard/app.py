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
import streamlit as st
import pandas as pd
from pathlib import Path
from io import BytesIO, StringIO
import re, unicodedata
from difflib import get_close_matches

# ---------- helpers ----------

def _clean_col(s: str) -> str:
    """Unicode-normalise, collapse whitespace, strip."""
    s = unicodedata.normalize("NFKC", str(s)).replace("\u00A0"," ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _normalise_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_clean_col(c) for c in df.columns]
    return df

def _try_read_csv(buf):
    """Try multiple CSV read strategies (handles BOM, odd delimiters)."""
    attempts = [
        dict(sep=None, engine="python", encoding="utf-8-sig"),  # auto-delim + BOM
        dict(sep=None, engine="python", encoding="utf-8"),
        dict(sep=",", engine="python", encoding="utf-8-sig"),
        dict(sep=";", engine="python", encoding="utf-8-sig"),
        dict(sep="\t", engine="python", encoding="utf-8-sig"),
        dict(sep=",", engine="python", on_bad_lines="skip"),
    ]
    last_err = None
    for opts in attempts:
        try:
            return pd.read_csv(buf, **opts)
        except Exception as e:
            last_err = e
    raise last_err

def _read_any(file_or_url_or_path):
    """Read CSV/Excel from upload, URL or path with fallbacks."""
    # Uploaded file-like?
    if hasattr(file_or_url_or_path, "read"):
        raw = file_or_url_or_path.read()
        name = getattr(file_or_url_or_path, "name", "").lower()
        bio = BytesIO(raw)
        if name.endswith((".xlsx",".xls")):
            bio.seek(0)
            return pd.read_excel(bio)
        if name.endswith(".gz"):
            bio.seek(0)
            return pd.read_csv(bio, compression="infer", engine="python")
        # try text path first (may reveal HTML)
        try:
            txt = raw.decode("utf-8", errors="replace")
            if "<html" in txt.lower():
                raise ValueError("Got HTML instead of CSV (check the link).")
            return _try_read_csv(StringIO(txt))
        except Exception:
            bio.seek(0)
            return _try_read_csv(bio)

    # String path/URL
    p = str(file_or_url_or_path)
    low = p.lower()
    if low.endswith((".xlsx",".xls")):
        return pd.read_excel(p)
    if low.endswith((".gz",".zip")):
        return pd.read_csv(p, compression="infer", engine="python")
    # Quick attempt; if it fails, try robust
    try:
        return pd.read_csv(p, nrows=5) or pd.read_csv(p)  # fast probe
    except Exception:
        return _try_read_csv(p)

def _resolve_col(df: pd.DataFrame, wanted_variants: list[str], fuzzy_target: str):
    """Find a column by exact-cleaned or fuzzy match."""
    cleaned = {_clean_col(c): c for c in df.columns}  # cleaned->original
    # exact (cleaned) match
    for w in wanted_variants:
        w_clean = _clean_col(w)
        if w_clean in cleaned:
            return cleaned[w_clean]
    # fuzzy
    close = get_close_matches(_clean_col(fuzzy_target).lower(),
                              [c.lower() for c in cleaned.keys()],
                              n=1, cutoff=0.7)
    if close:
        # map back to original case
        for k, orig in cleaned.items():
            if k.lower() == close[0]:
                return orig
    return None

# ---------- unified loader: URL (secrets) -> local -> upload ----------
@st.cache_data
def load_data_flexible():
    url = st.secrets.get("DATA_URL", "").strip() if "DATA_URL" in st.secrets else ""
    # 1) Try URL
    if url:
        try:
            df = _read_any(url)
            return _normalise_headers(df), f"URL (secrets): {url}"
        except Exception as e:
            st.warning(f"DATA_URL failed: {e}")
    # 2) Try local
    try:
        local = Path(__file__).parent / "data" / "cleaned_data.csv"
    except NameError:
        local = Path("data/cleaned_data.csv")
    if local.exists():
        try:
            df = _read_any(str(local))
            return _normalise_headers(df), f"Local file: {local}"
        except Exception as e:
            st.warning(f"Local file failed: {e}")
    # 3) Upload
    up = st.sidebar.file_uploader("Upload cleaned_data (CSV/Excel)", type=["csv","xlsx","xls","gz"])
    if up is not None:
        df = _read_any(up)
        return _normalise_headers(df), f"Uploaded: {getattr(up,'name','file')}"
    st.error("No data source available. Set DATA_URL, add data/cleaned_data.csv, or upload a file.")
    st.stop()

df, src = load_data_flexible()
st.sidebar.success(f"Loaded data from {src}")


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
#import plotly.express as px

st.subheader("Outcome Distribution")

# Try common header variants for outcome column
outcome_col = _resolve_col(
    df,
    wanted_variants=["Last outcome category", "Last outcome", "Outcome"],
    fuzzy_target="Last outcome category",
)

if outcome_col is None:
    st.error("Could not find the outcome column (e.g., 'Last outcome category'). "
             "See the printed column names below and adjust variants.")
    st.code("\n".join(repr(c) for c in df.columns))
    st.stop()

top_n = st.sidebar.slider("Top N outcomes", min_value=5, max_value=30, value=15, step=1)

outc = (
    df[outcome_col]
    .astype("string")
    .fillna("Missing/Unknown")
    .value_counts(dropna=False)
    .reset_index()
    .rename(columns={"index": "Outcome", outcome_col: "Count"})
)
outc["Percent"] = (outc["Count"] / outc["Count"].sum() * 100).round(2)

show = outc.head(top_n).iloc[::-1]  # reverse for horizontal plot
fig2 = px.bar(
    show,
    x="Count",
    y="Outcome",
    orientation="h",
    text="Count",
    title=f"Outcome Distribution (Top {top_n})",
)
fig2.update_traces(
    hovertemplate="<b>%{y}</b><br>Count: %{x}<br>Share: %{customdata:.2f}%<extra></extra>",
    customdata=show["Percent"].values
)
fig2.update_layout(margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)

with st.expander("See full outcome table / download"):
    st.dataframe(outc, use_container_width=True)
    st.download_button("Download outcome summary (CSV)",
                       outc.to_csv(index=False).encode("utf-8"),
                       file_name="outcome_summary.csv")




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
