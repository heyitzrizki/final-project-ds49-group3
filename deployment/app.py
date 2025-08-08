import os
import json
import re
import time
import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="Subtotal Predictor (LGBM)", page_icon="🧮", layout="wide")

# ===== Load artifacts =====
@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    def p(name: str) -> str: return os.path.join(base_dir, name)

    try:
        model = joblib.load(p("lgbm_model.pkl"))
        scaler = joblib.load(p("scaler.pkl"))

        with open(p("feature_order.json"), "r", encoding="utf-8") as f:
            feature_order = json.load(f)
        with open(p("skewed_cols.json"), "r", encoding="utf-8") as f:
            skewed_cols = json.load(f)

        # Load kategori lengkap untuk UI
        with open(p("categories.json"), "r", encoding="utf-8") as f:
            categories = json.load(f)

        # Normalisasi format
        if isinstance(feature_order, dict):
            feature_order = feature_order.get("feature_order") or feature_order.get("columns")
        if not isinstance(feature_order, list):
            raise ValueError("feature_order.json harus berisi list kolom.")

        if isinstance(skewed_cols, dict):
            skewed_cols = skewed_cols.get("skewed_cols") or skewed_cols.get("columns") or []
        if not isinstance(skewed_cols, list):
            raise ValueError("skewed_cols.json harus berisi list.")

        return model, scaler, feature_order, skewed_cols, categories
    except FileNotFoundError as e:
        here = os.getcwd()
        listing = "\n".join(os.listdir(base_dir))
        raise FileNotFoundError(f"{e}\n\ncwd={here}\nfiles_in_dir:\n{listing}")

try:
    model, scaler, feature_order, skewed_cols, categories = load_artifacts()
except Exception as e:
    st.error("❌ Gagal memuat artifacts.")
    st.exception(e)
    st.stop()

# ===== Helpers =====
def ensure_columns(df: pd.DataFrame, expected_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in expected_cols:
        if c not in df.columns:
            df[c] = 0
    return df[expected_cols]

def preprocess(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    for c in skewed_cols:
        if c in df.columns:
            df[c] = np.log1p(df[c].astype(float))
    df = ensure_columns(df, feature_order)
    Xs = scaler.transform(df)
    return pd.DataFrame(Xs, columns=feature_order, index=df.index)

def predict_df(df_features: pd.DataFrame) -> pd.DataFrame:
    X = preprocess(df_features)
    yhat_log = model.predict(X)
    yhat = np.expm1(yhat_log)
    out = df_features.copy()
    out["predicted_subtotal"] = yhat
    return out

# ===== Feature grouping untuk numeric & categorical =====
from collections import defaultdict
_tmp_groups = defaultdict(list)
for col in feature_order:
    if "_" in col:
        prefix = "_".join(col.split("_")[:-1])
        _tmp_groups[prefix].append(col)

one_hot_groups = {p: cols for p, cols in _tmp_groups.items() if len(cols) >= 2}
in_one_hot = set(sum(one_hot_groups.values(), []))
numeric_features = [c for c in feature_order if c not in in_one_hot]

# Fungsi utilitas mapping label ke kolom model
def label_to_col(prefix: str, label: str) -> str:
    return f"{prefix}_{label}"

# ===== UI =====
st.markdown(
    "<h2 style='text-align:center;margin-bottom:0'>Subtotal Prediction</h2>"
    "<p style='text-align:center;opacity:0.7;margin-top:4px'>LightGBM · auto-generated form from feature_order.json + categories.json</p>",
    unsafe_allow_html=True,
)

tab_form, tab_csv = st.tabs(["🧍 Single Input", "📤 Batch CSV"])

# ---------- Single Input ----------
with tab_form:
    st.write("Isi form di bawah. Numeric diisi angka, kategori pilih salah satu jika ada.")

    row = pd.DataFrame([[0] * len(feature_order)], columns=feature_order)

    st.subheader("Numeric Features")
    cols = st.columns(3)
    for i, col in enumerate(numeric_features):
        with cols[i % 3]:
            row.at[0, col] = st.number_input(col, value=0.0, step=1.0)

    st.subheader("Categorical Features")
    for prefix, opts in categories.items():
        full_opts = ["(none)"] + opts
        choice = st.selectbox(prefix, full_opts)
        # reset kolom model yang terkait prefix ini
        for c in one_hot_groups.get(prefix, []):
            row[c] = 0
        if choice != "(none)":
            cand = label_to_col(prefix, choice)
            if cand in row.columns:
                row[cand] = 1

    if st.button("🚀 Predict"):
        t0 = time.time()
        try:
            out = predict_df(row)
            st.success(f"Done in {time.time()-t0:.2f}s")
            st.metric("predicted_subtotal", f"{float(out['predicted_subtotal'].iloc[0]):,.0f}")
            with st.expander("Show feature vector used"):
                st.dataframe(row.T.rename(columns={0: "value"}))
        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)

# ---------- Batch CSV ----------
with tab_csv:
    st.caption("Upload CSV berisi fitur saja. Kolom boleh subset; yang hilang akan diisi 0.")
    f = st.file_uploader("Upload CSV (features only)", type=["csv"])
    if f and st.button("Predict (CSV)"):
        try:
            df_in = pd.read_csv(f)
            out = predict_df(df_in)
            st.dataframe(out.head(20), use_container_width=True)
            st.download_button("⬇️ Download", out.to_csv(index=False).encode(), "predictions.csv", "text/csv")
        except Exception as e:
            st.error("Batch prediction failed.")
            st.exception(e)
