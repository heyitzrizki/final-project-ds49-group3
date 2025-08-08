import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import time
import re

st.set_page_config(page_title="Subtotal Predictor (LGBM)", page_icon="🧮", layout="wide")

# ===== Load artifacts =====
@st.cache_resource
def load_artifacts():
    model = joblib.load("lgbm_model.pkl")          # predicts subtotal_log
    scaler = joblib.load("scaler.pkl")
    feature_order = json.load(open("feature_order.json"))
    skewed_cols = json.load(open("skewed_cols.json"))
    return model, scaler, feature_order, skewed_cols

model, scaler, feature_order, skewed_cols = load_artifacts()

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
            df[c] = np.log1p(df[c])
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

# ===== Detect feature groups =====
# One-hot detection: prefix before last underscore + unique values
one_hot_groups = {}
numeric_features = []

for col in feature_order:
    if re.search(r"_.", col) and not re.match(r".*\d+\.\d+$", col):
        prefix = "_".join(col.split("_")[:-1])
        one_hot_groups.setdefault(prefix, []).append(col)
    elif re.match(r".*\d+\.\d+$", col):  # special numeric-like suffix (e.g., order_protocol_4.0)
        prefix = "_".join(col.split("_")[:-1])
        one_hot_groups.setdefault(prefix, []).append(col)
    else:
        numeric_features.append(col)

# ===== UI =====
st.markdown(
    "<h2 style='text-align:center;margin-bottom:0'>Subtotal Prediction</h2>"
    "<p style='text-align:center;opacity:0.7;margin-top:4px'>LightGBM · auto-generated form from feature_order.json</p>",
    unsafe_allow_html=True,
)

tab_form, tab_csv = st.tabs(["🧍 Single Input", "📤 Batch CSV"])

# ================= Single Input =================
with tab_form:
    st.write("Isi form di bawah. Semua fitur diambil dari feature_order.json.")

    row = pd.DataFrame([[0]*len(feature_order)], columns=feature_order)

    # --- Numeric inputs ---
    st.subheader("Numeric Features")
    col_blocks = st.columns(3)
    for i, col in enumerate(numeric_features):
        with col_blocks[i % 3]:
            row.at[0, col] = st.number_input(col, value=0.0, step=1.0)

    # --- One-hot groups ---
    st.subheader("Categorical Features")
    for prefix, cols in one_hot_groups.items():
        options = [c.replace(f"{prefix}_", "") for c in cols]
        choice = st.selectbox(prefix, ["(none)"] + options)
        row.loc[:, cols] = 0
        if choice != "(none)":
            chosen_col = f"{prefix}_{choice}"
            if chosen_col in row.columns:
                row.at[0, chosen_col] = 1

    # ---- Predict button ----
    if st.button("🚀 Predict"):
        t0 = time.time()
        try:
            out = predict_df(row)
            st.success(f"Done in {time.time()-t0:.2f}s")
            st.metric("predicted_subtotal", f"{float(out['predicted_subtotal'].iloc[0]):,.0f}")
            with st.expander("Show feature vector used"):
                st.dataframe(row.T.rename(columns={0: "value"}))
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ================= Batch CSV =================
with tab_csv:
    st.caption("Upload CSV dengan kolom sesuai feature_order.json.")
    f = st.file_uploader("Upload CSV (features only)", type=["csv"])
    if f and st.button("Predict (CSV)"):
        df_in = pd.read_csv(f)
        out = predict_df(df_in)
        st.dataframe(out.head(20), use_container_width=True)
        st.download_button("⬇️ Download", out.to_csv(index=False).encode(), "predictions.csv", "text/csv")
