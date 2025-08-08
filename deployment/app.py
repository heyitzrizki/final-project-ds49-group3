import os
import json
import re
import time
import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="Subtotal Predictor (LGBM)", page_icon="🧮", layout="wide")

# ================= Load artifacts =================
@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    j = lambda name: os.path.join(base_dir, name)

    try:
        model = joblib.load(j("lgbm_model.pkl"))
        scaler = joblib.load(j("scaler.pkl"))

        # Prefer the column order stored inside the scaler (most reliable)
        feature_order_json = None
        try:
            with open(j("feature_order.json"), "r", encoding="utf-8") as f:
                tmp = json.load(f)
                feature_order_json = tmp.get("feature_order") if isinstance(tmp, dict) else tmp
        except FileNotFoundError:
            pass

        if hasattr(scaler, "feature_names_in_"):
            feature_order = list(scaler.feature_names_in_)
        elif isinstance(feature_order_json, list):
            feature_order = feature_order_json
        else:
            raise ValueError("Could not find feature list in scaler.feature_names_in_ or feature_order.json.")

        with open(j("skewed_cols.json"), "r", encoding="utf-8") as f:
            tmp = json.load(f)
            skewed_cols = tmp.get("skewed_cols") if isinstance(tmp, dict) else tmp
        if not isinstance(skewed_cols, list):
            skewed_cols = []

        return model, scaler, feature_order, skewed_cols
    except FileNotFoundError as e:
        listing = "\n".join(os.listdir(base_dir))
        raise FileNotFoundError(f"{e}\n\nassets_dir={base_dir}\nfiles:\n{listing}")

try:
    model, scaler, feature_order, skewed_cols = load_artifacts()
except Exception as e:
    st.error("❌ Failed to load artifacts.")
    st.exception(e)
    st.stop()

# ================= Feature grouping =================
def split_groups(feats: list[str]):
    """Detect one-hot groups for order_protocol and store_primary_category; others are numeric."""
    order_cols = [c for c in feats if c.startswith("order_protocol_")]
    store_cols = [c for c in feats if c.startswith("store_primary_category_")]
    in_cat = set(order_cols + store_cols)
    numeric = [c for c in feats if c not in in_cat]

    order_levels = [c.replace("order_protocol_", "", 1) for c in order_cols]
    store_levels = [c.replace("store_primary_category_", "", 1) for c in store_cols]
    return numeric, order_cols, store_cols, order_levels, store_levels

numeric_features, order_cols, store_cols, order_levels, store_levels = split_groups(feature_order)

# Friendly labels for UI
LABEL_MAP = {
    "total_items": "Total Items",
    "num_distinct_items": "Distinct Items",
    "min_item_price": "Min Item Price",
    "max_item_price": "Max Item Price",
    "total_onshift_partners": "Partners On-Shift",
    "total_busy_partners": "Busy Partners",
    "total_outstanding_orders": "Outstanding Orders",
    "delivery_time": "Estimated Delivery Time (min)",
    "item_price_range": "Item Price Range",
    "order_protocol": "Order Protocol",
    "store_primary_category": "Store Primary Category",
}

def nice_label(raw: str) -> str:
    return LABEL_MAP.get(raw, raw.replace("_", " ").title())

def cat_pretty(val: str) -> str:
    # Keep 'Unknown' as is; convert '4.0' -> '4'; tidy hyphens
    if re.fullmatch(r"\d+\.0", val):
        return val.split(".")[0]
    return val.replace("-", " ").title()

# ================= Helpers =================
def ensure_columns(df: pd.DataFrame, expected_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in expected_cols:
        if c not in df.columns:
            df[c] = 0
    return df[expected_cols]

def preprocess(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # log1p for skewed numeric columns if present
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

# ================= UI =================
st.markdown(
    "<h2 style='text-align:center;margin-bottom:0'>Subtotal Prediction</h2>"
    "<p style='text-align:center;opacity:0.8;margin-top:4px'>LightGBM · auto-generated from artifacts</p>"
    "<p style='text-align:center;opacity:0.6;margin-top:2px'>Dataset: "
    "<a href='https://www.kaggle.com/datasets/ranitsarkar01/porter-delivery-time-estimation/data' target='_blank'>Porter Delivery Time Estimation (Kaggle)</a>"
    "</p>",
    unsafe_allow_html=True,
)

tab_form, tab_csv = st.tabs(["🧍 Single Input", "📤 Batch CSV"])

# ---------- Single Input ----------
with tab_form:
    st.write("Fill in the form below. Numeric fields accept numbers; categorical fields are single-choice.")

    # empty row (all zeros) with the exact training feature order
    row = pd.DataFrame([[0] * len(feature_order)], columns=feature_order)

    # --- Numeric inputs ---
    st.subheader("Numeric Features")
    cols = st.columns(3)
    for i, col in enumerate(numeric_features):
        with cols[i % 3]:
            row.at[0, col] = st.number_input(nice_label(col), value=0.0, step=1.0)

    # --- Categorical inputs ---
    st.subheader("Categorical Features")

    # Order Protocol
    if order_cols:
        opts = ["(none)"] + [cat_pretty(x) for x in order_levels]
        choice = st.selectbox(nice_label("order_protocol"), opts)
        row.loc[:, order_cols] = 0
        if choice != "(none)":
            raw_suffix = [s for s in order_levels if cat_pretty(s) == choice][0]
            target_col = f"order_protocol_{raw_suffix}"
            if target_col in row.columns:
                row.at[0, target_col] = 1

    # Store Primary Category
    if store_cols:
        opts = ["(none)"] + [cat_pretty(x) for x in store_levels]
        choice = st.selectbox(nice_label("store_primary_category"), opts)
        row.loc[:, store_cols] = 0
        if choice != "(none)":
            raw_suffix = [s for s in store_levels if cat_pretty(s) == choice][0]
            target_col = f"store_primary_category_{raw_suffix}"
            if target_col in row.columns:
                row.at[0, target_col] = 1

    # ---- Predict button ----
    if st.button("🚀 Predict"):
        t0 = time.time()
        try:
            out = predict_df(row)
            st.success(f"Done in {time.time()-t0:.2f}s")
            st.metric("Predicted Subtotal", f"{float(out['predicted_subtotal'].iloc[0]):,.0f}")
            with st.expander("Show feature vector used"):
                st.dataframe(row.T.rename(columns={0: "value"}))
        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)

# ---------- Batch CSV ----------
with tab_csv:
    st.caption("Upload a CSV containing features only (column names must match training artifacts). Missing columns will be filled with 0.")
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
