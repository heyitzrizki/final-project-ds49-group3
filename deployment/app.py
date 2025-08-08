import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import time

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
    # log1p exactly the same cols used during training
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

# ===== UI =====
st.markdown(
    "<h2 style='text-align:center;margin-bottom:0'>Subtotal Prediction</h2>"
    "<p style='text-align:center;opacity:0.7;margin-top:4px'>LightGBM · identical preprocessing to training</p>",
    unsafe_allow_html=True,
)
st.write("")

tab_form, tab_csv = st.tabs(["🧍 Single Input", "📤 Batch CSV"])

# ================= Single Input =================
with tab_form:
    st.write("Isi form di bawah. Fitur lain yang tidak ditampilkan akan otomatis diisi 0 (defensif).")
    # start with zero vector containing ALL training features
    row = pd.DataFrame([[0]*len(feature_order)], columns=feature_order)

    # ---- numeric inputs (the core continuous features you used) ----
    numeric_cols = [
        "total_items", "num_distinct_items",
        "min_item_price", "max_item_price",
        "total_onshift_partners", "total_busy_partners",
        "total_outstanding_orders"
    ]
    # only show those that really exist in your feature set
    numeric_cols = [c for c in numeric_cols if c in row.columns]

    c1,c2,c3 = st.columns(3)
    cols = [c1,c2,c3]
    for i, col in enumerate(numeric_cols):
        with cols[i % 3]:
            row.at[0, col] = st.number_input(col, min_value=0.0, value=0.0, step=1.0)

    # ---- order_protocol (numeric code) ----
    if "order_protocol" in row.columns:
        with c1:
            row.at[0, "order_protocol"] = st.number_input("order_protocol", min_value=0.0, value=0.0, step=1.0)

    # ---- store_primary_category (one‑hot group -> select one) ----
    cat_prefix = "store_primary_category_"
    cat_cols = [c for c in feature_order if c.startswith(cat_prefix)]
    if cat_cols:
        categories = [c.replace(cat_prefix, "") for c in cat_cols]
        with c2:
            ch = st.selectbox("store_primary_category", ["(none)"] + categories)
        row.loc[:, cat_cols] = 0
        if ch and ch != "(none)":
            row.at[0, f"{cat_prefix}{ch}"] = 1

    # (OPTIONAL) you can replicate the block above for other one-hot groups,
    # e.g., market_id_*, store_id_*, etc. But beware: it will explode the form.

    # ---- Predict button ----
    st.write("")
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

# ================= Batch CSV (optional) =================
with tab_csv:
    st.caption("Gunakan ini kalau mau prediksi banyak baris sekaligus.")
    f = st.file_uploader("Upload CSV (features only)", type=["csv"])
    if f and st.button("Predict (CSV)"):
        df_in = pd.read_csv(f)
        out = predict_df(df_in)
        st.dataframe(out.head(20), use_container_width=True)
        st.download_button("⬇️ Download", out.to_csv(index=False).encode(), "predictions.csv", "text/csv")

