import streamlit as st, pandas as pd
from model_loader import load_artifacts
st.set_page_config(layout="wide", page_title="Batch Predict")

st.title("📦 Batch Predict")
model, features, thr = load_artifacts()

uploaded = st.file_uploader("Upload CSV with these columns", type=["csv"])
st.caption(", ".join(features))

if uploaded is not None:
    df = pd.read_csv(uploaded)
    X = df[features].copy()
    proba = model.predict_proba(X)[:,1]
    pred = (proba >= thr).astype(int)
    out = df.copy()
    out["probability"] = proba
    out["prediction"] = pred
    st.dataframe(out.head(20))
    out.to_csv("predictions.csv", index=False)
    st.download_button("Download predictions.csv", data=out.to_csv(index=False), file_name="predictions.csv")
