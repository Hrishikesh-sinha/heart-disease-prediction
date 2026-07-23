import streamlit as st, pandas as pd, numpy as np, matplotlib.pyplot as plt
from model_loader import load_and_prepare_data, get_unscaled_data
st.set_page_config(layout="wide", page_title="Explore Data")

st.title("🧭 Explore the Data")
X_train, X_test, y_train, y_test, _, features = load_and_prepare_data(".")
X_all = get_unscaled_data()
st.caption(f"Rows: {len(X_all)}, Features: {len(features)}")

st.subheader("Missingness")
miss = X_all.isna().mean().sort_values(ascending=False)
st.bar_chart(miss)

st.subheader("Distributions")
col = st.selectbox("Select feature", features)
fig = plt.figure()
plt.hist(X_all[col].dropna(), bins=30)
plt.xlabel(col); plt.ylabel("count"); st.pyplot(fig)

st.subheader("Correlations")
corr = X_all.corr(numeric_only=True)
st.dataframe(corr.style.background_gradient(axis=None))
