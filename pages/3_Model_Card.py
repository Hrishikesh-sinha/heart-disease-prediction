import streamlit as st, datetime as dt, json, joblib
from model_loader import load_and_prepare_data, load_artifacts
st.set_page_config(layout="wide", page_title="Model Card")

st.title("📄 Model Card – Heart Disease Predictor")

_, X_test, _, y_test, _, features = load_and_prepare_data(".")
model, features_saved, thr = load_artifacts()

st.markdown("""
**Intended Use:** Educational demo of risk prediction from UCI Heart datasets.  
**Datasets:** Cleveland, Hungary, Switzerland, VA (UCI).  
**Preprocessing:** KNN imputation → Standardization; class_weight='balanced'.  
**Validation:** Stratified 5-fold CV with randomized search; calibration (isotonic).  
**Primary Metric:** ROC-AUC; Operating threshold chosen via PR-curve F1.  
**Limitations:** Small, historical, non-representative; not a clinical device.  
**Safety:** Probabilities are illustrative; not for diagnosis or treatment.
""")

st.subheader("Artifacts")
st.write(f"- Trained: {dt.datetime.now().strftime('%Y-%m-%d')} (local)")
st.write(f"- Features used ({len(features_saved)}): {', '.join(features_saved)}")
st.write(f"- Decision threshold: **{thr:.2f}**")

st.subheader("Fairness/Robustness notes")
st.markdown("""
- Sex feature present; *do not* deploy without a bias review.
- Train/test split is random; results vary with seed.
- No leakage; calibration applied for honest probabilities.
""")
