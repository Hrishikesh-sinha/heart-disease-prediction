import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, classification_report
from model_loader import load_and_prepare_data, train_models, load_artifacts

st.set_page_config(layout="wide", page_title="Global Model Analysis")

st.title("📊 Global Model Analysis (Leaderboard + Diagnostics)")

X_train, X_test, y_train, y_test, pipe_base, feature_names = load_and_prepare_data(".")
try:
    model, features, threshold = load_artifacts()
    leaderboard = None
except Exception:
    model, threshold, leaderboard = train_models(X_train, y_train, feature_names)

st.subheader("Leaderboard (CV AUC)")
if leaderboard:
    for name, info in sorted(leaderboard.items(), key=lambda kv: -kv[1]["cv_auc"]):
        st.write(f"- **{name}** — AUC: {info['cv_auc']:.3f}")

# Test-set diagnostics
proba = model.predict_proba(X_test)[:,1]
pred = (proba >= threshold).astype(int)

st.subheader("ROC & PR Curves")
fpr, tpr, _ = roc_curve(y_test, proba); roc_auc = auc(fpr, tpr)
prec, rec, _ = precision_recall_curve(y_test, proba)

fig1 = plt.figure(); plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'--'); plt.title(f"ROC AUC={roc_auc:.3f}"); st.pyplot(fig1)
fig2 = plt.figure(); plt.plot(rec, prec); plt.title("Precision-Recall"); st.pyplot(fig2)

st.subheader("Confusion Matrix (at chosen threshold)")
cm = confusion_matrix(y_test, pred)
fig3 = plt.figure(); 
plt.imshow(cm); plt.colorbar(); 
plt.xticks([0,1],["Pred Low","Pred High"]); plt.yticks([0,1],["Actual Low","Actual High"]);
for (i,j),v in np.ndenumerate(cm): plt.text(j,i,int(v),ha="center",va="center")
st.pyplot(fig3)

st.subheader("Classification Report")
st.text(classification_report(y_test, pred, digits=3))
