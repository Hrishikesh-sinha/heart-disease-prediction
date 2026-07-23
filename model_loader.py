# model_loader.py
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

import shap

RANDOM_STATE = 42
ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

# ---- 13 features (UCI Heart) ----
FEATURE_ORDER = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

# ---------- Loaders ----------
def _read_uci_like(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, na_values=["?"], sep=None, engine="python")
    if df.shape[1] >= 13:
        df = df.iloc[:, :13].copy()
        df.columns = FEATURE_ORDER
    return df

def _load_all_sites(base="."):
    paths = [
        "cleveland.data", "hungarian.data", "switzerland.data", "long-beach-va.data",
        "processed.hungarian.data", "processed.switzerland.data", "processed.va.data",
        "reprocessed.hungarian.data", "new.data"
    ]

    frames, y_list = [], []
    for p in paths:
        full = Path(base) / p
        if not full.exists():
            continue
        try:
            df = pd.read_csv(full, header=None, na_values=["?"], sep=None, engine="python")
        except Exception:
            continue

        ncols = df.shape[1]
        if ncols >= 14:
            X = df.iloc[:, :13].copy()
            y = df.iloc[:, 13].copy()
        elif ncols == 13:
            X = df.iloc[:, :13].copy()
            y = pd.Series([np.nan] * len(X))
        else:
            continue

        X.columns = FEATURE_ORDER
        frames.append(X)
        y_list.append(y)

    if not frames:
        raise FileNotFoundError("No UCI heart-disease files found in the project folder.")

    X_all = pd.concat(frames, axis=0, ignore_index=True)
    y_all = pd.concat(y_list, axis=0, ignore_index=True)

    # fallback labels from heart_disease.csv
    if y_all.isna().all():
        csvp = Path(base) / "heart_disease.csv"
        if csvp.exists():
            dfc = pd.read_csv(csvp)
            target_col = "target" if "target" in dfc.columns else ("num" if "num" in dfc.columns else None)
            if target_col is not None:
                y_all = pd.Series(dfc[target_col].values[:len(X_all)])

    y_bin = pd.to_numeric(y_all, errors="coerce")
    y_bin = (y_bin.fillna(0) > 0).astype(int)

    mask_validX = ~X_all.isna().all(axis=1)
    mask_validY = y_bin.notna()
    mask = mask_validX & mask_validY

    X_all = X_all.loc[mask].reset_index(drop=True)
    y_bin = y_bin.loc[mask].reset_index(drop=True)
    return X_all, y_bin

def load_and_prepare_data(base="."):
    X, y = _load_all_sites(base)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    pipe_base = Pipeline([
        ("imputer", KNNImputer(n_neighbors=5)),
        ("scaler", StandardScaler(with_mean=True)),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    return X_train, X_test, y_train, y_test, pipe_base, X.columns.tolist()

# ---------- Search spaces ----------
def _rf_space():
    return {
        "model__n_estimators": [200, 400, 600],
        "model__max_depth": [None, 4, 6, 10],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", None],
    }

def _lr_space():
    return {
        "model__C": np.logspace(-3, 2, 10),
        "model__penalty": ["l2"],
        "model__solver": ["lbfgs", "liblinear"],
    }

# ---------- Utilities ----------
def _threshold_from_pr(y_true, proba):
    p, r, t = precision_recall_curve(y_true, proba)
    f1s = (2 * p * r) / (p + r + 1e-9)
    best_idx = np.nanargmax(f1s)
    return float(t[max(0, best_idx - 1)])

# ---------- Training ----------
def train_models(X_train, y_train, feature_names):
    candidates = {
        "LogisticRegression": Pipeline([
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "RandomForest": Pipeline([
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    best_name, best_est, best_auc = None, None, -1.0

    for name, pipe in candidates.items():
        search_space = _lr_space() if "Logistic" in name else _rf_space()
        rs = RandomizedSearchCV(
            pipe, search_space, n_iter=20, cv=cv, scoring="roc_auc",
            n_jobs=-1, random_state=RANDOM_STATE, refit=True, verbose=0
        )
        rs.fit(X_train, y_train)
        tuned = rs.best_estimator_

        calibrated = CalibratedClassifierCV(tuned, cv=cv, method="isotonic")
        calibrated.fit(X_train, y_train)

        oof_auc = rs.best_score_
        results[name] = {"estimator": calibrated, "cv_auc": float(oof_auc)}

        if oof_auc > best_auc:
            best_auc, best_name, best_est = oof_auc, name, calibrated

    train_proba = best_est.predict_proba(X_train)[:, 1]
    best_threshold = _threshold_from_pr(y_train, train_proba)

    joblib.dump(best_est, ARTIFACTS / "best_model.joblib")
    joblib.dump(feature_names, ARTIFACTS / "features.joblib")
    joblib.dump(best_threshold, ARTIFACTS / "threshold.joblib")
    joblib.dump(results, ARTIFACTS / "model_leaderboard.joblib")
    return best_est, best_threshold, results

def load_artifacts():
    model = joblib.load(ARTIFACTS / "best_model.joblib")
    features = joblib.load(ARTIFACTS / "features.joblib")
    threshold = joblib.load(ARTIFACTS / "threshold.joblib")
    return model, features, threshold

# ---------- Explainability ----------
def _unwrap_pipeline(calibrated):
    """
    Return the inner sklearn Pipeline from a (possibly) calibrated estimator.
    Handles sklearn versions using 'base_estimator' (old) or 'estimator' (new).
    """
    base = calibrated
    if isinstance(calibrated, CalibratedClassifierCV):
        base = getattr(calibrated, "base_estimator", None)
        if base is None:
            base = getattr(calibrated, "estimator", None)  # sklearn >= 1.3
        if base is None:
            raise AttributeError("Cannot access wrapped estimator from CalibratedClassifierCV.")
    from sklearn.pipeline import Pipeline as _SkPipeline
    if not isinstance(base, _SkPipeline):
        raise ValueError(f"Expected a sklearn Pipeline inside the calibrated model, got: {type(base)}")
    return base

def create_explainer(fitted_estimator, X_background):
    base_pipe = _unwrap_pipeline(fitted_estimator)
    imputer = base_pipe.named_steps["imputer"]
    scaler = base_pipe.named_steps["scaler"]
    model = base_pipe.named_steps["model"]

    bg_raw = shap.sample(X_background, 200, random_state=RANDOM_STATE)

    if isinstance(model, RandomForestClassifier):
        bg_trans = scaler.transform(imputer.transform(bg_raw))
        explainer = shap.TreeExplainer(model)
        return explainer, bg_trans
    else:
        f = lambda data: fitted_estimator.predict_proba(pd.DataFrame(data, columns=X_background.columns))[:, 1]
        explainer = shap.KernelExplainer(f, bg_raw)
        return explainer, bg_raw

def get_unscaled_data():
    X, _ = _load_all_sites(".")
    return X.copy()
