from pathlib import Path

import joblib
import pandas as pd
import sklearn
import streamlit as st
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

from preprocessing import FEATURES, prepare_input

st.set_page_config(page_title="Liver Disease Screening", page_icon="🩺", layout="centered")


@st.cache_resource
def load_artifact():
    return joblib.load(Path(__file__).parent / "liver_model.pkl")


art = load_artifact()
model, scaler = art["model"], art["scaler"]
ag_median = art.get("ag_median", 0.95)


def predict_proba(df_raw: pd.DataFrame):
    X = prepare_input(df_raw, ag_median)
    return model.predict_proba(scaler.transform(X))[:, 1]


# ---------------------------------------------------------------- header
st.title("🩺 Liver Disease Screening")
st.caption("Predicts the likelihood of liver disease from routine blood-test values "
           "(Indian Liver Patient Dataset, Logistic Regression).")
st.warning("Screening aid only. This is not a medical diagnosis. "
           "Always consult a qualified clinician.")

if art.get("sklearn_version") and art["sklearn_version"] != sklearn.__version__:
    st.info(f"Model was saved with scikit-learn {art['sklearn_version']}, "
            f"running {sklearn.__version__}. Pin the same version in requirements.txt.")

with st.sidebar:
    st.header("Settings")
    default_thr = round(float(art.get("threshold", 0.5)), 2)
    threshold = st.slider("Decision threshold", 0.30, 0.80, min(max(default_thr, 0.30), 0.80), 0.01,
                          help="Lower = catches more diseased patients (higher recall) but "
                               "raises more false alarms. Default = threshold chosen in the notebook.")
    m = art.get("metrics")
    if m:
        st.subheader("Model performance (test set)")
        for k, v in m.items():
            st.write(f"**{k}:** {v:.3f}")

tab1, tab2 = st.tabs(["Single patient", "Batch / evaluate on CSV"])

# ---------------------------------------------------------------- single
with tab1:
    with st.form("patient"):
        c1, c2 = st.columns(2)
        age = c1.number_input("Age", 1, 90, 45)
        gender = c2.selectbox("Gender", ["Male", "Female"])
        tb = c1.number_input("Total Bilirubin (mg/dL)", 0.1, 80.0, 1.0, 0.1)
        db = c2.number_input("Direct Bilirubin (mg/dL)", 0.1, 25.0, 0.3, 0.1)
        alp = c1.number_input("Alkaline Phosphotase (IU/L)", 20, 2500, 208)
        alt = c2.number_input("Alamine Aminotransferase / ALT (IU/L)", 5, 2500, 35)
        ast = c1.number_input("Aspartate Aminotransferase / AST (IU/L)", 5, 5000, 42)
        tp = c2.number_input("Total Proteins (g/dL)", 2.0, 10.0, 6.6, 0.1)
        alb = c1.number_input("Albumin (g/dL)", 0.5, 6.0, 3.1, 0.1)
        ag = c2.number_input("Albumin / Globulin ratio", 0.1, 3.0, 0.95, 0.05)
        submitted = st.form_submit_button("Predict")

    if submitted:
        if db > tb:
            st.error("Direct bilirubin cannot be higher than total bilirubin. Please check the values.")
        else:
            row = pd.DataFrame([{
                "Age": age, "Gender": gender, "Total Bilirubin": tb, "Direct Bilirubin": db,
                "Alkaline Phosphotase": alp, "Alamine Aminotransferase": alt,
                "Aspartate Aminotransferase": ast, "Total Protiens": tp,
                "Albumin": alb, "Albumin and Globulin": ag,
            }])
            p = float(predict_proba(row)[0])
            st.metric("Risk score (not a literal probability)", f"{p:.1%}")
            st.progress(min(max(p, 0.0), 1.0))
            st.caption("The model is heavily regularised and trained on balanced data, so scores "
                       "cluster between about 0.4 and 0.8. Compare the score with the threshold, "
                       "not with 50%.")
            if p >= threshold:
                st.error(f"Higher risk of liver disease (score {p:.1%} ≥ threshold {threshold:.0%}). "
                         "Further clinical tests are recommended.")
            else:
                st.success(f"Lower risk of liver disease (score {p:.1%} < threshold {threshold:.0%}).")

# ---------------------------------------------------------------- batch
with tab2:
    st.write("Upload a CSV with these columns: " + ", ".join(FEATURES) +
             ". If it also has a **Target** column (1 = disease, 2 = no disease) the app reports "
             "performance, so you can compare with the notebook.")
    file = st.file_uploader("CSV file", type="csv")

    if file is not None:
        df = pd.read_csv(file)
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            probs = predict_proba(df)
            out = df.copy()
            out["Probability"] = probs.round(4)
            out["Prediction"] = (probs >= threshold).astype(int)
            st.dataframe(out, use_container_width=True)
            st.download_button("Download predictions", out.to_csv(index=False), "predictions.csv")

            if "Target" in df.columns:
                y_true = (df["Target"] == 1).astype(int)
                y_pred = out["Prediction"]
                tn, fp, fn, tp_ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
                st.subheader("Performance on uploaded data")
                a, b, c, d, e = st.columns(5)
                a.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.3f}")
                b.metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.3f}")
                c.metric("Recall", f"{recall_score(y_true, y_pred):.3f}")
                d.metric("F1", f"{f1_score(y_true, y_pred):.3f}")
                e.metric("ROC-AUC", f"{roc_auc_score(y_true, probs):.3f}")
                st.write("Confusion matrix (rows = actual, columns = predicted)")
                st.dataframe(pd.DataFrame([[tn, fp], [fn, tp_]],
                                          index=["Actual: no disease", "Actual: disease"],
                                          columns=["Pred: no disease", "Pred: disease"]))