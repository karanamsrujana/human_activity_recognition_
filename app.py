import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Phone Activity Recognition", page_icon="📱", layout="centered")

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "har_model.joblib"
DATA_DIR = ROOT / "data"
ACTIVITIES = {
    1: "Walking",
    2: "Walking upstairs",
    3: "Walking downstairs",
    4: "Sitting",
    5: "Standing",
    6: "Laying"
}

st.title("📱 Human Activity Recognition")
st.write("A machine-learning demo that classifies activities using smartphone motion-sensor features.")
st.info("This starter version uses the UCI HAR feature dataset. It can classify a sample from the dataset or a CSV containing the same 561 features. It does **not** read your phone's sensors live from the browser yet.")

if not MODEL_PATH.exists():
    st.warning("The trained model isn't here yet. Run `python train_model.py` in the project folder first, then refresh this page.")
    st.code("python train_model.py")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
feature_names = bundle["feature_names"]

tab1, tab2, tab3 = st.tabs(["Try a demo", "Upload feature CSV", "About"])

with tab1:
    st.subheader("Try a sample from the test dataset")
    st.write("Choose a sample row and see the model's prediction. This is useful for demonstrating the app.")
    sample_path = DATA_DIR / "demo_samples.csv"
    if sample_path.exists():
        samples = pd.read_csv(sample_path)
        idx = st.number_input("Sample row", min_value=0, max_value=max(0, len(samples)-1), value=0, step=1)
        row = samples.iloc[[int(idx)]]
        true_label = int(row["activity_label"].iloc[0])
        X = row[feature_names]
        pred = int(model.predict(X)[0])
        probs = model.predict_proba(X)[0]
        st.metric("Predicted activity", ACTIVITIES.get(pred, str(pred)))
        st.caption(f"Dataset label for this sample: {ACTIVITIES.get(true_label, str(true_label))}")
        st.bar_chart(pd.DataFrame({"Probability": probs}, index=[ACTIVITIES.get(int(c), str(c)) for c in model.classes_]))
    else:
        st.warning("Demo samples are missing. Run `python train_model.py` again.")

with tab2:
    st.subheader("Upload a CSV")
    st.write("Your CSV must contain the 561 UCI HAR feature columns. An optional `activity_label` column is ignored for prediction.")
    uploaded = st.file_uploader("Choose feature CSV", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            missing = [c for c in feature_names if c not in df.columns]
            if missing:
                st.error(f"This CSV is missing {len(missing)} required feature columns. Please use the UCI HAR feature format. First missing columns: {missing[:8]}")
            else:
                X = df[feature_names].apply(pd.to_numeric, errors="coerce")
                if X.isna().any().any():
                    st.error("Some feature cells are blank or non-numeric. Please clean the CSV and upload again.")
                else:
                    preds = model.predict(X)
                    result = df.copy()
                    result["predicted_activity"] = [ACTIVITIES.get(int(p), str(p)) for p in preds]
                    st.success(f"Predicted activities for {len(result)} row(s).")
                    st.dataframe(result[["predicted_activity"]].join(df.head(0)), use_container_width=True)
                    st.dataframe(result[["predicted_activity"]], use_container_width=True)
                    st.download_button("Download predictions CSV", result.to_csv(index=False).encode("utf-8"), "har_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Couldn't read this file: {e}")

with tab3:
    st.subheader("What this app does")
    st.markdown("""
- **Dataset:** UCI Human Activity Recognition Using Smartphones (30 volunteers, six activities).
- **Model:** Random Forest classifier trained on the dataset's 561 engineered sensor features.
- **Input:** one row per 2.56-second sensor window, represented by 561 feature columns.
- **Important limitation:** the deployed website does not automatically access accelerometer/gyroscope sensors on your phone. To support live sensor capture, the app needs an additional browser sensor interface and a compatible data pipeline.
""")
    st.caption("For an academic project, describe the current version as a deployed HAR prediction demo unless you add and test live phone-sensor capture.")
