from pathlib import Path
import joblib
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Live Human Activity Recognition", page_icon="📱", layout="centered")
ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "live_har_model.joblib"
st.title("📱 Live Human Activity Recognition")
st.write("Use your phone's motion sensors to estimate activity after pressing Start Detection.")
st.warning("Research prototype only—not a medical or safety device. Phone placement and sensor differences can affect predictions.")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

model = load_model()
if model is None:
    st.error("The model file live_har_model.joblib is missing. Follow README.md to train it first.")
    st.stop()

sensor_component = components.declare_component("har_motion_sensor", path=str(ROOT / "motion_sensor_frontend"))
event = sensor_component(key="motion_sensor", default={"status": "idle"})
status = event.get("status", "idle") if isinstance(event, dict) else "idle"
if status == "started":
    st.info("Sensor capture started. The first prediction appears after a short window of movement.")
elif status == "permission_denied":
    st.error("Motion sensor permission was denied. Check browser permissions and try again.")
elif status == "unsupported":
    st.error("This browser does not provide motion data. Try the HTTPS site in Chrome on Android.")
elif status == "stopped":
    st.info("Detection stopped.")
elif status == "error":
    st.error(event.get("message", "Could not start motion sensing."))

def feature_vector(samples):
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 9 or arr.shape[0] < 32:
        return None
    feats = []
    for j in range(9):
        x = arr[:, j]
        dx = np.diff(x)
        feats.extend([np.mean(x), np.std(x), np.min(x), np.max(x), np.sqrt(np.mean(x*x)), np.mean(np.abs(x)), np.mean(dx), np.std(dx), np.sqrt(np.mean(dx*dx))])
    for cols in [(0,1,2), (3,4,5), (6,7,8)]:
        mag = np.sqrt(np.sum(arr[:, cols] ** 2, axis=1))
        feats.extend([np.mean(mag), np.std(mag), np.min(mag), np.max(mag), np.sqrt(np.mean(mag*mag))])
    return np.asarray(feats, dtype=np.float32).reshape(1, -1)

if isinstance(event, dict) and event.get("status") == "window":
    X = feature_vector(event.get("samples", []))
    if X is not None:
        pred = model.predict(X)[0]
        st.subheader("Latest prediction")
        st.metric("Predicted activity", str(pred).replace("_", " ").title())
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            classes = list(model.classes_)
            st.caption("Model scores (not calibrated probabilities):")
            for i in np.argsort(probs)[::-1][:3]:
                st.write(f"**{str(classes[i]).replace('_', ' ').title()}** — {probs[i]:.0%}")
                st.progress(float(np.clip(probs[i], 0, 1)))
    else:
        st.warning("Waiting for enough sensor samples. Keep moving naturally.")

st.caption("Activities: walking, walking upstairs/downstairs, sitting, standing, and laying. Browser sensor readings may vary by phone and operating system.")
