from pathlib import Path
from collections import Counter, deque

import joblib
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Live Human Activity Recognition",
    page_icon="📱",
    layout="centered",
)

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "live_har_model.joblib"

st.title("📱 Live Human Activity Recognition")


@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


model = load_model()

if model is None:
    st.error("The model file live_har_model.joblib is missing.")
    st.stop()


sensor_component = components.declare_component(
    "har_motion_sensor",
    path=str(ROOT / "motion_sensor_frontend"),
)

event = sensor_component(
    key="motion_sensor",
    default={"status": "idle"},
)

status = event.get("status", "idle") if isinstance(event, dict) else "idle"

if status == "permission_denied":
    st.error("Motion sensor permission was denied. Check browser permissions.")
elif status == "unsupported":
    st.error("Motion sensors are unsupported. Try Chrome on Android using HTTPS.")
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

        feats.extend([
            np.mean(x),
            np.std(x),
            np.min(x),
            np.max(x),
            np.sqrt(np.mean(x * x)),
            np.mean(np.abs(x)),
            np.mean(dx),
            np.std(dx),
            np.sqrt(np.mean(dx * dx)),
        ])

    for cols in [(0, 1, 2), (3, 4, 5), (6, 7, 8)]:
        mag = np.sqrt(np.sum(arr[:, cols] ** 2, axis=1))

        feats.extend([
            np.mean(mag),
            np.std(mag),
            np.min(mag),
            np.max(mag),
            np.sqrt(np.mean(mag * mag)),
        ])

    return np.asarray(feats, dtype=np.float32).reshape(1, -1)


# Keep recent predictions between Streamlit reruns.
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = deque(maxlen=5)

if status == "stopped":
    st.session_state.prediction_history.clear()

if status == "window":
    X = feature_vector(event.get("samples", []))

    if X is not None:
        raw_pred = model.predict(X)[0]
        st.session_state.prediction_history.append(raw_pred)

        # Show the most frequent of the last five predictions.
        history = list(st.session_state.prediction_history)
        pred = Counter(history).most_common(1)[0][0]

        st.metric("Predicted activity", str(pred).replace("_", " ").title())
