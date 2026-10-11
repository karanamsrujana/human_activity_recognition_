from pathlib import Path
from collections import Counter, deque

import joblib
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Live HAR",
    page_icon="📱",
    layout="centered",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "live_har_model.joblib"

# Compact, app-like styling.
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: radial-gradient(circle at 50% 0%, #eaf2ff 0%, #f7f9fc 42%, #f7f9fc 100%); }
    .block-container { max-width: 560px; padding-top: 2.1rem; padding-bottom: 2rem; }
    #MainMenu, footer, header { visibility: hidden; }
    .brand { display:flex; align-items:center; gap:12px; margin-bottom:1.5rem; }
    .brand-icon { width:46px; height:46px; border-radius:15px; display:flex; align-items:center; justify-content:center; background:#172554; color:white; font-size:23px; }
    .brand-name { color:#111827; font-size:1.25rem; font-weight:800; letter-spacing:-.04em; }
    .brand-sub { color:#64748b; font-size:.8rem; margin-top:2px; }
    .result-card { background:#fff; border:1px solid #e7ebf2; border-radius:24px; padding:28px 24px 24px; box-shadow:0 12px 35px rgba(15,23,42,.06); text-align:center; margin:8px 0 18px; }
    .result-label { color:#64748b; font-size:.78rem; font-weight:700; letter-spacing:.13em; text-transform:uppercase; }
    .activity { color:#111827; font-size:clamp(1.8rem, 7vw, 2.65rem); font-weight:800; letter-spacing:-.055em; margin:12px 0 22px; line-height:1.12; }
    .confidence-label { display:flex; justify-content:space-between; color:#64748b; font-size:.85rem; font-weight:600; margin-bottom:9px; }
    .confidence-value { color:#111827; font-weight:800; }
    .progress-track { width:100%; height:8px; background:#edf1f7; border-radius:99px; overflow:hidden; }
    .progress-fill { height:100%; background:linear-gradient(90deg,#2563eb,#38bdf8); border-radius:99px; }
    .waiting { background:white; border:1px solid #e7ebf2; border-radius:24px; padding:34px 20px; text-align:center; color:#64748b; box-shadow:0 12px 35px rgba(15,23,42,.04); }
    .waiting-icon { font-size:2rem; margin-bottom:8px; }
    .stButton > button { border-radius:12px; font-weight:700; min-height:44px; }
    div[data-testid="stAlert"] { border-radius:14px; }
</style>
<div class="brand">
  <div class="brand-icon">⌁</div>
  <div><div class="brand-name">MotionSense</div><div class="brand-sub">Live activity recognition</div></div>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


model = load_model()
if model is None:
    st.error("Model file not found: live_har_model.joblib")
    st.stop()

sensor_component = components.declare_component(
    "har_motion_sensor",
    path=str(ROOT / "motion_sensor_frontend"),
)
event = sensor_component(key="motion_sensor", default={"status": "idle"})
status = event.get("status", "idle") if isinstance(event, dict) else "idle"


def feature_vector(samples):
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 9 or arr.shape[0] < 32:
        return None

    feats = []
    for j in range(9):
        x = arr[:, j]
        dx = np.diff(x)
        feats.extend([
            np.mean(x), np.std(x), np.min(x), np.max(x),
            np.sqrt(np.mean(x * x)), np.mean(np.abs(x)),
            np.mean(dx), np.std(dx), np.sqrt(np.mean(dx * dx)),
        ])

    for cols in [(0, 1, 2), (3, 4, 5), (6, 7, 8)]:
        mag = np.sqrt(np.sum(arr[:, cols] ** 2, axis=1))
        feats.extend([np.mean(mag), np.std(mag), np.min(mag), np.max(mag), np.sqrt(np.mean(mag * mag))])

    return np.asarray(feats, dtype=np.float32).reshape(1, -1)


if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = deque(maxlen=5)
if "confidence_history" not in st.session_state:
    st.session_state.confidence_history = deque(maxlen=5)

if status == "stopped":
    st.session_state.prediction_history.clear()
    st.session_state.confidence_history.clear()

if status == "permission_denied":
    st.error("Allow motion sensor access in your browser to begin.")
elif status == "unsupported":
    st.error("Motion sensors are not supported by this browser. Try Chrome on Android over HTTPS.")
elif status == "error":
    st.error(event.get("message", "Could not start motion sensing."))

if status == "window":
    X = feature_vector(event.get("samples", []))
    if X is not None:
        raw_pred = model.predict(X)[0]
        confidence = None
        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(X)[0]
                classes = list(getattr(model, "classes_", []))
                if raw_pred in classes:
                    confidence = float(probabilities[classes.index(raw_pred)]) * 100
            except (ValueError, TypeError, IndexError):
                confidence = None

        st.session_state.prediction_history.append(raw_pred)
        st.session_state.confidence_history.append((raw_pred, confidence))
        pred = Counter(st.session_state.prediction_history).most_common(1)[0][0]
        matching_confidences = [c for label, c in st.session_state.confidence_history if label == pred and c is not None]
        shown_confidence = sum(matching_confidences) / len(matching_confidences) if matching_confidences else None

        activity = str(pred).replace("_", " ").title()
        if shown_confidence is None:
            confidence_html = '<div class="confidence-label"><span>Confidence</span><span class="confidence-value">N/A</span></div><div class="progress-track"><div class="progress-fill" style="width:0%"></div></div>'
        else:
            pct = max(0, min(100, shown_confidence))
            confidence_html = f'<div class="confidence-label"><span>Confidence</span><span class="confidence-value">{pct:.1f}%</span></div><div class="progress-track"><div class="progress-fill" style="width:{pct:.1f}%"></div></div>'
        st.markdown(f'<div class="result-card"><div class="result-label">Detected activity</div><div class="activity">{activity}</div>{confidence_html}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="waiting"><div class="waiting-icon">📱</div><b>Collecting motion data…</b></div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="waiting"><div class="waiting-icon">📱</div><b>Ready to detect</b></div>', unsafe_allow_html=True)

# This is model confidence when predict_proba is available, not measured test accuracy.
if status not in ("permission_denied", "unsupported", "error"):
    st.caption("Allow motion access when prompted. Confidence is the model's estimate, not measured accuracy.")
