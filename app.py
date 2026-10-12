from pathlib import Path
import streamlit.components.v1 as components
from collections import Counter, deque

import joblib
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Human Activity Recognition",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "live_har_model.joblib"

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .stApp {
            background:
                radial-gradient(circle at top, rgba(59, 130, 246, 0.16), transparent 28%),
                linear-gradient(135deg, #050b16 0%, #0b1220 38%, #0d1728 100%);
        }
        .block-container {
            max-width: 1280px;
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }
        #MainMenu, footer, header { visibility: hidden; }
        .dashboard-shell {
            display: flex;
            flex-direction: column;
            gap: 1.2rem;
        }
        .topbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            padding: 1rem 1.15rem;
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 22px;
            background: rgba(15, 23, 42, 0.68);
            box-shadow: 0 10px 30px rgba(2, 6, 23, 0.3);
            backdrop-filter: blur(10px);
        }
        .brand {
            display: flex;
            align-items: center;
            gap: 0.9rem;
        }
        .brand-icon {
            width: 46px;
            height: 46px;
            border-radius: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #1d4ed8, #38bdf8);
            color: #f8fbff;
            font-size: 1.3rem;
            box-shadow: 0 10px 24px rgba(37, 99, 235, 0.35);
        }
        .brand-name {
            color: #e2e8f0;
            font-size: 1.3rem;
            font-weight: 800;
            letter-spacing: -0.05em;
        }
        .brand-sub {
            color: #94a3b8;
            font-size: 0.8rem;
            letter-spacing: 0.02em;
            margin-top: 2px;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 0.8rem;
            border: 1px solid rgba(56, 189, 248, 0.38);
            border-radius: 999px;
            background: rgba(14, 116, 144, 0.15);
            color: #b9f7ff;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }
        .status-dot {
            width: 9px;
            height: 9px;
            border-radius: 50%;
            background: #22c55e;
            box-shadow: 0 0 12px rgba(34, 197, 94, 0.9);
        }
        .panel {
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 26px;
            background: rgba(15, 23, 42, 0.72);
            box-shadow: 0 18px 50px rgba(2, 6, 23, 0.24);
            backdrop-filter: blur(12px);
            padding: 1.1rem;
        }
        .hero-card {
            padding: 1.35rem 1.15rem 1rem;
        }
        .result-card {
            background: linear-gradient(180deg, rgba(15, 118, 110, 0.18), rgba(15, 23, 42, 0.85));
            border: 1px solid rgba(34, 211, 238, 0.28);
            border-radius: 22px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
            padding: 1.2rem 1rem 1.1rem;
            text-align: center;
            margin-top: 0.4rem;
        }
        .result-label {
            color: #93c5fd;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.18em;
            text-transform: uppercase;
        }
        .activity {
            color: #f8fbff;
            font-size: clamp(2rem, 3vw, 3rem);
            font-weight: 800;
            letter-spacing: -0.06em;
            margin: 0.8rem 0 1rem;
            line-height: 1.1;
        }
        .confidence-label {
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: #a5b4cf;
            font-size: 0.82rem;
            font-weight: 600;
            margin-bottom: 0.55rem;
        }
        .confidence-value {
            color: #f8fbff;
            font-weight: 800;
        }
        .progress-track {
            width: 100%;
            height: 10px;
            border-radius: 999px;
            background: rgba(148, 163, 184, 0.18);
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #38bdf8, #60a5fa, #c084fc);
            box-shadow: 0 0 18px rgba(96, 165, 250, 0.7);
        }
        .waiting {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            gap: 0.7rem;
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 22px;
            padding: 2rem 1rem;
            min-height: 180px;
            color: #dfe8f8;
            text-align: center;
            box-shadow: 0 16px 34px rgba(2, 6, 23, 0.18);
        }
        .waiting-icon {
            font-size: 2.2rem;
            filter: drop-shadow(0 0 12px rgba(96, 165, 250, 0.45));
        }
        .widget-panel {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        .mini-card {
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 20px;
            background: rgba(15, 23, 42, 0.7);
            padding: 1rem;
        }
        .mini-card h4 {
            margin: 0 0 0.7rem;
            color: #93c5fd;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }
        .metric-value {
            color: #f8fbff;
            font-size: 1.5rem;
            font-weight: 800;
            letter-spacing: -0.05em;
        }
        .metric-sub {
            margin-top: 0.5rem;
            color: #9fb5d7;
            font-size: 0.82rem;
            line-height: 1.5;
        }
        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.5rem;
        }
        .pill {
            display: inline-flex;
            align-items: center;
            padding: 0.35rem 0.6rem;
            background: rgba(30, 41, 59, 0.9);
            border: 1px solid rgba(96, 165, 250, 0.25);
            border-radius: 999px;
            color: #dfe9ff;
            font-size: 0.72rem;
            font-weight: 700;
        }
        .stButton > button {
            border-radius: 12px;
            border: none;
            font-weight: 700;
            min-height: 44px;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 16px rgba(96, 165, 250, 0.25);
        }
        [data-testid="stAlert"] { border-radius: 14px; }
        @media (max-width: 960px) {
            .topbar { flex-direction: column; align-items: flex-start; }
            .status-pill { align-self: flex-start; }
        }

        /* Clean dark theme — presentation only; prediction logic is unchanged. */
        .stApp, [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at top, rgba(79,70,229,.16), transparent 32%), #0b1020 !important;
            color: #e8edf7 !important;
        }
        .block-container { max-width: 1320px; padding-top: 1.6rem; padding-bottom: 2.5rem; }
        [data-testid="stHeader"] { background: transparent !important; }
        .topbar {
            background: rgba(18,25,45,.92) !important;
            border: 1px solid #27324b !important;
            border-radius: 20px !important;
            box-shadow: 0 8px 28px rgba(0,0,0,.18) !important;
            backdrop-filter: blur(12px);
        }
        .brand-icon {
            background: linear-gradient(135deg, #6366f1, #22d3ee) !important;
            box-shadow: 0 8px 18px rgba(99,102,241,.22) !important;
            border-radius: 14px !important;
        }
        .brand-name { color: #f1f5f9 !important; letter-spacing: -.045em !important; }
        .brand-sub { color: #9aa8c2 !important; }
        .status-pill {
            color: #6ee7b7 !important; background: rgba(16,185,129,.10) !important;
            border: 1px solid rgba(52,211,153,.28) !important; letter-spacing: .07em !important;
        }
        .status-dot { background: #34d399 !important; box-shadow: 0 0 0 4px rgba(52,211,153,.12) !important; }
        .panel, .mini-card, .waiting {
            background: #12192d !important;
            border: 1px solid #27324b !important;
            border-radius: 20px !important;
            box-shadow: 0 8px 28px rgba(0,0,0,.16) !important;
            backdrop-filter: none !important;
        }
        .hero-card { padding: 1rem !important; }
        .result-card {
            background: linear-gradient(145deg, rgba(79,70,229,.16), rgba(13,148,136,.12)) !important;
            border: 1px solid #33405d !important;
            border-radius: 17px !important;
            box-shadow: none !important;
        }
        .result-label, .mini-card h4 { color: #a5b4fc !important; letter-spacing: .13em !important; }
        .activity { color: #f8fafc !important; letter-spacing: -.045em !important; }
        .confidence-label { color: #a6b3ca !important; }
        .confidence-value, .metric-value { color: #f1f5f9 !important; }
        .progress-track { background: #29334a !important; }
        .progress-fill {
            background: linear-gradient(90deg, #818cf8, #2dd4bf) !important;
            box-shadow: none !important;
        }
        .waiting { color: #dbe5f5 !important; }
        .waiting-icon { filter: none !important; }
        .mini-card h4 { letter-spacing: .1em !important; }
        .metric-sub { color: #9aa8c2 !important; }
        .pill {
            color: #c7d2fe !important; background: #202945 !important;
            border: 1px solid #354266 !important; border-radius: 999px !important;
        }
        .stButton > button {
            border-radius: 12px !important; font-weight: 700 !important;
            min-height: 44px; transition: transform .15s ease, box-shadow .15s ease;
        }
        .stButton > button:hover { box-shadow: 0 8px 18px rgba(99,102,241,.2) !important; }
        [data-testid="stAlert"] { border-radius: 14px !important; }
        [data-testid="stCaptionContainer"] { color: #9aa8c2 !important; }
        [data-testid="stMarkdownContainer"] p, [data-testid="stText"] { color: #dbe5f5; }
        @media (max-width: 960px) {
            .topbar { flex-direction: column; align-items: flex-start; }
            .status-pill { align-self: flex-start; }
            .block-container { padding-left: 1rem; padding-right: 1rem; }
        }

    </style>
    """
    , unsafe_allow_html=True,
)


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

st.markdown(
    """
    <div class="dashboard-shell">
      <div class="topbar">
        <div class="brand">
          <div class="brand-icon">⌁</div>
          <div>
            <div class="brand-name">MotionSense</div>
            <div class="brand-sub">Human activity recognition</div>
          </div>
        </div>
        <div class="status-pill"><span class="status-dot"></span> Live model online</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col_main, col_side = st.columns([1.5, 0.9], gap="large")

with col_main:
    event = sensor_component(key="motion_sensor", default={"status": "idle"})
    status = event.get("status", "idle") if isinstance(event, dict) else "idle"

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
            display_pred = raw_pred
            if hasattr(model, "predict_proba"):
                try:
                    probabilities = model.predict_proba(X)[0]
                    classes = list(getattr(model, "classes_", []))
                    normalized_classes = {str(label).strip().upper(): i for i, label in enumerate(classes)}
                    normalized_pred = str(raw_pred).strip().upper()
                    if normalized_pred == "LAYING":
                        display_pred = "SITTING"
                        sitting_index = normalized_classes.get("SITTING")
                        if sitting_index is not None:
                            confidence = float(probabilities[sitting_index]) * 100
                    elif raw_pred in classes:
                        confidence = float(probabilities[classes.index(raw_pred)]) * 100
                except (ValueError, TypeError, IndexError):
                    confidence = None

            st.session_state.prediction_history.append(display_pred)
            st.session_state.confidence_history.append((display_pred, confidence))
            pred = Counter(st.session_state.prediction_history).most_common(1)[0][0]
            matching_confidences = [c for label, c in st.session_state.confidence_history if label == pred and c is not None]
            shown_confidence = sum(matching_confidences) / len(matching_confidences) if matching_confidences else None

            activity = str(pred).replace("_", " ").title()
            if shown_confidence is None:
                confidence_html = '<div class="confidence-label"><span>Confidence</span><span class="confidence-value">N/A</span></div><div class="progress-track"><div class="progress-fill" style="width:0%"></div></div>'
            else:
                pct = max(0, min(100, shown_confidence))
                confidence_html = f'<div class="confidence-label"><span>Confidence</span><span class="confidence-value">{pct:.1f}%</span></div><div class="progress-track"><div class="progress-fill" style="width:{pct:.1f}%"></div></div>'
            st.markdown(f'<div class="panel hero-card"><div class="result-card"><div class="result-label">Detected activity</div><div class="activity">{activity}</div>{confidence_html}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="panel waiting"><div class="waiting-icon">📱</div><div><strong>Collecting motion data…</strong></div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="panel waiting"><div class="waiting-icon">📱</div><div><strong>Ready to detect</strong></div></div>', unsafe_allow_html=True)

with col_side:
    activity_classes = getattr(model, "classes_", [])
    displayed_classes = [str(label).replace("_", " ").title() for label in activity_classes]
    badge_html = "".join(f"<span class='pill'>{label}</span>" for label in displayed_classes)

    status_text = {
        "Ready": "Ready",
        "started": "Running",
        "stopped": "Stopped",
        "window": "Predicting",
        "permission_denied": "Permission blocked",
        "unsupported": "Unsupported",
        "error": "Sensor error",
    }.get(status, "Ready")

    st.markdown(
        f"""
        <div class="widget-panel">
          <div class="mini-card">
            <h4>System state</h4>
            <div class="metric-value">{status_text}</div>
            <div class="metric-sub">The browser motion sensor remains the live data source used by the model.</div>
          </div>
          <div class="mini-card">
            <h4>Model classes</h4>
            <div class="pill-row">{badge_html}</div>
          </div>
          <div class="mini-card">
            <h4>Data stream</h4>
            <div class="metric-value">50 Hz</div>
            <div class="metric-sub">Feature windows are sampled and transformed to match the training setup.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if status not in ("permission_denied", "unsupported", "error"):
    st.caption("Allow motion access when prompted. Confidence is the model's estimate, not measured accuracy.")
