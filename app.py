from collections import Counter, deque
from html import escape
from pathlib import Path

import joblib
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="MotionSense", page_icon="📱", layout="centered",
                   initial_sidebar_state="collapsed")

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "live_har_model.joblib"
FRONTEND_DIR = ROOT / "motion_sensor_frontend"

SMOOTHING = 5                      # majority vote over the last N windows
MERGE_LAYING_INTO_SITTING = True   # display rule: LAYING is shown as SITTING
LOW_CONFIDENCE = 50.0              # below this %, show a "low confidence" hint

ICONS = {
    "STANDING": "🧍", "SITTING": "🪑", "LAYING": "🛏️",
    "WALKING": "🚶", "WALKING_UPSTAIRS": "🪜", "WALKING_DOWNSTAIRS": "⬇️",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
.stApp { background: radial-gradient(circle at 50% 0%, #1e1b3a 0%, #0b0d14 45%, #090b10 100%); color: #f3f4f6; }
.block-container { max-width: 560px; padding: 2rem 1rem 2.5rem; }
#MainMenu, footer, header { visibility: hidden; }
iframe { border: 0 !important; background: transparent !important; }
.brand { display: flex; align-items: center; gap: 12px; margin-bottom: 1.2rem; }
.brand-icon { width: 46px; height: 46px; border-radius: 15px; display: flex; align-items: center; justify-content: center;
  background: linear-gradient(135deg, #8b5cf6, #6366f1); font-size: 22px; box-shadow: 0 8px 24px rgba(139,92,246,.25); }
.brand-name { font-size: 1.25rem; font-weight: 800; letter-spacing: -.03em; color: #f9fafb; }
.brand-sub { font-size: .8rem; color: #9ca3af; margin-top: 2px; }
.card { background: linear-gradient(145deg, #181824, #11121a); border: 1px solid rgba(167,139,250,.2);
  border-radius: 24px; padding: 26px 22px; margin: 14px 0; box-shadow: 0 12px 35px rgba(0,0,0,.25); }
.center { text-align: center; }
.eyebrow { color: #a78bfa; font-size: .75rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; }
.emoji { font-size: 3rem; margin-top: 10px; }
.activity { font-size: clamp(1.7rem, 7vw, 2.5rem); font-weight: 800; letter-spacing: -.05em; margin: 6px 0 18px; line-height: 1.1; }
.row { display: flex; justify-content: space-between; color: #9ca3af; font-size: .85rem; font-weight: 600; margin-bottom: 8px; }
.row b { color: #f9fafb; font-weight: 800; }
.track { width: 100%; height: 8px; background: #292936; border-radius: 99px; overflow: hidden; }
.fill { height: 100%; background: linear-gradient(90deg, #8b5cf6, #22d3ee); border-radius: 99px; }
.hint { color: #fbbf24; font-size: .8rem; margin-top: 12px; }
.waiting { color: #9ca3af; }
.waiting .emoji { margin: 0 0 8px; font-size: 2.2rem; }
.sec { color: #9ca3af; font-size: .75rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; margin-bottom: 12px; }
.prob { margin-bottom: 11px; }
.prob:last-child { margin-bottom: 0; }
.chips { display: flex; gap: 6px; flex-wrap: wrap; }
.chip { background: #232433; border-radius: 99px; padding: 5px 11px; font-size: .78rem; color: #d1d5db; }
.foot { color: #6b7280; font-size: .75rem; text-align: center; margin-top: 18px; }
div[data-testid="stAlert"] { border-radius: 14px; }
</style>
<div class="brand"><div class="brand-icon">⌁</div>
<div><div class="brand-name">MotionSense</div><div class="brand-sub">Live human activity recognition</div></div></div>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


model = load_model()
if model is None:
    st.error("Model file not found: live_har_model.joblib (keep it next to app.py).")
    st.stop()
if not (FRONTEND_DIR / "index.html").exists():
    st.error("Missing motion_sensor_frontend/index.html (keep the folder next to app.py).")
    st.stop()

sensor = components.declare_component("har_motion_sensor", path=str(FRONTEND_DIR))
event = sensor(key="motion_sensor", default={"status": "idle"})
if not isinstance(event, dict):
    event = {"status": "idle"}
status = event.get("status", "idle")


def feature_vector(samples):
    """96 features: 9 stats x 9 channels + 5 magnitude stats x 3 sensors (unchanged from training)."""
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 9 or arr.shape[0] < 32:
        return None
    feats = []
    for j in range(9):
        x = arr[:, j]
        dx = np.diff(x)
        feats.extend([np.mean(x), np.std(x), np.min(x), np.max(x),
                      np.sqrt(np.mean(x * x)), np.mean(np.abs(x)),
                      np.mean(dx), np.std(dx), np.sqrt(np.mean(dx * dx))])
    for cols in [(0, 1, 2), (3, 4, 5), (6, 7, 8)]:
        mag = np.sqrt(np.sum(arr[:, cols] ** 2, axis=1))
        feats.extend([np.mean(mag), np.std(mag), np.min(mag), np.max(mag), np.sqrt(np.mean(mag * mag))])
    return np.asarray(feats, dtype=np.float32).reshape(1, -1)


def predict_window(samples):
    X = feature_vector(samples)
    if X is None:
        return None
    label = str(model.predict(X)[0]).strip().upper()
    probs = {}
    if hasattr(model, "predict_proba"):
        try:
            p = model.predict_proba(X)[0]
            for c, v in zip(getattr(model, "classes_", []), p):
                key = str(c).strip().upper()
                probs[key] = probs.get(key, 0.0) + float(v) * 100
        except Exception:
            probs = {}
    if MERGE_LAYING_INTO_SITTING:
        if "LAYING" in probs:
            probs["SITTING"] = probs.get("SITTING", 0.0) + probs.pop("LAYING")
        if label == "LAYING":
            label = "SITTING"
    return label, probs


def pretty(label):
    return label.replace("_", " ").title()


# ---- session state ----
ss = st.session_state
ss.setdefault("history", deque(maxlen=SMOOTHING))
ss.setdefault("counts", Counter())
ss.setdefault("recent", deque(maxlen=8))
ss.setdefault("result", None)
ss.setdefault("last_ts", None)

if status == "started":
    ss.history.clear(); ss.counts.clear(); ss.recent.clear(); ss.result = None; ss.last_ts = None
elif status == "stopped":
    ss.history.clear(); ss.result = None

if status == "permission_denied":
    st.error("Allow motion sensor access in your browser, then press Start again.")
elif status == "unsupported":
    st.error("Motion sensors are not supported here. Open the app in Chrome (Android) or Safari (iOS) over HTTPS.")
elif status == "error":
    st.error(event.get("message", "Could not start motion sensing."))

# ---- process each new window exactly once ----
if status == "window":
    ts = event.get("capturedAt")
    if ts != ss.last_ts:
        ss.last_ts = ts
        out = predict_window(event.get("samples", []))
        if out is not None:
            label, probs = out
            ss.history.append((label, probs))
            top = Counter(l for l, _ in ss.history).most_common(1)[0][0]
            confs = [p[top] for l, p in ss.history if l == top and top in p]
            avg = {}
            for _, p in ss.history:
                for k, v in p.items():
                    avg[k] = avg.get(k, 0.0) + v / len(ss.history)
            ss.result = {"label": top, "conf": (sum(confs) / len(confs)) if confs else None, "probs": avg}
            ss.counts[top] += 1
            ss.recent.append(top)

res = ss.result

# ---- main result card ----
if res:
    icon = ICONS.get(res["label"], "📱")
    conf = res["conf"]
    if conf is None:
        bar = '<div class="row"><span>Confidence</span><b>N/A</b></div><div class="track"><div class="fill" style="width:0%"></div></div>'
        warn = ""
    else:
        pct = max(0.0, min(100.0, conf))
        bar = (f'<div class="row"><span>Confidence</span><b>{pct:.1f}%</b></div>'
               f'<div class="track"><div class="fill" style="width:{pct:.1f}%"></div></div>')
        warn = '<div class="hint">⚠ Low confidence — hold the phone steadily for a few seconds.</div>' if pct < LOW_CONFIDENCE else ""
    st.markdown(f'<div class="card center"><div class="eyebrow">Detected activity</div>'
                f'<div class="emoji">{icon}</div><div class="activity">{escape(pretty(res["label"]))}</div>{bar}{warn}</div>',
                unsafe_allow_html=True)
elif status == "window":
    st.markdown('<div class="card center waiting"><div class="emoji">⏳</div><b>Collecting motion data…</b></div>', unsafe_allow_html=True)
elif status == "started":
    st.markdown('<div class="card center waiting"><div class="emoji">📡</div><b>Listening to sensors…</b><br>First result in about 3 seconds.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="card center waiting"><div class="emoji">📱</div><b>Ready to detect</b><br>Press Start and allow motion access.</div>', unsafe_allow_html=True)

# ---- class probabilities ----
if res and res["probs"]:
    rows = ""
    for k, v in sorted(res["probs"].items(), key=lambda kv: -kv[1]):
        rows += (f'<div class="prob"><div class="row"><span>{ICONS.get(k, "")} {escape(pretty(k))}</span><b>{v:.1f}%</b></div>'
                 f'<div class="track"><div class="fill" style="width:{min(v, 100):.1f}%"></div></div></div>')
    st.markdown(f'<div class="card"><div class="sec">Class probabilities</div>{rows}</div>', unsafe_allow_html=True)

# ---- session summary ----
if ss.counts:
    total = sum(ss.counts.values())
    chips = "".join(f'<span class="chip">{ICONS.get(k, "")} {escape(pretty(k))} · {n}</span>' for k, n in ss.counts.most_common())
    recent = " → ".join(ICONS.get(k, k) for k in ss.recent)
    st.markdown(f'<div class="card"><div class="sec">Session · {total} predictions</div>'
                f'<div class="chips">{chips}</div><div class="sec" style="margin-top:16px">Recent</div>'
                f'<div style="font-size:1.4rem;letter-spacing:.1em">{recent}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="foot">Confidence is the model\'s own probability estimate, not measured accuracy.</div>', unsafe_allow_html=True)
