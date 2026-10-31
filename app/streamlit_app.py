"""Live interview behaviour dashboard."""

import sys
from collections import Counter
from pathlib import Path

import av
import pandas as pd
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.video_processor import CLASS_NAMES, LiveInferenceProcessor

MODEL_PATH = ROOT / "models" / "cnn_lstm_classifier_35epochs_SUBJECT_06.pt"

st.set_page_config(page_title="Interview Behaviour Monitor", layout="wide", page_icon="🎥")
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #000000; color: #f4f4f4; }
[data-testid="stHeader"] { background: #000000; }
[data-testid="stSidebar"] { background: #080808; width: 220px !important; min-width: 220px !important; }
[data-testid="stSidebarUserContent"] { padding: 1rem 0.8rem; }
[data-testid="stSidebar"] div { color: #f4f4f4; }
[data-testid="stToolbar"] { background: transparent; }
.card { background: #111111; border: 1px solid #343434; border-radius: 14px; padding: 18px; margin-bottom: 12px; box-shadow: 0 8px 24px rgba(255,255,255,0.025); }
.alert { background: #301d00; border: 1px solid #d89a2b; border-radius: 12px; padding: 14px; color: #ffcf70; }
.metric-value { font-size: 28px; font-weight: 700; color: #ffffff; }
.probability-row { margin-bottom: 12px; }
.probability-track { height: 10px; background: #292929; border-radius: 6px; overflow: hidden; }
.probability-fill-red { height: 100%; background: #d94a4a; border-radius: 6px; }
.probability-fill-green { height: 100%; background: #35b768; border-radius: 6px; }
[data-testid="stIFrame"] { max-width: 560px; margin: 0 auto; }
[data-testid="stIFrame"] iframe { max-height: 360px; }
</style>
""", unsafe_allow_html=True)

st.title("Interview Behaviour Monitor")
st.caption("Live, non-diagnostic behaviour review • CNN+LSTM • 7.5 FPS • 38-frame windows")

if "processor" not in st.session_state:
    try:
        st.session_state.processor = LiveInferenceProcessor(MODEL_PATH)
    except Exception as exc:
        st.session_state.processor_error = str(exc)

processor = st.session_state.get("processor")
if "processor_error" in st.session_state:
    st.error(f"Could not initialize the model or MediaPipe: {st.session_state.processor_error}")

if processor is not None:
    class WebcamProcessor(VideoProcessorBase):
        def __init__(self):
            self.live_processor = processor

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="bgr24")
            result = self.live_processor.process_frame(image)
            if result.get("face_visible"):
                import cv2
                cv2.circle(image, (24, 24), 8, (0, 220, 120), -1)
            else:
                if result.get("face_visible") is False:
                    import cv2
                    cv2.circle(image, (24, 24), 8, (0, 90, 220), -1)
            return av.VideoFrame.from_ndarray(image, format="bgr24")

    left, right = st.columns([2, 1])
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Live camera")
        context = webrtc_streamer(
            key="interview-camera",
            video_processor_factory=WebcamProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        if context and context.state.playing:
            st.success("Camera active")
        else:
            st.info("Camera idle — use the single START/STOP control above the video.")
        if st.button("Refresh metrics", use_container_width=True):
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    snapshot = processor.snapshot()
    prediction = snapshot.get("prediction")
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Current prediction")
        if prediction:
            display_prediction = "Talking / Listening" if prediction["class_name"] == "Talking" else prediction["class_name"]
            st.markdown(f'<div class="metric-value">{display_prediction}</div>', unsafe_allow_html=True)
            st.write(f"Confidence: {prediction['confidence']:.1%}")
            for name in CLASS_NAMES:
                display_name = "Talking / Listening" if name == "Talking" else name
                probability = prediction['probabilities'][name]
                fill_class = "probability-fill-red" if name in ("Looking down", "Reading") else "probability-fill-green"
                st.write(f"{display_name}: {probability:.1%}")
                st.markdown(
                    f'<div class="probability-row"><div class="probability-track">'
                    f'<div class="{fill_class}" style="width: {probability * 100:.1f}%"></div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.write("Waiting for a complete valid 38-frame sequence")
            st.caption(f"Buffered samples: {snapshot['buffer_size']} / 38")
        st.markdown('</div>', unsafe_allow_html=True)

    threshold = st.sidebar.slider("Reading alert confidence", 0.50, 0.95, 0.75, 0.05)
    consecutive = st.sidebar.slider("Consecutive reading predictions", 2, 10, 2)
    st.sidebar.markdown("---")
    st.sidebar.warning("Reading alerts are review signals, not proof of misconduct.")

    history = processor.history_snapshot()
    recent_reading = [item for item in history[-consecutive:] if item[1] == "Reading" and item[2] >= threshold]
    if len(recent_reading) >= consecutive:
        st.markdown('<div class="alert"><strong>Potential reading behaviour detected — review required.</strong></div>', unsafe_allow_html=True)
    else:
        st.info("No sustained reading alert is currently active.")

    timeline_col, duration_col = st.columns([2, 1])
    with timeline_col:
        st.subheader("Rolling prediction history")
        if history:
            recent = history[-60:]
            timeline = pd.DataFrame(
                {"Prediction": [item[1] for item in recent], "Confidence": [item[2] for item in recent]},
                index=pd.RangeIndex(len(recent), name="Window")
            )
            st.bar_chart(timeline.set_index("Prediction", drop=False)[["Confidence"]])
        else:
            st.caption("Predictions will appear once valid sequences are available.")
    with duration_col:
        st.subheader("Detected durations")
        counts = Counter(item[1] for item in history[-300:])
        # Each prediction corresponds to one sampled temporal window.
        for name in CLASS_NAMES:
            seconds = counts[name] * (38 / 7.5)
            st.write(f"**{name}:** {seconds:.1f} s")

