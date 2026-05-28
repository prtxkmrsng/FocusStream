import streamlit as st
import cv2
import xgboost as xgb
import numpy as np
import time
import os

# Import the custom multi-threaded components we engineered
from camera import VideoCameraStream
from biometrics import extract_live_biometrics

# -------------------------------------------------------------------
# PAGE SETUP & CONFIGURATION
# -------------------------------------------------------------------
st.set_page_config(
    page_title="FocusStream | AI Engagement Monitor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 FocusStream: Real-Time AI Engagement & Fatigue Monitor")
st.markdown("---")

# -------------------------------------------------------------------
# CACHED INITIALIZATION RESOURCE WORKERS
# -------------------------------------------------------------------
@st.cache_resource
def load_production_xgb_model():
    """Loads and caches our optimized tree ensemble checkpoint."""
    if not os.path.exists("focus_model.json"):
        st.error("Missing 'focus_model.json'. Run train_model.py first!")
        st.stop()
    model = xgb.XGBClassifier()
    model.load_model("focus_model.json")
    return model

@st.cache_resource
def initialize_camera_thread():
    """Starts our thread-isolated background windows webcam capture."""
    return VideoCameraStream(src=0).start()

# Load our saved model structure
model = load_production_xgb_model()

# -------------------------------------------------------------------
# DASHBOARD GRID METRIC DESIGN
# -------------------------------------------------------------------
# Create layout split columns for streaming metrics and the webcam video window
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("💡 Biometric Tracking Telemetry")
    ear_metric = st.metric(label="Eye Aspect Ratio (EAR)", value="0.00")
    mar_metric = st.metric(label="Mouth Aspect Ratio (MAR)", value="0.00")
    dev_metric = st.metric(label="Head Position Deviation", value="0.00")
    
    st.markdown("---")
    st.subheader("🚨 System Status Matrix")
    status_placeholder = st.empty()

with col2:
    st.subheader("📹 Live Computer Vision Analysis Feed")
    video_placeholder = st.empty()

# Add a user-controlled execution kill switch
run_system = st.checkbox("Toggle Engine Active State", value=True)

# Initialize the camera background worker thread
cam = initialize_camera_thread()

# -------------------------------------------------------------------
# MAIN CORE REAL-TIME EXECUTION ITERATOR LOOP
# -------------------------------------------------------------------
if run_system:
    st.toast("FocusStream engine initialized. Live evaluation starting...", icon="🚀")
    
    # State mapping translations matching training indices
    state_classes = {0: "Focused", 1: "Distracted", 2: "Fatigued"}
    
    while run_system:
        # Fetch the latest available video frame matrix from the background thread safely
        grabbed, frame = cam.read()
        if not grabbed or frame is None:
            time.sleep(0.03) # Match 30FPS synchronization pacing
            continue
            
        # Parse frame matrices through the biometric feature engine
        processed_frame, feature_space = extract_live_biometrics(frame)
        
        # Format the values for our XGBoost inference pipeline
        # Order: [ear, blink_duration, mar, head_deviation]
        input_data = np.array([feature_space], dtype=np.float32)
        
        # Execute local model class probability prediction
        prediction_index = int(model.predict(input_data)[0])
        current_state = state_classes[prediction_index]
        
        # ---------------------------------------------------------------
        # FRONTEND DASHBOARD RENDERING & ALERTS UPDATE
        # ---------------------------------------------------------------
        # Update raw numerical biometrics displays
        ear_metric.metric(label="Eye Aspect Ratio (EAR)", value=f"{feature_space[0]:.2f}")
        mar_metric.metric(label="Mouth Aspect Ratio (MAR)", value=f"{feature_space[2]:.2f}")
        dev_metric.metric(label="Head Position Deviation", value=f"{feature_space[3]:.2f}")
        
        # Dynamic alert block formatting conditional logic
        if current_state == "Focused":
            status_placeholder.success("🎯 STATE: FOCUSED — System optimal.")
        elif current_state == "Distracted":
            status_placeholder.warning("⚠️ STATE: DISTRACTED — Please look back at the display screen.")
        elif current_state == "Fatigued":
            status_placeholder.error("🚨 STATE: FATIGUED DETECTED — Take a micro break!")
            # Native Windows audio buzzer alert execution (Frequency=800Hz, Duration=250ms)
            import winsound
            winsound.Beep(800, 250)
            
        # Convert frame color space matrices from OpenCV standard BGR to Streamlit RGB
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
        
        # Give a microscopic window for rendering operations to cycle
        time.sleep(0.01)
else:
    st.info("System Engine Paused. Toggle checkbox to reboot tracking sensors.")