import streamlit as st
import cv2
import xgboost as xgb
import numpy as np
import time
import os
import joblib

# Import custom multi-threaded components
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
    if not os.path.exists("focus_model.json"):
        st.error("Missing 'focus_model.json'. Run train_model.py first!")
        st.stop()
    model = xgb.XGBClassifier()
    model.load_model("focus_model.json")
    return model

@st.cache_resource
def load_scaler():
    if not os.path.exists("feature_scaler.pkl"):
        st.error("Missing 'feature_scaler.pkl'. Run train_model.py first!")
        st.stop()
    return joblib.load("feature_scaler.pkl")

@st.cache_resource
def initialize_camera_thread():
    return VideoCameraStream(src=0).start()

# Load our saved model and scaler
model = load_production_xgb_model()
scaler = load_scaler()

# -------------------------------------------------------------------
# DASHBOARD GRID METRIC DESIGN
# -------------------------------------------------------------------
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

run_system = st.checkbox("Toggle Engine Active State", value=True)

cam = initialize_camera_thread()

# -------------------------------------------------------------------
# MAIN CORE REAL-TIME EXECUTION ITERATOR LOOP
# -------------------------------------------------------------------
if run_system:
    st.toast("FocusStream engine initialized. Live evaluation starting...", icon="🚀")
    
    # Binary State mapping
    state_classes = {0: "Focused", 1: "Fatigued"}
    
    while run_system:
        grabbed, frame = cam.read()
        if not grabbed or frame is None:
            time.sleep(0.03) 
            continue
            
        processed_frame, feature_space = extract_live_biometrics(frame)
        
        # Only predict if a face was actually detected
        if feature_space is not None:
            # Format the raw values
            raw_input_data = np.array([feature_space], dtype=np.float32)
            
            # SCALE THE DATA using the exact math from training
            scaled_input_data = scaler.transform(raw_input_data)
            
            # Execute local model prediction
            prediction_index = int(model.predict(scaled_input_data)[0])
            current_state = state_classes[prediction_index]
            
            # Update frontend metrics
            ear_metric.metric(label="Eye Aspect Ratio (EAR)", value=f"{feature_space[0]:.2f}")
            mar_metric.metric(label="Mouth Aspect Ratio (MAR)", value=f"{feature_space[2]:.2f}")
            dev_metric.metric(label="Head Position Deviation", value=f"{feature_space[3]:.2f}")
            
            # Dynamic alert block
            if current_state == "Focused":
                status_placeholder.success("🎯 STATE: FOCUSED — System optimal.")
            elif current_state == "Fatigued":
                status_placeholder.error("🚨 STATE: FATIGUED DETECTED — Take a micro break!")
                import winsound
                winsound.Beep(800, 250)
        else:
            status_placeholder.warning("⚠️ Waiting for face detection...")
            
        # Convert frame color space matrices from OpenCV standard BGR to Streamlit RGB
        if processed_frame is not None:
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
            
        time.sleep(0.01)
else:
    st.info("System Engine Paused. Toggle checkbox to reboot tracking sensors.")