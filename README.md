# FocusStream: AI Engagement & Fatigue Monitor 📊

FocusStream is an edge-based real-time computer vision tracking application constructed to analyze user focus profiles and trigger automated fatigue mitigation responses. It captures live webcam video streams, extracts structural facial landmarks, and utilizes a trained gradient boosting tree architecture to classify user behavioral states.

---

## ⚙️ Core Technical Architecture

- **Multi-Threaded CV Pipeline:** OpenCV running on an isolated background execution worker thread to fetch webcam frame matrices continuously at 30 FPS without blocking the UI rendering cycle.
- **Biometric Feature Engineering:** Automated mathematical extraction of Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and Head Pose Space Deviation using structural pre-trained Haar Cascades.
- **Classification Engine:** XGBoost Multiclass Classifier trained locally on a regularized objective function to minimize sequential residual errors with built-in overfitting mitigations.
- **Analytical Front-End Layer:** Streamlit runtime hosting live tracking telemetry charts, biometric metrics data, and automatic Windows system audio alarm triggers (`winsound`).

---

## 📁 Repository Blueprint

```text
FocusStream/
│
├── .gitattributes                     # Line-ending normalizations across operating systems
├── .gitignore                         # Tracking safe-list configuration
├── app.py                             # Live Streamlit dashboard layout entry-point
├── biometrics.py                      # Facial landmark mapping & feature calculator
├── camera.py                          # Thread-isolated safe background webcam handler
├── train_model.py                     # Regularized XGBoost dataset trainer
├── focus_model.json                   # Serialized production-ready model weights
├── download_cascades.py               # Downloads Landmark Parameters
├── haarcascade_eye.xml                # Pre-trained Haar Cascade eye-tracking parameters
└── haarcascade_frontalface_default.xml # Pre-trained Haar Cascade face-tracking parameters
```

## 🚀 Local Windows Setup Instruction

### 1. Prerequisite Checklist

- Python: Ensure Python 3.10+ is installed and mapped to your system environment path.
- Hardware: A functional, built-in or external USB webcam.
- Landmark Parameters: Ensure both haarcascade_frontalface_default.xml and haarcascade_eye.xml are located in your root directory or run `python download_cascade.py` from your terminal.

### 2. Environment Configuration

Clone this repository, navigate to your root directory, and set up your virtual space:

```
python -m venv venv
call venv\Scripts\activate
pip install -r requirements
```

### 3. Model Training & Serialization

> **_Skip this step if focus_model.json is present in your root._**

Train your high-performance gradient booster model on your local system before initiating the tracking client:

```
python train_model.py
```

This generates your local model validation metrics score alongside the serialized focus_model.json weight files.

### 4. Launch the Monitor Dashboard

Deploy the interactive tracking panel engine:

```
streamlit run app.py
```

Open your browser and navigate to:

- Dashboard Host Interface: http://localhost:8501/
