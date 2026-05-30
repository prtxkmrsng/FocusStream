# FocusStream: AI Engagement & Fatigue Monitor 📊

FocusStream is an edge-based, real-time computer vision tracking application designed to analyze user focus profiles and trigger automated fatigue mitigation responses. It captures live webcam video streams, extracts high-precision 3D facial geometry, and utilizes a regularized gradient boosting tree architecture to classify binary behavioral states (Focused vs. Fatigued).

---

## ⚙️ Core Technical Architecture

- **Multi-Threaded CV Pipeline:** OpenCV running on an isolated background execution worker thread to fetch native Windows webcam frame matrices continuously at 30 FPS without blocking the UI rendering cycle.
- **Biometric Feature Engineering:** Automated mathematical extraction of Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and Head Pose Space Deviation using the cutting-edge **Google MediaPipe Tasks API (`vision.FaceLandmarker`)** for 468-point sub-millimeter precision.
- **Data Preprocessing & Exploration:** Dimensionality reduction (PCA) and algorithmic grouping (K-Means) via Jupyter Notebooks to validate biometric separation. Real-time data is normalized via Scikit-learn's `StandardScaler`.
- **Classification Engine:** XGBoost Binary Logistic Classifier trained locally with an early-stopping regularization loop to prevent overfitting and maximize recall on consumer hardware.
- **Analytical Front-End Layer:** Streamlit runtime hosting live tracking telemetry charts, biometric metrics data, and automatic Windows system audio alarm triggers (`winsound`).

---

## 📁 Repository Blueprint

```text
FocusStream/
│
├── raw_data/                          # Local Kaggle image datasets (focused/fatigued)
├── .gitattributes                     # Line-ending normalizations across operating systems
├── .gitignore                         # Tracking safe-list configuration
├── app.py                             # Live Streamlit dashboard layout entry-point
├── biometrics.py                      # MediaPipe FaceLandmarker geometry & feature calculator
├── build_dataset.py                   # Automated image extraction bridge to generate CSVs
├── camera.py                          # Thread-isolated safe background webcam handler
├── data_exploration.ipynb             # PCA and K-Means clustering validation (Unsupervised ML)
├── train_model.py                     # Regularized XGBoost binary classifier trainer
├── face_landmarker.task               # Downloaded MediaPipe Tasks API lightweight model
├── feature_scaler.pkl                 # Serialized Scikit-learn data normalization math
├── focus_model.json                   # Serialized production-ready XGBoost weights
└── drowsiness_data.csv                # Extracted tabular biometric training data
```

## 🚀 Local Windows Setup Instruction

### 1. Prerequisite Checklist
- Python: Ensure Python 3.10+ is installed and mapped to your system environment path.
- Hardware: A functional, built-in or external USB webcam.

### 2. Environment Configuration

Clone this repository, open your Command Prompt, navigate to your root directory, and set up your virtual space:

```
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Data Pipeline & Exploration

``Skip this step if you want to use our drowsiness_data.csv``

- Put your categorized image dataset in:
```
FocusStream/
│
├── raw_data/
    ├── focused/
    └── fatigued/
```
- Generate the tabular training data from raw images using the MediaPipe extraction bridge:

```
python build_dataset.py
```

- ***(Optional)*** Validate the mathematical separation of the extracted biometrics using Unsupervised Learning:

```
jupyter lab
```

Open `data_exploration.ipynb` to view the PCA and K-Means clustering visualizations.

### 4. Model Training & Serialization

``Skip this step if you want to use our pretrained model and have the files`` **focus_model.json** and **feature_scaler.pkl**.

Train the high-performance gradient booster model on your local system with automated early stopping:

```
python train_model.py
```

This generates your validation metrics alongside the serialized `focus_model.json` **(model weights)** and `feature_scaler.pkl` **(normalization parameters)**.

### 5. Launch the Monitor Dashboard

Deploy the interactive tracking panel engine:

```
streamlit run app.py
```

- Open your browser and navigate to the dashboard host interface (typically `http://localhost:8501/`).