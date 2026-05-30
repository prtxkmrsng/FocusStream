import cv2
import numpy as np
import math
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Auto-Download the modern Task API model from Google
MODEL_PATH = 'face_landmarker.task'
if not os.path.exists(MODEL_PATH):
    print("[System] Downloading Google's modern Face Landmarker model...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("[System] Download complete!")

# 2. Initialize the modern Tasks API (Completely bypasses mp.solutions!)
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

def calculate_distance(point1, point2):
    """Calculates pixel distance between two dots."""
    return math.hypot(point2[0] - point1[0], point2[1] - point1[1])

def extract_live_biometrics(frame):
    """
    Extracts high-precision facial landmarks using the modern Tasks API.
    Returns: (processed_frame, [ear, blink_duration, mar, head_deviation])
    """
    if frame is None:
        return None, None

    h, w, _ = frame.shape
    
    # 3. Task API requires a specific MediaPipe Image object
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # 4. Detect faces
    detection_result = detector.detect(mp_image)

    if not detection_result.face_landmarks:
        return None, None # No face found

    # 5. Extract the 468 precise coordinates
    landmarks = detection_result.face_landmarks[0]

    def get_pt(index):
        return [int(landmarks[index].x * w), int(landmarks[index].y * h)]

    # --- 1. PRECISE EYE ASPECT RATIO (EAR) ---
    left_h = calculate_distance(get_pt(159), get_pt(145))
    left_w = calculate_distance(get_pt(33), get_pt(133))
    left_ear = left_h / (left_w + 1e-6)

    right_h = calculate_distance(get_pt(386), get_pt(374))
    right_w = calculate_distance(get_pt(362), get_pt(263))
    right_ear = right_h / (right_w + 1e-6)

    ear = (left_ear + right_ear) / 2.0

    # --- 2. PRECISE MOUTH ASPECT RATIO (MAR) ---
    mouth_h = calculate_distance(get_pt(13), get_pt(14))
    mouth_w = calculate_distance(get_pt(78), get_pt(308))
    mar = mouth_h / (mouth_w + 1e-6)

    # --- 3. HEAD DEVIATION ---
    nose_x = landmarks[1].x
    head_deviation = abs(0.5 - nose_x)
    
    blink_duration = 0.55 if ear < 0.20 else 0.11

    return frame, [ear, blink_duration, mar, head_deviation]