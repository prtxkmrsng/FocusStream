import cv2
import numpy as np

# Load our local structural landmark detectors
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def extract_live_biometrics(frame) -> tuple:
    """
    Parses an OpenCV image matrix to extract tracking landmarks.
    Returns: (processed_frame, feature_list)
    """
    if frame is None:
        return None, [0.34, 0.12, 0.08, 0.04] # Baseline fallback safe array

    # Convert to grayscale for Haar processing cascade operations
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Initialize baseline metrics (representing our focused state)
    ear = 0.34
    blink_duration = 0.12
    mar = 0.08
    head_deviation = 0.04

    for (x, y, w, h) in faces:
        # Draw bounding rectangle around detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Calculate dynamic head pose variance based on center bounding box movement
        # (Simulating real roll/yaw tracker deviations mathematically)
        head_deviation = abs(0.5 - (x + w/2) / frame.shape[1])
        
        # Region of Interest (ROI) isolated on face region for eye hunting
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3, minSize=(30, 30))
        
        if len(eyes) >= 2:
            # We have both eyes tracked successfully!
            # Draw rectangles and calculate mock real-time Eye Aspect Ratio (EAR)
            for (ex, ey, ew, eh) in eyes[:2]:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
            # Simulate real-time biological adjustments
            # Lower eye height detection relative to width translates to dropping EAR
            eye_widths = [ew for (_, _, ew, _) in eyes[:2]]
            eye_heights = [eh for (_, _, _, eh) in eyes[:2]]
            avg_ratio = np.mean(eye_heights) / np.mean(eye_widths)
            
            # Map structural measurements cleanly to model parameters
            ear = float(np.clip(avg_ratio * 0.5, 0.15, 0.40))
            
            if ear < 0.24:
                # Prolonged low ear reading registers micro-naps
                blink_duration = 0.55
                mar = 0.35 # Fatigue yawns
            else:
                blink_duration = 0.11

    # Format the explicit feature array order matching training setup
    features = [ear, blink_duration, mar, head_deviation]
    return frame, features