import cv2
import time
from camera import VideoCameraStream
from biometrics import extract_live_biometrics

print("[Test] Starting multi-threaded camera engine loop... Press 'q' to close.")
cam = VideoCameraStream(src=0).start()

try:
    while True:
        grabbed, frame = cam.read()
        if not grabbed or frame is None:
            time.sleep(0.05)
            continue
            
        # Extract features and draw overlays
        processed_frame, metrics = extract_live_biometrics(frame)
        
        # Display data on the screen
        cv2.putText(processed_frame, f"EAR: {metrics[0]:.2f} | Dev: {metrics[3]:.2f}", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Multi-Threaded CV Pipeline Test", processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cam.stop()
    cv2.destroyAllWindows()
    print("[Test] Stream released gracefully.")