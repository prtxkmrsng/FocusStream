import cv2
import threading
import time

class VideoCameraStream:
    def __init__(self, src=0):
        """
        Initializes the camera stream on a background thread.
        src=0 targets your native Windows primary webcam.
        """
        self.stream = cv2.VideoCapture(src)
        # Configure lower buffer size to minimize lag frame processing
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        (self.grabbed, self.frame) = self.stream.read()
        
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return self
        self.started = True
        # Launch frame tracking in an isolated background execution worker thread
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()
        return self

    def update(self):
        """
        Background loop continually reading matrices directly from hardware buffer.
        """
        while self.started:
            (grabbed, frame) = self.stream.read()
            if grabbed:
                # Thread-safe write access to prevent tearing
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
            time.sleep(0.01) # Small sleep to prevent thread thrashing

    def read(self):
        """
        Returns the latest frame read by the background thread.
        """
        with self.read_lock:
            # Create a shallow copy of the frame matrix safely
            frame_copy = self.frame.copy() if self.frame is not None else None
            return self.grabbed, frame_copy

    def stop(self):
        self.started = False
        if self.thread.is_alive():
            self.thread.join()
        self.stream.release()