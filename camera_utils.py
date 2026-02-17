"""
Camera Utilities for High-Performance Capture
"""

import cv2
import threading
import time
import copy

class ThreadedCamera:
    """
    Non-blocking camera capture using a daemon thread.
    
    Standard cv2.VideoCapture.read() blocks until a frame is ready (latency).
    This class continuously reads frames in a background thread, so the 
    main loop gets the latest frame instantly w/o waiting.
    """
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        # Try to set 30FPS / HD
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()  # Lock for thread-safe frame access
        
        # Start background thread
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True # Kill thread when main program exits
        self.thread.start()

    def update(self):
        """Review frames from stream continuously"""
        while True:
            if self.stopped:
                return
            
            (grabbed, frame) = self.stream.read()
            if grabbed:
                with self.lock:
                    self.frame = frame
            else:
                self.stopped = True

    def read(self):
        """Return the most recent frame (thread-safe copy)"""
        with self.lock:
            return copy.copy(self.frame) if self.frame is not None else None

    def release(self):
        """Stop thread and release camera"""
        self.stopped = True
        self.thread.join()
        self.stream.release()

    def isOpened(self):
        return self.stream.isOpened()
