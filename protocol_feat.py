"""
PROTOCOL FEAT: OpenFace 2.0 (Python Edition)
High-Accuracy Facial Behavior Analysis using Py-Feat.

Powered by:
- RetinaFace (Face Detection)
- MobileNet/PFLD (Landmarks)
- XGBoost/SVM (Action Units) - "Academic Grade"
- ResMaskNet (Emotion)

WARNING: Heavy Resource Usage. Requires GPU for decent FPS.
"""

import cv2
import time
import numpy as np
import pandas as pd
import threading
import queue
from typing import Dict, Optional

# Attempt import (will fail if not installed)
try:
    from feat import Detector
except ImportError:
    print("\n[ERROR] Py-Feat not found.")
    print("Please run: pip install py-feat pandas scikit-learn")
    print("Note: Installation requires Internet and may take time (Torch + Models).")
    exit(1)

from action_units_v2 import ActionUnits
from flash_detector import FlashDetector, FlashEvent
from camera_utils import ThreadedCamera

# --- CONFIGURATION ---
SKIP_FRAMES = 2 # Process every Nth frame to maintain FPS
SCALE_FACTOR = 0.5 # Resize frame for faster detection

def map_feat_to_aus(pred_row) -> ActionUnits:
    """Map Py-Feat output columns to ActionUnits dataclass"""
    # Py-Feat headers are usually 'AU01', 'AU02', etc.
    # Depending on model, scales might be 0-1 or raw.
    
    def get(name):
        return float(pred_row[name].values[0]) if name in pred_row else 0.0

    return ActionUnits(
        AU1=get('AU01'),
        AU2=get('AU02'),
        AU4=get('AU04'),
        AU5=get('AU05'),
        AU6=get('AU06'),
        AU9=get('AU09'),
        AU12=get('AU12'),
        AU15=get('AU15'),
        AU20=get('AU20'),
        AU26=get('AU26'),
        # Gaze/Forehead might need custom mapping from landmarks
        # Py-Feat landmarks are 0-67 (68 points).
    )

def main():
    print("="*50)
    print("PROTOCOL FEAT: OpenFace 2.0 Engine")
    print("Initializing Py-Feat Detector... (First run downloads models)")
    print("="*50)
    
    # Initialize Detector
    # au_model='xgb' is fast and accurate.
    # face_model='retinaface' is accurate but slow. 'img2pose' is faster?
    try:
        detector = Detector(
            face_model="retinaface",
            landmark_model="mobilenet",
            au_model="xgb",
            emotion_model="resmasknet",
            device="cuda" # autodetect if possible
        )
    except Exception as e:
        print(f"Error initializing Detector: {e}")
        return

    print("Detector Ready.")

    cap = ThreadedCamera(0)
    flash_detector = FlashDetector(baseline_frames=15, deviation_threshold=2.5) # Lower baseline due to FPS

    frame_count = 0
    fps_start = time.time()
    
    current_aus = ActionUnits()
    latest_emotion = "Initializing..."
    
    while True:
        frame = cap.read()
        if frame is None:
            time.sleep(0.01)
            continue
            
        frame_count += 1
        
        # Display Loop (Always runs)
        display_frame = frame.copy()
        
        # Detection Loop (Throttled)
        if frame_count % SKIP_FRAMES == 0:
            # Resize for speed
            small_frame = cv2.resize(frame, (0,0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
            
            try:
                # Detect
                detected = detector.detect_image(small_frame)
                
                if not detected.empty:
                    # Extract AUs
                    current_aus = map_feat_to_aus(detected.aus)
                    
                    # Extract Emotion
                    if not detected.emotions.empty:
                        # Get max emotion
                        emotions = detected.emotions.iloc[0]
                        latest_emotion = emotions.idxmax()
                        conf = emotions.max()
                        
                    # Flash Check
                    is_flashing, event = flash_detector.detect(current_aus)
                    if is_flashing:
                         cv2.putText(display_frame, "MICRO DETECTED", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                         
            except Exception as e:
                print(f"Inference error: {e}")

        # Draw UI
        h, w = display_frame.shape[:2]
        cv2.putText(display_frame, f"Emotion: {latest_emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw AUs
        for i, val in enumerate(current_aus.to_array()):
            y = 60 + i*15
            cv2.rectangle(display_frame, (10, y), (10 + int(val*50), y+10), (255, 0, 0), -1)

        fps = frame_count / (time.time() - fps_start)
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (w-100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        cv2.imshow("Protocol Feat", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
