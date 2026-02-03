"""
PROTOCOL FEAT DEEP: The Hybrid Accuracy Engine
===============================================
Combines Py-Feat (OpenFace 2.0) for Action Units with 
DeepFace (AffectNet) for Emotion Recognition.

Powered by:
- Py-Feat (AUs):    XGBoost Model (Academic Grade)
- DeepFace (Emo):   VGG-Face / AffectNet (SOTA Accuracy)

WARNING: extremely Resource Intensive.
"""

import cv2
import time
import numpy as np
import threading
import queue
from typing import Optional

# --- DEPENDENCY CHECK ---
try:
    from feat import Detector as FeatDetector
except ImportError:
    print("[ERROR] Py-Feat not found. Run: pip install py-feat pandas scikit-learn")
    exit(1)

try:
    from deepface import DeepFace
except ImportError:
    print("[ERROR] DeepFace not found. Run: pip install deepface tf-keras")
    exit(1)

# Local imports
from action_units_v2 import ActionUnits
from flash_detector import FlashDetector
from camera_utils import ThreadedCamera


# --- CONFIGURATION ---
SKIP_FRAMES = 2          # Run Py-Feat every N frames
DEEPFACE_INTERVAL = 10   # Run DeepFace every N frames (Heavy!)
SCALE_FACTOR = 0.5       # Resize for speed

def map_feat_to_aus(pred_row) -> ActionUnits:
    """Map Py-Feat output columns to ActionUnits dataclass"""
    def get(name):
        return float(pred_row[name].values[0]) if name in pred_row else 0.0

    return ActionUnits(
        AU1=get('AU01'), AU2=get('AU02'), AU4=get('AU04'),
        AU5=get('AU05'), AU6=get('AU06'), AU9=get('AU09'),
        AU12=get('AU12'), AU15=get('AU15'), AU20=get('AU20'),
        AU26=get('AU26')
    )

def main():
    print("="*50)
    print("PROTOCOL FEAT DEEP: Hybrid Engine")
    print("Initializing Models... (This may take time)")
    print("="*50)

    # 1. Initialize Py-Feat (AUs only)
    print("[1/2] Loading Py-Feat (XGBoost)...")
    try:
        feat_detector = FeatDetector(
            face_model="retinaface",
            landmark_model="mobilenet",
            au_model="xgb",
            emotion_model=None, # We use DeepFace for this
            device="cuda"
        )
    except Exception as e:
        print(f"Py-Feat Init Error: {e}")
        return

    # 2. Initialize DeepFace (Warmup)
    print("[2/2] Warming up DeepFace (AffectNet)...")
    try:
        # Dummy forward pass to load weights
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        DeepFace.analyze(dummy, actions=['emotion'], enforce_detection=False, silent=True)
    except Exception as e:
        print(f"DeepFace Warmup Warning: {e}")

    print("\n>>> HYBRID ENGINE READY <<<")
    print("Controls: 'q' quit | 'd' toggle debug (view raw probs)")

    cap = ThreadedCamera(0)
    flash_detector = FlashDetector(baseline_frames=15, deviation_threshold=2.5)

    frame_count = 0
    fps_start = time.time()
    
    current_aus = ActionUnits()
    latest_emotion = "Neutral"
    deepface_conf = 0.0
    deepface_probs = {} # Store raw probabilities
    
    DEBUG_MODE = False
    
    # Async DeepFace Worker
    deepface_queue = queue.Queue(maxsize=1)
    
    def deepface_worker():
        nonlocal latest_emotion, deepface_conf, deepface_probs
        while True:
            frame_bgr = deepface_queue.get()
            try:
                # DeepFace analyze
                results = DeepFace.analyze(
                    frame_bgr, 
                    actions=['emotion'], 
                    enforce_detection=False, 
                    detector_backend='skip', 
                    silent=True
                )
                if results:
                    top_res = results[0]
                    # Update State
                    latest_emotion = top_res['dominant_emotion']
                    # Normalize confidence 0-1 (DeepFace is 0-100)
                    deepface_conf = top_res['emotion'][latest_emotion] / 100.0
                    deepface_probs = top_res['emotion']
            except Exception as e:
                print(f"[DeepFace] Error: {e}")
            finally:
                deepface_queue.task_done()

    # Start Worker Thread
    t = threading.Thread(target=deepface_worker, daemon=True)
    t.start()
    
    # --- CALIBRATION PHASE ---
    print("\n[CALIBRATION] Calibrating DeepFace Baseline...")
    print("Keep your face NEUTRAL for 5 seconds.")
    baseline_probs = {}
    calibration_buffer = []
    
    # Collect samples
    cal_start = time.time()
    while len(calibration_buffer) < 5 or (time.time() - cal_start) < 5.0:
        if not deepface_probs:
            time.sleep(0.1)
            continue
            
        # Store copy of current probs
        calibration_buffer.append(deepface_probs.copy())
        print(f"Sampling... {len(calibration_buffer)}/5 minimum")
        
        # Force a few updates
        if deepface_queue.empty():
            frame = cap.read()
            if frame is not None:
                deepface_queue.put(frame.copy())
        
        time.sleep(1.0) # Slow sampling
        
    # Calculate Average Baseline
    print("Calculating baseline...")
    keys = calibration_buffer[0].keys()
    for k in keys:
        values = [sample[k] for sample in calibration_buffer]
        baseline_probs[k] = sum(values) / len(values)
    
    print("Baseline Emotions:")
    for k, v in baseline_probs.items():
        if v > 10.0: print(f"  {k}: {v:.1f}%")
    print("Calibration Complete. Applying 'Resting Face' Subtraction.\n")

    while True:
        frame = cap.read()
        if frame is None:
            time.sleep(0.01)
            continue
            
        frame_count += 1
        display_frame = frame.copy()
        
        # --- PY-FEAT (AUs) ---
        if frame_count % SKIP_FRAMES == 0:
             # ... (Py-Feat logic preserved) ...
            small_frame = cv2.resize(frame, (0,0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
            try:
                detected = feat_detector.detect_image(small_frame)
                if not detected.empty:
                    current_aus = map_feat_to_aus(detected.aus)
                    is_flashing, event = flash_detector.detect(current_aus)
                    if is_flashing:
                        if deepface_queue.empty(): deepface_queue.put(frame.copy())
                        cv2.putText(display_frame, "MICRO DETECTED", (100, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            except Exception: pass

        # --- DEEPFACE (Periodic) ---
        if frame_count % DEEPFACE_INTERVAL == 0 and deepface_queue.empty():
            deepface_queue.put(frame.copy())
            
        # --- APPLY CALIBRATION ---
        # Logic: Adjusted = Current - Baseline
        # We find the emotion with the highest POSITIVE increase
        adjusted_emotion = "Neutral"
        adjusted_conf = 0.0
        
        if deepface_probs:
            max_increase = -100.0
            
            for emo, score in deepface_probs.items():
                baseline = baseline_probs.get(emo, 0.0)
                increase = score - baseline
                
                # Boost 'Neutral' logic: If everything is close to baseline, it's Neutral
                # But here we just want the biggest spikes.
                
                if increase > max_increase:
                    max_increase = increase
                    adjusted_emotion = emo
                    adjusted_conf = score / 100.0 # Raw score for confidence display
                    
            # Threshold: If max increase is tiny (< 10%), assume Neutral
            if max_increase < 10.0:
                adjusted_emotion = "Neutral (Resting)"

        # --- DRAW UI ---
        h, w = display_frame.shape[:2]
        
        # Emotion Label
        color = (0, 255, 0) if "Neutral" not in adjusted_emotion else (200, 200, 200)
        cv2.putText(display_frame, f"DeepFace: {adjusted_emotion}", (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        if DEBUG_MODE:
             cv2.putText(display_frame, "DEBUG ON: Check Console", (10, h-40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # AU Bars
        for i, val in enumerate(current_aus.to_array()):
            y = 70 + i*15
            color = (0, 255, 255) if val > 0.5 else (100, 100, 100)
            cv2.rectangle(display_frame, (10, y), (10 + int(val*60), y+10), color, -1)

        fps = frame_count / (time.time() - fps_start)
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (w-120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        cv2.imshow("Protocol Feat Deep", display_frame)
        
        # --- CONTROLS ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            DEBUG_MODE = not DEBUG_MODE
            print(f"Debug Mode: {DEBUG_MODE}")

        # --- DEBUG LOGGING ---
        if DEBUG_MODE and frame_count % 30 == 0:
            print(f"\n[DeepFace] Display: {adjusted_emotion} (Increase: {max_increase:.1f}%)")
            sorted_probs = sorted(deepface_probs.items(), key=lambda x: x[1], reverse=True)
            for emo, score in sorted_probs:
                base = baseline_probs.get(emo, 0.0)
                diff = score - base
                print(f"  {emo}: {score:.1f}% (Base: {base:.1f}%, Diff: {diff:+.1f}%)")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
