"""
Protocol Senses v2 - Q-Learning Edition
========================================
Modified to use Q-Learning for adaptive sensitivity tuning.
Merged with Behavioral Analysis Module.

Changes from original:
- Replaced gradient descent with Q-Learning agent
- Added policy visualization
- Enhanced status display with Q-Learning metrics
- Integrated Behavioral Analysis (Chewing, Scratching Zones)
"""

import cv2
import numpy as np
import urllib.request
import os
import time
import concurrent.futures
from collections import deque

# MediaPipe Tasks imports
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat

# Local modules
from action_units_v2 import StrictAUEstimator as AUEstimator, ActionUnits
from flash_detector import FlashDetector, FlashEvent, classify_emotion_from_aus
from emotion_classifier import EmotionClassifier, crop_face_from_landmarks
from rl_agent import QLearningThresholdAgent
from facs_decoder import FACSDecoder  # [NEW] Vector Decoder
from camera_utils import ThreadedCamera


# --- CONFIGURATION ---
DROIDCAM_INDEX = 0
HAND_FACE_THRESHOLD = 0.15
DEBUG_MODE = False
MICRO_MODE = True
USE_CNN_CLASSIFIER = True

# --- MODEL PATHS ---
FACE_MODEL_PATH = "face_landmarker.task"
HAND_MODEL_PATH = "hand_landmarker.task"

FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def download_model(url, path):
    """Download model if not present"""
    if not os.path.exists(path):
        print(f"Downloading {path}...")
        urllib.request.urlretrieve(url, path)
        print(f"Downloaded {path}")


def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def draw_au_bars(frame, aus: ActionUnits, x=10, y=60):
    """Draw AU activation bars on frame"""
    au_dict = aus.to_dict()
    bar_width = 80
    bar_height = 12
    
    for i, (name, value) in enumerate(au_dict.items()):
        y_pos = y + i * 18
        
        # Background
        cv2.rectangle(frame, (x, y_pos), (x + bar_width, y_pos + bar_height), (50, 50, 50), -1)
        
        # Fill based on value
        fill_width = int(value * bar_width)
        color = (0, 255, 0) if value < 0.65 else (0, 255, 255) if value < 0.85 else (0, 0, 255)
        cv2.rectangle(frame, (x, y_pos), (x + fill_width, y_pos + bar_height), color, -1)
        
        # Label
        cv2.putText(frame, name, (x + bar_width + 5, y_pos + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)


def draw_flash_indicator(frame, is_flashing: bool, flash_event: FlashEvent = None, 
                         emotion: str = None, confidence: float = 0.0):
    """Draw flash detection indicator"""
    h, w = frame.shape[:2]
    
    if is_flashing:
        # Flash border effect
        cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 255), 4)
        cv2.putText(frame, "MICRO-EXPRESSION DETECTED!", (w//2 - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    if flash_event and emotion:
        conf_pct = int(confidence * 100)
        text = f"Flash: {emotion.upper()} ({conf_pct}% conf, {flash_event.duration_ms:.0f}ms)"
        cv2.putText(frame, text, (w//2 - 150, h - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


def run_calibration(cap, face_landmarker, au_estimator, start_timestamp=0):
    """Run calibration for AU baseline. Returns (success, final_timestamp)"""
    print("\n" + "="*50)
    print("MICRO-EXPRESSION CALIBRATION")
    print("="*50)
    print("Keep your face NEUTRAL and look at the camera.")
    print("Collecting baseline for 3 seconds...")
    print("="*50 + "\n")
    
    frame_timestamp = start_timestamp
    collected = 0
    target_frames = 90
    
    while collected < target_frames:
        frame = cap.read()
        if frame is None:
            continue
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
        frame_timestamp += 33
        
        face_result = face_landmarker.detect_for_video(mp_image, frame_timestamp)
        
        if face_result.face_landmarks:
            landmarks = face_result.face_landmarks[0]
            au_estimator.add_calibration_sample(landmarks)
            collected += 1
            
            progress = int((collected / target_frames) * 100)
            cv2.putText(frame, f"CALIBRATING... {progress}%", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, "Keep face NEUTRAL", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Face not detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Protocol Senses v2 - Calibration', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            return False, frame_timestamp
    
    baseline = au_estimator.finalize_calibration()
    
    print("\n" + "="*50)
    print("CALIBRATION COMPLETE!")
    print("="*50)
    print("Baseline AU Values:")
    for name, value in baseline.to_dict().items():
        print(f"  {name}: {value:.3f}")
    print("="*50 + "\n")
    
    return True, frame_timestamp


def main():
    global DEBUG_MODE, MICRO_MODE
    
    # Download models
    download_model(FACE_MODEL_URL, FACE_MODEL_PATH)
    download_model(HAND_MODEL_URL, HAND_MODEL_PATH)
    
    # Initialize modules
    au_estimator = AUEstimator(smoothing_factor=0.7)
    flash_detector = FlashDetector(
        baseline_frames=30,
        deviation_threshold=2.0,  # Will be dynamically adjusted by Q-Learning
        min_duration_ms=50,
        max_duration_ms=500,
        cooldown_ms=500
    )
    emotion_classifier = EmotionClassifier(use_gpu=True)
    facs_decoder = FACSDecoder() # [NEW] Initialize Decoder
    
    # Initialize Q-Learning Agent
    print("\n" + "="*50)
    print("INITIALIZING Q-LEARNING AGENT")
    print("="*50)
    rl_agent = QLearningThresholdAgent(
        min_threshold=1.5,
        max_threshold=4.0,
        n_states=15,           # 15 discrete threshold levels
        learning_rate=0.1,     # How fast to learn
        discount_factor=0.9,   # Value future rewards at 90% of immediate
        epsilon=0.4,           # Start with 40% exploration
        epsilon_decay=0.995,   # Gradually reduce exploration
        epsilon_min=0.05       # Always keep 5% exploration
    )
    
    BRAIN_PATH = "brain_qlearning.json"
    if rl_agent.load_state(BRAIN_PATH):
        print("Continuing from previous training session.")
    else:
        print("Starting fresh training session.")
    print("="*50 + "\n")
    
    # Initialize Async Executor for Classifier
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    classifier_future = None
            
    # Create MediaPipe options
    face_options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    hand_options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    print("Starting Threaded Camera...")
    cap = ThreadedCamera(DROIDCAM_INDEX)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    frame_timestamp = 0
    
    # For FPS tracking
    fps_start = time.time()
    fps_frames = 0
    current_fps = 0
    
    # State
    last_emotion = "neutral"
    last_flash_event = None
    last_confidence = 0.0
    chew_buffer = deque(maxlen=60) # 2 seconds history for chewing detection
    
    with vision.FaceLandmarker.create_from_options(face_options) as face_landmarker, \
         vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker:
        
        # Run calibration
        success, frame_timestamp = run_calibration(cap, face_landmarker, au_estimator, frame_timestamp)
        if not success:
            cap.release()
            cv2.destroyAllWindows()
            return
        
        print("--- PROTOCOL SENSES v2 - Q-LEARNING MODE ---")
        print("Controls: 'q' quit | 'r' recalibrate | 'm' toggle micro-mode | 'd' debug | 'p' print policy")
        
        # Main Loop
        while True:
            frame = cap.read()
            if frame is None:
                print("Ignoring empty camera frame.")
                continue
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
            frame_timestamp += 33
            
            # Detect face and hands
            face_result = face_landmarker.detect_for_video(mp_image, frame_timestamp)
            hand_result = hand_landmarker.detect_for_video(mp_image, frame_timestamp)
            
            status_report = []
            is_flashing = False
            flash_event = None
            current_aus = None
            
            if face_result.face_landmarks:
                landmarks = face_result.face_landmarks[0]
                
                if MICRO_MODE:
                    # === MICRO-EXPRESSION DETECTION ===
                    current_aus = au_estimator.compute(landmarks)
                    display_aus = au_estimator.compute_relative(landmarks)
                    
                    is_flashing, flash_event = flash_detector.detect(current_aus)
                    
                    if flash_event:
                        last_flash_event = flash_event
                        
                    if is_flashing:
                        # === ASYNC CLASSIFIER LOGIC ===
                        result = None
                        if classifier_future and classifier_future.done():
                            try:
                                result = classifier_future.result()
                            except Exception as e:
                                if DEBUG_MODE: print(f"Classifier error: {e}")
                            classifier_future = None
                        
                        if USE_CNN_CLASSIFIER and emotion_classifier.is_available() and classifier_future is None:
                            face_crop = crop_face_from_landmarks(frame, landmarks)
                            if face_crop is not None:
                                classifier_future = executor.submit(emotion_classifier.classify, face_crop)
                        
                        # === EMOTION DECODING (FACS VECTOR) ===
                        # Replaced Black Box CNN with Transparent Vector Math
                        au_emotion, au_conf, all_scores = facs_decoder.decode(display_aus)

                        # Integrate Results
                        # We trust FACS Decoder more than the CNN now.
                        # CNN is only used if explicitly enabled and high confidence.
                        
                        use_cnn_result = False
                        if USE_CNN_CLASSIFIER and emotion_classifier.is_available() and result:
                             if result.confidence > 0.85: # Only trust CNN if SUPER confident
                                 use_cnn_result = True
                        
                        if use_cnn_result:
                            last_emotion = result.emotion
                            detected_confidence = result.confidence
                        else:
                            last_emotion = au_emotion
                            detected_confidence = au_conf
                                
                        # === Q-LEARNING UPDATE ===
                        rl_result = (last_emotion, detected_confidence)
                        new_threshold = rl_agent.update(rl_result, True)
                        flash_detector.deviation_threshold = new_threshold
                        
                        if DEBUG_MODE:
                            print(f"\n[Q-LEARN] Flash detected!")
                            print(f"  Emotion: {last_emotion} ({detected_confidence:.0%})")
                            print(f"  New threshold: {new_threshold:.2f}")
                            print(f"  Exploration rate: {rl_agent.epsilon:.3f}")

                        elif last_emotion == "neutral" or last_emotion == "":
                            last_emotion = au_emotion
                            detected_confidence = au_conf

                        last_confidence = detected_confidence
                    
                    # Non-flash updates (keep exploring)
                    if not is_flashing and not flash_event:
                        new_threshold = rl_agent.update(None, False)
                        flash_detector.deviation_threshold = new_threshold
                    
                    # Draw AU bars
                    if display_aus:
                        draw_au_bars(frame, display_aus)
                    
                    # Draw flash indicator
                    draw_flash_indicator(frame, is_flashing, last_flash_event, last_emotion, last_confidence)
                
                # === BEHAVIOR ANALYSIS ===
                
                # 1. Chewing Detection
                if current_aus:
                    chew_buffer.append(current_aus.AU26)
                    if len(chew_buffer) > 10:
                        jaw_activity = np.var(list(chew_buffer))
                        if jaw_activity > 0.02: 
                            status_report.append("CHEWING/TALKING")

                # 2. Hand-Face Interaction
                nose = (landmarks[1].x, landmarks[1].y)
                chin = (landmarks[152].x, landmarks[152].y)
                
                if hand_result.hand_landmarks:
                    for hand_landmarks in hand_result.hand_landmarks:
                        finger_tip = (hand_landmarks[8].x, hand_landmarks[8].y)
                        
                        # Distance to Nose
                        dist_nose = calculate_distance(nose, finger_tip)
                        
                        # Distance to Chin (for Neck check)
                        dist_chin = calculate_distance(chin, finger_tip)
                        
                        # 1. Nose
                        if dist_nose < 0.08:
                            status_report.append("SCRATCHING NOSE")
                        
                        # 2. Chin (Direct contact) - High priority over Neck
                        elif dist_chin < 0.06:
                            status_report.append("SCRATCHING CHIN")
                            
                        # 3. Neck (Strictly Below Chin & Further away)
                        elif finger_tip[1] > chin[1] and dist_chin < 0.40:
                            status_report.append("SCRATCHING NECK")
                            
                        # 4. General Face
                        elif dist_nose < HAND_FACE_THRESHOLD:
                            status_report.append("FACE TOUCHING")
            
            # Display status
            h, w = frame.shape[:2]
            
            if status_report:
                status_text = f"ALERT: {', '.join(set(status_report))}"
                cv2.putText(frame, status_text, (w - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Mode indicator
            mode_text = "Q-LEARNING MODE" if MICRO_MODE else "BASIC MODE"
            cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Q-Learning Agent Status
            if MICRO_MODE:
                agent_status = rl_agent.get_status()
                cv2.putText(frame, agent_status, (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            
            # FPS counter
            fps_frames += 1
            if time.time() - fps_start >= 1.0:
                current_fps = fps_frames
                fps_frames = 0
                fps_start = time.time()
            cv2.putText(frame, f"FPS: {current_fps}", (w - 80, h - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Controls hint
            cv2.putText(frame, "'q' quit | 'r' recal | 'm' mode | 'd' debug | 'p' policy", (10, h - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

            # === UI: EMOTION SPECTRUM (Right Side) ===
            # Show all vector scores transparently
            if 'all_scores' in locals() and all_scores:
                ui_y = 150
                cv2.putText(frame, "VECTOR MATCH SCORES:", (w - 220, ui_y - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Sort by score desc
                sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
                
                for emo, score in sorted_scores:
                    # Label
                    cv2.putText(frame, f"{emo[:3].upper()}", (w - 220, ui_y + 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    
                    # Bar
                    bar_len = int(score * 150)
                    color = (0, 255, 0) if emo == last_emotion else (100, 100, 100)
                    if score > 0.8: color = (0, 255, 255) # High confidence
                    if score < 0.1: bar_len = 2 # Minimum visibility
                    
                    cv2.rectangle(frame, (w - 180, ui_y), (w - 180 + bar_len, ui_y + 12), color, -1)
                    cv2.putText(frame, f"{score:.2f}", (w - 180 + bar_len + 5, ui_y + 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    ui_y += 25
            
            cv2.imshow('Protocol Senses v2 - Q-Learning', frame)
            
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                print("\nSaving Q-Learning brain...")
                rl_agent.save_state(BRAIN_PATH)
                print("Brain saved. Quitting.")
                break
            elif key == ord('r'):
                print("\nRecalibrating...")
                au_estimator = AUEstimator()
                flash_detector.reset()
                success, frame_timestamp = run_calibration(cap, face_landmarker, au_estimator, frame_timestamp)
                if not success:
                    break
            elif key == ord('m'):
                MICRO_MODE = not MICRO_MODE
                print(f"Micro-expression mode: {'ON' if MICRO_MODE else 'OFF'}")
            elif key == ord('d'):
                DEBUG_MODE = not DEBUG_MODE
                print(f"Debug mode: {'ON' if DEBUG_MODE else 'OFF'}")
            elif key == ord('p'):
                # Print learned policy
                rl_agent.print_policy()
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final statistics
    print("\n" + "="*60)
    print("SESSION COMPLETE")
    print("="*60)
    rl_agent.print_policy()
    print("\nBest learned threshold:", rl_agent.get_best_threshold())
    print("="*60)


if __name__ == "__main__":
    main()
