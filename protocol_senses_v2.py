"""
Protocol Senses v2 - Micro-Expression Detection System

Integrates:
- MediaPipe Face Mesh (478 landmarks)
- Action Unit Estimation (10 AUs from geometry)
- Flash Detector (temporal micro-expression detection)
- Emotion Classifier (FER+ model with GPU acceleration)

Controls:
- 'q': Quit
- 'r': Recalibrate
- 'm': Toggle micro-expression mode (AU/Flash vs basic)
- 'd': Toggle debug output
"""

import cv2
import numpy as np
import urllib.request
import os
import time
import concurrent.futures

# MediaPipe Tasks imports
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat

# Local modules
from action_units_v2 import StrictAUEstimator as AUEstimator, ActionUnits
from flash_detector import FlashDetector, FlashEvent, classify_emotion_from_aus
from emotion_classifier import EmotionClassifier, crop_face_from_landmarks
from rl_agent import SensitivityAgent
from camera_utils import ThreadedCamera



# --- CONFIGURATION ---
DROIDCAM_INDEX = 0
HAND_FACE_THRESHOLD = 0.15
DEBUG_MODE = False  # Terminal output
MICRO_MODE = True   # Enable micro-expression detection

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
        # Green for neutral/low (up to 0.65), Yellow for medium, Red for high
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
        # Show detected emotion with confidence (no emoji - OpenCV Unicode issues)
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
        deviation_threshold=2.0,
        min_duration_ms=50,
        max_duration_ms=500,
        cooldown_ms=500
    )
    emotion_classifier = EmotionClassifier(use_gpu=True)
    
    # Initialize RL Agent
    rl_agent = SensitivityAgent(initial_threshold=2.0)
    
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

    
    with vision.FaceLandmarker.create_from_options(face_options) as face_landmarker, \
         vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker:
        
        # Run calibration
        success, frame_timestamp = run_calibration(cap, face_landmarker, au_estimator, frame_timestamp)
        if not success:
            cap.release()
            cv2.destroyAllWindows()
            return
        
        print("--- PROTOCOL SENSES v2 ACTIVE ---")
        print("Controls: 'q' quit | 'r' recalibrate | 'm' toggle micro-mode | 'd' debug")
        
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
                    # Raw AUs for flash detection (needs absolute values)
                    current_aus = au_estimator.compute(landmarks)
                    # Relative AUs for display (0.5 = YOUR baseline)
                    display_aus = au_estimator.compute_relative(landmarks)
                    
                    is_flashing, flash_event = flash_detector.detect(current_aus)
                    
                    if flash_event:
                        last_flash_event = flash_event
                        
                    if is_flashing:
                        # === ASYNC CLASSIFIER LOGIC ===
                        
                        # 1. Check for completed task
                        result = None
                        if classifier_future and classifier_future.done():
                            try:
                                result = classifier_future.result()
                            except Exception as e:
                                if DEBUG_MODE: print(f"Classifier error: {e}")
                            classifier_future = None
                        
                        # 2. Trigger new task if idle
                        if USE_CNN_CLASSIFIER and emotion_classifier.is_available() and classifier_future is None:
                            face_crop = crop_face_from_landmarks(frame, landmarks)
                            if face_crop is not None:
                                classifier_future = executor.submit(emotion_classifier.classify, face_crop)
                        
                        # 3. Fast Path (AU-Based) - Immediate Result
                        au_emotion, au_conf = classify_emotion_from_aus(display_aus)
                        
                        # 4. Integrate Results
                        # If async result just arrived, use it (with Override sync)
                        if result:
                            if result.confidence > 0.50:
                                if result.emotion == 'neutral' and au_emotion != 'neutral' and au_conf > 0.4:
                                    last_emotion = au_emotion
                                    detected_confidence = au_conf
                                else:
                                    last_emotion = result.emotion
                                    detected_confidence = result.confidence
                            else:
                                last_emotion = au_emotion
                                detected_confidence = au_conf
                                
                            # Feed RL only on finished classification
                            rl_result = (last_emotion, detected_confidence)
                            new_threshold = rl_agent.update(rl_result, True)
                            flash_detector.deviation_threshold = new_threshold
                            
                            if DEBUG_MODE:
                                print(f"        [RL] Sensitivity tuned to: {new_threshold:.2f}")

                        elif last_emotion == "neutral" or last_emotion == "":
                            # While waiting for CNN, show AU estimation
                            last_emotion = au_emotion
                            detected_confidence = au_conf

                        # (Visualization uses last_emotion continuously)
                        last_confidence = detected_confidence
                        
                        if DEBUG_MODE and result:
                            print(f"\n[FLASH] Emotion: {last_emotion} ({detected_confidence:.0%}), Duration: {flash_event.duration_ms:.0f}ms")
                            print(f"        Dominant AUs: {flash_event.dominant_aus}")
                    
                    # Draw AU bars
                    if display_aus:
                        draw_au_bars(frame, display_aus)
                    
                    # Draw flash indicator
                    draw_flash_indicator(frame, is_flashing, last_flash_event, last_emotion, last_confidence)
                    
                    # RL Decay (Exploration)
                    # If not flashing and no event just happened, slowly increase sensitivity
                    if not is_flashing and not flash_event and MICRO_MODE:
                        d_threshold = rl_agent.decay()
                        flash_detector.deviation_threshold = d_threshold
                
                # === FACE TOUCHING (always active) ===
                nose = (landmarks[1].x, landmarks[1].y)
                if hand_result.hand_landmarks:
                    for hand_landmarks in hand_result.hand_landmarks:
                        finger_tip = (hand_landmarks[8].x, hand_landmarks[8].y)
                        dist = calculate_distance(nose, finger_tip)
                        if dist < HAND_FACE_THRESHOLD:
                            status_report.append("FACE_TOUCHING")
                            break
            
            # Display status
            h, w = frame.shape[:2]
            
            if status_report:
                status_text = f"ALERT: {', '.join(set(status_report))}"
                cv2.putText(frame, status_text, (w - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Mode indicator
            mode_text = "MICRO-EXPRESSION MODE" if MICRO_MODE else "BASIC MODE"
            cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # RL Agent Status
            if MICRO_MODE:
                agent_status = rl_agent.get_status()
                cv2.putText(frame, agent_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # FPS counter
            fps_frames += 1
            if time.time() - fps_start >= 1.0:
                current_fps = fps_frames
                fps_frames = 0
                fps_start = time.time()
            cv2.putText(frame, f"FPS: {current_fps}", (w - 80, h - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Controls hint
            cv2.putText(frame, "'q' quit | 'r' recal | 'm' mode | 'd' debug", (10, h - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            
            cv2.imshow('Protocol Senses v2', frame)
            
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
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
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
