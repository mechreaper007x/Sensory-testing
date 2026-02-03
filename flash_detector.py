"""
Flash Detector: Temporal Micro-Expression Detection
====================================================
Detects sudden logical spikes in Action Units relative to a baseline.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass
import time
from action_units_v2 import ActionUnits

@dataclass
class FlashEvent:
    start_time: float
    duration_ms: float
    max_deviation: float
    involved_aus: list

class FlashDetector:
    def __init__(self, baseline_frames=30, deviation_threshold=2.5, min_duration_ms=50, max_duration_ms=500, cooldown_ms=500):
        self.baseline_frames = baseline_frames
        self.deviation_threshold = deviation_threshold
        self.min_duration_ms = min_duration_ms
        self.max_duration_ms = max_duration_ms
        self.cooldown_ms = cooldown_ms
        
        # History for baseline
        self.history = deque(maxlen=baseline_frames)
        
        # State
        self.last_trigger_time = 0
        self.is_flashing = False
        self.flash_start_time = 0
        
    def reset(self):
        self.history.clear()
        self.is_flashing = False
        
    def detect(self, current_aus: ActionUnits):
        """
        Detect micro-expressions based on deviation from baseline.
        Returns: (is_flashing, event)
        """
        # Convert AUs to vector
        values = np.array(list(current_aus.to_dict().values()))
        
        # 1. Update Baseline
        if len(self.history) < self.baseline_frames:
            self.history.append(values)
            return False, None
            
        # Calculate Logic
        history_arr = np.array(self.history)
        mean = np.mean(history_arr, axis=0)
        std = np.std(history_arr, axis=0) + 1e-6 # Avoid div/0
        
        # Z-Score
        z_scores = (values - mean) / std
        max_z = np.max(z_scores)
        
        now = time.time() * 1000
        
        # 2. Check Triggers
        if not self.is_flashing:
            # Check cooldown
            if (now - self.last_trigger_time) < self.cooldown_ms:
                self.history.append(values) # Keep updating baseline
                return False, None
                
            if max_z > self.deviation_threshold:
                # START FLASH
                self.is_flashing = True
                self.flash_start_time = now
                return True, None
                
        else:
            # Currently Flashing
            if max_z < (self.deviation_threshold * 0.8): # Hysteresis exit
                # END FLASH
                duration = now - self.flash_start_time
                self.is_flashing = False
                self.last_trigger_time = now
                
                # Check metrics
                if self.min_duration_ms <= duration <= self.max_duration_ms:
                    # Valid Micro-Expression
                    event = FlashEvent(
                        start_time=self.flash_start_time,
                        duration_ms=duration,
                        max_deviation=max_z,
                        involved_aus=[] # Todos: identify specific AUs
                    )
                    return False, event
                else:
                    return False, None # Too short/long
            else:
                return True, None # Still flashing
        
        # Update baseline
        self.history.append(values)
        return False, None

def classify_emotion_from_aus(aus: ActionUnits):
    """
    Simple heuristic rule-based classifier for AUs.
    Returns: (emotion_name, confidence)
    """
    scores = {
        "happiness": (aus.AU6 + aus.AU12) / 2,
        "sadness": (aus.AU1 + aus.AU4 + aus.AU15) / 3,
        "surprise": (aus.AU1 + aus.AU2 + aus.AU5 + aus.AU26) / 4,
        "fear": (aus.AU1 + aus.AU2 + aus.AU4 + aus.AU5 + aus.AU20) / 5,
        "anger": (aus.AU4 + aus.AU5 + aus.AU6 + aus.AU9) / 4,
        "disgust": (aus.AU9 + aus.AU15) / 2
    }
    
    # Get max score
    best_emotion = max(scores, key=scores.get)
    confidence = scores[best_emotion]
    
    threshold = 0.3
    if confidence < threshold:
        return "neutral", 1.0 - confidence
        
    return best_emotion, confidence
