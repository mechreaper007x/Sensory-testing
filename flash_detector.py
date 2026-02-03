"""
Flash Detector - Temporal Analysis for Micro-Expression Detection

Tracks Action Unit values over time to detect sudden spikes that indicate
micro-expressions (involuntary emotional leakage lasting 50-500ms).

Key concepts:
- Baseline: Rolling average of recent AU values
- Flash: Sudden deviation > 2.5 std dev from baseline
- Duration filter: Only expressions in 50-500ms range qualify
"""

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Tuple
from action_units_v2 import ActionUnits
import time

@dataclass
class FlashEvent:
    """Represents a detected micro-expression flash"""
    timestamp: float          # When flash started
    peak_deviation: float     # Max deviation from baseline
    dominant_aus: List[str]   # Which AUs fired most
    duration_ms: float        # Duration in milliseconds
    confidence: float         # 0-1 confidence score


class FlashDetector:
    """
    Detects micro-expression "flashes" by tracking AU deviations over time.
    
    Algorithm:
    1. Maintain rolling baseline of last N frames
    2. Compute standard deviation for each AU
    3. Flag when current AU exceeds baseline by K std devs
    4. Track duration - only 50-500ms qualifies as micro-expression
    """
    
    AU_NAMES = ['AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU9', 'AU12', 'AU15', 'AU20', 'AU26', 'GAZE_X', 'GAZE_Y', 'AU_FOREHEAD']
    
    def __init__(
        self,
        baseline_frames: int = 30,       # ~1 second at 30 FPS
        deviation_threshold: float = 2.0, # Std devs from baseline
        min_duration_ms: float = 50,     # Minimum flash duration
        max_duration_ms: float = 500,    # Maximum (micro only)
        cooldown_ms: float = 500         # Time between detections
    ):
        self.baseline_frames = baseline_frames
        self.deviation_threshold = deviation_threshold
        self.min_duration_ms = min_duration_ms
        self.max_duration_ms = max_duration_ms
        self.cooldown_ms = cooldown_ms
        
        # Rolling buffer for baseline computation
        self.au_buffer: deque = deque(maxlen=baseline_frames)
        
        # Flash tracking state
        self.flash_start_time: Optional[float] = None
        self.flash_start_aus: Optional[np.ndarray] = None
        self.peak_deviation: float = 0.0
        self.in_flash: bool = False
        
        # Cooldown tracking
        self.last_flash_time: float = 0.0
        
        # Statistics
        self.baseline_mean: Optional[np.ndarray] = None
        self.baseline_std: Optional[np.ndarray] = None
    
    def update_baseline(self, aus: ActionUnits) -> None:
        """Add current AU values to rolling baseline"""
        self.au_buffer.append(aus.to_array())
        
        if len(self.au_buffer) >= 10:  # Need minimum samples
            buffer_array = np.array(self.au_buffer)
            self.baseline_mean = np.mean(buffer_array, axis=0)
            self.baseline_std = np.std(buffer_array, axis=0)
            # Prevent division by zero
            self.baseline_std = np.maximum(self.baseline_std, 0.01)
    
    def detect(self, aus: ActionUnits) -> Tuple[bool, Optional[FlashEvent]]:
        """
        Process current AU values and detect flash if present.
        
        Args:
            aus: Current ActionUnits from AUEstimator
            
        Returns:
            (is_flashing, flash_event)
            - is_flashing: True if currently in a flash
            - flash_event: FlashEvent if flash just ended, None otherwise
        """
        current_time = time.time() * 1000  # ms
        current_aus = aus.to_array()
        
        # Need baseline first
        if self.baseline_mean is None:
            self.update_baseline(aus)
            return False, None
        
        # Check cooldown
        if current_time - self.last_flash_time < self.cooldown_ms:
            self.update_baseline(aus)
            return False, None
        
        # Compute z-scores (deviation in std devs)
        z_scores = np.abs(current_aus - self.baseline_mean) / self.baseline_std
        max_z = np.max(z_scores)
        
        # Detect flash start
        if not self.in_flash and max_z > self.deviation_threshold:
            self.in_flash = True
            self.flash_start_time = current_time
            self.flash_start_aus = current_aus.copy()
            self.peak_deviation = max_z
            return True, None
        
        # During flash - track peak
        if self.in_flash:
            self.peak_deviation = max(self.peak_deviation, max_z)
            
            # Check if flash ended (back to normal)
            if max_z < self.deviation_threshold * 0.7:
                duration = current_time - self.flash_start_time
                
                # Reset flash state
                self.in_flash = False
                self.last_flash_time = current_time
                self.update_baseline(aus)
                
                # Validate duration
                if self.min_duration_ms <= duration <= self.max_duration_ms:
                    # Find dominant AUs
                    deviation = np.abs(self.flash_start_aus - self.baseline_mean)
                    top_indices = np.argsort(deviation)[-3:][::-1]
                    dominant_aus = [self.AU_NAMES[i] for i in top_indices if deviation[i] > 0.05]
                    
                    # Compute confidence
                    confidence = min(1.0, self.peak_deviation / 4.0)
                    
                    event = FlashEvent(
                        timestamp=self.flash_start_time,
                        peak_deviation=self.peak_deviation,
                        dominant_aus=dominant_aus,
                        duration_ms=duration,
                        confidence=confidence
                    )
                    return False, event
                
                return False, None
            
            # Still in flash
            return True, None
        
        # Normal operation - update baseline
        self.update_baseline(aus)
        return False, None
    
    def get_current_z_scores(self, aus: ActionUnits) -> Optional[np.ndarray]:
        """Get current z-scores for visualization"""
        if self.baseline_mean is None:
            return None
        return np.abs(aus.to_array() - self.baseline_mean) / self.baseline_std
    
    def reset(self) -> None:
        """Reset detector state"""
        self.au_buffer.clear()
        self.baseline_mean = None
        self.baseline_std = None
        self.in_flash = False
        self.flash_start_time = None
        self.peak_deviation = 0.0


# Emotion mapping based on AU combinations
# Emotion mapping based on AU combinations
# Simplified and tuned to avoid false positives
EMOTION_AU_PATTERNS = {
    'happiness': {'AU6': 1.0, 'AU12': 1.0},
    'sadness': {'AU1': 1.0, 'AU4': 0.8, 'AU15': 1.0},
    'fear': {'AU1': 0.8, 'AU2': 0.8, 'AU4': 1.0, 'AU5': 0.8, 'AU20': 1.0},
    'disgust': {'AU9': 1.0, 'AU15': 0.3},
    'anger': {'AU4': 1.0, 'AU5': 0.8},
    'surprise': {'AU1': 1.0, 'AU2': 1.0, 'AU5': 0.8, 'AU26': 1.0},
    'contempt': {'AU12': 0.5, 'AU15': 0.0}, 
}


def classify_emotion_from_aus(aus: ActionUnits) -> Tuple[str, float]:
    """
    Classify emotion based on AU pattern matching.
    
    This is a heuristic classifier based on Ekman's research.
    For better accuracy, use the MobileNetV3 classifier.
    
    Returns:
        (emotion_name, confidence)
    """
    au_dict = aus.to_dict()
    
    best_emotion = 'neutral'
    best_score = 0.0
    
    for emotion, pattern in EMOTION_AU_PATTERNS.items():
        score = 0.0
        max_possible = 0.0
        
        for au_name, weight in pattern.items():
            max_possible += weight
            # Only count if AU is activated (> 0.5)
            val = au_dict.get(au_name, 0.0)
            if val > 0.5: 
                score += (val - 0.5) * 2 * weight # Scale 0.5-1.0 to 0.0-1.0
        
        normalized_score = score / max_possible if max_possible > 0 else 0
        
        if normalized_score > best_score:
            best_score = normalized_score
            best_emotion = emotion
    
    # Threshold for detection - stricter now
    if best_score < 0.55:
        return 'neutral', 0.0
    
    return best_emotion, best_score
