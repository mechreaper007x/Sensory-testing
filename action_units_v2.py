"""
Action Unit (AU) Estimation v2 - Strict Geometric Rigidity
Corrects for Head Pose (Pitch/Yaw) and Scale variations.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

# --- STRICT CONSTANTS ---
# Rigid Skull Anchors (Do not move with expression)
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
NOSE_TIP = 1
NOSE_ROOT = 168  # Between eyes, rigid

# Dynamic Features
LEFT_BROW_INNER = 107
RIGHT_BROW_INNER = 336
LEFT_BROW_OUTER = 70
RIGHT_BROW_OUTER = 300

LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

LEFT_CHEEK = 117
RIGHT_CHEEK = 346

NOSE_LEFT = 129
NOSE_RIGHT = 358

MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
JAW_BOTTOM = 152

@dataclass
class ActionUnits:
    AU1: float = 0.0   # Inner Brow Raise
    AU2: float = 0.0   # Outer Brow Raise
    AU4: float = 0.0   # Brow Lowerer (frown)
    AU5: float = 0.0   # Upper Lid Raise
    AU6: float = 0.0   # Cheek Raise (smile)
    AU9: float = 0.0   # Nose Wrinkle
    AU12: float = 0.0  # Lip Corner Pull (smile)
    AU15: float = 0.0  # Lip Corner Depress (frown)
    AU20: float = 0.0  # Lip Stretch
    AU26: float = 0.0  # Jaw Drop
    
    def to_array(self) -> np.ndarray:
        return np.array([self.AU1, self.AU2, self.AU4, self.AU5, self.AU6,
                         self.AU9, self.AU12, self.AU15, self.AU20, self.AU26])

class StrictAUEstimator:
    def __init__(self, smoothing_factor: float = 0.7):
        self.baseline = None
        self.smoothing = smoothing_factor
        self.prev_landmarks = None
        
        # Calibration stats
        self.calibrated = False
        self.baseline_samples = []
    
    def _get_vec(self, landmarks, idx) -> np.ndarray:
        """Extract 3D vector for a landmark"""
        lm = landmarks[idx]
        return np.array([lm.x, lm.y, lm.z])

    def _rotate_points(self, points: np.ndarray, pitch: float, yaw: float) -> np.ndarray:
        """
        Mathematically un-rotate points to frontal view.
        Strict Rotation Matrix application.
        """
        # Rotation Matrix for Pitch (X-axis)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(-pitch), -np.sin(-pitch)],
            [0, np.sin(-pitch), np.cos(-pitch)]
        ])
        # Rotation Matrix for Yaw (Y-axis)
        Ry = np.array([
            [np.cos(-yaw), 0, np.sin(-yaw)],
            [0, 1, 0],
            [-np.sin(-yaw), 0, np.cos(-yaw)]
        ])
        
        # Apply R = Ry * Rx
        R = np.dot(Ry, Rx)
        return np.dot(points, R.T)

    def compute_pose(self, landmarks) -> Tuple[float, float]:
        """Estimate crude Pitch and Yaw for correction"""
        nose = self._get_vec(landmarks, NOSE_TIP)
        left_eye = self._get_vec(landmarks, LEFT_EYE_OUTER)
        right_eye = self._get_vec(landmarks, RIGHT_EYE_OUTER)
        
        # Simple geometric approximation
        # Yaw: Asymmetry of nose relative to eyes
        yaw = (nose[0] - (left_eye[0] + right_eye[0]) / 2) * 3.0
        
        # Pitch: Angle between nose vector and vertical axis
        pitch = (nose[1] - (left_eye[1] + right_eye[1]) / 2) * 3.0
        
        return pitch, yaw

    def compute(self, landmarks) -> ActionUnits:
        # 1. Smooth Landmarks (Exponential Moving Average)
        # For simplicity, we assume 'landmarks' is raw frame data
        # In a real pipeline, we'd store prev_landmarks and lerp
        
        # 2. Estimate Pose
        pitch, yaw = self.compute_pose(landmarks)
        
        # 3. Get Key Points and Un-rotate them
        # We collect all critical points into a matrix for batch rotation
        indices = [
            # 0-1
            LEFT_EYE_OUTER, RIGHT_EYE_OUTER, 
            # 2-3
            NOSE_ROOT, NOSE_TIP,
            # 4-7
            LEFT_BROW_INNER, RIGHT_BROW_INNER, LEFT_BROW_OUTER, RIGHT_BROW_OUTER,
            # 8-11
            LEFT_EYE_TOP, LEFT_EYE_BOTTOM, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM,
            # 12-13
            LEFT_CHEEK, RIGHT_CHEEK,
            # 14-15
            NOSE_LEFT, NOSE_RIGHT,
            # 16-19
            MOUTH_LEFT, MOUTH_RIGHT, MOUTH_TOP, MOUTH_BOTTOM,
            # 20
            JAW_BOTTOM
        ]
        
        raw_points = np.array([self._get_vec(landmarks, i) for i in indices])
        
        # Center points around Nose Root before rotation
        center = raw_points[2] # NOSE_ROOT
        centered_points = raw_points - center
        
        # Apply correction
        corrected_points = self._rotate_points(centered_points, pitch, yaw)
        
        # Map back to a dictionary for easy access using original indices
        pt = {idx: corrected_points[i] for i, idx in enumerate(indices)}
        
        # 4. Strict Normalization (Inter-Ocular Width)
        # This is your NEW '1.0' scale unit. It never changes.
        skull_scale = np.linalg.norm(pt[LEFT_EYE_OUTER] - pt[RIGHT_EYE_OUTER])
        if skull_scale < 0.001: return ActionUnits()

        aus = ActionUnits()
        
        # === AU1: Inner Brow Raise ===
        # Measure vertical distance from Inner Brow to Eye Corner (Outer) ? No, Inner Eye is better but moving.
        # User defined: Inner Brow - Eye Outer.
        # Y is positive DOWN. 
        # Brow (y) < Eye (y). So Brow - Eye is negative.
        # Higher Brow = More Negative.
        l_brow_h = pt[LEFT_BROW_INNER][1] - pt[LEFT_EYE_OUTER][1] 
        r_brow_h = pt[RIGHT_BROW_INNER][1] - pt[RIGHT_EYE_OUTER][1]
        
        # We want Positive value for Raise.
        # Default (Rest): -0.X. Raise: -0.X - Delta (More negative? No, Higher is smaller Y)
        # Wait, Y=0 is top. Y=1 is bottom.
        # Brow Y=0.3, Eye Y=0.4. Diff = -0.1.
        # Raise: Brow Y=0.2. Diff = -0.2.
        # So Raise makes it MORE NEGATIVE.
        # Let's invert signs to make math intuitive.
        # Height = Eye - Brow (Positive distance).
        # Raise = Larger Distance.
        
        l_brow_dist = pt[LEFT_EYE_OUTER][1] - pt[LEFT_BROW_INNER][1]
        r_brow_dist = pt[RIGHT_EYE_OUTER][1] - pt[RIGHT_BROW_INNER][1]
        
        aus.AU1 = ((l_brow_dist + r_brow_dist) / 2.0) / skull_scale
        
        # === AU2: Outer Brow Raise ===
        l_obrow_dist = pt[LEFT_EYE_OUTER][1] - pt[LEFT_BROW_OUTER][1]  # Using Eye Outer as anchor
        r_obrow_dist = pt[RIGHT_EYE_OUTER][1] - pt[RIGHT_BROW_OUTER][1]
        aus.AU2 = ((l_obrow_dist + r_obrow_dist) / 2.0) / skull_scale
        
        # === AU4: Brow Lowerer (Frown) ===
        # Distance between inner brows. Rigid Skull Normalization.
        brow_width = np.linalg.norm(pt[LEFT_BROW_INNER] - pt[RIGHT_BROW_INNER])
        # Smaller width = Frown.
        # We want to map standard width (~X) to 0, and Squeezed (~X-d) to 1.
        # We'll normalize raw ratio here; compute_relative handles the 0-1 mapping.
        aus.AU4 = brow_width / skull_scale
        
        # === AU5: Upper Lid Raise ===
        # Eye opening height.
        l_eye_open = np.linalg.norm(pt[LEFT_EYE_TOP] - pt[LEFT_EYE_BOTTOM])
        r_eye_open = np.linalg.norm(pt[RIGHT_EYE_TOP] - pt[RIGHT_EYE_BOTTOM])
        aus.AU5 = ((l_eye_open + r_eye_open) / 2.0) / skull_scale
        
        # === AU6: Cheek Raise ===
        # Cheek vs Eye Bottom.
        # Y dist: Eye Bottom (y) - Cheek (y).
        # Cheek is below eye. Eye Y < Cheek Y. Diff is Negative.
        # Raise: Cheek y decreases. Diff becomes less negative (approaches 0).
        l_cheek_h = pt[LEFT_CHEEK][1] - pt[LEFT_EYE_BOTTOM][1]
        r_cheek_h = pt[RIGHT_CHEEK][1] - pt[RIGHT_EYE_BOTTOM][1]
        aus.AU6 = -((l_cheek_h + r_cheek_h) / 2.0) / skull_scale
        
        # === AU9: Nose Wrinkle ===
        # Nose Width (scrunching narrows it? or raises wings?)
        # Let's use distance between nose wings.
        nose_w = np.linalg.norm(pt[NOSE_LEFT] - pt[NOSE_RIGHT])
        aus.AU9 = nose_w / skull_scale
        
        # === AU12: Lip Corner Pull (Smile) ===
        # Mouth Corner Y relative to Mouth Top Y.
        # Smile: Corner Y < Top Y (Corners go up).
        # Reference: Mouth Top is dynamic? Yes.
        # Better anchor: Nose Root?
        # Let's use Nose Root to Mouth Corner Y distance.
        # Smile = Smaller distance.
        l_smile_dist = pt[MOUTH_LEFT][1] - pt[NOSE_ROOT][1]
        r_smile_dist = pt[MOUTH_RIGHT][1] - pt[NOSE_ROOT][1]
        # We want Raise (Smile) to increase value? 
        # No, compute_relative handles direction. Just store the metric.
        aus.AU12 = ((l_smile_dist + r_smile_dist) / 2.0) / skull_scale
        
        # === AU15: Lip Corner Depressor (Frown) ===
        # Same metric as AU12, just direction differs.
        aus.AU15 = aus.AU12 # Will differentiate in relative step
        
        # === AU20: Lip Stretch ===
        # Mouth Width
        mouth_w = np.linalg.norm(pt[MOUTH_LEFT] - pt[MOUTH_RIGHT])
        aus.AU20 = mouth_w / skull_scale
        
        # === AU26: Jaw Drop ===
        # Distance from Nose Root (Rigid) to Jaw Bottom (Dynamic)
        jaw_dist = np.linalg.norm(pt[JAW_BOTTOM] - pt[NOSE_ROOT])
        aus.AU26 = (jaw_dist / skull_scale)
        
        return aus

    def add_calibration_sample(self, landmarks) -> None:
        """Add a sample during calibration phase"""
        aus = self.compute(landmarks)
        self.baseline_samples.append(aus)
    
    def finalize_calibration(self) -> ActionUnits:
        """Finalize calibration by averaging samples."""
        if not self.baseline_samples:
            self.baseline = ActionUnits()
        else:
            arrays = [s.to_array() for s in self.baseline_samples]
            avg = np.mean(arrays, axis=0)
            self.baseline = ActionUnits(
                AU1=avg[0], AU2=avg[1], AU4=avg[2], AU5=avg[3], AU6=avg[4],
                AU9=avg[5], AU12=avg[6], AU15=avg[7], AU20=avg[8], AU26=avg[9]
            )
        
        self.calibrated = True
        self.baseline_samples = []
        return self.baseline

    def compute_relative(self, landmarks) -> ActionUnits:
        """
        Returns deviation from YOUR neutral face.
        Maps raw metric deviation to 0.0-1.0 intensity.
        """
        current = self.compute(landmarks)
        
        if self.baseline is None or not self.calibrated:
            return current 
            
        rel = ActionUnits()
        b = self.baseline
        
        # Scaling factors (Sensitivity)
        # Based on how much the metric changes for a full expression relative to skull width
        
        def norm(val, sensitivity=0.1):
            return max(0.0, min(1.0, val / sensitivity))

        # AU1: Inner Brow Raise (Higher Distance = Raise)
        rel.AU1 = norm(current.AU1 - b.AU1, 0.05)
        
        # AU2: Outer Brow Raise
        rel.AU2 = norm(current.AU2 - b.AU2, 0.05)
        
        # AU4: Frown (Smaller Width = Frown) -> Baseline > Current
        rel.AU4 = norm(b.AU4 - current.AU4, 0.04)
        
        # AU5: Lid Raise (Larger Open = Raise)
        rel.AU5 = norm(current.AU5 - b.AU5, 0.02)
        
        # AU6: Cheek Raise (Higher Cheek = Less Negative Y Diff? My math was -((...)). )
        # I did -((cheek - eye)). Cheek > Eye. Diff is positive.
        # Wait, Y increases down. Cheek > Eye. Cheek - Eye is Positive.
        # Raise: Cheek moves UP (smaller Y). Cheek - Eye becomes SMALLER.
        # Negation: -Small > -Big. So value INCREASES.
        rel.AU6 = norm(current.AU6 - b.AU6, 0.03)

        # AU9: Nose Wrinkle (Smaller Width?)
        # Wrinkle pulls wings up/together. Width decreases? Or Scrunch?
        # Usually vertical scrunch.
        # Let's assume Width decreases or vertical distance decreases.
        # For now, use deviation magnitude.
        rel.AU9 = norm(abs(current.AU9 - b.AU9), 0.03)
        
        # AU12: Smile (Corners UP -> Smaller Y dist to Nose Root)
        # Baseline Dist > Current Dist
        rel.AU12 = norm(b.AU12 - current.AU12, 0.06)
        
        # AU15: Sad (Corners DOWN -> Larger Y dist to Nose Root)
        # Current Dist > Baseline Dist
        rel.AU15 = norm(current.AU15 - b.AU15, 0.04)
        
        # AU20: Lip Stretch (Wider Mouth)
        rel.AU20 = norm(current.AU20 - b.AU20, 0.10)
        
        # AU26: Jaw Drop (Larger Dist)
        rel.AU26 = norm(current.AU26 - b.AU26, 0.08)
        
        return rel
