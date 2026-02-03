"""
Reinforcement Learning Agent for Sensitivity Tuning
"The Reactor-Discriminator Loop"

This agent learns the optimal sensitivity threshold for the user's face
by observing the feedback from the Emotion Classifier (Discriminator).

Logic:
- If Flash Detector triggers AND Emotion Classifier is confident:
  -> Reward = Positive
  -> Action: Increase sensitivity (Lower threshold) to catch simpler cues

- If Flash Detector triggers BUT Emotion Classifier is unsure/neutral:
  -> Reward = Negative
  -> Action: Decrease sensitivity (Raise threshold) to reduce noise
"""

import numpy as np

class SensitivityAgent:
    def __init__(
        self, 
        initial_threshold: float = 2.0, 
        min_threshold: float = 1.5,
        max_threshold: float = 4.0,
        learning_rate: float = 0.05
    ):
        self.threshold = initial_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.learning_rate = learning_rate
        
        # Stats
        self.rewards = []
        self.history = []
    
    def get_threshold(self) -> float:
        """Get current optimal threshold"""
        return self.threshold
    
    def update(self, classifier_result, is_flash: bool) -> float:
        """
        Update policy based on feedback.
        
        Args:
            classifier_result: EmotionResult object (or tuple)
            is_flash: Whether a flash was detected
            
        Returns:
            New threshold value
        """
        if not is_flash:
            # Decay towards stability if nothing is happening for a long time?
            # For now, do nothing.
            return self.threshold
            
        reward = 0.0
        
        # Calculate Reward
        # ----------------
        if classifier_result:
            if hasattr(classifier_result, 'confidence'):
                # New AffectNet model
                confidence = classifier_result.confidence
                emotion = classifier_result.emotion
            else:
                # Fallback tuple (emotion, score)
                emotion, confidence = classifier_result
            
            if emotion == 'neutral':
                # False Positive (Phantom Flash)
                # We saw a flash, but it turned out to be nothing.
                # Penalty proportional to how sure we were it was nothing.
                reward = -1.0
            elif confidence > 0.6:
                # True Positive (Strong Signal)
                # We saw a flash and it was a real emotion.
                # Reward: +1.0
                reward = 1.0
            else:
                # Weak Signal (Unsure)
                # Penalty for uncertainty
                reward = -0.5
        else:
            # No result/classifier failure
            reward = 0.0
            
        # Update Policy (Gradient Ascent)
        # -------------------------------
        # If Reward > 0 (Good): Decrease threshold (More Sensitive)
        # If Reward < 0 (Bad): Increase threshold (Less Sensitive)
        # Formula: new = old - (lr * reward)
        
        step = self.learning_rate * reward
        self.threshold -= step
        
        # Clip to safe bounds
        self.threshold = np.clip(self.threshold, self.min_threshold, self.max_threshold)
        
        # Log
        self.rewards.append(reward)
        self.history.append(self.threshold)
        
        return self.threshold

    def get_status(self) -> str:
        """Get status string for display"""
        trend = "STABLE"
        if len(self.history) > 5:
            recent = self.history[-5:]
            if recent[-1] > recent[0]: trend = "DECR SENS" # Threshold rising
            if recent[-1] < recent[0]: trend = "INCR SENS" # Threshold falling
            
        return f"SENS: {4.5 - self.threshold:.1f} | {trend}"

    def decay(self) -> float:
        """
        Slowly decay threshold towards minimum (increase sensitivity)
        when no flashes are detected.
        Prevents getting stuck in "insensitivity trap".
        """
        decay_rate = 0.001  # Very slow drift
        self.threshold -= decay_rate
        self.threshold = max(self.threshold, self.min_threshold)
        return self.threshold
