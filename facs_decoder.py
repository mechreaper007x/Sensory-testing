"""
FACS DECODER: Vector-Based Emotion Logic
=========================================
Determines emotion by comparing the current Action Unit (AU) Vector
against defined "Prototype Vectors" for each universal emotion.

Theory: Cosine Similarity
1. We define an 'Ideal' vector for Happiness (e.g., High AU6 + High AU12).
2. We compare the Live AU Vector to this Ideal.
3. The result is a 'Similarity Score' (0.0 to 1.0).
4. No black boxes. No training bias. Pure geometry.

Supported AUs: 1, 2, 4, 5, 6, 9, 12, 15, 20, 26
"""

import numpy as np
from typing import Dict, Tuple
from action_units_v2 import ActionUnits

class FACSDecoder:
    def __init__(self):
        # Define Prototype Vectors (The "Ideal" Face for each emotion)
        # Weights normalized roughly to 0.0-1.0 range
        self.prototypes = {
            "happiness": {
                "AU6": 1.0,  # Cheek Raiser
                "AU12": 1.0, # Lip Corner Puller
                "AU26": 0.2  # Slight jaw drop allowed
            },
            "sadness": {
                "AU1": 1.0,  # Inner Brow Raise
                "AU4": 0.8,  # Brow Lowerer
                "AU15": 1.0  # Lip Corner Depressor
            },
            "surprise": {
                "AU1": 1.0,  # Inner Brow Raise
                "AU2": 1.0,  # Outer Brow Raise
                "AU5": 1.0,  # Upper Lid Raise
                "AU26": 1.0  # Jaw Drop
            },
            "fear": {
                "AU1": 1.0,  # Inner Brow Raise
                "AU2": 1.0,  # Outer Brow Raise
                "AU4": 1.0,  # Brow Lowerer
                "AU5": 0.8,  # Upper Lid Raise
                "AU20": 1.0, # Lip Stretch
                "AU26": 0.8  # Jaw Drop
            },
            "anger": {
                "AU4": 1.3,  # PRIMARY: Brow Lowerer + Procerus Drop
                "AU5": 0.8,  # Upper Lid Raise (Stare)
                "AU7": 0.5,  # Lid Tightener (Squint) - Not tracked yet, using AU20/AU26
                "AU20": 0.3, # Lip Stretch
                "AU10": 0.2  # Slight snarl maybe?
            },
            "disgust": {
                "AU9": 1.0,  # Nose Wrinkler 
                "AU10": 1.0, # PRIMARY: Upper Lip Raise (Snarl)
                "AU15": 0.5, # Lip Corner Depressor
                "AU4": 0.3   # Brow Lowerer
            },
            "contempt": {
                "AU12": 0.5, 
                "AU15": 0.0  
            }
        }
        
        # Pre-compile vectors for speed
        self.proto_vectors = {}
        self.au_indices = [
            'AU1', 'AU2', 'AU4', 'AU5', 'AU6', 
            'AU9', 'AU10', 'AU12', 'AU15', 'AU20', 'AU26'
        ]
        
        for emotion, au_dict in self.prototypes.items():
            vec = np.zeros(len(self.au_indices))
            for i, au_name in enumerate(self.au_indices):
                vec[i] = au_dict.get(au_name, 0.0)
            
            # Normalize prototype vector
            norm = np.linalg.norm(vec)
            if norm > 0:
                self.proto_vectors[emotion] = vec / norm
            else:
                self.proto_vectors[emotion] = vec

    def decode(self, aus: ActionUnits) -> Tuple[str, float, Dict[str, float]]:
        """
        Input: ActionUnits object
        Output: (Best Emotion, Confidence, All Scores)
        """
        # 1. Convert Live AUs to Vector
        live_vec = np.array([
            aus.AU1, aus.AU2, aus.AU4, aus.AU5, aus.AU6,
            aus.AU9, aus.AU10, aus.AU12, aus.AU15, aus.AU20, aus.AU26
        ])
        
        # 2. Similarity Check
        scores = {}
        
        # Avoid div/0 if live face is completely neutral
        live_norm = np.linalg.norm(live_vec)
        if live_norm < 0.1:
            return "neutral", 1.0, {}

        live_unit = live_vec / live_norm
        
        for emotion, proto_unit in self.proto_vectors.items():
            # Cosine Similarity: A . B
            # Since both are unit vectors, Dot Product = Cosine Similarity
            similarity = np.dot(live_unit, proto_unit)
            
            # Clip negative matches (opposite expressions)
            scores[emotion] = max(0.0, similarity)
            
        # 3. Find Winner
        # Filter low scores
        valid_scores = {k: v for k, v in scores.items() if v > 0.4}
        
        if not valid_scores:
            return "neutral", 1.0 - (live_norm / 3.0), scores
            
        best_emotion = max(valid_scores, key=valid_scores.get)
        confidence = valid_scores[best_emotion]
        
        # Heuristic Corrections (Disambiguation)
        # Fear vs Surprise: High AU4 (Brow Lower) pushes Fear
        if best_emotion == 'surprise' and aus.AU4 > 0.4:
            if scores.get('fear', 0) > 0.3:
                best_emotion = 'fear'
                confidence = scores['fear']
                
        # Anger vs Disgust: High AU9 (Nose) pushes Disgust
        if best_emotion == 'anger' and aus.AU9 > 0.6:
            best_emotion = 'disgust'
        
        return best_emotion, confidence, scores
