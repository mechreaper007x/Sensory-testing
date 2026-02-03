"""
Emotion Classifier using HSEmotion (EfficientNet-B0)
Trained on AffectNet (~450k images) for robust, unbiased detection.

Optimized for:
- GTX 1650 (CUDA) - <10ms per inference
- 224x224 RGB input
- 8 Classes: Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise
"""

import numpy as np
import cv2
import os
import urllib.request
from typing import Optional, Dict
from dataclasses import dataclass

# Try to import ONNX Runtime with GPU support
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnxruntime not installed. Emotion classifier disabled.")
    print("Install with: pip install onnxruntime-gpu")


@dataclass
class EmotionResult:
    """Result from emotion classification"""
    emotion: str
    confidence: float
    all_scores: Dict[str, float]


class EmotionClassifier:
    """
    Emotion classifier using HSEmotion (EfficientNet-B0) trained on AffectNet.
    
    References:
    - https://github.com/av-savchenko/face-emotion-recognition
    - Model: enet_b0_8_best_vgaf.onnx (Acquires ~8 classes)
    """
    
    # Direct download link for the AffectNet-trained EfficientNet-B0 model
    MODEL_URL = "https://github.com/av-savchenko/hsemotion-onnx/raw/main/demo/enet_b0_8_best_vgaf.onnx"
    MODEL_PATH = "enet_b0_8_best_vgaf.onnx"
    
    # AffectNet labels (8 classes)
    EMOTIONS = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
    
    def __init__(self, use_gpu: bool = True):
        self.session = None
        self.input_name = None
        self.use_gpu = use_gpu
        
        if not ONNX_AVAILABLE:
            return
            
        # Check/Download model
        if not os.path.exists(self.MODEL_PATH):
            print(f"Downloading AffectNet model ({self.MODEL_PATH})...")
            try:
                urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)
                print("Download complete.")
            except Exception as e:
                print(f"Error downloading model: {e}")
                return

        # Initialize ONNX Runtime
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
            self.session = ort.InferenceSession(self.MODEL_PATH, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            print(f"Emotion Classifier loaded (AffectNet/EfficientNet). Device: {ort.get_device()}")
        except Exception as e:
            print(f"Failed to load Emotion Classifier: {e}")

    def is_available(self) -> bool:
        return self.session is not None

    def preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for EfficientNet-B0.
        Expected: 224x224 RGB, normalized with ImageNet mean/std.
        """
        # Resize to 224x224
        img = cv2.resize(face_image, (224, 224))
        
        # Convert to float32 and scale to 0-1
        img = img.astype(np.float32) / 255.0
        
        # Normalize (ImageNet mean/std)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        
        # Transpose to (C, H, W) -> (1, 3, 224, 224)
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        
        return img

    def classify(self, face_image: np.ndarray) -> Optional[EmotionResult]:
        """Classify emotion from face crop"""
        if not self.session:
            return None
            
        try:
            # Preprocess
            input_tensor = self.preprocess(face_image)
            
            # Inference
            outputs = self.session.run(None, {self.input_name: input_tensor})
            scores = outputs[0][0]  # Raw logits or probabilities
            
            # Softmax to get probabilities if model returns logits
            probs = np.exp(scores) / np.sum(np.exp(scores))
            
            # Get best class
            best_idx = np.argmax(probs)
            confidence = probs[best_idx]
            emotion = self.EMOTIONS[best_idx]
            
            # Map simplified scores
            score_dict = {self.EMOTIONS[i]: float(probs[i]) for i in range(len(self.EMOTIONS))}
            
            return EmotionResult(
                emotion=emotion.lower(),
                confidence=float(confidence),
                all_scores=score_dict
            )
            
        except Exception as e:
            print(f"Inference error: {e}")
            return None


def crop_face_from_landmarks(frame: np.ndarray, landmarks, padding: float = 0.2) -> Optional[np.ndarray]:
    """
    Crop face region from frame using MediaPipe landmarks.
    
    Args:
        frame: Original BGR frame
        landmarks: MediaPipe face landmarks
        padding: Extra padding around face (0.2 = 20%)
        
    Returns:
        Cropped face image or None if invalid
    """
    h, w = frame.shape[:2]
    
    # Get bounding box from landmarks
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    
    # Add padding
    width = x_max - x_min
    height = y_max - y_min
    
    x_min = max(0, x_min - width * padding)
    x_max = min(1, x_max + width * padding)
    y_min = max(0, y_min - height * padding)
    y_max = min(1, y_max + height * padding)
    
    # Convert to pixel coordinates
    x1, x2 = int(x_min * w), int(x_max * w)
    y1, y2 = int(y_min * h), int(y_max * h)
    
    # Validate
    if x2 <= x1 or y2 <= y1:
        return None
    
    return frame[y1:y2, x1:x2]
