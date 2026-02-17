# Protocol Senses v2 - Q-Learning Edition

A real-time micro-expression detection system using Q-Learning for adaptive sensitivity tuning, MediaPipe for facial landmark detection, and FACS (Facial Action Coding System) for emotion classification.

## Features

- **Micro-expression Detection**: Detects brief facial expressions using Action Unit (AU) analysis
- **Q-Learning Adaptive Sensitivity**: Automatically adjusts detection thresholds using reinforcement learning
- **Real-time Emotion Classification**: Classifies emotions using FACS decoder and optional CNN classifier
- **Behavioral Analysis**: Detects chewing, talking, and face-touching behaviors
- **Hand-Face Interaction Tracking**: Identifies scratching and touching behaviors
- **Calibration System**: Personalizes detection to individual neutral expressions

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam or camera device
- (Optional) NVIDIA GPU with CUDA for GPU acceleration

### Step 1: Clone the Repository

```bash
git clone https://github.com/mechreaper007x/Sensory-testing.git
cd Sensory-testing
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Linux/Mac:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# For GPU acceleration (if you have NVIDIA GPU):
pip install onnxruntime-gpu

# OR for CPU-only:
pip install onnxruntime

# Optional: Install additional features
pip install -r requirements-optional.txt
```

## Usage

### Basic Usage

```bash
python protocol_senses_v2.py
```

### Command-Line Options

```bash
# Use specific camera index
python protocol_senses_v2.py --camera 1

# Or set via environment variable
set CAMERA_INDEX=1  # Windows
export CAMERA_INDEX=1  # Linux/Mac
python protocol_senses_v2.py
```

### Controls

- **'q'**: Quit application
- **'r'**: Recalibrate (reset baseline)
- **'m'**: Toggle micro-expression mode
- **'d'**: Toggle debug mode
- **'p'**: Print Q-Learning policy

### Calibration

On startup, the system will:
1. Display a calibration screen
2. Collect baseline AU values for 3 seconds
3. Require you to keep your face neutral
4. Use this baseline for personalized detection

## Configuration

Key configuration constants can be modified in `protocol_senses_v2.py`:

```python
# Camera settings
DEFAULT_CAMERA_INDEX = 0

# Flash Detection thresholds
BASELINE_FRAMES = 30
DEFAULT_DEVIATION_THRESHOLD = 2.0
MIN_FLASH_DURATION_MS = 50
MAX_FLASH_DURATION_MS = 500

# Hand-Face Interaction
HAND_FACE_THRESHOLD = 0.15
NOSE_SCRATCH_THRESHOLD = 0.08
CHIN_SCRATCH_THRESHOLD = 0.06
```

## Project Structure

```
Sensory-testing/
├── protocol_senses_v2.py      # Main application (Q-Learning edition)
├── action_units_v2.py          # Action Unit estimation
├── flash_detector.py           # Micro-expression detection
├── emotion_classifier.py       # CNN-based emotion classification
├── facs_decoder.py             # FACS-based emotion decoding
├── rl_agent.py                 # Q-Learning agent
├── camera_utils.py              # Threaded camera capture
├── benchmark_rl.py             # Q-Learning vs Gradient Descent comparison
├── protocol_feat.py            # Py-Feat integration (optional)
├── protocol_feat_deep.py      # DeepFace integration (optional)
├── requirements.txt            # Core dependencies
└── README.md                   # This file
```

## How It Works

1. **Face Detection**: MediaPipe detects facial landmarks in real-time
2. **Action Unit Estimation**: Calculates FACS Action Units from landmarks
3. **Baseline Calibration**: Establishes personal neutral expression baseline
4. **Micro-expression Detection**: Identifies deviations from baseline using statistical analysis
5. **Emotion Classification**: 
   - Primary: FACS decoder (vector-based, transparent)
   - Optional: CNN classifier (high confidence only)
6. **Q-Learning Adaptation**: Adjusts detection sensitivity based on classification confidence
7. **Behavioral Analysis**: Tracks chewing, scratching, and face-touching

## Troubleshooting

### Camera Issues

- **"Could not open camera"**: 
  - Check camera is connected
  - Ensure no other application is using the camera
  - Try different camera index: `python protocol_senses_v2.py --camera 1`

### Model Download Issues

- **Model download fails**:
  - Check internet connection
  - The system will retry automatically (3 attempts)
  - Models are downloaded to the current directory

### GPU Issues

- **ONNX Runtime GPU errors**:
  - System automatically falls back to CPU
  - Install `onnxruntime-gpu` for GPU acceleration
  - Requires CUDA-compatible GPU

### Performance Issues

- **Low FPS**:
  - Reduce camera resolution in `camera_utils.py`
  - Disable CNN classifier: Set `USE_CNN_CLASSIFIER = False`
  - Use CPU-only ONNX runtime if GPU is slow

## Advanced Usage

### Using Different Protocols

```bash
# Py-Feat protocol (requires py-feat)
python protocol_feat.py

# DeepFace protocol (requires deepface)
python protocol_feat_deep.py

# Benchmark Q-Learning vs Gradient Descent
python benchmark_rl.py
```

### Q-Learning Brain

The Q-Learning agent saves its learned state to `brain_qlearning.json`. This file:
- Persists between sessions
- Contains learned threshold policy
- Can be deleted to reset learning

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Acknowledgments

- MediaPipe for facial landmark detection
- HSEmotion for AffectNet-trained emotion classifier
- FACS (Facial Action Coding System) for emotion encoding

## Citation

If you use this code in your research, please cite:

[Add citation information]
