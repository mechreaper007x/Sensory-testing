# Issues Found in Sensory-testing Repository

## Critical Issues

### 1. **Logic Error: Incorrect Conditional Structure (Line 331 in `protocol_senses_v2.py`)**
   - **Location**: `protocol_senses_v2.py:331`
   - **Problem**: There's an `elif` statement that matches `if DEBUG_MODE:` on line 325, but logically it doesn't make sense. The code updates `last_emotion` only when DEBUG_MODE is False AND `last_emotion == "neutral"`. This means the emotion update logic is tied to debug mode, which is likely unintended.
   - **Code**:
     ```python
     if DEBUG_MODE:
         print(f"\n[Q-LEARN] Flash detected!")
         print(f"  Emotion: {last_emotion} ({detected_confidence:.0%})")
         print(f"  New threshold: {new_threshold:.2f}")
         print(f"  Exploration rate: {rl_agent.epsilon:.3f}")
     
     elif last_emotion == "neutral" or last_emotion == "":  # Logic error: tied to DEBUG_MODE
         last_emotion = au_emotion
         detected_confidence = au_conf
     ```
   - **Impact**: Emotion detection logic only works when DEBUG_MODE is False, which is likely a bug. The emotion should be updated regardless of debug mode.
   - **Fix**: Change `elif` to a separate `if` statement, or restructure the logic so emotion updates happen independently of DEBUG_MODE.

### 2. **Missing Dependencies - No requirements.txt File**
   - **Problem**: The repository lacks a `requirements.txt` file listing all dependencies.
   - **Missing Dependencies Identified**:
     - `mediapipe` (required for face/hand landmark detection)
     - `opencv-python` (cv2)
     - `numpy`
     - `onnxruntime` or `onnxruntime-gpu` (for emotion classifier)
     - `matplotlib` (for benchmark_rl.py)
   - **Impact**: Users cannot easily install dependencies, leading to `ModuleNotFoundError` exceptions.
   - **Fix**: Create a `requirements.txt` file with all dependencies.

### 3. **Variable Scope Issue: `all_scores` Stale Data**
   - **Location**: `protocol_senses_v2.py:302` and `protocol_senses_v2.py:420`
   - **Problem**: `all_scores` is only assigned when `is_flashing` is True (line 302), but it's accessed outside that block (line 420). While `locals()` check prevents `NameError`, `all_scores` may contain stale data from a previous iteration when `is_flashing` was True but is now False.
   - **Code**:
     ```python
     # Inside if is_flashing: block (line 302)
     au_emotion, au_conf, all_scores = facs_decoder.decode(display_aus)
     
     # Later, outside the block (line 420)
     if 'all_scores' in locals() and all_scores:  # Uses stale data if is_flashing is now False
     ```
   - **Impact**: UI may display emotion scores from a previous flash event, causing confusion.
   - **Fix**: Initialize `all_scores = {}` at the start of each loop iteration, or only display it when `is_flashing` is True.

## Logic Errors

### 4. **Inconsistent AUEstimator Initialization on Recalibration**
   - **Location**: `protocol_senses_v2.py:166` vs `protocol_senses_v2.py:455`
   - **Problem**: When initializing `AUEstimator` at the start, `smoothing_factor=0.7` is used, but when recalibrating (line 455), a new `AUEstimator()` is created without the smoothing factor.
   - **Code**:
     ```python
     # Initial creation (line 166)
     au_estimator = AUEstimator(smoothing_factor=0.7)
     
     # Recalibration (line 455)
     au_estimator = AUEstimator()  # Missing smoothing_factor!
     ```
   - **Impact**: Recalibration will use default smoothing factor, potentially causing inconsistent behavior.
   - **Fix**: Use `AUEstimator(smoothing_factor=0.7)` on line 455.

### 5. **Potential Race Condition in ThreadedCamera**
   - **Location**: `camera_utils.py:39-43`
   - **Problem**: The `update()` method modifies `self.frame` without proper locking, and `read()` accesses it without synchronization. This could lead to reading partially updated frames.
   - **Code**:
     ```python
     def update(self):
         while True:
             ...
             (grabbed, frame) = self.stream.read()
             if grabbed:
                 self.frame = frame  # No lock!
     ```
   - **Impact**: Potential race conditions leading to corrupted frames or crashes.
   - **Fix**: Add threading locks or use thread-safe data structures.

### 6. **Unused Import: `classify_emotion_from_aus`**
   - **Location**: `protocol_senses_v2.py:29`
   - **Problem**: `classify_emotion_from_aus` is imported from `flash_detector` but never used in the code.
   - **Impact**: Code clutter, but no functional impact.
   - **Fix**: Remove unused import.

## Code Quality Issues

### 7. **Hardcoded Camera Index**
   - **Location**: `protocol_senses_v2.py:37`
   - **Problem**: `DROIDCAM_INDEX = 0` is hardcoded, making it difficult to use different cameras without modifying code.
   - **Impact**: Poor flexibility for users with multiple cameras or different setups.
   - **Fix**: Make it configurable via command-line argument or environment variable.

### 8. **Missing Error Handling for Model Downloads**
   - **Location**: `protocol_senses_v2.py:51-56`
   - **Problem**: `download_model()` function doesn't handle network errors, timeouts, or partial downloads gracefully.
   - **Impact**: Program may crash or hang if download fails.
   - **Fix**: Add try-except blocks and retry logic.

### 9. **Missing Error Handling for ONNX Runtime Initialization**
   - **Location**: `emotion_classifier.py:71-77`
   - **Problem**: If ONNX Runtime fails to initialize (e.g., CUDA unavailable), the error is printed but the program continues. The `is_available()` check may not catch all failure modes.
   - **Impact**: Silent failures or unexpected behavior.
   - **Fix**: Add more robust error handling and validation.

### 10. **Potential Division by Zero in Softmax**
   - **Location**: `emotion_classifier.py:118`
   - **Problem**: While unlikely, if all scores are extremely negative, `np.sum(np.exp(scores))` could theoretically underflow to 0, causing division by zero.
   - **Code**:
     ```python
     probs = np.exp(scores) / np.sum(np.exp(scores))  # Potential div/0
     ```
   - **Impact**: Potential `ZeroDivisionError` in edge cases.
   - **Fix**: Add a small epsilon or use `scipy.special.softmax` which handles edge cases.

### 11. **Inconsistent Emotion Label Casing**
   - **Location**: Multiple files
   - **Problem**: Emotion labels are sometimes lowercase (`emotion.lower()` in `emotion_classifier.py:129`), sometimes capitalized (FACS decoder uses lowercase), which could cause comparison issues.
   - **Impact**: Potential string comparison bugs.
   - **Fix**: Standardize emotion label casing throughout the codebase.

### 12. **Missing Documentation/README**
   - **Problem**: No README.md file explaining:
     - What the project does
     - How to install dependencies
     - How to run the code
     - What each script does
   - **Impact**: Difficult for new users to understand and use the project.
   - **Fix**: Create comprehensive README.md.

### 13. **No Unit Tests**
   - **Problem**: No test files or test framework setup.
   - **Impact**: Difficult to verify correctness and prevent regressions.
   - **Fix**: Add unit tests for core modules (AUEstimator, FlashDetector, QLearningAgent, etc.).

### 14. **Magic Numbers Throughout Code**
   - **Examples**:
     - `baseline_frames=30` (line 168)
     - `deviation_threshold=2.0` (line 169)
     - `min_duration_ms=50` (line 170)
     - `HAND_FACE_THRESHOLD = 0.15` (line 38)
   - **Impact**: Hard to understand and modify thresholds.
   - **Fix**: Define constants with descriptive names and document their purpose.

### 15. **Potential Memory Leak in ThreadPoolExecutor**
   - **Location**: `protocol_senses_v2.py:200`
   - **Problem**: `ThreadPoolExecutor` is created but never explicitly shut down. While it's a daemon thread, best practice is to use context manager or call `shutdown()`.
   - **Code**:
     ```python
     executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
     # Never calls executor.shutdown()
     ```
   - **Impact**: Potential resource leak (minor).
   - **Fix**: Use context manager: `with ThreadPoolExecutor(...) as executor:` or call `executor.shutdown()` before exit.

## Runtime Issues

### 16. **No Validation for Model File Existence After Download**
   - **Location**: `protocol_senses_v2.py:162-163`
   - **Problem**: After calling `download_model()`, the code doesn't verify that the file actually exists before trying to use it.
   - **Impact**: Program may crash with `FileNotFoundError` if download failed silently.
   - **Fix**: Add file existence check after download.

### 17. **Missing Validation for Landmark Indices**
   - **Location**: `action_units_v2.py` (multiple locations)
   - **Problem**: The code accesses MediaPipe landmark indices (e.g., `landmarks[1]`, `landmarks[152]`) without checking if the landmarks array has enough elements.
   - **Impact**: Potential `IndexError` if MediaPipe returns fewer landmarks than expected.
   - **Fix**: Add bounds checking or use MediaPipe's landmark count.

### 18. **No Graceful Degradation if Camera Fails**
   - **Location**: `protocol_senses_v2.py:222-224`
   - **Problem**: If camera fails to open, the program just returns without cleanup or informative error message.
   - **Impact**: Poor user experience.
   - **Fix**: Add better error messages and cleanup.

## Summary

**Total Issues Found: 18**
- **Critical**: 3 (Syntax error, missing dependencies, variable scope)
- **Logic Errors**: 3
- **Code Quality**: 9
- **Runtime Issues**: 3

**Priority Fixes:**
1. Fix syntax error on line 331 (CRITICAL - prevents code from running)
2. Create requirements.txt (CRITICAL - prevents installation)
3. Fix variable scope issue with `all_scores` (HIGH)
4. Fix AUEstimator initialization inconsistency (MEDIUM)
5. Add error handling for model downloads (MEDIUM)
6. Create README.md (MEDIUM - improves usability)
