# Emotion & Sign Language Detection (WebRTC + OpenCV)

Real-time **facial emotion recognition** and **sign language action recognition** in the browser using **Streamlit + WebRTC**, **TensorFlow/Keras**, **OpenCV**, and **MediaPipe**.  
Works with your webcam and includes optional training utilities.

https://github.com/<your-username>/<your-repo> (replace with your link)

---

## ✨ Features

- 🧠 **Emotion detection** (7 classes): `angry, disgust, fear, happy, neutral, sad, surprise`
- 🤟 **Sign language detection** using MediaPipe Holistic keypoints (sample actions included)
- 🌐 **Runs in the browser** via Streamlit WebRTC (no separate client needed)
- 📦 Fallbacks for face detector path & utils (works out of the box)
- 🏋️ **Training pipeline** for emotion model (optional)

---

## 📁 Project Structure

```
FACE-RECOGNITION/
├── app.py                            # Streamlit WebRTC app (Emotion + Sign tabs)
├── webcam.py                         # Simple OpenCV webcam runner (emotion)
├── video.py                          # (optional) another webcam script
├── model.py                          # Emotion prediction helper
├── utils.py                          # Preprocessing + MediaPipe helpers
├── emotion_detection_module.py       # (optional) helpers
├── human-emotion-calssification.py   # (typo preserved) training playground
├── sign_language_detection.py        # OpenCV-based sign detection runner
├── trained_emotion_recognition_model.keras  # (optional) saved model
├── best_model.keras                  # Emotion model (expected by app.py)
├── action.h5                         # Sign language model (expected by app.py)
├── haarcascades/
│   └── haarcascade_frontalface_default.xml  # Face detector
└── Facial_Images/
    ├── train/<class folders>         # for training (optional)
    └── validation/<class folders>    # for validation (optional)
```

> **Models expected by the app**
> - `best_model.keras` – emotion classifier (48×48 grayscale input)  
> - `action.h5` – sign language sequence model (30-frame keypoint window)  
> Put them in the project root (same folder as `app.py`). If you don’t have them yet, see **Training** below or replace with your own.

---

## 🔧 Requirements

- Python **3.9 – 3.11** recommended
- Webcam access

Create a virtual env and install:

```bash
pip install -r requirements.txt
```

If you don’t keep a `requirements.txt`, this minimal set works:

```txt
streamlit
streamlit-webrtc
opencv-python
tensorflow        # or tensorflow-cpu on machines without GPU
mediapipe
numpy
matplotlib
scikit-learn
```

---

## 🚀 Run the App (WebRTC)

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (e.g., `http://localhost:8501`), allow camera access, and use the sidebar to switch between **Emotion Detection** and **Sign Language Detection**.

> For remote deployment, WebRTC usually needs HTTPS and proper STUN/TURN. The app uses Google’s public STUN by default:
> ```
> stun:stun.l.google.com:19302
> ```
> For production, configure your own TURN server.

---

## 🎥 Run the Simple Webcam Scripts (OpenCV)

Emotion (OpenCV window):
```bash
python webcam.py
```

Sign language (OpenCV window):
```bash
python sign_language_detection.py
```

Press `q` to quit the OpenCV window.

---

## 🧪 Quick Test on a Single Image

Replace the `image_path` in the snippet inside your training/playground script with a validation image.  
The model expects **grayscale 48×48** normalized input.

---

## 🏋️ Training (Optional)

The repo includes a simple training pipeline for the emotion model. Expected dataset layout:

```
Facial_Images/
├── train/
│   ├── angry/ ... .jpg
│   ├── disgust/ ...
│   ├── fear/ ...
│   ├── happy/ ...
│   ├── neutral/ ...
│   ├── sad/ ...
│   └── surprise/ ...
└── validation/
    ├── angry/ ...
    └── ... (same class folders)
```

High-level steps:
1. Prepare the dataset folders above.
2. In the training script, ensure `train_data_path` and `val_data_path` point to your dataset.
3. Run the training cell/section to produce `best_model.keras` (early stopping + checkpoint are set up).
4. Drop `best_model.keras` into the project root.

> **Tip:** If class distribution is imbalanced, the script uses `class_weight` to compensate.

---

## ⚙️ Notes & Troubleshooting

- **Haarcascade path**: the app tries `haarcascades/haarcascade_frontalface_default.xml` first, then falls back to OpenCV’s built-in path.
- **Keras imports**: this project uses `tensorflow.keras`. Avoid mixing legacy `keras` and `tf.keras`.
- **Missing utils**: `app.py` gracefully falls back if `utils.py` isn’t available, but keeping it improves drawing/keypoint extraction.
- **Streamlit “camera not starting”**:
  - Close any other app using the camera.
  - Try a different browser (Chrome/Edge).
  - If remote, ensure HTTPS or proper TURN server.
- **GPU/CPU**: If TensorFlow GPU isn’t available, install `tensorflow-cpu`.

---

## 🔐 Privacy

All inference runs locally in your session. Video frames are processed in memory and not stored unless you add code to do so.



---

## 🙌 Acknowledgements

- TensorFlow/Keras
- OpenCV
- MediaPipe
- Streamlit & streamlit-webrtc
- Haar cascade from OpenCV’s data distribution

---

## 🗺️ Roadmap (nice-to-have)

- Export to ONNX / TFLite
- Add more sign actions & language packs
- Better UX (device selector, record/recap, latency stats)
- Model cards & evaluation metrics

---

### Badges (optional)

```md
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![Streamlit](https://img.shields.io/badge/Streamlit-WebRTC-red)
```
