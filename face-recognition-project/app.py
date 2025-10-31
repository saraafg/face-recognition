import os
import cv2
import numpy as np
import streamlit as st

# Use TF/Keras consistently
import tensorflow as tf
from tensorflow.keras.models import load_model

# WebRTC for webcam in browser
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode


# Try to use your utils if available
try:
    from utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints
    HAVE_UTILS = True
except Exception:
    HAVE_UTILS = False

# Optional: mediapipe (needed for sign language)
import mediapipe as mp

st.set_page_config(page_title="Emotion & Sign Language Detection", layout="centered")

st.title("🖥️ Emotion & Sign Language Detection (WebRTC)")
st.sidebar.title("Navigation")
mode = st.sidebar.selectbox("Choose an option", ["Emotion Detection", "Sign Language Detection"])

# ====== Cached loaders ======
@st.cache_resource
def load_emotion_model(path="best_model.keras"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Emotion model not found at: {path}")
    return load_model(path)

@st.cache_resource
def load_sign_model(path="action.h5"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Sign model not found at: {path}")
    return load_model(path)

@st.cache_resource
def load_face_detector():
    # Works if file is at ./haarcascades/haarcascade_frontalface_default.xml
    local_path = os.path.join("haarcascades", "haarcascade_frontalface_default.xml")
    if os.path.exists(local_path):
        return cv2.CascadeClassifier(local_path)
    # Fallback to OpenCV data
    default_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(default_path)

# WebRTC needs a config for some environments (TURN server recommended for prod; STUN is OK for localhost)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ====== Emotion transformer ======
class EmotionTransformer(VideoTransformerBase):
    def __init__(self, model, classes, face_detector):
        self.model = model
        self.classes = classes
        self.face_detector = face_detector

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi, (48, 48))
            roi_norm = (roi_resized.astype("float32") / 255.0)[..., np.newaxis]
            roi_batch = np.expand_dims(roi_norm, axis=0)

            preds = self.model.predict(roi_batch, verbose=0)[0]
            idx = int(np.argmax(preds))
            label = self.classes[idx]
            conf = float(preds[idx])

            # Draw
            cv2.rectangle(img, (x, y-50), (x+w, y+h+10), (0, 255, 0), 2)
            cv2.putText(img, f"{label} ({conf:.2f})", (x+5, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)
        return img

# ====== Sign language transformer ======
class SignTransformer(VideoTransformerBase):
    def __init__(self, model, actions):
        self.model = model
        self.actions = actions
        self.threshold = 0.5
        self.sequence = []
        self.sentence = []
        self.colors = [(245,117,16), (117,245,16), (16,117,245), (16,245,245), (245,16,117),
                       (245,245,16), (117,16,245), (16,245,117)]
        # Mediapipe holistic
        self.holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

    def _mp_detection(self, image):
        # Fallback if user doesn't have utils.py
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.holistic.process(image_rgb)
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        return image, results

    def _draw_landmarks(self, image, results):
        # Fallback drawing if utils not present
        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        mp_holistic = mp.solutions.holistic
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    def _extract_keypoints(self, results):
        pose = np.zeros(33*4)
        face = np.zeros(468*3)
        lh = np.zeros(21*3)
        rh = np.zeros(21*3)
        if results.pose_landmarks:
            pose = np.array([[res.x, res.y, res.z, res.visibility]
                             for res in results.pose_landmarks.landmark]).flatten()
        if results.face_landmarks:
            face = np.array([[res.x, res.y, res.z]
                             for res in results.face_landmarks.landmark]).flatten()
        if results.left_hand_landmarks:
            lh = np.array([[res.x, res.y, res.z]
                           for res in results.left_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks:
            rh = np.array([[res.x, res.y, res.z]
                           for res in results.right_hand_landmarks.landmark]).flatten()
        return np.concatenate([pose, face, lh, rh])

    def _prob_viz(self, probs, image):
        for i, p in enumerate(probs):
            cv2.rectangle(image, (0, 60 + i*40), (int(p*300), 90 + i*40), self.colors[i % len(self.colors)], -1)
            cv2.putText(image, f'{self.actions[i]}: {p:.2f}', (10, 85 + i*40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        return image

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")

        # Use your utils if present; otherwise fallback
        if HAVE_UTILS:
            image, results = mediapipe_detection(image, self.holistic)
            draw_styled_landmarks(image, results)
            keypoints = extract_keypoints(results)
        else:
            image, results = self._mp_detection(image)
            self._draw_landmarks(image, results)
            keypoints = self._extract_keypoints(results)

        # Sequence buffer
        self.sequence.append(keypoints)
        self.sequence = self.sequence[-30:]

        if len(self.sequence) == 30:
            res = self.model.predict(np.expand_dims(self.sequence, axis=0), verbose=0)[0]
            pred_idx = int(np.argmax(res))
            pred_label = self.actions[pred_idx]
            pred_conf = float(res[pred_idx])

            if pred_conf > self.threshold:
                if not self.sentence or pred_label != self.sentence[-1]:
                    self.sentence.append(pred_label)
            if len(self.sentence) > 5:
                self.sentence = self.sentence[-5:]

            image = self._prob_viz(res, image)

        # Header bar with sentence
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, " ".join(self.sentence[-5:]), (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
        return image

# ====== UI ======
if mode == "Emotion Detection":
    st.header("😊 Webcam Emotion Recognition")

    # Classes (adjust to your dataset ordering)
    EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

    try:
        emotion_model = load_emotion_model("best_model.keras")
    except Exception as e:
        st.error(f"Could not load emotion model: {e}")
        st.stop()

    face_detector = load_face_detector()

    st.info("Allow camera access. Press the ❌/Stop button to end.")
    webrtc_streamer(
        key="emotion",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=lambda: EmotionTransformer(
            model=emotion_model,
            classes=EMOTION_CLASSES,
            face_detector=face_detector,
        ),
    )

else:
    st.header("🤟 Webcam Sign Language Recognition")

    # Actions must match your training
    ACTIONS = np.array(['Salam', 'Khubi', 'kojast', 'Khune', 'Man', 'raftan', 'Maman', 'Bayad'])

    try:
        sign_model = load_sign_model("action.h5")
    except Exception as e:
        st.error(f"Could not load sign language model: {e}")
        st.stop()

    st.info("Allow camera access. Press the ❌/Stop button to end.")
    webrtc_streamer(
        key="sign",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=lambda: SignTransformer(
            model=sign_model,
            actions=ACTIONS,
        ),
        async_processing=True,  # keeps UI responsive
    )

st.caption("Tip: run with `streamlit run app.py` and open the local URL (HTTPS needed on the web).")
