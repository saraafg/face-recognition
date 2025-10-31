import os
import numpy as np
import cv2
from keras._tf_keras.keras.models import load_model
import mediapipe as mp

from utils import draw_styled_landmarks, extract_keypoints, mediapipe_detection

# Load your saved model
model = load_model('action.h5')

# Actions to detect
actions = np.array(['Salam', 'Khubi', 'kojast', 'Khune', 'Man', 'raftan', 'Maman', 'Bayad'])

# Color array for the probability visualization
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (16, 245, 245), (245, 16, 117), 
          (245, 245, 16), (117, 16, 245), (16, 245, 117)]

# Function to visualize predictions with confidence
def prob_viz(res, actions, input_frame, colors):
    for num, prob in enumerate(res):
        cv2.rectangle(input_frame, (0, 60 + num * 40), (int(prob * 300), 90 + num * 40), colors[num], -1)
        cv2.putText(input_frame, f'{actions[num]}: {prob:.2f}', (10, 85 + num * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return input_frame

# Sign Language Detection Function
def start_sign_language_detection():
    # Detection variables
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5

    # Initialize mediapipe holistic model
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    # Start video capture
    cap = cv2.VideoCapture(0)

    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            # Read feed
            ret, frame = cap.read()

            # Make detections (replace 'mediapipe_detection' with your actual implementation)
            image, results = mediapipe_detection(frame, holistic)

            # Draw landmarks (replace 'draw_styled_landmarks' with your actual implementation)
            draw_styled_landmarks(image, results)

            # Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                # Make prediction from the model
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))

                # Display the detected word with confidence
                if res[np.argmax(res)] > threshold: 
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5: 
                    sentence = sentence[-5:]

                # Visualize probabilities
                image = prob_viz(res, actions, image, colors)
                
            # Display the detected word at the top center of the screen
            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Show video feed with annotations
            cv2.imshow('Sign Language Detection', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()

