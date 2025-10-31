import numpy as np
from keras._tf_keras.keras import models
from utils import preprocess_image

# Load the trained model
model = models.load_model('best_model.keras')

# Define the emotion categories (adjust according to your dataset)
emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def predict_emotion(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_emotion = emotion_classes[np.argmax(prediction)]
    return predicted_emotion
