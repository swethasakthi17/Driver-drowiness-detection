import cv2
import dlib
import numpy as np
from keras.models import load_model

# Load pre-trained models
emotion_model = load_model('emotion_detection_model.h5')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Dictionary for mapping emotion labels
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

def detect_emotion(gray_frame):
    faces = detector(gray_frame)
    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        gray_face = gray_frame[y:y1, x:x1]
        gray_face = cv2.resize(gray_face, (48, 48))
        gray_face = gray_face / 255.0
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_model.predict(gray_face)
        emotion_label = np.argmax(emotion_prediction)
        return emotion_dict[emotion_label]

def detect_yawn(shape):
    top_lip = shape[50:53] + shape[61:64]
    top_lip = np.array(top_lip, dtype=np.int32)
    bottom_lip = shape[65:68] + shape[56:59]
    bottom_lip = np.array(bottom_lip, dtype=np.int32)
    top_mean = np.mean(top_lip, axis=0)
    bottom_mean = np.mean(bottom_lip, axis=0)
    distance = abs(top_mean[1] - bottom_mean[1])
    return distance

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        shape = np.array([[p.x, p.y] for p in landmarks.parts()])
        
        # Detect emotion
        emotion = detect_emotion(gray)
        
        # Detect yawn
        yawn_distance = detect_yawn(shape)
        
        # Display emotion and yawn status
        cv2.putText(frame, f"Emotion: {emotion}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #if yawn_distance > 20:  # Adjust threshold as needed
            #cv2.putText(frame, "Yawning Detected!", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow('Mood and Drowsiness Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
