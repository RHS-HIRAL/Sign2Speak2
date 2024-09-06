import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from preprocess import preprocess_image  # Ensure this function is correctly defined
from PIL import Image

# Initialize MediaPipe hands detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

tf.get_logger().setLevel('ERROR')
# Load the trained model
model = tf.keras.models.load_model('trained_model.keras')

# Define label mapping (A-Z, space, nothing)
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["space", "nothing"]

# Real-time video capture using OpenCV
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Convert the frame to RGB (required by MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect hand landmarks
    result = hands.process(rgb_frame)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks on the hand
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract the bounding box of the hand region
            height, width, _ = frame.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * width)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * width)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * height)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * height)
            
            # Ensure coordinates are within frame bounds
            x_min = max(x_min, 0)
            y_min = max(y_min, 0)
            x_max = min(x_max, width)
            y_max = min(y_max, height)
            
            # Ensure the region is valid and not empty
            if x_max - x_min > 0 and y_max - y_min > 0:
                hand_region = frame[y_min:y_max, x_min:x_max]
                
                # Debugging: Print the shape of the hand_region
                print(f"Hand region shape: {hand_region.shape}")
                
                # Ensure hand_region is valid and not None
                if hand_region is not None and hand_region.size > 0:
                    # Convert hand region to RGB
                    hand_region_rgb = cv2.cvtColor(hand_region, cv2.COLOR_BGR2RGB)
                    
                    # Check if hand_region_rgb is valid
                    if hand_region_rgb is not None and hand_region_rgb.size > 0:
                        # Convert NumPy array to PIL Image before preprocessing
                        hand_image_pil = Image.fromarray(hand_region_rgb)
                        
                        # Preprocess the hand image (resize, grayscale, normalize)
                        hand_image = preprocess_image(hand_image_pil)  # Ensure this function is correct
                        
                        # Expand dimensions to match model input (1, 64, 64, 1)
                        hand_image = np.expand_dims(hand_image, axis=0)
                        
                        # Predict gesture
                        prediction = model.predict(hand_image)
                        predicted_class = np.argmax(prediction)
                        predicted_label = labels[predicted_class]
                        
                        # Debugging: Print the raw prediction
                        print("Raw prediction:", prediction)
                        
                        # Display predicted gesture on the frame
                        cv2.putText(frame, predicted_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                    else:
                        print("Invalid or empty hand region after RGB conversion.")
                else:
                    print("Invalid hand region detected or empty hand_region.")
            else:
                print("Hand region coordinates are invalid or out of bounds.")
    
    # Display the frame
    cv2.imshow('Sign Language Detection', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
