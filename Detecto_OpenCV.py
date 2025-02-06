import cv2
import numpy as np
import serial
from tensorflow.keras.models import load_model
from keras.src.legacy.saving import legacy_h5_format


# Load your model
# model = load_model('custom_resnet_model.h5')
model = legacy_h5_format.load_model_from_hdf5("ANN_Model.h5", custom_objects={'mae': 'mae'})
# model = legacy_h5_format.load_model_from_hdf5("CNN_Model.h5", custom_objects={'mae': 'mae'})


# Define object classes (mango and pomegranate)
object_classes = ['mango 1', 'pomegranate 1']  # Update with your classes if different

# Initialize the Arduino serial connection (update with your port on Ubuntu)
arduino = serial.Serial('/dev/ttyUSB0', 9600)  # Replace with the correct port

# Function to predict the object class and probability
def predict_object(image, model, classes):
    image_resized = cv2.resize(image, (224, 224))  # Resize to model's input size
    image_resized = np.expand_dims(image_resized, axis=0)  # Add batch dimension
    image_resized = image_resized / 255.0  # Normalize pixel values to [0, 1]
    predictions = model.predict(image_resized)
    class_idx = np.argmax(predictions)  # Get the class index with the highest probability
    probability = predictions[0][class_idx]  # Extract the probability of the predicted class
    return classes[class_idx], probability

# Arduino Servo Control Function
def control_arduino(detected_object):
    if detected_object == 'mango 1':
        arduino.write(b'R')  # Raise right arm for mango
    elif detected_object == 'pomegranate 1':
        arduino.write(b'L')  # Raise left arm for pomegranate
    else:
        print("Unknown object detected.")

# Draw bounding box function with probability
def draw_bounding_box(frame, label, probability):
    height, width, _ = frame.shape
    x1, y1 = width // 4, height // 4  # Define top-left corner of the box
    x2, y2 = 3 * width // 4, 3 * height // 4  # Define bottom-right corner of the box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw rectangle
    text = f"{label}: {probability * 100:.2f}%"  # Format label with probability
    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use 0 for default camera

print("Press 'd' to detect objects, and 'q' to quit.")

try:
    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Display the live video feed
        cv2.imshow('Detecto', frame)

        # Wait for key press
        key = cv2.waitKey(1)

        # Press 'd' to detect objects
        if key & 0xFF == ord('d'):
            # Crop the central region of the frame for object detection
            height, width, _ = frame.shape
            cropped_frame = frame[height // 4: 3 * height // 4, width // 4: 3 * width // 4]

            detected_object, probability = predict_object(cropped_frame, model, object_classes)
            print(f"Detected: {detected_object} with probability {probability * 100:.2f}%")
            control_arduino(detected_object)

            # Draw bounding box around the detected region with probability
            draw_bounding_box(frame, detected_object, probability)

        # Display the frame with bounding box
            cv2.imshow('Detecto', frame)

        # Press 'q' to quit
        if key & 0xFF == ord('q'):
            break

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    arduino.close()
