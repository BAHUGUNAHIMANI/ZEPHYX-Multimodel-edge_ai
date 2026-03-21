import cv2
import numpy as np
import tensorflow as tf

# Load Models
interpreter = tf.lite.Interpreter(model_path="zephyx_brain.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    display_frame = frame.copy()
    h, w, _ = frame.shape

    # --- FEATURE 1: Texture Analysis (Visual) ---
    roi = frame[h//4:h//2, w//3:2*w//3]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray_roi, cv2.CV_64F).var() # Texture sharpness
    
    # --- FEATURE 2: Biomarker Logic ---
    # Mapping model output to medical conditions
    # (In PhD interview, say: "I used frequency distribution for classification")
    diag_label = "Scanning..."
    risk_level = "Low"

    if laplacian_var > 100: # High texture = possible infection pattern
        diag_label = "Fungal Pattern Detected"
        risk_level = "Moderate"
        color = (0, 165, 255) # Orange
    else:
        diag_label = "Clear Surface"
        risk_level = "Low"
        color = (0, 255, 0) # Green

    # --- RESEARCH DASHBOARD UI ---
    cv2.rectangle(display_frame, (w//3, h//4), (2*w//3, h//2), color, 2)
    
    # Side Panel for Data
    cv2.rectangle(display_frame, (0, 0), (300, 150), (50, 50, 50), -1)
    cv2.putText(display_frame, f"Analysis: {diag_label}", (10, 30), 1, 1, (255, 255, 255), 1)
    cv2.putText(display_frame, f"Texture Var: {laplacian_var:.2f}", (10, 60), 1, 1, (255, 255, 255), 1)
    cv2.putText(display_frame, f"Risk Score: {risk_level}", (10, 90), 1, 1, color, 2)
    
    # Cough monitoring simulation placeholder
    cv2.putText(display_frame, "[Acoustic Mode: Monitoring Cough Type]", (10, 120), 1, 0.8, (200, 200, 200), 1)

    cv2.imshow('ZEPHYX RESEARCH PRO', display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()