import cv2
import numpy as np
import tensorflow as tf
import sounddevice as sd # pip install sounddevice
import queue

# 1. Setup Models & Queues
interpreter = tf.lite.Interpreter(model_path="zephyx_brain.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

audio_q = queue.Queue()
LABELS = ["Healthy", "Melanoma Suspected", "Basal Cell", "Fungal", "Other"]

# 2. Audio Callback Function
def audio_callback(indata, frames, time, status):
    volume_norm = np.linalg.norm(indata)*10
    audio_q.put(volume_norm)

# 3. Start Stream
cap = cv2.VideoCapture(0)
with sd.InputStream(callback=audio_callback):
    print("ZEPHYX Multimodal Fusion Active...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        display_frame = frame.copy()
        h, w, _ = frame.shape

        # --- VISION LOGIC ---
        roi_x, roi_y, roi_w, roi_h = w//3, h//4, w//3, h//2
        cv2.rectangle(display_frame, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (255, 255, 0), 2)
        
        # Prediction
        roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        img = cv2.resize(roi, (224, 224)).astype(np.float32) / 255.0
        interpreter.set_tensor(input_details[0]['index'], [img])
        interpreter.invoke()
        pred_idx = np.argmax(interpreter.get_tensor(output_details[0]['index'])[0]) % len(LABELS)

        # --- AUDIO LOGIC (Real-time Fusion) ---
        vol = audio_q.get() if not audio_q.empty() else 0
        audio_status = "Acoustic: MONITORING" if vol < 5 else "Acoustic: BREATH/COUGH DETECTED"
        audio_color = (0, 255, 0) if vol < 5 else (0, 165, 255)

        # --- MULTIMODAL DASHBOARD OVERLAY ---
        # Top Bar
        cv2.rectangle(display_frame, (0, 0), (w, 60), (0,0,0), -1)
        cv2.putText(display_frame, f"VISION: {LABELS[pred_idx]}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Side Bar for Audio
        cv2.putText(display_frame, audio_status, (20, h-30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, audio_color, 2)

        cv2.imshow('ZEPHYX: Multimodal Fusion Prototype', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()