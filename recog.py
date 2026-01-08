import cv2
import numpy as np
import socket
import json
import base64
import time

# Configuration
HOST = '172.20.168.79'  # Your WSL IP
PORT = 5000

# Emotion labels - adjust order to match your model
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if face_cascade.empty():
    print("Error: Could not load face cascade classifier!")
    exit()

# Connect to WSL server
print(f"Connecting to WSL server at {HOST}:{PORT}...")
try:
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))
    client_socket.settimeout(5)
    print("✓ Connected to WSL server!")
except Exception as e:
    print(f"Error connecting: {e}")
    exit()

# Start webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

print("Press 'q' to quit")
print("Camera started...")

latest_emotion = None
latest_confidence = 0
frame_count = 0
process_every = 5  # Process every 5th frame for speed

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    
    # Process first face every N frames
    if len(faces) > 0 and frame_count % process_every == 0:
        x, y, w, h = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (224, 224))
        face_normalized = (face_resized / 255.0).astype(np.float32)
        
        try:
            # Encode as base64 JSON (more reliable than pickle)
            face_bytes = face_normalized.tobytes()
            face_b64 = base64.b64encode(face_bytes).decode('utf-8')
            
            message = json.dumps({'face': face_b64}) + '\n'
            client_socket.sendall(message.encode())
            
            # Receive response
            response = b''
            while b'\n' not in response:
                chunk = client_socket.recv(1024)
                if not chunk:
                    break
                response += chunk
            
            if response:
                result = json.loads(response.decode().strip())
                latest_emotion = emotion_labels[result['emotion_idx']]
                latest_confidence = result['confidence']
                
        except socket.timeout:
            pass  # Skip this frame
        except Exception as e:
            print(f"Error: {e}")
    
    # Draw all faces
    for i, (x, y, w, h) in enumerate(faces):
        # Draw box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Show prediction on first face only
        if i == 0 and latest_emotion:
            text = f"{latest_emotion}: {latest_confidence:.0f}%"
            
            # Background for text
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.rectangle(frame, (x, y-40), (x+tw+10, y-5), (0, 255, 0), -1)
            
            # Text
            cv2.putText(frame, text, (x+5, y-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    
    # Status
    status = f"FPS: ~{30//process_every} | Faces: {len(faces)}"
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
               0.7, (255, 255, 255), 2)
    
    cv2.imshow('Emotion Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
client_socket.close()
print("Done!")