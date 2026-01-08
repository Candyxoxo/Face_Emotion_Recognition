import socket
import json
import base64
import numpy as np
from tensorflow.keras.models import load_model

print("Loading model...")
model = load_model('/mnt/c/Users/OMEN/Downloads/code/FACE_EMOTION_RECOG/MODEL/best_model2.keras')
print("✓ Model loaded!")

HOST = '0.0.0.0'
PORT = 5000

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print(f"Server listening on {HOST}:{PORT}")
print("Waiting for client...")

client_socket, addr = server_socket.accept()
print(f"✓ Connected: {addr}")

count = 0

try:
    buffer = b''
    while True:
        # Read until newline
        data = client_socket.recv(4096)
        if not data:
            break
        
        buffer += data
        
        # Process complete messages
        while b'\n' in buffer:
            line, buffer = buffer.split(b'\n', 1)
            
            try:
                count += 1
                request = json.loads(line.decode())
                
                # Decode face data
                face_bytes = base64.b64decode(request['face'])
                face_array = np.frombuffer(face_bytes, dtype=np.float32).reshape(224, 224)
                face_input = face_array.reshape(1, 224, 224, 1)
                
                # Predict
                prediction = model.predict(face_input, verbose=0)
                emotion_idx = int(np.argmax(prediction))
                confidence = float(np.max(prediction) * 100)
                
                # Send response
                response = json.dumps({
                    'emotion_idx': emotion_idx,
                    'confidence': confidence
                }) + '\n'
                
                client_socket.sendall(response.encode())
                
                if count % 10 == 0:
                    print(f"Processed {count} requests")
                    
            except Exception as e:
                print(f"Error: {e}")
                continue

except KeyboardInterrupt:
    print("\nShutting down...")
finally:
    client_socket.close()
    server_socket.close()
    print("Server closed")