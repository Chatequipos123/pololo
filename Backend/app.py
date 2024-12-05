import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, Response, jsonify
from tensorflow.keras.models import load_model

# Cargar el modelo
model = load_model('Backend/modelo.h5')

# Inicialización de MediaPipe para detección de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

app = Flask(__name__)

# Función para predecir la seña
def predict_gesture(landmarks):
    # Aquí se extraen los puntos clave de los landmarks
    # Usamos las coordenadas de los puntos clave de las manos
    data = np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in range(len(landmarks))])
    data = data.flatten().reshape(1, -1)  # Aplanar los puntos para que coincidan con la entrada del modelo
    prediction = model.predict(data)
    return prediction[0]  # Devuelve la predicción

# Ruta de video en tiempo real
@app.route('/video')
def video():
    def generate_frames():
        cap = cv2.VideoCapture(0)  # Captura desde la cámara
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convertir a RGB para MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for landmarks in results.multi_hand_landmarks:
                    # Dibujar los puntos clave en la mano
                    mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

                    # Predecir la seña utilizando el modelo
                    prediction = predict_gesture(landmarks.landmark)
                    cv2.putText(frame, f"Predicción: {prediction}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Convertir la imagen a formato JPEG para enviarla al cliente
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

        cap.release()

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Ruta para obtener la predicción
@app.route('/prediction')
def prediction():
    # Aquí puedes devolver la última predicción realizada
    # En un caso real, probablemente necesitarías almacenar y gestionar las predicciones
    return jsonify({"prediction": "Seña detectada"})  # Cambiar según la lógica real

# Página principal (HTML)
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
