import os
import sys
import time
import base64
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, Response
from tensorflow.keras.models import load_model
import mediapipe as mp
import cv2

# Ruta al modelo
model_path = os.path.join(os.getcwd(), 'Backend', 'modelo.h5')  # Ruta local cuando el script no está empaquetado

# Inicializar Flask
app = Flask(__name__)

# Intentar cargar el modelo de Keras y manejar el error si el formato es incorrecto
try:
    model = load_model(model_path, compile=False)
except ValueError as e:
    sys.exit(1)  # Terminar el script si el modelo no se puede cargar

# Inicializar MediaPipe para la detección de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Mapeo manual de las clases
class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "a", "e", "i", "u", "o", "b", "c", "d", "f", "g", "h", 
               "l", "m", "n", "p", "r", "s", "t", "v", "w", "y", "k", "q", "x", "z", "te amo", "mucho", "yo"]

# Ruta principal de la aplicación web
@app.route('/')
def home():
    return render_template('index.html')  # Renderizar la página principal

# Función para capturar el video y procesarlo
def generate_frames():
    global current_word, last_letter, words_history, last_detection_time  # Hacer que las variables sean globales

    # Abrir la cámara
    cap = cv2.VideoCapture(0)  # Captura de la cámara local

    # Verificar si la cámara se abrió correctamente
    if not cap.isOpened():
        return jsonify({"error": "No se pudo acceder a la cámara."}), 500

    while True:
        success, image = cap.read()
        
        # Verificar si se obtuvo una imagen válida
        if not success or image is None:
            continue  # O terminar el ciclo si prefieres no seguir ejecutando

        # Convertir la imagen a RGB para MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Procesar la imagen con MediaPipe
        results = hands.process(image_rgb)

        class_label = ""  # Predicción de la seña
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                keypoints = []
                for lm in hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])

                keypoints = np.array(keypoints).reshape(1, -1)
                prediction = model.predict(keypoints, verbose=0)
                class_index = np.argmax(prediction)
                class_label = class_names[class_index]

        # Mostrar las palabras formadas
        text_word = " ".join(words_history) + current_word
        image_height, image_width, _ = image.shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2

        # Mostrar la palabra formada
        (text_width, text_height), _ = cv2.getTextSize(text_word, font, font_scale, font_thickness)
        text_x = (image_width - text_width) // 2  # Posición centrada en la parte inferior
        text_y = image_height - 50  # Parte inferior

        # Añadir contorno y sombra al texto de la palabra
        cv2.putText(image, text_word, (text_x + 2, text_y + 2), font, font_scale, (0, 0, 0), font_thickness + 2, lineType=cv2.LINE_AA)
        cv2.putText(image, text_word, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness, lineType=cv2.LINE_AA)

        # Convertir la imagen a formato JPEG para enviarla como un frame
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()

        # Enviar el frame como un flujo de bytes
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Ruta para mostrar el video en el navegador (streaming)
@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


