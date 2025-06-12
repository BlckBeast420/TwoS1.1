#  LIBRERÍAS 
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyttsx3
import time
import unicodedata

#  CONFIGURACIÓN 
FRAMES_LSTM = 30  # Número de frames que requiere el modelo LSTM
DELAY_SEGUNDOS = 2  # Tiempo de espera entre predicciones para evitar repeticiones

#  MODELOS ENTRENADOS 
modelo = tf.keras.models.load_model("modelo_manos_rostro.h5")  # Modelo LSTM entrenado con manos + rostro
clases = np.load("clases_manos_rostro.npy", allow_pickle=True)

#  VOZ (pyttsx3) 
voz = pyttsx3.init()
voz.setProperty('rate', 140)  # Velocidad de lectura

#  UTILIDAD: quitar acentos para visualización en pantalla 
def quitar_acentos(texto):
    return ''.join(c for c in unicodedata.normalize('NFKD', texto) if not unicodedata.combining(c))

#  MEDIAPIPE SETUP 
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Dibujo de los landmarks y deteccion de solo 2 manos
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                             min_detection_confidence=0.7, min_tracking_confidence=0.7)

#  VARIABLES DE FLUJO 
cap = cv2.VideoCapture(0)
buffer = []
palabra_actual = ""
ultimo_tiempo = time.time() - DELAY_SEGUNDOS

print("🟢 Sistema iniciado. Realiza un gesto para comenzar.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results_hands = hands.process(rgb)
    results_face = face_mesh.process(rgb)

    #  OBTENER LANDMARKS DE MANOS 
    mano_1 = [0.0] * 42
    mano_2 = [0.0] * 42

    if results_hands.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y])
            if len(coords) == 42:
                if i == 0:
                    mano_1 = coords
                elif i == 1:
                    mano_2 = coords
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    #  OBTENER LANDMARKS DE ROSTRO 
    rostro = [0.0] * 36
    if results_face.multi_face_landmarks:
        face_landmarks = results_face.multi_face_landmarks[0]
        indices_clave = [33, 263, 61, 291, 199, 4, 0, 17, 267, 37, 164, 393, 78, 308, 13, 14, 312, 82]
        coords_rostro = []
        for idx in indices_clave:
            lm = face_landmarks.landmark[idx]
            coords_rostro.extend([lm.x, lm.y])
        if len(coords_rostro) == 36:
            rostro = coords_rostro

        # Dibujar contornos básicos
        color_linea = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
        for connection_type in [
            mp_face.FACEMESH_CONTOURS,
            mp_face.FACEMESH_LEFT_EYE,
            mp_face.FACEMESH_RIGHT_EYE,
            mp_face.FACEMESH_LEFT_EYEBROW,
            mp_face.FACEMESH_RIGHT_EYEBROW,
            mp_face.FACEMESH_FACE_OVAL,
            mp_face.FACEMESH_NOSE,
            mp_face.FACEMESH_LIPS,
        ]:
            mp_drawing.draw_landmarks(frame, face_landmarks, connection_type,
                                      landmark_drawing_spec=None, connection_drawing_spec=color_linea)

    #  VERIFICAR QUE HAYA AL MENOS UNA MANO PARA EMPEZAR 
    if any(mano_1) or any(mano_2):
        datos = mano_1 + mano_2 + rostro
        buffer.append(datos)

        if len(buffer) > FRAMES_LSTM:
            buffer.pop(0)

        if len(buffer) == FRAMES_LSTM and (time.time() - ultimo_tiempo >= DELAY_SEGUNDOS):
            entrada = np.array(buffer).reshape(1, FRAMES_LSTM, len(datos))
            pred = modelo.predict(entrada, verbose=0)
            indice = np.argmax(pred)
            palabra_actual = clases[indice]

            # Mostrar sin acentos en pantalla
            texto_mostrar = quitar_acentos(palabra_actual).capitalize()
            cv2.putText(frame, texto_mostrar, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 3)

            # Reproducir con voz 
            voz.say(palabra_actual)
            voz.runAndWait()

            buffer = []
            ultimo_tiempo = time.time()

    # Mostrar texto actual
    if palabra_actual:
        texto_mostrar = quitar_acentos(palabra_actual).capitalize()
        cv2.putText(frame, texto_mostrar, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 3)

    cv2.imshow("Traductor LSM - Texto y Voz", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

