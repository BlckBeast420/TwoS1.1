# lsm_a_texto.py
# === Librerías ===
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyttsx3
import time
import unicodedata

# === CONFIGURACIÓN GENERAL ===
FRAMES_LSTM = 30           # Número de frames requeridos para el LSTM
COOLDOWN_SEGUNDOS = 2      # Tiempo entre predicciones para evitar repeticiones
buffer = []                # Almacena secuencia de frames
ultima_palabra = ""        # Guarda la última predicción
ultimo_tiempo = time.time() - COOLDOWN_SEGUNDOS  # Controla cooldown inicial

# === CARGAR MODELO Y CLASES ===
modelo = tf.keras.models.load_model("modelo_manos_rostro.h5")
clases = np.load("clases_manos_rostro.npy", allow_pickle=True)

# === CONFIGURAR SÍNTESIS DE VOZ ===
voz = pyttsx3.init()
voz.setProperty('rate', 140)  # Velocidad de lectura

# === UTILERÍA: Quitar acentos para pantalla ===
def quitar_acentos(texto):
    return ''.join(
        c for c in unicodedata.normalize('NFKD', texto)
        if not unicodedata.combining(c)
    )

# === MEDIAPIPE: Manos y Rostro ===
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.7, min_tracking_confidence=0.7
)

# === CONEXIONES DEL ROSTRO ===
FACE_CONNECTIONS = [
    mp_face_mesh.FACEMESH_CONTOURS,
    mp_face_mesh.FACEMESH_LEFT_EYE,
    mp_face_mesh.FACEMESH_RIGHT_EYE,
    mp_face_mesh.FACEMESH_LEFT_EYEBROW,
    mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
    mp_face_mesh.FACEMESH_FACE_OVAL,
    mp_face_mesh.FACEMESH_NOSE,
    mp_face_mesh.FACEMESH_LIPS
]

# === INICIAR CÁMARA ===
cap = cv2.VideoCapture(0)
print("🟢 Sistema listo. Haz un gesto para comenzar...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results_hands = hands.process(rgb)
    results_face = face_mesh.process(rgb)

    # === INICIALIZAR VECTORES DE MANOS Y ROSTRO ===
    mano_1 = [0.0] * 42
    mano_2 = [0.0] * 42
    rostro = [0.0] * 36

    # === EXTRAER MANOS ===
    if results_hands.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
            coords = [lm.x for lm in hand_landmarks.landmark] + [lm.y for lm in hand_landmarks.landmark]
            if len(coords) == 42:
                if i == 0:
                    mano_1 = coords
                elif i == 1:
                    mano_2 = coords
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # === EXTRAER ROSTRO ===
    if results_face.multi_face_landmarks:
        face_landmarks = results_face.multi_face_landmarks[0]
        indices_clave = [33, 263, 61, 291, 199, 4, 0, 17, 267, 37, 164, 393, 78, 308, 13, 14, 312, 82]
        coords_rostro = []
        for idx in indices_clave:
            lm = face_landmarks.landmark[idx]
            coords_rostro.extend([lm.x, lm.y])
        if len(coords_rostro) == 36:
            rostro = coords_rostro

        # DIBUJAR contornos clave del rostro
        linea = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
        for grupo in FACE_CONNECTIONS:
            mp_drawing.draw_landmarks(frame, face_landmarks, grupo, landmark_drawing_spec=None, connection_drawing_spec=linea)

    # === SOLO AGREGAR SI HAY AL MENOS UNA MANO ===
    if any(mano_1) or any(mano_2):
        datos = mano_1 + mano_2 + rostro
        buffer.append(datos)
        if len(buffer) > FRAMES_LSTM:
            buffer.pop(0)

    # === PREDICCIÓN SI EL BUFFER ESTÁ COMPLETO Y CUMPLE COOLDOWN ===
    if len(buffer) == FRAMES_LSTM and (time.time() - ultimo_tiempo >= COOLDOWN_SEGUNDOS):
        entrada = np.array(buffer).reshape(1, FRAMES_LSTM, 120).astype(np.float32)
        pred = modelo.predict(entrada, verbose=0)
        palabra = clases[np.argmax(pred)]
        ultima_palabra = palabra

        # Mostrar texto sin acentos en pantalla
        texto_mostrar = quitar_acentos(ultima_palabra).capitalize()
        cv2.putText(frame, texto_mostrar, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 3)

        # Reproducir con voz
        voz.say(ultima_palabra)
        voz.runAndWait()

        buffer = []  # Reiniciar buffer
        ultimo_tiempo = time.time()  # Reiniciar cooldown

    # === Mostrar palabra en pantalla (mientras dure el cooldown) ===
    if ultima_palabra:
        texto_mostrar = quitar_acentos(ultima_palabra).capitalize()
        cv2.putText(frame, texto_mostrar, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 3)

    cv2.imshow("Traductor LSM -> Texto + Voz", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === LIMPIEZA FINAL ===
cap.release()
cv2.destroyAllWindows()

