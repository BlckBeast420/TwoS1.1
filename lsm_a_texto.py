import os
# Silenciar logs de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=INFO,1=WARNING,2=ERROR,3=FATAL

# Silenciar logs de absl (usados por MediaPipe y TensorFlow)
import logging
logging.getLogger('absl').setLevel(logging.ERROR)
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)

# Importar módulos necesarios
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyttsx3
import time
import unicodedata

# —————  CONFIG  —————
FRAMES_LSTM    = 30    # Frames por predicción
COOLDOWN_SEC   = 2.0   # Retraso entre predicciones

# —————  MODELO  —————
modelo = tf.keras.models.load_model("modelo_manos_rostro.h5")
clases = np.load("clases_manos_rostro.npy", allow_pickle=True)

# —————  VOZ  —————
voz = pyttsx3.init()
voz.setProperty('rate', 140)

# —————  UTIL  —————
def quitar_acentos(txt):
    return ''.join(c for c in unicodedata.normalize('NFKD', txt)
                   if not unicodedata.combining(c))

# —————  MEDIAPIPE  —————
mp_hands     = mp.solutions.hands
mp_face      = mp.solutions.face_mesh
mp_drawing   = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=2,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

face_mesh = mp_face.FaceMesh(static_image_mode=False,
                             max_num_faces=1,
                             refine_landmarks=True,
                             min_detection_confidence=0.7,
                             min_tracking_confidence=0.7)

# Conexiones faciales a dibujar
FACE_CONNECTIONS = [
    mp_face.FACEMESH_CONTOURS,
    mp_face.FACEMESH_LEFT_EYE,
    mp_face.FACEMESH_RIGHT_EYE,
    mp_face.FACEMESH_LEFT_EYEBROW,
    mp_face.FACEMESH_RIGHT_EYEBROW,
    mp_face.FACEMESH_FACE_OVAL,
    mp_face.FACEMESH_NOSE,
    mp_face.FACEMESH_LIPS,
]

def run_lsm_to_text(batch_mode: bool):
    """
    Modo LSM→Texto/Voz. 
    Si batch_mode=False: habla cada vez que detecta.
    Si batch_mode=True: acumula predicciones hasta tecla '5'.
    Devuelve (tecla_presionada, #palabras_en_batch).
    Teclas especiales: 
      3 = cambia de modo  
      4 = iniciar batch  
      5 = finalizar batch  
      q = salir
    """
    cap = cv2.VideoCapture(0)
    buffer = []
    batch = []
    last_time = time.time() - COOLDOWN_SEC
    tecla = None

    while True:
        ret, frame = cap.read()
        if not ret:
            tecla = 'q'
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #–– Procesar manos y rostro ––
        res_h = hands.process(rgb)
        res_f = face_mesh.process(rgb)

        #–– Inicializar vectores ––
        m1 = [0.0]*42; m2 = [0.0]*42; r = [0.0]*36

        # —— Extraer manos ——
        if res_h.multi_hand_landmarks:
            for i, hand in enumerate(res_h.multi_hand_landmarks):
                pts = [lm.x for lm in hand.landmark] + [lm.y for lm in hand.landmark]
                if len(pts)==42:
                    (m1 if i==0 else m2)[:] = pts
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # —— Extraer rostro ——
        if res_f.multi_face_landmarks:
            face = res_f.multi_face_landmarks[0]
            idxs = [33,263,61,291,199,4,0,17,267,37,164,393,78,308,13,14,312,82]
            coords = []
            for idx in idxs:
                lm = face.landmark[idx]
                coords += [lm.x, lm.y]
            if len(coords)==36:
                r[:] = coords
            spec = mp_drawing.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=1)
            for grp in FACE_CONNECTIONS:
                mp_drawing.draw_landmarks(frame, face, grp,
                    landmark_drawing_spec=None, connection_drawing_spec=spec)

        # —— Lograr que empiece SOLO con manos ——
        if any(m1) or any(m2):
            buffer.append(m1 + m2 + r)
            if len(buffer)>FRAMES_LSTM:
                buffer.pop(0)

        # —— Predicción cuando buffer lleno y cooldown ——
        now = time.time()
        if len(buffer)==FRAMES_LSTM and (now - last_time)>=COOLDOWN_SEC:
            inp = np.array(buffer).reshape(1, FRAMES_LSTM, 120).astype(np.float32)
            pred = modelo.predict(inp, verbose=0)
            palabra = clases[np.argmax(pred)]
            last_time = now
            buffer.clear()

            if batch_mode:
                batch.append(palabra)
            else:
                # Modo instantáneo
                voz.say(palabra); voz.runAndWait()
                txt = quitar_acentos(palabra).capitalize()
                cv2.putText(frame, txt, (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0,255,0), 3)

        # —— Mostrar frase parcial en batch ——
        if batch_mode and batch:
            frase = " ".join(quitar_acentos(w).capitalize() for w in batch)
            cv2.putText(frame, frase, (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 2)

        cv2.imshow("LSM → Texto/Voz", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('3'), ord('4'), ord('5'), ord('q')):
            tecla = chr(key); break

    cap.release()
    cv2.destroyAllWindows()

    # Al cerrar batch con '5', reproducir toda la frase
    if batch_mode and tecla=='5' and batch:
        texto = " ".join(batch)
        voz.say(texto); voz.runAndWait()

    return tecla, len(batch)
