import os
# Silenciar logs de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=INFO,1=WARNING,2=ERROR,3=FATAL

# Silenciar logs de absl (usados por MediaPipe y TensorFlow)
import logging
logging.getLogger('absl').setLevel(logging.ERROR)
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)

# Importar módulos necesarios
import cv2, os, unicodedata, re

RUTA_VIDEOS = "dataBaseVideos"

# Quita acentos y puntuación
def normalize(txt):
    s = unicodedata.normalize('NFKD', txt)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^\w\s]", "", s)  # elimina signos (?,¡!,:;)
    return s.lower().strip()

def run_text_to_lsm():
    """
    Pide al usuario una frase, busca en dataBaseVideos los mp4
    (ignorando acentos y símbolos), y los reproduce.
    Devuelve:
      '3' si el usuario pulsa esa tecla para volver al LSM,
      'q' si decide salir.
    """
    # Cargar y normalizar lista de videos existentes (sin .mp4)
    files = [f for f in os.listdir(RUTA_VIDEOS) if f.lower().endswith(".mp4")]
    base, norm_map = {}, {}
    for f in files:
        name = f[:-4]  # sin .mp4
        norm = normalize(name)
        base[norm] = f

    while True:
        frase = input("\nEscribe texto (o '3' para LSM, 'q' para salir): ").strip()
        if frase.lower() in ('3','q'):
            return frase.lower()
        # Normalizar entrada y dividir
        words = normalize(frase).split()
        i, seq = 0, []
        while i < len(words):
            found = False
            # Buscar fragmentos largos primero
            for j in range(len(words), i, -1):
                frag = " ".join(words[i:j])
                if frag in base:
                    seq.append(base[frag]); i = j; found = True; break
            if not found:
                print(f"⚠️ No hay video para: {words[i]}")
                i += 1

        # Reproducir la secuencia
        for fn in seq:
            path = os.path.join(RUTA_VIDEOS, fn)
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                print(f"❌ No abre: {fn}"); continue
            print(f"▶️ Reproduciendo {fn}")
            while True:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.resize(frame, (700,700))
                cv2.imshow("Texto → LSM", frame)
                k = cv2.waitKey(30) & 0xFF
                if k in (ord('q'), ord('3')):
                    cap.release(); cv2.destroyAllWindows()
                    return chr(k)
            cap.release()
            cv2.destroyAllWindows()

        print("✅ ¡Frase completa reproducida!")
