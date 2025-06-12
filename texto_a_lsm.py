import cv2
import os

ruta_videos = "dataBaseVideos"  # Carpeta donde están tus videos

def obtener_palabras_validas(frase, archivos_video):
    palabras = frase.lower().strip().split()
    resultado = []
    i = 0
    while i < len(palabras):
        encontrado = False
        for j in range(len(palabras), i, -1):
            fragmento = " ".join(palabras[i:j])
            if f"{fragmento}.mp4" in archivos_video:
                resultado.append(fragmento)
                i = j
                encontrado = True
                break
        if not encontrado:
            print(f"⚠️ No se encontró video para: {' '.join(palabras[i:i+1])}")
            i += 1
    return resultado

# Obtener archivos de video disponibles
archivos_disponibles = [archivo.lower() for archivo in os.listdir(ruta_videos) if archivo.endswith(".mp4")]

while True:
    entrada = input("\n📝 Escribe una frase (o 'salir' para volver al menú): ").strip().lower()
    if entrada == "salir":
        break

    frases_video = obtener_palabras_validas(entrada, archivos_disponibles)

    for palabra in frases_video:
        nombre_video = os.path.join(ruta_videos, f"{palabra}.mp4")
        cap = cv2.VideoCapture(nombre_video)

        if not cap.isOpened():
            print(f"⚠️ No se pudo abrir el video: '{nombre_video}'")
            continue

        print(f"▶️ Reproduciendo: {palabra}.mp4 (presiona Q para interrumpir)")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (700, 700))
            cv2.imshow("Video LSM", frame)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
