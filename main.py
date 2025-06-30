import os
import sys
import logging

# Silenciar logs de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0''1''2'  # 0=INFO,1=WARNING,2=ERROR,3=FATAL

# Silenciar logs de absl (usados por MediaPipe y TensorFlow)
logging.getLogger('absl').setLevel(logging.ERROR)
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)

# Importar módulos necesarios
from lsm_a_texto import run_lsm_to_text
from texto_a_lsm import run_text_to_lsm

MENU = """
=======================================
     TwoS Traductor LSM v1.1
=======================================
1) LSM → Texto y Voz
2) Texto → LSM (videos)
m) Menú principal
q) Salir
"""

def mostrar_menu():
    print(MENU)

def main():
    mode = 'menu'   # 'menu', 'lsm' o 'text'
    batch = False   # Sólo válido en modo 'lsm'

    while True:
        if mode == 'menu':
            mostrar_menu()
            sel = input("Selecciona (1/2/m/q): ").strip().lower()
            if sel == '1':
                mode = 'lsm'
                batch = False
            elif sel == '2':
                mode = 'text'
            elif sel == 'm':
                continue
            elif sel == 'q':
                sys.exit(0)
            else:
                print("⚠️ Opción inválida.")
            continue

        if mode == 'lsm':
            # Modo LSM → Texto/Voz
            tecla, cnt = run_lsm_to_text(batch)
            if tecla == '3':
                mode = 'text'    # Conmutar a Texto→LSM
            elif tecla == '4':
                batch = True     # Iniciar modo batch
            elif tecla == '5':
                batch = False    # Finalizar batch
            elif tecla == 'm':
                mode = 'menu'    # Volver al menú principal
            elif tecla == 'q':
                sys.exit(0)
            # Sino, sigo en mismo modo
            continue

        if mode == 'text':
            # Modo Texto → LSM
            tecla = run_text_to_lsm()
            if tecla == '3':
                mode = 'lsm'     # Conmutar a LSM→Texto
            elif tecla == 'm':
                mode = 'menu'    # Volver al menú principal
            elif tecla == 'q':
                sys.exit(0)
            # Sino, sigo en mismo modo
            continue

if __name__ == "__main__":
    main()

