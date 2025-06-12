import os

def mostrar_menu():
    print("\n🎬 Bienvenido al Traductor Bidireccional LSM")
    print("1. Traducir señas (LSM) a texto y voz")
    print("2. Traducir texto a LSM (videos)")
    print("q. Salir")

def ejecutar_menu():
    while True:
        mostrar_menu()
        opcion = input("\n📝 Selecciona una opción (1/2/q): ").strip().lower()

        if opcion == '1':
            os.system("python lsm_a_texto.py")
        elif opcion == '2':
            os.system("python texto_a_lsm.py")
        elif opcion == 'q':
            print("👋 Saliendo del sistema. ¡Hasta luego!")
            break
        else:
            print("⚠️ Opción inválida. Intenta nuevamente.")

if __name__ == "__main__":
    ejecutar_menu()