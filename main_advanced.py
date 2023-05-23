import cv2
import numpy as np
import sys
import os

# Cargar el clasificador frontal de Haar para detección de rostros utilizado en la práctica 4
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Inicializar la cámara 
# Está inicializada a 1 para que utilice app móvil IRIUN WEBCAM
video = cv2.VideoCapture(1)

# Definir y cargar usuarios registrados
usuarios_dict = {}  # Diccionario para mapear nombres de usuarios a etiquetas
usuarios_imagenes = []
usuarios_etiquetas = []
carpeta_usuarios = 'usuarios_registrados'

usuarios = os.listdir(carpeta_usuarios)
for i, usuario in enumerate(usuarios):
    usuarios_dict[usuario] = i

    carpeta_usuario = os.path.join(carpeta_usuarios, usuario)
    if os.path.isdir(carpeta_usuario):
        for imagen_nombre in os.listdir(carpeta_usuario):
            imagen_path = os.path.join(carpeta_usuario, imagen_nombre)
            imagen = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)
            usuarios_imagenes.append(imagen)
            usuarios_etiquetas.append(i)

# Crear el modelo de reconocimiento facial
modelo = cv2.face.LBPHFaceRecognizer_create()
modelo.train(usuarios_imagenes, np.array(usuarios_etiquetas, dtype=np.int32))

# Variables para contar las imágenes capturadas y el usuario identificado
image_count = 0
usuario_identificado = None

# Variables para la detección de movimiento
prev_frame = None
static_threshold = 5000  # Umbral para considerar que la imagen es estática

# Bucle principal
while True:
    # Leer el fotograma actual desde la cámara
    ret, frame = video.read()

    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en la imagen
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    k = cv2.waitKey(1)

    if k == 27:
        break

    # Verificar si hay movimiento en la imagen
    if prev_frame is not None:
        diff_frame = cv2.absdiff(prev_frame, gray)
        motion_area = np.sum(diff_frame > 30)
        if motion_area > static_threshold:
            usuario_identificado = None
            prev_frame = None
            continue

    prev_frame = gray.copy()

    # Iterar sobre los rostros detectados
    for (x, y, w, h) in faces:

        # Recortar el rostro de la imagen original
        face_image = gray[y:y+h, x:x+w]

        # Realizar la predicción del usuario
        etiqueta_predicha, confianza = modelo.predict(face_image)

        # Buscar el nombre del usuario en base a la etiqueta predicha
        nombre_usuario = [usuario for usuario, etiqueta in usuarios_dict.items() if etiqueta == etiqueta_predicha][0]

        # Dibujar un rectángulo alrededor del rostro
        if usuario_identificado == nombre_usuario:
            color_rectangulo = (0, 255, 0)  # Color verde para usuarios identificados
        else:
            color_rectangulo = (0, 0, 255)  # Color rojo para usuarios desconocidos

        cv2.rectangle(frame, (x, y), (x+w, y+h), color_rectangulo, 2)

        if confianza < 70:  # Umbral de confianza para la identificación del usuario
            usuario_identificado = nombre_usuario
            color_rectangulo = (0, 255, 0)  # Color verde para usuarios registrados
        else:
            usuario_identificado = None
            color_rectangulo = (0, 0, 255)  # Color rojo para usuarios no registrados

        # Mostrar el nombre del usuario en la interfaz gráfica
        cv2.putText(frame, usuario_identificado or "Desconocido", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_rectangulo, 1, cv2.LINE_AA)
        
    cv2.putText(frame, f'Presione ESC para salir', (10, 20), 2, 0.5, (128, 0, 255), 1, cv2.LINE_AA)
    # Mostrar el fotograma con los rectángulos dibujados
    cv2.imshow('Streaming de Video', frame)

# Liberar los recursos
video.release()
cv2.destroyAllWindows()
