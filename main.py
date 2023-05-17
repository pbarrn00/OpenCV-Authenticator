import cv2 
import numpy as np
import sys
import os

try:
    max_images = int(sys.argv[1])  # Número máximo de imágenes que se guardarán
except IndexError:
    print('Error: No se ha proporcionado el número máximo de imágenes como parámetro.')
    sys.exit(1)
    
# Creamos una carpeta para almacenar los rostros si esta no existe
carpeta_rostros = 'rostros'

if not os.path.exists('rostros'):
    print('Carpeta creada: rostros')
    os.makedirs(carpeta_rostros, exist_ok=True)

# Cargar el clasificador frontal de Haar para detección de rostros utilizado en la práctica 4 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Inicializar la cámara
video = cv2.VideoCapture(0)

# Variable para contar las imágenes capturadas
image_count = 0
max_images = int(sys.argv[1])  # Número máximo de imágenes que se guardarán

# Bucle principal
while True:
    # Leer el fotograma actual desde la cámara
    ret, frame = video.read()

    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en la imagen
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    k = cv2.waitKey(1)

    if k == 27 or image_count >= max_images:
        break

    # Iterar sobre los rostros detectados
    for (x, y, w, h) in faces:
        # Dibujar un rectángulo alrededor del rostro
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Recortar el rostro de la imagen original
        face_image = frame[y:y+h, x:x+w]

        # Guardar la imagen del rostro recortado

        if k == ord('s'):
            image_filename = os.path.join(carpeta_rostros, 'rostro_{}.jpg'.format(image_count))
            image_count += 1
            cv2.imwrite(image_filename, face_image)
            print(f"Imagen {image_filename} guardada.")
            cv2.imshow('Rostros', face_image)

    # Mostrar el fotograma con los rectángulos dibujados
    #cv2.rectangle(frame, (10, 5), (450, 25), (255, 255, 255), -1)

    # Construir el texto a mostrar
    text = f'Presione ESC para salir\nPresione S para guardar imagenes.\nImagenes guardadas: {image_count}/{max_images}'

    # Mostrar el texto en varias líneas con fondo blanco
    lines = text.split('\n')
    y_pos = 20
    line_height = 20
    rectangle_padding = 5
    for line in lines:
        # Obtener las dimensiones del texto
        (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Dibujar el rectángulo de fondo blanco
        cv2.rectangle(frame, (10, y_pos - h - rectangle_padding), (10 + w + 2*rectangle_padding, y_pos + rectangle_padding), (255, 255, 255), cv2.FILLED)

        # Mostrar el texto
        cv2.putText(frame, line, (10 + rectangle_padding, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 255), 1, cv2.LINE_AA)

        y_pos += line_height    

    cv2.imshow('Streaming de Video', frame)
    
# Liberar los recursos
video.release()
cv2.destroyAllWindows()
