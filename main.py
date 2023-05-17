import cv2 
import numpy as np

# Cargar el clasificador frontal de Haar para detección de rostros utilizado en la práctica 4 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Variable para contar las imágenes capturadas
image_count = 0

# Bucle principal
while True:
    # Leer el fotograma actual desde la cámara
    ret, frame = cap.read()

    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en la imagen
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Iterar sobre los rostros detectados
    for (x, y, w, h) in faces:
        # Dibujar un rectángulo alrededor del rostro
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Recortar el rostro de la imagen original
        face_image = frame[y:y+h, x:x+w]

        # Guardar la imagen del rostro recortado
        image_count += 1
        image_filename = f'rostro_{image_count}.png'
        cv2.imwrite(image_filename, face_image)
        print(f"Imagen {image_filename} guardada.")

    # Mostrar el fotograma con los rectángulos dibujados
    cv2.imshow('Streaming de Video', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()