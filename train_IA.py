import cv2
import os
import sys


# Creamos una carpeta para almacenar los rostros si esta no existe
carpeta_rostros = 'rostros'

if not os.path.exists('rostros'):
    print('Carpeta creada: rostros')
    os.makedirs(carpeta_rostros, exist_ok=True)

try:
    max_images = int(sys.argv[1])  # Número máximo de imágenes que se guardarán
except IndexError:
    print('Error: No se ha proporcionado el número máximo de imágenes como parámetro.')
    sys.exit(1)

# Inicializar la cámara 
# Está inicializada a 1 para que utilice app móvil IRIUN WEBCAM
cap = cv2.VideoCapture(0)
faceClassif = cv2.CascadeClassifier('clasificadores\haarcascade_frontalface_default.xml')
count = 0
max_images = int(sys.argv[1])  # Número máximo de imágenes que se guardarán

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)
    k = cv2.waitKey(1)

    if k == 27 or count >= max_images:
        break

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (128, 0, 255), 2)
        rostro = auxFrame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)

        nombre_rostro = os.path.join(carpeta_rostros, 'rostro_{}.jpg'.format(count))
        cv2.imwrite(nombre_rostro, rostro)
        cv2.imshow('rostro', rostro)
        count = count + 1

    cv2.rectangle(frame, (10, 5), (450, 25), (255, 255, 255), -1)
    cv2.putText(frame, f'Presione ESC para salir | Imágenes guardadas: {count}/{max_images}', (10, 20), 2, 0.5, (128, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow('frame', frame)

cap.release()
cv2.destroyAllWindows()
