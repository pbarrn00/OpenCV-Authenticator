import cv2
import os
import numpy as np
import json
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils
import time

def guardar_label_map(label_map):
    with open('json/label_map.json', 'w') as file:
        json.dump(label_map, file)

def cargar_label_map():
    with open('json/label_map.json', 'r') as file:
        label_map = json.load(file)
    return label_map

def entrenar_modelo():
    data_dir = 'usuarios_registrados'
    image_paths = []
    labels = []
    label_id = 0
    label_map = {}

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                image_path = os.path.join(root, file)
                label = os.path.basename(root)

                if label not in label_map:
                    label_map[label] = label_id
                    label_id += 1

                image_paths.append(image_path)
                labels.append(label_map[label])

    face_images = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_images.append(gray)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(face_images, np.array(labels, dtype=np.int32))
    recognizer.save('modelos/modelo_LBPHF.xml')
    print("Modelo entrenado y guardado con éxito.")

    return label_map

def calcular_aspecto_ojo(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    aspect_ratio = (A + B) / (2.0 * C)
    return aspect_ratio

def detectar_parpadeo(roi_gray, shape, umbral):
    left_eye = shape[36:42]
    right_eye = shape[42:48]

    left_eye_aspect_ratio = calcular_aspecto_ojo(left_eye)
    right_eye_aspect_ratio = calcular_aspecto_ojo(right_eye)
    aspect_ratio = (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2.0

    if aspect_ratio < umbral:
        return False  # No se detecta parpadeo
    else:
        return True  # Se detecta parpadeo

def autenticar(label_map):
    face_cascade = cv2.CascadeClassifier('clasificadores/haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('modelos/modelo_LBPHF.xml')

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape/shape_predictor_68_face_landmarks.dat')

    video_capture = cv2.VideoCapture(1)

    umbral_parpadeo = 0.2  # Ajusta este valor para el umbral de detección de parpadeo
    tiempo_parpadeo = 2.0  # Tiempo en segundos para considerar que no hay parpadeo

    tiempo_inicio = time.time()

    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            rects = detector(gray, 0)

            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                parpadeo = detectar_parpadeo(roi_gray, shape, umbral_parpadeo)

                if parpadeo:
                    print("Parpadeo detectado")
                    tiempo_inicio = time.time()  # Reiniciar el tiempo si se detecta parpadeo

                tiempo_transcurrido = time.time() - tiempo_inicio

                if tiempo_transcurrido >= tiempo_parpadeo:
                    print("No se detecta parpadeo, imagen estatica")
                    # No se detecta parpadeo durante el tiempo determinado, considerar como imagen estática
                    color = (255, 0, 0)  # Azul
                    label = "Imagen estática"
                    text = label
                else:
                    label_id, confidence = recognizer.predict(roi_gray)
                    if confidence < 70:
                        label = [k for k, v in label_map.items() if v == label_id][0]
                        color = (0, 255, 0)  # Verde
                        text = label
                    else:
                        label = "Desconocido"
                        color = (0, 0, 255)  # Rojo
                        text = label

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.putText(frame, 'Presione ESC para salir', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 255), 1,
                    cv2.LINE_AA)
        cv2.imshow('Sistema de Reconocimiento Facial', frame)

        if cv2.waitKey(1) == 27:
            break

    video_capture.release()
    cv2.destroyAllWindows()


# Entrenar el modelo (ejecutar solo una vez)
#label_map = entrenar_modelo()

# Guardar label_map en un archivo JSON
#guardar_label_map(label_map)

# Autenticar a partir del modelo entrenado
# Cargar label_map desde el archivo JSON
label_map = cargar_label_map()
autenticar(label_map)
