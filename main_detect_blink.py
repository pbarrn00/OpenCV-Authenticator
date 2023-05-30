import dlib
import cv2
from imutils import face_utils
import numpy as np
import json
import time
import os

def guardar_label_map(label_map):
    with open('json/label_map.json', 'w') as file:
        json.dump(label_map, file)

def cargar_label_map():
    with open('json/label_map.json', 'r') as file:
        label_map = json.load(file)
    return label_map

def detectar_parpadeo(roi_gray, shape):
    ojo_izquierdo = shape[36:42]
    ojo_derecho = shape[42:48]

    ratio_izquierdo = calcular_ratio_parpadeo(ojo_izquierdo)
    ratio_derecho = calcular_ratio_parpadeo(ojo_derecho)

    ratio_parpadeo = (ratio_izquierdo + ratio_derecho) / 2

    if ratio_parpadeo < 0.18:
        return True
    else:
        return False

def calcular_ratio_parpadeo(ojo):
    a = np.linalg.norm(ojo[1] - ojo[5])
    b = np.linalg.norm(ojo[2] - ojo[4])
    c = np.linalg.norm(ojo[0] - ojo[3])

    ratio = (a + b) / (2.0 * c)
    return ratio

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
    recognizer.save('modelos/modelo_LBPHF_advanced.xml')
    print("Modelo entrenado y guardado con éxito.")

    return label_map

def autenticar(label_map):
    face_cascade = cv2.CascadeClassifier('clasificadores/haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('modelos/modelo_LBPHF_advanced.xml')

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape/shape_predictor_68_face_landmarks.dat')

    video_capture = cv2.VideoCapture(1)

    tiempo_parpadeo = 0.3  # Tiempo en segundos para considerar un parpadeo
    tiempo_estatico = 5.0  # Tiempo en segundos para considerar una imagen estática

    tiempo_inicial_parpadeo = time.time()
    tiempo_inicial_estatico = time.time()
    parpadeo_detectado = False
    rostro_estatico = False

    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            if rostro_estatico:
                tiempo_actual_estatico = time.time()
                tiempo_transcurrido_estatico = tiempo_actual_estatico - tiempo_inicial_estatico
                if tiempo_transcurrido_estatico >= tiempo_estatico:
                    label = "Imagen estática"
                    color = (255, 0, 0)
                    text = label
                    rostro_estatico = True

                    # Mostrar el recuadro azul de "Imagen estática"
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), color, cv2.FILLED)
                    cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        else:
            rostro_estatico = False
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]

                rects = detector(gray, 0)

                for rect in rects:
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)

                    ojo_izquierdo = shape[36:42]
                    ojo_derecho = shape[42:48]

                    ojo_izquierdo_hull = cv2.convexHull(ojo_izquierdo)
                    ojo_derecho_hull = cv2.convexHull(ojo_derecho)

                    cv2.drawContours(frame, [ojo_izquierdo_hull], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [ojo_derecho_hull], -1, (0, 255, 0), 1)

                    if not detectar_parpadeo(roi_gray, shape):
                        tiempo_inicial_parpadeo = time.time()  # Reiniciar el contador de tiempo de parpadeo
                        parpadeo_detectado = True

                        label_id, confidence = recognizer.predict(roi_gray)
                        if confidence < 70:
                            label = [k for k, v in label_map.items() if v == label_id][0]
                            color = (0, 255, 0)
                            text = label
                        else:
                            label = "Desconocido"
                            color = (0, 0, 255)
                            text = label

                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                    elif parpadeo_detectado:
                        print("parpadeo detectado")
                        tiempo_actual_parpadeo = time.time()
                        tiempo_transcurrido_parpadeo = tiempo_actual_parpadeo - tiempo_inicial_parpadeo
                        if tiempo_transcurrido_parpadeo >= tiempo_parpadeo:
                            parpadeo_detectado = False

        cv2.putText(frame, f'Presione ESC para salir', (10, 20), 2, 0.5, (128, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow(f'Sistema de Reconocimiento Facial', frame)

        if cv2.waitKey(1) == 27:
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Entrenar el modelo (ejecutar solo una vez)
# label_map = entrenar_modelo()

# Guardar label_map en un archivo JSON
# guardar_label_map(label_map)

# Autenticar a partir del modelo entrenado
# Cargar label_map desde el archivo JSON
label_map = cargar_label_map()
autenticar(label_map)
