import cv2
import os
import numpy as np
import json

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

def autenticar(label_map):
    face_cascade = cv2.CascadeClassifier('clasificadores/haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('modelos/modelo_LBPHF.xml')

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            label_id, confidence = recognizer.predict(roi_gray)
            print(confidence)
            if confidence < 100:
                label = [k for k, v in label_map.items() if v == label_id][0]
                color = (0, 255, 0)
                text = label
            else:
                label = "Desconocido"
                color = (0, 0, 255)
                text = label

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.putText(frame, f'Presione ESC para salir', (10, 20), 2, 0.5, (128, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow(f'Sistema de Reconocimiento Facial', frame)

        if cv2.waitKey(1) == 27 :
            break

    video_capture.release()
    cv2.destroyAllWindows()


# Entrenar el modelo (ejecutar solo una vez)
label_map = entrenar_modelo()

# Guardar label_map en un archivo JSON
guardar_label_map(label_map)

# Autenticar a partir del modelo entrenado
# Cargar label_map desde el archivo JSON
label_map = cargar_label_map()
autenticar(label_map)