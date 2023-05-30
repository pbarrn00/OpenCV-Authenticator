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

def detectar_movimiento(frames, umbral, area_minima):

    # Calcular la diferencia acumulativa entre los frames
    diff_acumulada = np.zeros_like(frames[0], dtype=np.float32)
    for frame in frames:
        diff_frame = cv2.absdiff(frame, frames[0])
        diff_acumulada += diff_frame.astype(np.float32)
    
    # Aplicar un umbral para obtener una imagen binaria del movimiento
    umbralizado = cv2.threshold(diff_acumulada, umbral, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
    
    # Encontrar los contornos de los objetos en movimiento
    _, contornos, _ = cv2.findContours(umbralizado, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Verificar si se detecta movimiento significativo
    movimiento_detectado = False
    for contorno in contornos:
        if cv2.contourArea(contorno) > area_minima:
            movimiento_detectado = True
            break
    
    return movimiento_detectado

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
    recognizer.read('modelos/modelo_LBPHF.xml')

    video_capture = cv2.VideoCapture(1)

    # Parámetros de ajuste
    n_frames = 7  # Número de frames a acumular
    umbral = 30  # Ajusta este valor según tus necesidades
    area_minima = 100  # Ajusta este valor según tus necesidades
    
    # Leer el primer frame
    _, frame = video_capture.read()

    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Inicializar la lista de frames acumulados
    frames_acumulados = [gray] * n_frames


    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Agregar el frame actual a la lista de frames acumulados
        frames_acumulados.append(gray)
        
        # Mantener solo los últimos n_frames en la lista
        frames_acumulados = frames_acumulados[-n_frames:]
        
        # Comprobar si hay movimiento significativo
        hay_movimiento = detectar_movimiento(frames_acumulados, umbral, area_minima)
        
        # Mostrar el resultado
        if hay_movimiento:
            print("Persona real detectada")
        else:
            print("Imagen estática o movimiento insignificante")

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))


        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            label_id, confidence = recognizer.predict(roi_gray)
            #print(confidence)
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

        cv2.putText(frame, f'Presione ESC para salir', (10, 20), 2, 0.5, (128, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow(f'Sistema de Reconocimiento Facial', frame)

        if cv2.waitKey(1) == 27 :
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