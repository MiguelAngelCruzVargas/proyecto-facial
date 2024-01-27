import cv2
import mediapipe as mp

# Inicializar el detector de MediaPipe para la detección y seguimiento facial
mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)





# Configurar la captura de video
cap = cv2.VideoCapture(1)  # Usar la cámara por defecto

# Variable para controlar si se ha detectado movimiento de cabeza
head_moved = False

# Variables para almacenar la posición anterior de la nariz
prev_nose_y = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No se puede capturar el fotograma de la cámara.")
        break

    # Convertir la imagen al formato de MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Realizar la detección de rostros con MediaPipe
    results_detection = mp_face_detection.process(rgb_frame)

    if results_detection.detections:
        for detection in results_detection.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            
            # Dibujar el cuadro delimitador del rostro
            cv2.rectangle(frame, bbox, (255, 0, 0), 2)

            # Convertir la imagen a escala de grises para el seguimiento de rostros
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Realizar el seguimiento facial con MediaPipe
            results_mesh = mp_face_mesh.process(rgb_frame)

            if results_mesh.multi_face_landmarks:
                for face_landmarks in results_mesh.multi_face_landmarks:
                    # Extraer las coordenadas del punto de la nariz para detectar el movimiento de la cabeza
                    nose_landmark = face_landmarks.landmark[2]
                    nose_y = int(nose_landmark.y * ih)

                    # Si la posición de la nariz cambia significativamente, se considera un movimiento de cabeza
                    if abs(nose_y - prev_nose_y) > 5:  # Ajustar el umbral según la sensibilidad deseada
                        head_moved = True
                    else:
                        head_moved = False

                    prev_nose_y = nose_y

                    # Dibujar un punto en la nariz para visualización
                    cv2.circle(frame, (int(nose_landmark.x * iw), nose_y), 2, (0, 255, 0), -1)

                    # Dibujar la malla facial
                    for face_landmarks in results_mesh.multi_face_landmarks:
                        for landmark in face_landmarks.landmark:
                            x = int(landmark.x * iw)
                            y = int(landmark.y * ih)
                            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                    # Mostrar mensaje si no se detecta ningún rostro o si se detecta movimiento de cabeza
                    if not results_mesh.multi_face_landmarks:
                        cv2.putText(frame, 'Rostro no detectado', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    elif head_moved:
                        cv2.putText(frame, 'Movimiento de cabeza detectado. Autenticación exitosa', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
    # Mostrar el resultado
    cv2.imshow('Facial Recognition and Tracking', frame)

    # Salir del bucle si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura de video y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
