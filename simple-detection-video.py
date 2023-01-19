import time

import cv2
import face_recognition as fr

IMAGE_FOLDER = "assets"

video = cv2.VideoCapture(rf"{IMAGE_FOLDER}\videos\manoel-gomes.mp4")
pTime = 0
face_locations = []
while True:
    # Lê o próximo quadro do vídeo
    ret, frame = video.read()
    cTime = time.time()

    # Verifica se chegamos ao final do vídeo
    if not ret:
        break

    # Redimensiona tamanho do video
    frame_small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame_small[:, :, ::-1]

    # Find all the faces in the current frame of video
    face_locations = fr.face_locations(rgb_frame)

    # Display the results
    for top, right, bottom, left in face_locations:
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    # Display the resulting image
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow('Video', frame)
    cv2.waitKey(2)
# Libera os recursos
video.release()
cv2.destroyAllWindows()
