import math
import os

import cv2
import face_recognition as fr
import numpy as np


def face_confidence(face_distance, face_match_threshold=0.6):
    escala = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (escala * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) *
                 math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'


# Display annotations
def draw_rectangle(frame, face_locations, face_names):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top),
                      (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35),
                      (right, bottom), (0, 0, 255), -1)
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)


def compare_min_distance(know_face_encodings, know_face_names, face_encoding):
    def remove_ext(name): return name.split(".")[0]

    name = 'Unknown'
    confidence = 'Unknown'
    matches = fr.compare_faces(
        know_face_encodings, face_encoding
    )
    face_distances = fr.face_distance(
        know_face_encodings, face_encoding
    )
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = know_face_names[best_match_index]
        confidence = face_confidence(face_distances[best_match_index])

    return remove_ext(name), confidence


class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    know_face_encodings = []
    know_face_names = []
    process_current_frame = True

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir('assets/images/know'):
            face_image = fr.load_image_file(f"assets/images/know/{image}")
            face_enconding = fr.face_encodings(face_image)[0]

            self.know_face_encodings.append(face_enconding)
            self.know_face_names.append(image)
        print(self.know_face_names)

    def run_recognition(self):
        video_capture = cv2.VideoCapture("assets/videos/manoel-gomes.mp4")

        while True:
            ret, frame = video_capture.read()

            # Verifica se chegamos ao final do vídeo
            if not ret:
                break

            if self.process_current_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame[:, :, ::-1]

                # find all the faces in current frame
                self.face_locations = fr.face_locations(rgb_small_frame)
                self.face_encodings = fr.face_encodings(
                    rgb_small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    name, confidence = compare_min_distance(
                        self.know_face_encodings, self.know_face_names, face_encoding)

                    self.face_names.append(
                        f"{name} ({confidence})")

            self.process_current_frame = not self.process_current_frame

            draw_rectangle(frame, self.face_locations, self.face_names)

            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    faceRecognition = FaceRecognition()
    faceRecognition.run_recognition()
