import math
import os

import cv2
import face_recognition as fr
import numpy as np

IMAGE_FOLDER = "assets/images"
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)


def face_confidence(face_distance, face_match_threshold=0.6):
    escala = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (escala * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) *
                 math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'


def load_image(path):
    image = fr.load_image_file(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    encode = fr.face_encodings(image)

    return image, encode


def draw_rectangle(image, face_locations, face_names):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(image, (left, top),
                      (right, bottom), COLOR_RED, 2)
        cv2.rectangle(image, (left, bottom - 35),
                      (right, bottom), COLOR_RED, -1)
        cv2.putText(image, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, COLOR_WHITE, 1)


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


def showImage(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)


class FaceRecognitionImage:
    face_encodings = []
    know_face_names = []
    know_face_encodings = []
    face_locations = []
    face_names = []

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir(f'{IMAGE_FOLDER}/know'):
            face_image, face_enconding = load_image(
                f"{IMAGE_FOLDER}/know/{image}")

            self.know_face_encodings.append(face_enconding[0])
            self.know_face_names.append(image)

    def run(self):
        image, face_encodings = load_image(
            f"{IMAGE_FOLDER}/unknown/BillElonTest.jpg")
        self.face_locations = fr.face_locations(image)

        for face_encoding in face_encodings:
            name, confidence = compare_min_distance(
                self.know_face_encodings, self.know_face_names, face_encoding)

            self.face_names.append(f"{name} ({confidence})")

        draw_rectangle(image, self.face_locations, self.face_names)

        showImage("Other peoples", image)


if __name__ == "__main__":
    faceRecognitionImage = FaceRecognitionImage()
    faceRecognitionImage.run()
