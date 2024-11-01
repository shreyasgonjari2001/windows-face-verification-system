from architecture import *
import cv2
import numpy as np
import tensorflow as tf


class Verifier:
    def __init__(self, weights_path):
        self.face_extraction = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.facenet = InceptionResNetV2().load_weights(weights_path)

    def get_embedding(self, image_path, use_as_ref=False):
        faces_s = []
        embeddings = []
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_extraction.detectMultiScale(gray_image, 1.3, 5)
        for (x, y, w, h) in faces:
            img = image[y:y+h, x:x+w]
            gray_img = gray_image[y:y+h, x:x+w]

            eyes = self.eye_detector.detectMultiScale(gray_img)

            if len(eyes) >= 2:
                faces_s.append(img)

        for i in faces:
            frame = np.expand_dims(i, axis=0)/127 -1
            frame = tf.keras.preprocessing.image.smart_resize(frame, (160, 160))
            ebd = self.facenet.predict_on_batch(frame)
            embeddings.append(ebd/np.linalg.norm(ebd, ord=2))

        if use_as_ref:
            assert len(faces_s) == 1, "Provide image with one number of faces"
            

        return embeddings
