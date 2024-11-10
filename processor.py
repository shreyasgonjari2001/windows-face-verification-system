from core.architecture import *
import cv2
import numpy as np
import tensorflow as tf
import sqlite3
from concurrent.futures import ThreadPoolExecutor


class Verifier:
    def __init__(self, weights_path="models/facenet_keras_weights.h5"):
        self.face_extraction = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        try:
            self.facenet = InceptionResNetV2()
            print("Model initialized successfully.")
        except Exception as e:
            print(f"Error initializing model: {e}")
            self.facenet = None
        self.db_data = None
        self.conn = sqlite3.connect("employee_embeddings.db")
        self.cursor = self.conn.cursor()
        self.initialize_db()
        

    def __enter__(self):
        return self
    

    def get_db_data(self, update=False):
        self.cursor.execute("""
                            SELECT name, embedding
                            FROM EmployeeEmbeddings""")
        if update:
            self.db_data = self.cursor.fetchall()
            return
        return self.cursor.fetchall()


    def get_facenet_summary(self):
        return self.facenet.summary()

    def get_embedding(self, image, from_array=False):
        faces_s = []
        embeddings = []

        if not from_array:
            image = cv2.imread(image)
        
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_extraction.detectMultiScale(gray_image, 1.3, 5)
        for (x, y, w, h) in faces:
            img = image[y:y+h, x:x+w]
            gray_img = gray_image[y:y+h, x:x+w]

            eyes = self.eye_detector.detectMultiScale(gray_img)

            if len(eyes) >= 2:
                faces_s.append(img)

        for i in faces_s:
            frame = np.expand_dims(i, axis=0)/127 -1
            frame = tf.keras.preprocessing.image.smart_resize(frame, (160, 160))
            ebd = self.facenet.predict_on_batch(frame)
            embeddings.append(ebd/np.linalg.norm(ebd, ord=2))
            
        return embeddings
    

    def get_euclidean_dist(self, array1, array2):
        assert type(array1) == np.ndarray and type(array2) == np.ndarray, "please provide numpy array"
        return np.linalg.norm(array1 - array2)


    def __exit__(self, exc_type, exc_value, traceback):
        if self.conn:
            self.conn.close()
            # print("connection is closed")
        if exc_type:
            print(f"error occured: {exc_type}")
        return False

    
    def initialize_db(self):
        self.cursor.execute("""CREATE TABLE IF NOT EXISTS EmployeeEmbeddings(
                        employeeId INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE,
                        embedding BLOB,
                        path TEXT UNIQUE
                        )""")
        self.conn.commit()
    

    def store_reference(self, ref, name):
        assert iter(ref), "Argument 'data' must be iterable"
        assert type(name) == str
        blob_ref = ref.tobytes()
        self.cursor.execute("""INSERT INTO EmployeeEmbeddings(name, embedding)
                            VALUES (?, ?)
                         """, (name, ref))
        self.conn.commit()


    def get_ref_data(self, name):
        self.cursor.execute("""
                            SELECT embedding
                            FROm EmployeeEmbeddings
                            WHERE name = ?""", (name, ))
        self.conn.commit()
        result = self.cursor.fetchone()[0]
        if result:
            return np.frombuffer(result, dtype=np.float32)
        else:
            return result

      
    def identify(self, ebd, threshold=0.7):
        if not self.db_data:
            self.get_db_data(update=True)
        identity = None
        min_dist = float("inf")
        for n, e in self.db_data:
            e_np = np.frombuffer(e, np.float32)
            dist = self.get_euclidean_dist(e_np, ebd)
            if dist < min_dist:
                min_dist = dist
                if min_dist<threshold:
                    identity = n
        return identity, min_dist

        
