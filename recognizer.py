import face_recognition
import cv2
import numpy as np
import face_database

class Recognizer:
    def __init__(self, path, image, show = None):
        self.path = path
        self.frame = cv2.imread(image)
        self.show = show
        self.known_face_names = []
        self.known_face_encodings = []
        self.face_names = []   #detected faces in frame
        self.face_locations = []

    def processImage(self):
        small_frame = cv2.resize(self.frame, (0, 0), fx=0.25, fy=0.25)
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        return rgb_small_frame
    
    def identify(self):
        rgb_small_frame = self.processImage()
        self.face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            # matches = Recognizer.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            
            # Or instead, use the known face with the smallest distance to the new face
            face_id = Recognizer.face_distance(self.known_face_encodings, face_encoding, 0.6)
            if face_id != -1:
                name = self.known_face_names[face_id]
                print(name)

            self.face_names.append(name)
            if self.show:
                self.display()
        return self.face_locations, self.face_names

    def loadDatabase(self):
        database = face_database.FaceDatabase(self.path) 
        self.known_face_names, self.known_face_encodings = database.get_encodings()
    
    @staticmethod
    def face_distance(face_encodings, face_to_compare, tolerance):
        """
        Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
        for each comparison face. The distance tells you how similar the faces are.
        :param faces: List of face encodings to compare
        :param face_to_compare: A face encoding to compare against
        :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
        """
        if len(face_encodings) == 0:
            return np.empty((0))
        distance = np.zeros(len(face_encodings))
        for i, face_encoding in enumerate(face_encodings):
            distance[i] = (np.max(np.linalg.norm(face_encoding - face_to_compare, axis=1)))

        if np.min(distance) <= tolerance:
            return np.argmin(distance)
        else:
            return -1

    def display(self):
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(self.frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(self.frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(self.frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # Display the resulting image
            cv2.imshow('Video', frame)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def startEngine(self):
        # self.processImage()
        self.loadDatabase()
        self.identify()

if __name__ == "__main__":
    face_recognizer = Recognizer("face_gallery", "ash.jpg")
    face_recognizer.startEngine()