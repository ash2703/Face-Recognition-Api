import face_recognition
import os
import os.path as osp
import numpy as np
import cv2
import pickle
import hashlib


class FaceDatabase:
    IMAGE_EXTENSIONS = ['jpg', 'png']

    def __init__(self, path):
        self.path = path
        self.encodings = [[] for i in range(len(os.listdir(path)))]
        self.labels = []
        self.checksum = self.get_dir_md5()

    def encode(self):
        if not osp.isdir(self.path):
            raise FileNotFoundError
        for dirname, _, filenames in os.walk(self.path):
            files = [osp.join(dirname, f) for f in os.listdir(dirname) \
                      if f.split('.')[-1] in self.IMAGE_EXTENSIONS]
            label = ""
            if len(files) > 0: label = osp.basename(dirname)
            else: 
                continue
            _id = len(self.labels)
            self.labels.append(label)

            for filename in files:
                image = face_recognition.load_image_file(filename)
                print(image.shape)
                assert len(image.shape) == 3, \
                    "Expected an input image in (H, W, C) format"
                assert image.shape[2] in [3, 4], \
                    "Expected BGR or BGRA input"
                # print("f")
                try:
                    encoding = face_recognition.face_encodings(image)[0]
                    self.encodings[_id].append(encoding)
                    print("success{}".format(label), _id)
                except Exception as e:
                    raise Exception
                    # log.error("error raised {}".format(e))
        self.serializeDatabase()
        return self.labels, self.encodings
    
    def get_encodings(self):
        db = self.filesModified()
        if db:
            return db[1:]  #hash, labels, encodings
        else:    #database modified retrain faces
            return self.encode()
    
    def databaseExists(self):
        if osp.exists("database") :
            return True
        return False

    def serializeDatabase(self):
        dbfile = open('database', 'wb') 
        # source, destination 
        pickle.dump((self.checksum, self.labels, self.encodings), dbfile)                     
        dbfile.close() 

    def deSerializeDatabase(self):
        dbfile = open('database', 'rb')
        db = pickle.load(dbfile)
        # print(db)
        dbfile.close()
        return db 

    def filesModified(self):
        if self.databaseExists():
            db = self.deSerializeDatabase()
            # print(db[0], self.checksum)
            if db[0] == self.checksum:   #hash matches
                return db
            # return db[0]
        return False


    def get_dir_md5(self):
        hash = hashlib.md5()
        for dirpath, dirnames, filenames in os.walk(self.path, topdown=True):

            dirnames.sort(key=osp.normcase)
            filenames.sort(key=osp.normcase)

            for filename in filenames:
                filepath = osp.join(dirpath, filename)

                f = open(filepath, 'rb')
                for chunk in iter(lambda: f.read(65536), b''):
                    hash.update(chunk)

        return hash.hexdigest()

if __name__ == "__main__":
    face_database = FaceDatabase("face_gallery")
    labels, encoding = face_database.get_encodings()
    print(labels)
    print(len(encoding), len(encoding[0][0]))
    # face_database.deSerializeDatabase()