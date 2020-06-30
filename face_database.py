import face_recognition
import os
import os.path as osp
import numpy as np
import cv2


class FaceDatabase:
    IMAGE_EXTENSIONS = ['jpg', 'png']

    def __init__(self, path):
        if not osp.isdir(path):
            log.error("Wrong face images database path. Expected a " \
                      "path to the directory containing %s files, " \
                      "but got '%s'" % \
                      (" or ".join(self.IMAGE_EXTENSIONS), path))

        self.encodings = [[] for i in range(len(os.listdir(path)))]
        self.labels = []
        for dirname, _, filenames in os.walk(path):
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
                print("f")
                try:
                    encoding = face_recognition.face_encodings(image)[0]
                    self.encodings[_id].append(encoding)
                    print("success{}".format(label), _id)
                except Exception as e:
                    # print("error ", e)
                    log.error("error raised {}".format(e))
        # return self.labels, self.encodings
    def get_encodings(self):
        return self.labels, self.encodings



if __name__ == "__main__":
    face_database = FaceDatabase("face_gallery")