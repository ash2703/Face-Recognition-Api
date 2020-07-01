# Face-Recognition-API  

Utilising the face recognition library built on top of dlib to create a REST API using Flask

---
Dependencies:

* Python 3.6
* Visual Studio Build Tools
* Face-recognition
* openCV
* dlib
* cmake
* flask

---

# Setup Build Environment

* Clone the repository

    ```
    git clone https://github.com/ash2703/Face-Recognition-Api.git
    ```

    Navigate to the root of the above cloned repo.

* Install Dependencies

    [Refer this for detailed instructions](https://medium.com/analytics-vidhya/how-to-install-dlib-library-for-python-in-windows-10-57348ba1117f#:~:text=Now%20we%20can%20install%20dlib,need%20to%20install%20CMake%20library.&text=Then%2C%20you%20can%20install%20dlib%20library%20using%20pip%20install%20.&text=After%20passing%20enter%2C%20you%20laptop,run%20the%20C%2C%20C%2B%2B%20Complier)


    ```
    CMake : https://cmake.org/download/
    Visual Studio : https://visualstudio.microsoft.com/visual-cpp-build-tools/
    Add Cmake to system PATH

    pip install -r requirements.txt
    ```

* Create Face Database

    Follow this directory structure, store multiple photos of same person in a single folder

    ```bash

    face_gallery
    ├───person
    │       photo1.jpg
    │       photo2.jpg
    │
    ├───obama
    │       obama.jpg
    │
    └───trump
            donald-trump-premiere-king-kong
            trump.jpg

    ``` 

# Testing the pipleine

### **1. Face Recognition**

To run the recogniton alone

```python
python recognizer.py
```
### **2. Recogniton API**

```bash
python api.py   #Run the api
curl -XPOST -F "file=@sample/obamas.jpg" http://127.0.0.1:5001
```


