import cv2
import mediapipe as mp  
import time
import os
import tensorflow as tf
import numpy as np
from pygame import mixer


mixer.init()
sound = mixer.Sound('alarm.wav')

def loadCascadingClassfiers(modelPath, labels, loadModel) :
    if loadModel :
        face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
        leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
        reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')
        lbl=labels
        model = tf.keras.models.load_model(modelPath)
        return model, face, leye, reye 


Path = 'models\cnnCat2.h5'
Label = ['Close','Open']

model, face, leye, reye = loadCascadingClassfiers(Path, Label, True)

path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
detectionStyle = "contours-style"

cap = cv2.VideoCapture(0)
previousTime = 0
with mp_face_mesh.FaceMesh(
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        height,width = image.shape[:2] 
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
        left_eye = leye.detectMultiScale(gray)
        right_eye =  reye.detectMultiScale(gray)

        cv2.rectangle(image, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

        for (x,y,w,h) in faces:
            cv2.rectangle(image, (x, y) , (x + w, y + h) , (0, 255, 0) , 2)
 
        for (x, y, w, h) in right_eye :
            r_eye = image[y : y + h, x : x + w]
            count = count + 1
            r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye, (24, 24))
            r_eye= r_eye / 255
            r_eye=  r_eye.reshape(24, 24, -1)
            r_eye = np.expand_dims(r_eye, axis = 0)
            rpred = np.argmax(model.predict(r_eye), axis = 1)
            if(rpred[0] == 1) :
                lbl = 'Open' 
            if(rpred[0] == 0) :
                lbl = 'Closed'
            break

        for (x,y,w,h) in left_eye :
            l_eye = image[y : y + h, x : x + w]
            count = count + 1
            l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)  
            l_eye = cv2.resize(l_eye, (24, 24))
            l_eye= l_eye/255
            l_eye=l_eye.reshape(24, 24, -1)
            l_eye = np.expand_dims(l_eye, axis = 0)
            lpred = np.argmax(model.predict(l_eye), axis = 1)
            if(lpred[0] == 1) :
                lbl = 'Open'   
            if(lpred[0] == 0) :
                lbl = 'Closed'
            break

        if(rpred[0] == 0 and lpred[0] == 0) :
            score = score + 1
            cv2.putText(image, "Closed", (10, height - 20), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # if(rpred[0]==1 or lpred[0]==1):
        else:
            score=score-1
            cv2.putText(image, "Open", (10, height - 20), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Checking Validation Scores
        if(score < 0) :
            score = 0   
        cv2.putText(image, 'Score:'+str(score),(100, height - 20), font, 1, (255,255,255), 1, cv2.LINE_AA)
        
        if(score > 5) :
            #person is feeling sleepy so we beep the alarm
            cv2.imwrite(os.path.join(path, 'image.jpg'), image)
            try :
                sound.play()
                
            except Exception as exp:
                print(exp)
                pass

            if(thicc < 16) :
                thicc = thicc + 2
            else :
                thicc = thicc - 2
                if(thicc < 2) :
                    thicc = 2
            cv2.rectangle(image, (0, 0),(width, height),(0, 0, 255), thicc) 

        ### Media Pipe Detection
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if detectionStyle == "mesh-only-style" :
                    mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                elif detectionStyle == "contours-style" :
                    mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                elif detectionStyle == "iris-connection-style" :
                    mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
            
            currentTime = time.time()
            FPS = 1 / (currentTime - previousTime)
            previousTime = currentTime
            cv2.putText(image, "FPS : {}".format(int(FPS)), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            # Flip the image horizontally for a selfie-view display.

            cv2.imshow('MediaPipe Face Mesh', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

cap.release()


