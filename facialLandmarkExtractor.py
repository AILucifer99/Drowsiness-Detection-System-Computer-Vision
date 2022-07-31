# import cv2
# import mediapipe
# import time

# cap = cv2.VideoCapture(0)
# previousTime = 0 
# drawing = mediapipe.solutions.drawing_utils
# faceMesh = mediapipe.solutions.face_mesh
# mesh = faceMesh.FaceMesh(static_image_mode = False, max_num_faces = 2, 
#                             min_detection_confidence=0.5, min_tracking_confidence=0.5)

# while True :
#     success, image = cap.read()
#     imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = mesh.process(imgRGB)
#     if results.multi_face_landmarks :
#         for faceLandmarks in results.multi_face_landmarks :
#             drawing.draw_landmarks(imgRGB, faceLandmarks, faceMesh.FACEMESH_CONTOURS)    


#     currentTime = time.time()
#     FPS = 1 / (currentTime - previousTime)
#     previousTime = currentTime
#     cv2.putText(image, "FPS : {}".format(int(FPS)), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
#     cv2.imshow("Window", image)
#     cv2.waitKey(1)

import cv2
import mediapipe as mp  
import time


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
detectionStyle = "iris-connection-style"

cap = cv2.VideoCapture(0)
previousTime = 0
with mp_face_mesh.FaceMesh(
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

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


