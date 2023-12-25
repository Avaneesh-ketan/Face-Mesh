import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(0) #or 0

mpfacem = mp.solutions.face_mesh
facem = mpfacem.FaceMesh(max_num_faces=1)#parameters - 1)static_image_mode - if False, detection and tracking - If True, only detection 2) max_num_hands 3) Min_detection_confidence = 50% 4) Min_tracking_confidence = 50%
mpDraw = mp.solutions.drawing_utils #to draw external dots to hand.
drawSpec = mpDraw.DrawingSpec(thickness = 1, circle_radius = 2)
pTime = 0
cTime = 0
while True:
    suc, img = cap.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB) #Facemesh only accepts RGB images
    results = facem.process(imgRGB)#processes the frame
    #multi hand landmarks helps us detect hands and find the landmarks
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            for id, lm in enumerate(faceLms.landmark):
                #to print the ids of each landmark
                print(id, lm)
                #to get pixel val;ues
                h,w,c = img.shape
                cx, cy = int((lm.x)*w), int((lm.y)*h) #size. 
                print(cx, cy)
                
            mpDraw.draw_landmarks(img, faceLms, mpfacem.FACEMESH_CONTOURS, drawSpec, drawSpec)
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv.putText(img, str(int(fps)), (10,70), cv.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3) #1) img, 2)displayable text 3) position 4) Font 5)Scale 6) Color 7)thickness

    cv.imshow("Image", img)
    cv.waitKey(1)