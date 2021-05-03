import cv2 as cv
import numpy as np

# This Funtion Will Blur Faces in an Image
# Returns the final image

def blur_face():
    img = cv.imread('C:/Users/Linh/Documents/Opencv/Tester/obamaMODI.jpg')
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #cv.imshow('gray image',gray)

# xlm Code Classifier for face
    hard_cascade = cv.CascadeClassifier('Hard_Face.xml')
    #hard_cascade = cv.CascadeClassifier('Smile_Opencv.xml')

# Detect face
    faces_rect = hard_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)
    print(len(faces_rect))
# Apply Blur over the faces(ROI)
    for (x,y,w,h) in faces_rect:
    # get the ROI
    # Apply blur onto the ROI
        face_roi = img[y:y+h, x:x + w]
        face_roi = cv.GaussianBlur(face_roi,(49,49),0)

    # Applying blur faces onto original image
        img[y:y + h, x:x + w]=face_roi
        #cv.putText(img, f"Faces Discovered/Blurred:{len(faces_rect)}", (15,30), cv.FONT_ITALIC, 1, (0, 0, 255), thickness=2)

    cv.imshow('deteced faces',img)

    cv.waitKey(0)



# This Function will Blur the Faces in Video
# Return the Final Frame.

def video_blur_face():
    capture = cv.VideoCapture(0)
    #capture = cv.VideoCapture('C:/Users/Linh/Documents/Opencv/Photos/waking (3).mp4')

    while True:
        _, frame = capture.read()
        mask =  np.zeros_like(frame)

        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        #cv.imshow('gray image',gray)

    # xlm Code Classifier for face
        hard_cascade = cv.CascadeClassifier('Hard_Face.xml')
        #hard_cascade = cv.CascadeClassifier('Smile_Opencv.xml')

    # Detect face
        faces_rect = hard_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)
        print(len(faces_rect))
    # Apply Blur over the faces(ROI)
        for (x,y,w,h) in faces_rect:
        # get the ROI
        # Apply blur onto the ROI
        # Create the mask of the face(s)
            #mask = cv.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)
            #mask = cv.bitwise_and(frame, mask)
            face_roi = gray[y:y+h, x:x + w]
            face_roi = cv.GaussianBlur(face_roi,(49,49),0)

        # Applying blur faces onto original image
            gray[y:y + h, x:x + w]=face_roi
            #cv.putText(frame, f"Faces Discovered/Blurred:{len(faces_rect)}", (15,30), cv.FONT_ITALIC, 1, (0, 0, 255), thickness=2)

        cv.imshow('Deteced faces',gray)
        #cv.imshow('Blue faces',mask)

        if cv.waitKey(45) == ord('q'):
            break

    capture.release()
    cv.destroyAllWindows()
