# This Function will Blur the Background(Everything besides the Faces)
# Return the Final image.
# This can be refined to make ONLY BACKGROUND blur


def background_blur():
    img = cv.imread('C:/Users/Linh/Documents/Opencv/Photos/grace_hopper.png')
    mask = np.zeros_like(img)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# Blur the Orignal Image
    img_blured = cv.GaussianBlur(img, (49, 49), 0)

# xlm Code Classifier for face
    hard_cascade = cv.CascadeClassifier('Hard_Face.xml')
    #hard_cascade = cv.CascadeClassifier('Smile_Opencv.xml')

# Detect the face
    faces_rect = hard_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)
    print(len(faces_rect))

# Apply Non-Blur over the faces(ROI)
    for (x,y,w,h) in faces_rect:
    # Get the ROI
        face_roi = img[y:y+h, x:x + w]

    # Create the mask of the face(s)
        mask = cv.rectangle(mask,(x,y),(x+w,y+h),(255,255,255),-1)
        mask = cv.bitwise_and(img,mask)

    # Find Countors in mask

    # Applying NON-blur faces onto original image
        img_blured[y:y + h, x:x + w]=face_roi

        cv.putText(img_blured, f"Faces Discovered/UnBlurred:{len(faces_rect)}", (15,30), cv.FONT_ITALIC, 1, (0, 0, 255), thickness=2)

# Showing the final Image :)
    cv.imshow('Background Blur',img_blured)
    cv.imshow('mask',mask)

    cv.waitKey(0)
