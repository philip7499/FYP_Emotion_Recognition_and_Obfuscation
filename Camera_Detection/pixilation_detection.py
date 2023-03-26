import cv2
import numpy as np

# Load a cascade file for detecting faces
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

vid = cv2.VideoCapture(0)
pwidth, pheight = (10, 10)

while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    
    height, width = frame.shape[:2]
    # Convert into grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Look for faces in the image using the loaded cascade file
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)
    for (x,y,w,h) in faces:
        # Create rectangle around faces
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
        face_image = frame[y:y+h, x:x+w]
        
        # Pixilate faces
        temp = cv2.resize(face_image, (pwidth, pheight), interpolation=cv2.INTER_LINEAR)
        face_image = cv2.resize(temp, (h, w), interpolation=cv2.INTER_NEAREST)
        frame[y:y+face_image.shape[0], x:x+face_image.shape[1]] = face_image
  
    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
