
import cv2
import numpy as np

cap = cv2.VideoCapture("D:/tmp/vedio/123.mp4")
while(1):
    # get a frame
    ret, frame = cap.read()
    # show a frame
    cv2.imshow("capture", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
