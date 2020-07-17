import numpy as np
import cv2
img=cv2.imread('in4.png')
mask=cv2.imread('out4.png')
def maskedLane(img,mask):
    lowerGreen = np.array([0,150,0])
    upperGreen = np.array([0,255,0])
    maskedGreen = cv2.inRange(mask,lowerGreen,upperGreen)
    maskedGreen =cv2.resize(maskedGreen,(img.shape[1],img.shape[0]))
    roi = cv2.bitwise_and(img,img,mask=maskedGreen)
    return roi

while True:
    cv2.imshow("result",maskedLane(img,mask))
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cv2.destroyAllWindows()