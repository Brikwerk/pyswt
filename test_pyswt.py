import pyswt
import cv2

img_path = "./images/swt-example.png"
img = cv2.imread(img_path)
swt_img = pyswt.run(img)
cv2.imshow('image',swt_img)
cv2.waitKey(0) # Waiting for any keypress when focused on displayed image window
cv2.destroyAllWindows()