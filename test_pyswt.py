import pyswt
import cv2
import sys

img_path = "./images/swt-example-3.png"
# img_path = "./images/jpg-1.jpg"
img = cv2.imread(img_path)
swt_img = pyswt.run(img)
cv2.imwrite("output.png", swt_img)
