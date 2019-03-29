import pyswt
import cv2
import sys

#sys.setrecursionlimit(10000)
img_path = "./images/swt-example-2.png"
img = cv2.imread(img_path)
swt_img = pyswt.run(img)
#swt_img = cv2.cvtColor(swt_img, cv2.COLOR)
cv2.imwrite("output.png", swt_img)
