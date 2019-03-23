import pyswt
import cv2

img_path = "./images/swt-example-2.png"
img = cv2.imread(img_path)
swt_img = pyswt.run(img)
cv2.imwrite("output.png", swt_img)