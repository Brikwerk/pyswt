import pyswt
import cv2
import sys
import os

ext = ".png"
img_name = "swt-example-1"
img_path = "./images/" + img_name + ext
img_output_path = "./output/"

img = cv2.imread(img_path)
final_img, swt_ld, cc_ld, cc_boxes, filtered_cc = pyswt.run(img)

# Creating output directory if not available
if not os.path.exists(img_output_path):
    os.makedirs(img_output_path)

cv2.imwrite(img_output_path + img_name + "-final" + ext, final_img)
cv2.imwrite(img_output_path + img_name + "-swt-light" + ext, swt_ld[0])
cv2.imwrite(img_output_path + img_name + "-swt-dark" + ext, swt_ld[1])
cv2.imwrite(img_output_path + img_name + "-cc-bw-light" + ext, cc_ld[0])
cv2.imwrite(img_output_path + img_name + "-cc-bw-dark" + ext, cc_ld[1])
cv2.imwrite(img_output_path + img_name + "-cc-BB" + ext, cc_boxes)
cv2.imwrite(img_output_path + img_name + "-cc-filtered" + ext, filtered_cc)

