import timeit

setup = """\
import cv2
import pyswt
img_path = "./images/swt-example-1.png"
img = cv2.imread(img_path)
"""

print(timeit.timeit("pyswt.run(img)",setup,number=1))
