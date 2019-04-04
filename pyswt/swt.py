import cv2
import numpy as np
import math
import copy
from . import cast_ray as cr

def run(img, gradient_direction):
    """Applies the SWT to the input image"""

    # Getting Canny edges
    edges = cv2.Canny(img, 100, 300)
    # Getting gradient derivatives
    # Note: can also use a Scharr filter here if
    # ksize is set to -1. Potentially, provides better
    # results than a 3x3 sobel.
    gy = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=-1)
    gx = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=-1)

    # Setting up SWT image
    swt_img = np.empty(img.shape)
    swt_img[:] = np.Infinity  # Setting all values to infinite

    rays = []
    # Looping through each pixel, calculating rays
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            edge = edges[row, col]
            if edge > 0:  # Checking if we're on an edge
                # Passing in single derivative values for rows and cols
                # Along with edges and ray origin
                ray = cr.cast_ray(gx, gy, edges, row, col, gradient_direction, math.pi / 2)
                if ray != None:
                    # Adding ray to rays accumulator
                    rays.append(ray)
                    # Calculating the width of the ray
                    width = cr.magnitude(ray[len(ray) - 1][0] - ray[0][0], ray[len(ray) - 1][1] - ray[0][1])
                    # Assigning width to each pixel in the ray
                    for point in ray:
                        if swt_img[point[0], point[1]] > width:
                            swt_img[point[0], point[1]] = width

    # Set values of infinity to zero so that only values that had ray > 0
    swt_img[swt_img == np.Infinity] = 0

    # Creating a copy of the SWT image
    swt_median = copy.deepcopy(swt_img)

    # Looping through rays and assigning the median value
    # to ray pixels that are above the median
    for ray in rays:
        # Getting median of each ray's values
        median = cr.median_ray(ray, swt_img)

        # Loop through ray and change pixel values greater than median
        for coordinate in ray:
            if swt_img[coordinate[0], coordinate[1]] > median:
                swt_median[coordinate[0], coordinate[1]] = median

    return swt_median
