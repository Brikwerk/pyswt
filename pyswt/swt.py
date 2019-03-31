import cv2
import numpy as np
import math
import copy
from . import connected_component
from . import filter_connected_components
from . import letter_chains


def run(img):
    """Main SWT runner function.
    Applies the SWT algorithm steps and outputs bounding boxes.

    Keyword Arguments:
  
    img -- the image to apply SWT on
    """

    # Converting image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray_smoothed = cv2.GaussianBlur(gray, (5,5), 0)
    # Applying SWT to image
    swt_img = swt(gray)
    # Get connected component image and data. connected_component_data is defined in connected_component.py
    connected_components_img, connected_component_data = connected_component.run(gray, swt_img)
    filtered_components = filter_connected_components.run(connected_component_data)
    # Chains contain the final bounding boxes
    chains = letter_chains.run(filtered_components)

    """
    final_cc = []
    for chain in chains:
        for cc in chain.chain:
            final_cc.append(cc)

    return connected_component.get_connected_component_image(final_cc, swt_img.shape[0], swt_img.shape[1])
    """
    return letter_chains.make_image_with_bounding_boxes(img, chains)


def swt(img):
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

    edge_counter = 0
    # Looping through each pixel, calculating rays
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            edge = edges[row, col]
            edge_counter += 1
            if edge > 0:  # Checking if we're on an edge
                print(edge_counter)
                # Passing in single derivative values for rows and cols
                # Along with edges and ray origin
                ray = cast_ray(gx, gy, edges, row, col, -1, math.pi / 2)
                if ray != None:
                    # Adding ray to rays accumulator
                    rays.append(ray)
                    # Calculating the width of the ray
                    width = magnitude(ray[len(ray) - 1][0] - ray[0][0], ray[len(ray) - 1][1] - ray[0][1])
                    # Assigning width to each pixel in the ray
                    for point in ray:
                        if swt_img[point[0], point[1]] > width:
                            swt_img[point[0], point[1]] = width

    # Set values of infinity to zero so that only values that had ray > 0
    for row in range(swt_img.shape[0]):
      for col in range(swt_img.shape[1]):
         if swt_img[row, col] == np.Infinity:
           swt_img[row, col] = 0

    # Creating a copy of the SWT image
    swt_median = copy.deepcopy(swt_img)

    # Looping through rays and assigning the median value
    # to ray pixels that are above the median
    for ray in rays:
        # Getting median of each ray's values
        median = median_ray(ray, swt_img)

        # Loop through ray and change pixel values greater than median
        for coordinate in ray:
            if swt_img[coordinate[0], coordinate[1]] > median:
                swt_median[coordinate[0], coordinate[1]] = median

    return swt_median


"""Casts a ray in an image given a starting point, an edge set, and the gradient
  Applies the SWT algorithm steps and outputs bounding boxes.

  Keyword Arguments:
  
  gx -- verticle component of the gradient
  gy -- horizontal component of the gradient
  edges -- the edge set of the image
  row -- the starting row location in the image
  col -- the starting column location in the image
  dir -- either 1 (light text) or -1 (dark text), the direction the ray should be cast
  max_angle_diff -- Controls how far from directly opposite the two edge gradeints should be
  """


def cast_ray(gx, gy, edges, row, col, dir, max_angle_diff):
    i = 1
    ray = [[row, col]]
    # Getting origin gradients
    g_row = gx[row, col] * dir
    g_col = gy[row, col] * dir
    # Normalizing g_col and g_row to ensure we move ahead one pixel
    g_col_norm = g_col / magnitude(g_col, g_row)
    g_row_norm = g_row / magnitude(g_col, g_row)

    # TODO: Cap ray size based off of ratio?
    while True:
        # Calculating the next step ahead in the ray
        # Adding 0.5 to start in center of pixel
        col_step = math.floor(col + 0.5 + g_col_norm * i)
        row_step = math.floor(row + 0.5 + g_row_norm * i)
        i += 1
        try:
            # Checking if the next step is an edge
            if edges[row_step, col_step] > 0:
                # Checking that edge pixels gradient is approximately opposite the direction of travel
                g_opp_row = gx[row_step, col_step] * dir
                g_opp_col = gy[row_step, col_step] * dir
                theta = angle_between(g_row_norm, g_col_norm, -g_opp_row, -g_opp_col)

                if theta < max_angle_diff:
                    g_opp_row = g_opp_row / magnitude(g_opp_row, g_opp_col)
                    g_opp_col = g_opp_col / magnitude(g_opp_row, g_opp_col)
                    # print("Start Gradient: " + str(g_row_norm) + ", " + str(g_col_norm))
                    # print("End Gradient: " + str(-g_opp_row) + ", " + str(-g_opp_col))
                    return ray
                else:
                    return None
            else:
                ray.append([row_step, col_step])
        except IndexError:
            return None


def magnitude(x, y):
    return math.sqrt(x * x + y * y)


def dot(x1, y1, x2, y2):
    return x1 * x2 + y1 * y2


# assumes neither vector is zero
def angle_between(x1, y1, x2, y2):
    proportion = dot(x1, y1, x2, y2) / (magnitude(x1, y1) * magnitude(x2, y2))
    # print(proportion)
    if abs(proportion) > 1:
        return math.pi / 2
    else:
        return math.acos(dot(x1, y1, x2, y2) / (magnitude(x1, y1) * magnitude(x2, y2)))


def median_ray(ray, swt_img):
    # Accumulate pixel values and calculate median
    pixel_values = []
    for coordinate in ray:
        pixel_values.append(swt_img[coordinate[0], coordinate[1]])
    return np.median(pixel_values)


def print_image(img):
    for row in range(img.shape[0]):
        row_values = []
        for col in range(img.shape[1]):
            row_values.append(img[row, col])
        print(row_values)
