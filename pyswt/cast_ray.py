import numpy as np
import math

def cast_ray(gx, gy, edges, row, col, dir, max_angle_diff):
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

    i = 1
    ray = [[row, col]]
    # Getting origin gradients
    g_row = gx[row, col] * dir
    g_col = gy[row, col] * dir

    # If we encounter an edge with no direction
    if g_row == 0 and g_col == 0:
        return None;

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