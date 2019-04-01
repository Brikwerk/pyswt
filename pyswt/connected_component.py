import copy
import cv2
import numpy as np
from typing import List

# 8 connected relative directions
__directions8__ = [
    [-1, 1],
    [0, 1],
    [1, 1],
    [1, 0],
    [1, -1],
    [0, -1],
    [-1, -1],
    [-1, 0]
]

# 4 connected relative directions
__directions4__ = [
    [0, 1],
    [1, 0],
    [0, -1],
    [-1, 0]
]


def run(gray_img, swt_median_image):
    """Main runner for the connected components discovery algorithm.
    Applies the algorithm steps and outputs a connected component image
    and connected component data.

    Keyword Arguments
    
    swt_median_image -- An image with SWT applied to it
    """
    # Copying so we can remove pixels to keep track
    # of components found
    pixel_source = copy.deepcopy(swt_median_image)

    # Creating initial label
    label = 1

    # Creating an empty image to store connected components
    component_image = np.zeros(pixel_source.shape)
    component_image[:] = 0

    # connected component data
    connected_component_data = []

    for row in range(pixel_source.shape[0]):
        for col in range(pixel_source.shape[1]):
            if pixel_source[row, col] > 0:
                # Create a new data storage object
                component_data = ConnectedComponentData(row, col, label)
                region_grow_stack(gray_img, pixel_source, component_image, label, row, col, component_data)
                # Keep track of the component data
                if component_data.area > 5:
                    connected_component_data.append(component_data)
                label = label + 1

    return component_image, connected_component_data



# This method is more gross than the recusive one, but does not break number of frames allowed
def region_grow_stack(gray_img, pixel_source, component_image, label, row, col, component_data, connect8=True, max_ratio=3):
    """A stack based implementation of the region growing algorithm"""

    if connect8:
        num_directions = 8
        directions = __directions8__
    else:
        num_directions = 4
        directions = __directions4__

    initial_stroke_width = pixel_source[row, col]
    # Delete visited pixel
    pixel_source[row, col] = 0
    component_data.add_pixel(row, col, initial_stroke_width, gray_img[row, col])
    # label visited pixel
    component_image[row, col] = label

    pixel_stack = []

    # Initialize stack
    for i in range(num_directions):
        try:
            # Getting coords we're going to check to grow into
            row_shift = row + directions[i][0]
            col_shift = col + directions[i][1]

            adj_value = pixel_source[row_shift, col_shift]

            # Checking we're not growing into an empty region
            if adj_value > 0:
                if initial_stroke_width / adj_value < max_ratio and adj_value / initial_stroke_width < max_ratio:
                    # update connected component tracking data structures
                    component_data.add_pixel(row_shift, col_shift, adj_value, gray_img[row, col])
                    pixel_source[row_shift, col_shift] = 0
                    component_image[row_shift, col_shift] = label
                    # put on stack
                    pixel = Pixel(row_shift, col_shift, adj_value)
                    pixel_stack.append(pixel)

        except IndexError:
            continue

    # Now go through to find the connected components
    while len(pixel_stack) > 0:
        curr_pixel = pixel_stack.pop()
        for i in range(num_directions):
            try:
                # Getting coords we're going to check to grow into
                row_shift = curr_pixel.row + directions[i][0]
                col_shift = curr_pixel.col + directions[i][1]

                adj_value = pixel_source[row_shift, col_shift]

                # Checking we're not growing into an empty region
                if adj_value > 0:
                    # Checking stroke width ration does not exceed max ratio
                    if curr_pixel.stroke_width / adj_value < max_ratio and curr_pixel.stroke_width / initial_stroke_width < max_ratio:
                        # update connected component tracking data structures
                        component_data.add_pixel(row_shift, col_shift, adj_value, gray_img[row_shift, col_shift])
                        pixel_source[row_shift, col_shift] = 0
                        component_image[row_shift, col_shift] = label
                        # put on stack
                        pixel = Pixel(row_shift, col_shift, adj_value)
                        pixel_stack.append(pixel)

            except IndexError:
                continue


class ConnectedComponentData:
    """This class is utilized as a data container for the cc algorithm
    Do not call the get methods until all data points have been added
    """
    def __init__(self, row, col, label):
        # These values will define the bounding box of the component
        self.row_min = row
        self.row_max = row
        self.col_min = col
        self.col_max = col

        # label of component values and indexes of components
        self.label = label
        self.pixel_coordinates = []

        # Average center point of the components, calculate after
        self.__centroid = None

        # The total number of piels in this component
        self.area = 0

        # A list of all stroke widths, will be used to calculate variance, and median stroke width
        self.stroke_widths = []

        self.grays = []

        # Statistical values to be calculated later
        self.__median_sw = None
        self.__mean_gray = None

        self.__variance_stroke_width = None

        self.__mean_gray = None
        self.__variance_gray = None

    def get_centroid(self):
        if self.__centroid is None:
            row_sum = 0
            col_sum = 0
            for coord in self.pixel_coordinates:
                row_sum += coord[0]
                col_sum += coord[1]

            n = len(self.pixel_coordinates)
            self.__centroid = [row_sum / n, col_sum / n]
        return self.__centroid

    def get_mean_stroke_width(self):
        if self.__mean_gray is None:
            self.__mean_gray = np.average(self.stroke_widths)

        return self.__mean_gray

    def get_median_stroke_width(self):
        if self.__median_sw is None:
            self.__median_sw = np.median(self.stroke_widths)

        return self.__median_sw

    def get_variance_stroke_width(self):
        if self.__variance_stroke_width is None:
            squared_sum = 0
            mean = self.get_mean_stroke_width()
            for s in self.stroke_widths:
                squared_sum += (s - mean) ** 2

            self.__variance_stroke_width = squared_sum / len(self.stroke_widths)

        return self.__variance_stroke_width

    def get_mean_gray(self):
        if self.__mean_gray is None:
            self.__mean_gray = np.average(self.grays)

        return self.__mean_gray

    def get_variance_gray(self):
        if self.__variance_gray is None:
            squared_sum = 0
            mean = self.get_mean_gray()
            for g in self.grays:
                squared_sum += (g - mean) ** 2
            self.__variance_gray = squared_sum / len(self.grays)

        return self.__variance_gray

    # Returns the coordinates for the bounding box: [top-left, top-right, bottom-right, bottom-left]
    def get_bounding_box(self):
        return [
            [self.row_min, self.col_min],  # Top-left
            [self.row_min, self.col_max],  # Top-right
            [self.row_max, self.col_max],  # Bottom-right
            [self.row_max, self.col_min]  # Bottom-left
        ]

    def get_height(self):
        return self.row_max - self.row_min

    def get_width(self):
        return self.col_max - self.col_min

    # updates the values this component contains
    def add_pixel(self, row, col, stroke_width, gray_value):
        # add location and stroke width information
        self.pixel_coordinates.append([row, col])
        self.stroke_widths.append(stroke_width)
        self.grays.append(gray_value)

        # update bounds
        if row < self.row_min:
            self.row_min = row
        elif row > self.row_max:
            self.row_max = row

        if col < self.col_min:
            self.col_min = col
        elif col > self.col_max:
            self.col_max = col

        # update pixel total
        self.area += 1


class Pixel:
    def __init__(self, row, col, stroke_width):
        self.row = row
        self.col = col
        self.stroke_width = stroke_width


def get_connected_component_image(cc_data: List[ConnectedComponentData], num_rows: int, num_cols: int):
    blank = np.zeros([num_rows, num_cols, 1], np.uint8)
    for cc in cc_data:
        label = min(255, cc.label)
        for coord in cc.pixel_coordinates:
            blank[coord[0], coord[1]] = 255

    return blank


# Default color is red
def make_image_with_bounding_boxes(img, ccs: List[ConnectedComponentData], color=(0, 0, 255)):
    img_drawn = copy.deepcopy(img)
    for cc in ccs:
        # Bounding-box top-left clockwise
        bb = cc.get_bounding_box()
        top_left = (bb[0][1], bb[0][0])
        bottom_right = (bb[2][1], bb[2][0])
        cv2.rectangle(img_drawn, top_left, bottom_right, color)

    return img_drawn
