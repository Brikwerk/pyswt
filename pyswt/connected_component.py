import copy
import numpy as np
import math

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


def run(swt_median_image):
    # Copying so we can remove pixels to keep track
    # of components found
    pixel_source = copy.deepcopy(swt_median_image)
    # Creating initial label
    label = 1

    # Creating an empty image to store connected components
    component_image = np.empty(pixel_source.shape)
    component_image[:] = 0

    # connected component data
    connected_component_data = []

    for row in range(pixel_source.shape[0]):
        for col in range(pixel_source.shape[1]):
            if pixel_source[row, col] > 0:
                # Create a new data storage object
                component_data = ConnectedComponentData(row, col, label)
                region_grow(pixel_source, component_image, label, row, col, component_data)
                # Keep track of the component data
                connected_component_data.append(component_data)
                label = label + 1

    return component_image, connected_component_data


# We may consider removing component_image as we can store all this data in component_data
def region_grow(pixel_source, component_image, label, row, col, component_data, connect8=True, max_ratio=3):
    # Getting current pixel value
    curr_value = pixel_source[row, col]
    # Removing point from the pixel source
    pixel_source[row, col] = 0
    # Adding our label to the components
    component_image[row, col] = label
    component_data.add_pixel(row, col, curr_value)

    if connect8:
        num_directions = 8
        directions = __directions8__
    else:
        num_directions = 4
        directions = __directions4__

    # Looking at each direction
    for i in range(0, num_directions):
        try:
            # Getting coords we're going to check to grow into
            row_shift = row + directions[i][0]
            col_shift = row + directions[i][1]

            # Checking we're not growing into an empty region
            if pixel_source[row_shift, col_shift] > 0:
                adj_value = pixel_source[row_shift, col_shift]
                if curr_value / adj_value < max_ratio and adj_value / curr_value < max_ratio:
                    print("Current_value: " + str(curr_value))
                    print("adj_value: " + str(adj_value))
                    # Recursively grow to connected pieces
                    region_grow(pixel_source, component_image, label, row_shift, col_shift, component_data, connect8)
        except IndexError:
            continue


# This class is just a data container
# Do not call the get methods until all data points have been added
class ConnectedComponentData:
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

        # Statistical values to be calculated later
        self.__median_sw = None
        self.__mean_sw = None
        self.__variance_sw = None

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
        if self.__mean_sw is None:
            self.__mean_sw = np.average(self.stroke_widths)

        return self.__mean_sw

    def get_median_stroke_width(self):
        if self.__median_sw is None:
            self.__median_sw = np.median(self.stroke_widths)

        return self.__median_sw

    def get_variance_stroke_width(self):
        if self.__variance_sw is None:
            squared_sum = 0
            mean = self.get_mean_stroke_width()
            for s in self.stroke_widths:
                squared_sum += squared_sum + (s - mean) ** 2
            self.__variance_sw = squared_sum / len(self.stroke_widths)

        return self.__variance_sw

    # Returns the coordinates for the bounding box: [top-left, top-right, bottom-right, bottom-left]
    def get_bounding_box(self):
        return [
          [self.row_min, self.col_min],  # Top-left
          [self.row_min, self.col_max],  # Top-right
          [self.row_max, self.col_max],  # Bottom-right
          [self.row_max, self.col_min]   # Bottom-left
        ]

    # updates the values this component contains
    def add_pixel(self, row, col, stroke_width):
        # add location and stroke width information
        self.pixel_coordinates.append([row, col])
        self.stroke_widths.append(stroke_width)

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
