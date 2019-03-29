import copy
import numpy as np

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
  components = np.empty(pixel_source.shape)
  components[:] = 0;

  for row in range(pixel_source.shape[0]):
    for col in range(pixel_source.shape[1]):
      if pixel_source[row, col] > 0:
        region_grow(pixel_source, components, label, row, col)
        label = label + 1

  return components

def region_grow(pixel_source, components, label, row, col, connect8=True):
  # Getting current pixel value
  curr_value = pixel_source[row, col]
  # Removing point from the pixel source
  pixel_source[row, col] = 0
  # Adding our label to the components
  components[row, col] = label

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
        if curr_value/adj_value < 3 and adj_value/curr_value < 3:
          # Recursively grow to connected pieces
          region_grow(pixel_source, components, label, row_shift, col_shift, connect8)
    except IndexError:
      continue
