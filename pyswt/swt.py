import cv2
import numpy as np
import math

def run(img):
  """Main SWT runner function.
  Applies the SWT algorithm steps and outputs bounding boxes.

  Keyword Arguments:
  
  img -- the image to apply SWT on
  """

  # Converting image to grayscale
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # Applying SWT to image
  fin = swt(gray)
  return fin

def swt(img):
  """Applies the SWT to the input image"""

  # Getting Canny edges
  edges = cv2.Canny(img, 100, 300)
  # Getting gradient derivatives
  # Note: can also use a Scharr filter here if
  # ksize is set to -1. Potentially, provides better
  # results than a 3x3 sobel.
  g_cols = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=-1)
  g_rows = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=-1)
  # Getting gradient direction (rads)
  gdir = np.arctan2(g_cols, g_rows)

  # Setting up SWT image
  swt_img = np.empty(img.shape)
  swt_img[:] = np.Infinity # Setting all values to infinite

  rays = []
  # Looping through each pixel, calculating rays
  for row in range(img.shape[0]):
    for col in range(img.shape[1]):
      edge = edges[row,col]
      if edge > 0: # Checking if we're on an edge
        # Passing in single derivative values for rows and cols
        # Along with edges and ray origin
        ray = cast_ray(g_rows, g_cols, edges, row, col,1, math.pi/6)
        if ray != None:
          rays.append(ray)
          for point in ray:
            swt_img[point[0],point[1]] = 255
      else:
        swt_img[row,col] = 0
  
  return swt_img

"""Casts a ray in an image given a starting point, an edge set, and the gradient
  Applies the SWT algorithm steps and outputs bounding boxes.

  Keyword Arguments:
  
  g_rows -- verticle component of the gradient
  g_cols -- horizontal component of the gradient
  edges -- the edge set of the image
  row -- the starting row location in the image
  col -- the starting column location in the image
  dir -- either 1 (light text) or -1 (dark text), the direction the ray should be cast
  max_angle_diff -- Controls how far from directly opposite the two edge gradeints should be
  """

def cast_ray(g_rows, g_cols, edges, row, col, dir, max_angle_diff):
  i = 1
  ray = [[row,col]]
  # Getting origin gradients
  g_row = g_rows[row,col]*dir
  g_col = g_cols[row,col]*dir
  # Normalizing g_col and g_row to ensure we move ahead one pixel
  g_col_norm = g_col / norm(g_col, g_row)
  g_row_norm = g_row / norm(g_col, g_row)
  while True:
    # Calculating the next step ahead in the ray
    # Adding 0.5 to start in center of pixel
    col_step = math.floor(col + 0.5 + g_col_norm * i)
    row_step = math.floor(row + 0.5 + g_row_norm * i)
    i += 1
    try:
      # Checking if the next step is an edge
      if edges[row_step,col_step] > 0:
        # Checking that edge pixels gradient is approximately opposite the direction of travel
        g_opp_row = g_rows[row_step,col_step]*dir
        g_opp_col = g_cols[row_step,col_step]*dir
        theta = angleBetween(g_row_norm, g_col_norm, -g_opp_row, -g_opp_col)

        if theta < max_angle_diff:
          g_opp_row = g_opp_row/norm(g_opp_row, g_opp_col)
          g_opp_col = g_opp_col/norm(g_opp_row, g_opp_col)
         #print("Start Gradient: " + str(g_row_norm) + ", " + str(g_col_norm))
         #print("End Gradient: " + str(-g_opp_row) + ", " + str(-g_opp_col))
          return ray
        else:
          return None
      else:
        ray.append([row_step,col_step])
    except IndexError:
      return None

def norm(x,y):
  return math.sqrt(x * x + y * y)

def dot(x1,y1,x2,y2):
  return x1*x2+y1*y2    

# assumes neither vector is zero
def angleBetween(x1,y1,x2,y2):
  proportion = dot(x1,y1,x2,y2)/(norm(x1,y1)*norm(x2,y2))
  #print(proportion)
  if abs(proportion) > 1:
    return math.pi/2
  else:
    return math.acos(dot(x1,y1,x2,y2)/(norm(x1,y1)*norm(x2,y2)))

def find_letters(swt_image):
  return letter_comps

def letter_filtering(letter_comps):
  return letter_candidates

def textline_aggregate(letter_candidates):
  return textlines

def word_detection(textlines):
  return mask
