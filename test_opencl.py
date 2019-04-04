import pyopencl as cl
import numpy as np
import cv2
import os
import math
import timeit

# Showing compiler output
# os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

#Read in image
img = cv2.imread('./images/swt-example-2.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Getting Canny edges
edges = cv2.Canny(img, 100, 300)
# Getting gradient derivatives
# Note: can also use a Scharr filter here if
# ksize is set to -1. Potentially, provides better
# results than a 3x3 sobel.
gy = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=-1)
gx = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=-1)

# Max ray length
hypot = math.floor(math.sqrt(img.shape[0]*img.shape[0] + img.shape[1]*img.shape[1]))
# Create an empty array to keep track of all rays
rays = np.zeros(shape=(img.shape[0], img.shape[1], 4))

# Setting up for OpenCL
# Get platforms, both CPU and GPU
plat = cl.get_platforms()
CPU = plat[0].get_devices()
try:
    GPU = plat[1].get_devices()
except IndexError:
    GPU = "none"

#Create context for GPU/CPU
if GPU!= "none":
    ctx = cl.Context(GPU)
else:
    ctx = cl.Context(CPU)

# Create queue for each kernel execution
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags

# SWT Kernel function
swt_src = '''
float magnitude(int x, int y) {
    return sqrt((float)(x * x + y * y));
}

int rowColToIndex(int row, int col, int width) {
    return col + (row * width);
}

float dotprod(float x1, float y1, float x2, float y2) {
    return x1 * x2 + y1 * y2;
}

float angleBetween(float x1, float y1, float x2, float y2) {
    float proportion = dotprod(x1, y1, x2, y2) / (magnitude(x1, y1) * magnitude(x2, y2));
    if (abs((float)proportion) > 1) {
        return M_PI_2;
    } else {
        return acos(dotprod(x1, y1, x2, y2) / (magnitude(x1, y1) * magnitude(x2, y2)))
    }
}

__kernel void swt(__global float *img, 
    __global int *grad_dir, 
    __global float *edges, 
    __global int *gx, 
    __global int *gy, 
    __global int *result, 
    __global int *width, 
    __global int *height)
{
    int w = *width;
    int h = *height;

    int dir = *grad_dir;

    // Getting position
    int posx = get_global_id(1);
    int posy = get_global_id(0);
    // Getting pos. in image and list
    int i = w*posy + posx;
    int res_i = (w*posy + posx)*4;

    // Getting row and col of image
    int row = floor((float)(i/w));
    int col = i - (w * row);

    // Checking we're on an edge
    if (edges[i] > 0) {
        // Getting gradient values
        int g_row = gx[i];
        int g_col = gy[i];

        // Checking we aren't on an edge with no gradient
        if (g_row != 0 && g_col != 0) {
            // Getting normalized gradient direction
            float g_col_norm = g_col / magnitude(g_col, g_row);
            float g_row_norm = g_row / magnitude(g_col, g_row);

            // Creating step to keep track of ray cast iteration
            int step = 1;

            // Casting ray
            while (true) {
                // Calculating next step ahead in the ray
                // Adding 0.5 to start in center of the pixel
                int col_step = floor((float)(col + 0.5 + g_col_norm * step));
                int row_step = floor((float)(row + 0.5 + g_row_norm * step));
                step++;

                // Detecting if we left the image
                if (row_step < 0 || col_step < 0 || row_step > h || col_step > w) {
                    break;
                }

                // Checking if the next step is an edge
                int step_i = rowColToIndex(row, col, w);
                if (edges[step_i] > 0) {
                    // Checking that the pixel gradient is approximately opposite to the
                    // direction of travel
                    int g_opp_row = gx[step_i] * dir;
                    int g_opp_col = gy[step_i] * dir;
                    theta = angleBetween(g_row_norm, g_col_norm, (float)(-1*g_opp_row), (float)(-1*g_opp_col))

                    if (theta < M_PI_2) {
                        // Storing rows/col of start/end of ray
                        result[res_i] = row;
                        result[res_i+1] = col;
                        result[res_i+2] = row_step;
                        result[res_i+3] = col_step;
                        break;
                    } else {
                        break;
                    }
                }
            }
        }
    }
}
'''

# Kernel function instantiation
swt_prg = cl.Program(ctx, swt_src).build()

# Converting data for memory transfer
img = img.astype(np.float32)
edges = edges.astype(np.float32)
gx = gx.astype(np.int32)
gy = gy.astype(np.int32)
rays = rays.astype(np.int32)

# Allocate memory for variables on the device
img_g =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img)
result_g = cl.Buffer(ctx, mf.WRITE_ONLY, rays.nbytes)
grad_dir = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(1))
edges_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=edges)
gx_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gx)
gy_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gy)
width_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(img.shape[1]))
height_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(img.shape[0]))
hypot_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(hypot))

# Call SWT kernel. Automatically takes care of block/grid distribution
swt_prg.swt(queue, img.shape, None, img_g, grad_dir, edges_g, gx_g, gy_g, result_g, width_g, height_g)
rays_result = np.empty_like(rays)
cl.enqueue_copy(queue, rays_result, result_g)

print(len(edges[edges>0]))
print(rays_result[3262,1642])
# print(rays_result)

# Show the blurred image
# cv2.imwrite('medianFilter-OpenCL.png',result)