import cv2
import pyswt
import math

cap = cv2.VideoCapture(0)
scale = 2

if (cap.isOpened()== False): 
    print("Error opening video stream or file")

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        
        # Resizing video
        frame_resize = cv2.resize(frame, (math.floor(frame.shape[1]/scale), math.floor(frame.shape[0]/scale)), interpolation = cv2.INTER_LINEAR)

        # Applying SWT to the frame
        final_img, swt_ld, cc_ld, cc_boxes, filtered_cc = pyswt.run(frame_resize)

        # Scaling swt image size back up
        output = cv2.resize(final_img, (math.floor(final_img.shape[1]*scale), math.floor(final_img.shape[0]*scale)), interpolation = cv2.INTER_LINEAR)

        # Displaying SWT-applied frame
        cv2.imshow('frame', output)
 
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
 
    # Break the loop
    else: 
        break

cap.release()

cv2.destroyAllWindows()
