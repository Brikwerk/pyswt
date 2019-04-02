import cv2
import numpy as np
import math
import copy
from . import swt
from . import connected_component as con_comp
from . import filter_connected_components as filt_cc
from . import letter_chains
import timeit

def run(img, image_output=False, diagnostic=False, timing=False):
    """Main SWT runner function.
    Applies the SWT algorithm steps and outputs bounding boxes.

    Keyword Arguments:
  
    img -- the image to apply SWT on
    diagnostic -- enables diagnostic outputs from each step of the algorithm
    timing -- enables output of the runtimes for each step of the algorithm
    """

    # Converting image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Running in diagnostic mode
    # Enables diagnostic outputs from each step of the algorithm
    if diagnostic:
        # Running Stroke Width Transform with timing
        start = timeit.default_timer()
        swt_light, swt_dark, swt_light_dark = run_swt(gray, diagnostic=True)
        end = timeit.default_timer()
        swt_dur = end - start

        # Getting Connected Components with timing
        start = timeit.default_timer()
        cc_drawn_boxes, cc_light_dark, cc_data_light, cc_data_dark = run_connected_components(gray, img, swt_light, swt_dark, diagnostic=True)
        end = timeit.default_timer()
        cc_dur = end - start

        # Filtering Connected components with timing
        start = timeit.default_timer()
        filt_comps_light, filt_comps_dark, cc_filt_drawn_boxes = run_filtered_connected_components(cc_data_light, cc_data_dark, img, diagnostic=True)
        end = timeit.default_timer()
        cc_filt_dur = end - start

        # Getting letter chains with timing
        start = timeit.default_timer()
        image_with_bounding_boxes = run_letter_chains(img, filt_comps_light, filt_comps_dark, diagnostic=True)
        end = timeit.default_timer()
        lc_dur = end - start

        # Printing out durations
        if timing:
            print_durs(swt_dur, cc_dur, cc_filt_dur, lc_dur)

        return image_with_bounding_boxes, swt_light_dark, cc_light_dark, cc_drawn_boxes, cc_filt_drawn_boxes
    else:
        # Running Stroke Width Transform with timing
        start = timeit.default_timer()
        swt_light, swt_dark = run_swt(gray)
        end = timeit.default_timer()
        swt_dur = end - start

        # Getting Connected Components with timing
        start = timeit.default_timer()
        cc_data_light, cc_data_dark = run_connected_components(gray, img, swt_light, swt_dark)
        end = timeit.default_timer()
        cc_dur = end - start

        # Filtering Connected components with timing
        start = timeit.default_timer()
        filt_comps_light, filt_comps_dark = run_filtered_connected_components(cc_data_light, cc_data_dark, img)
        end = timeit.default_timer()
        cc_filt_dur = end - start

        # Getting letter chains with timing
        start = timeit.default_timer()
        bounding_boxes = run_letter_chains(img, filt_comps_light, filt_comps_dark, image=image_output)
        end = timeit.default_timer()
        lc_dur = end - start

        # Printing out durations
        if timing:
            print_durs(swt_dur, cc_dur, cc_filt_dur, lc_dur)

        return bounding_boxes

def run_swt(gray, diagnostic=False):
    # Applying SWT to image, once for light text, once for dark text
    swt_light = swt.run(gray, 1)
    swt_dark = swt.run(gray, -1)

    if diagnostic:
        # Returning a combo of SWT light/dark texts for diagnostic purposes
        swt_light_dark = [swt_light, swt_dark]
        return swt_light, swt_dark, swt_light_dark
    else:
        return swt_light, swt_dark

def run_connected_components(gray, img, swt_light, swt_dark, diagnostic=False):
    # Get connected component image 
    # and data. connected_component_data is defined in connected_component.py
    cc_img_light, cc_data_light = con_comp.run(gray, swt_light)
    cc_img_dark, cc_data_dark = con_comp.run(gray, swt_dark)

    if diagnostic:
        cc_light_dark = [cc_img_light, cc_img_dark]

        aggregate_cc_dark = []
        for cc in cc_data_dark:
            aggregate_cc_dark.append(cc)

        aggregate_cc_light = []
        for cc in cc_data_light:
            aggregate_cc_light.append(cc)

        cc_drawn_boxes = con_comp.make_image_with_bounding_boxes(img, aggregate_cc_light)
        cc_drawn_boxes = con_comp.make_image_with_bounding_boxes(cc_drawn_boxes, aggregate_cc_dark, (255, 0, 0))

        return cc_drawn_boxes, cc_light_dark, cc_data_light, cc_data_dark
    else:
        return cc_data_light, cc_data_dark

def run_filtered_connected_components(cc_data_light, cc_data_dark, img, diagnostic=False):
    # Apply single connected component filters to remove noise
    # Light and dark filters
    filt_comps_light = filt_cc.run(cc_data_light)
    filt_comps_dark = filt_cc.run(cc_data_dark)

    if diagnostic:
        aggregate_cc_dark = []
        for cc in filt_comps_dark:
            aggregate_cc_dark.append(cc)

        aggregate_cc_light = []
        for cc in filt_comps_light:
            aggregate_cc_light.append(cc)

        cc_filt_drawn_boxes = con_comp.make_image_with_bounding_boxes(img, aggregate_cc_light)
        cc_filt_drawn_boxes = con_comp.make_image_with_bounding_boxes(cc_filt_drawn_boxes, aggregate_cc_dark, (255, 0, 0))

        return filt_comps_light, filt_comps_dark, cc_filt_drawn_boxes
    else:
        return filt_comps_light, filt_comps_dark

def run_letter_chains(img, filt_comp_l, filt_comp_d, diagnostic=False, image=False):
    # Chains contain the final bounding boxes. Filter based on chain properties
    chains_light = letter_chains.run(filt_comp_l)
    chains_dark = letter_chains.run(filt_comp_d)
    chains_light_dark = [chains_light, chains_dark]

    if diagnostic or image: # If we're in diagnostic mode or image=True, output an image
        bbox_img = letter_chains.make_image_with_bounding_boxes(img, chains_light)
        bbox_img = letter_chains.make_image_with_bounding_boxes(bbox_img, chains_dark, (255, 0, 0))

        return bbox_img
    else: # Otherwise, output the bounding boxes
        bboxes_light = letter_chains.get_bounding_boxes(chains_light)
        bboxes_dark = letter_chains.get_bounding_boxes(chains_dark)
        return [bboxes_light, bboxes_dark]

def print_durs(swt_dur, cc_dur, cc_filt_dur, lc_dur):
    print("")
    print("-------------------")
    print("| Alg Step | Secs |")
    print("-------------------")
    print("| SWT Dur: | %.2f |" % swt_dur)
    print("-------------------")
    print("| CC Dur:  | %.2f |" % cc_dur)
    print("-------------------")
    print("| FC Dur:  | %.2f |" % cc_filt_dur)
    print("-------------------")
    print("| LC Dur:  | %.2f |" % lc_dur)
    print("-------------------")
    print(" Total: %.2f" % (swt_dur + cc_dur + cc_filt_dur + lc_dur))
    print("")