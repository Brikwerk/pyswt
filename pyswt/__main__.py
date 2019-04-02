import cv2
import numpy as np
import math
import copy
from . import swt
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

    # Applying SWT to image, once for light text, once for dark text
    swt_light = swt.run(gray, 1)
    swt_dark = swt.run(gray, -1)
    swt_light_dark = [swt_light, swt_dark]

    # Get connected component image and data. connected_component_data is defined in connected_component.py
    connected_components_img_light, connected_component_data_light = connected_component.run(gray, swt_light)
    connected_components_img_dark, connected_component_data_dark = connected_component.run(gray, swt_dark)
    cc_light_dark = [connected_components_img_light, connected_components_img_dark]

    aggregate_cc_dark = []
    for cc in connected_component_data_dark:
        aggregate_cc_dark.append(cc)

    aggregate_cc_light = []
    for cc in connected_component_data_light:
        aggregate_cc_light.append(cc)

    # return connected_component.get_connected_component_image(connected_component_data_light, img.shape[0], img.shape[1])
    cc_drawn_boxes = connected_component.make_image_with_bounding_boxes(img, aggregate_cc_light)
    cc_drawn_boxes = connected_component.make_image_with_bounding_boxes(cc_drawn_boxes, aggregate_cc_dark, (255, 0, 0))
    # apply single connected component filters to remove noise
    filtered_components_light = filter_connected_components.run(connected_component_data_light)
    filtered_components_dark = filter_connected_components.run(connected_component_data_dark)

    aggregate_cc_dark = []
    for cc in filtered_components_dark:
        aggregate_cc_dark.append(cc)

    aggregate_cc_light = []
    for cc in filtered_components_light:
        aggregate_cc_light.append(cc)

    cc_filt_drawn_boxes = connected_component.make_image_with_bounding_boxes(img, aggregate_cc_light)
    cc_filt_drawn_boxes = connected_component.make_image_with_bounding_boxes(cc_filt_drawn_boxes, aggregate_cc_dark, (255, 0, 0))

    # Chains contain the final bounding boxes. Filter based on chain properties
    chains_light = letter_chains.run(filtered_components_light)
    chains_dark = letter_chains.run(filtered_components_dark)
    chains_light_dark = [chains_light, chains_dark]

    """
    final_cc = []
    for chain in dark:
        for cc in chain.chain:
            final_cc.append(cc)

    return connected_component.get_connected_component_image(final_cc, swt_img.shape[0], swt_img.shape[1])
    """

    image_with_bounding_boxes = letter_chains.make_image_with_bounding_boxes(img, chains_light)
    image_with_bounding_boxes = letter_chains.make_image_with_bounding_boxes(image_with_bounding_boxes, chains_dark, (255, 0, 0))

    return image_with_bounding_boxes, swt_light_dark, cc_light_dark, cc_drawn_boxes, cc_filt_drawn_boxes
