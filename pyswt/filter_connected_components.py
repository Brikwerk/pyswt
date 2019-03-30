from .connected_component import ConnectedComponentData
from typing import List


def run(connected_components_data: List[ConnectedComponentData]):
    # Filter from cheapest to calculate to most expensive
    filtered_data = connected_components_data
    filtered_data = filter_by_component_height(filtered_data)
    filtered_data = filter_by_aspect_ratio(filtered_data)

    # if dropping text, it might be this method...
    filtered_data = filter_by_stroke_width_variance(filtered_data)

    return filtered_data


def filter_by_stroke_width_variance(cc_data: List[ConnectedComponentData]):
    filtered_set = []
    for cc in cc_data:
        # Remove the point if the variance is above half the average stroke width. See paper for details
        # This parameter is found empirically. Sometimes removes text
        if cc.get_variance_stroke_width() >= cc.get_mean_stroke_width() / 2:
            filtered_set.append(cc)

    return filtered_set


def filter_by_aspect_ratio(cc_data: List[ConnectedComponentData]):
    filtered_set = []
    for cc in cc_data:
        width = cc.col_max - cc.col_min
        # discard ccs that are only one pixel wide
        if width == 0:
            continue

        aspect_ratio = (cc.row_max - cc.row_min) / width
        # This constraint is also specified in the original SWT paper
        if 0.1 <= aspect_ratio <= 10:
            filtered_set.append(cc)

    return filtered_set


# TODO: Make this proportional to image dimensions
def filter_by_component_height(cc_data: List[ConnectedComponentData]):
    filtered_set = []
    for cc in cc_data:
        # Learned parameter, see paper
        if 10 <= (cc.row_max - cc.row_min) <= 300:
            filtered_set.append(cc)

    return filtered_set
