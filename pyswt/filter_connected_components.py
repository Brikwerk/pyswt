from .connected_component import ConnectedComponentData
from typing import List

# Magic numbers as specified by the paper
__stroke_width_variance_coeff = 0.5
__max_stroke_width_variance = 5
__aspect_ratio_upper_bound = 10
__aspect_ratio_lower_bound = 1.0 / __aspect_ratio_upper_bound
__height_lower_bound = 10
__height_upper_bound = 300

__num_components_embedded_max = 3


def run(connected_components_data: List[ConnectedComponentData]):
    # Filter from cheapest to calculate to most expensive
    filtered_data = connected_components_data
    filtered_data = filter_by_component_height(filtered_data)
    filtered_data = filter_by_aspect_ratio(filtered_data)

    # if dropping text, it might be this method...
    filtered_data = filter_by_stroke_width_variance(filtered_data)

    # Currently, there seems like there is a bug that causes a few components to have huge bounding boxes
    # TODO: components randomly have huge bounding boxes, causing this to break, fix this bug
    filtered_data = filter_if_contains_other_components(filtered_data)

    return filtered_data


def filter_by_stroke_width_variance(cc_data: List[ConnectedComponentData]):
    filtered_set = []
    for cc in cc_data:
        # Remove the point if the variance is above half the average stroke width. See paper for details
        # This parameter is found empirically. Sometimes removes text

        """
        # This is how the paper suggests to do it, but it is non-sense
        if cc.get_variance_stroke_width() <= cc.get_mean_stroke_width() * __stroke_width_variance_coeff:
            filtered_set.append(cc)
        # """
        # if cc.get_variance_stroke_width() <= __max_stroke_width_variance:
        if cc.get_variance_stroke_width()/cc.area < 0.05:
            filtered_set.append(cc)
            # print((cc.get_variance_stroke_width(), cc.area))
        # """

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
        if __aspect_ratio_lower_bound <= aspect_ratio <= __aspect_ratio_upper_bound:
            filtered_set.append(cc)

    return filtered_set


# TODO: Make this proportional to image dimensions
def filter_by_component_height(cc_data: List[ConnectedComponentData]):
    filtered_set = []
    for cc in cc_data:
        # Learned parameter, see paper
        if __height_lower_bound <= (cc.row_max - cc.row_min) <= __height_upper_bound:
            filtered_set.append(cc)

    return filtered_set


def filter_if_contains_other_components(cc_data: List[ConnectedComponentData]):
    filtered_set = []
    for i in range(len(cc_data)):
        num_components_embedded = 0
        cc_0 = cc_data[i]
        """
        if ((cc_0.row_max - cc_0.row_min) * (cc_0.col_max - cc_0.col_min)) / cc_0.area > 10:
            print("Max_Area: " + str((cc_0.row_max - cc_0.row_min) * (cc_0.col_max - cc_0.col_min)))
            print("Area: " + str(cc_0.area))
            print(str(cc_0.get_bounding_box()))
        """
        for j in range(len(cc_data)):
            if j == i:
                continue
            else:
                cc_1 = cc_data[j]
                if (cc_0.row_min <= cc_1.row_min
                        and cc_0.row_max >= cc_1.row_max
                        and cc_0.col_min <= cc_1.col_min
                        and cc_0.col_max >= cc_1.row_max
                ):
                    num_components_embedded += 1
                    # print("(" + str(cc_0.row_min) + ", " + str(cc_0.col_min) + "), (" + str(cc_0.row_max) + ", " + str(
                       # cc_0.col_max) + ")")

        if num_components_embedded <= __num_components_embedded_max:
            filtered_set.append(cc_0)

    return filtered_set
