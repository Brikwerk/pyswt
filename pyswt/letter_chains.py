import math
import copy
import cv2

from .connected_component import ConnectedComponentData
from typing import List

__sw_median_max_ratio = 2
__height_max_ratio = 2
__max_distance_multiplier = 3
__min_chain_size = 3
__max_average_gray_diff = 10


# Produce the final set of letter chains and get their bounding boxes
def run(cc_data_filtered: List[ConnectedComponentData]):
    chains = populate_pairs(cc_data_filtered)
    # Get rid of chains if component height ratio > 2
    chains = remove_if_heights_too_different(chains)
    chains = remove_if_grays_dissimilar(chains)
    chains = remove_if_stroke_widths_too_different(chains)
    chains = lengthen_chains(chains)
    chains = remove_short_chains(chains)

    # return chains

    return chains


# Check each pair of connected components and produce a tuple of sufficicently close letter candidates
def populate_pairs(cc_data_filtered: List[ConnectedComponentData]):
    # Only need to check each pair of elements once
    chains = []
    for i in range(len(cc_data_filtered)):
        for j in range(i + 1, len(cc_data_filtered)):
            # If the two components are close enough together, add them together in a chain
            if is_within_relative_distance(cc_data_filtered[i], cc_data_filtered[j]):
                chains.append(build_chain(cc_data_filtered[i], cc_data_filtered[j]))

    return chains


def is_within_relative_distance(cc_1: ConnectedComponentData, cc_2: ConnectedComponentData):
    # Ensure one letter candidate is not floating above the other
    if cc_1.row_min >= cc_2.row_max or cc_2.row_min >= cc_1.row_max:
        return False
    # Computing distance from bottom_corner right corner to bottom_corner left because this should not change much
    # Euclidean distance
    dist = math.sqrt((cc_2.row_max - cc_1.row_max) ** 2 + (cc_2.col_min - cc_1.col_max) ** 2)
    largest_width = max(cc_1.col_max - cc_1.col_min, cc_2.col_max - cc_2.col_min)
    return dist <= largest_width * __max_distance_multiplier


class Chain:

    def __init__(self):
        self.chain = []

        self.row_min = -1
        self.row_max = -1
        self.col_min = -1
        self.col_max = -1

    # Returns the coordinates for the bounding box: [top-left, top-right, bottom-right, bottom-left]
    def get_bounding_box(self):
        return [
            [self.row_min, self.col_min],  # Top-left
            [self.row_min, self.col_max],  # Top-right
            [self.row_max, self.col_max],  # Bottom-right
            [self.row_max, self.col_min]  # Bottom-left
        ]


# python does not allow multiple constructors....
def build_chain(cc_1: ConnectedComponentData, cc_2: ConnectedComponentData):
    chain = Chain()
    chain.row_min = min(cc_1.row_min, cc_2.row_min)
    chain.row_max = max(cc_1.row_max, cc_2.row_max)
    chain.col_min = min(cc_1.col_min, cc_2.col_min)
    chain.col_max = max(cc_1.col_max, cc_2.col_max)
    chain.chain = [cc_1, cc_2]

    return chain


def build_chain_from_merge(row_min: int, row_max: int, col_min: int, col_max: int, ccs: List[ConnectedComponentData]):
    chain = Chain()
    chain.row_min = row_min
    chain.row_max = row_max
    chain.col_min = col_min
    chain.col_max = col_max
    chain.chain = ccs

    return chain


def merge_chains(c1: Chain, c2: Chain):
    c1.row_min = min(c1.row_min, c2.row_min)
    c1.row_max = max(c1.row_max, c2.row_max)
    c1.col_min = min(c1.col_min, c2.col_min)
    c1.col_max = max(c1.col_max, c2.col_max)

    c1.chain = list(set().union(c1.chain, c2.chain))

    return c1


def remove_if_heights_too_different(chains: List[Chain]):
    filtered_chains = []
    for chain in chains:
        cc_0 = chain.chain[0]
        cc_1 = chain.chain[1]
        height_0 = cc_0.row_max - cc_0.row_min
        height_1 = cc_1.row_max - cc_1.row_min
        # heights are non-zero from the component filtering step
        if height_0 / height_1 <= __height_max_ratio or height_1 / height_0 <= __height_max_ratio:
            filtered_chains.append(chain)

    return filtered_chains


def remove_if_stroke_widths_too_different(chains: List[Chain]):
    filtered_chains = []
    for chain in chains:
        sw_median_0 = chain.chain[0].get_median_stroke_width()
        sw_median_1 = chain.chain[1].get_median_stroke_width()
        # see paper for reason for this magic number
        if sw_median_0 / sw_median_1 <= __sw_median_max_ratio or sw_median_1 / sw_median_0 <= __sw_median_max_ratio:
            filtered_chains.append(chain)

    return filtered_chains


def remove_if_grays_dissimilar(chains: List[Chain]):
    filtered_chains = []
    for chain in chains:
        avg_gray_0 = chain.chain[0].get_mean_gray()
        avg_gray_1 = chain.chain[1].get_mean_gray()
        if abs(avg_gray_1 - avg_gray_0) < __max_average_gray_diff:
            filtered_chains.append(chain)

    return filtered_chains


# return true if they share a connected component, false otherwise
def contain_new_chain_link(chain_1: Chain, chain_2: Chain):
    # if their bounding boxes do not over lap, then they cannot contain the same element. if too slow, implement
    # cycle through all cc elements to see if they contain the same cc
    if chain_1 is chain_2:
        return False

    for cc_1 in chain_1.chain:
        for cc_2 in chain_2.chain:
            if cc_1 is cc_2:
                return True
    return False


# TODO: build a doubly linked list implementation to reduce time complexity to n^2
def lengthen_chains(chains: List[Chain]):
    chains_copy = list.copy(chains)
    lengthened_chains = []
    i = 0
    while i < len(chains_copy):
        altered = False
        lengthened_chain = chains_copy[i]
        j = i + 1
        while j < len(chains_copy):

            if contain_new_chain_link(chains_copy[i], chains_copy[j]):
                altered = True
                # Make a new larger chain that is he union of the 2
                lengthened_chain = merge_chains(lengthened_chain, chains_copy[j])

                # This ensures we will catch all relations of near by components
                chains_copy[j] = lengthened_chain
            j += 1

        if not altered:
            i += 1
            lengthened_chains.append(lengthened_chain)
        else:
            chains_copy[i] = lengthened_chain


    # Return unique elements
    return list(set(lengthened_chains))


def remove_short_chains(chains: List[Chain]):
    long_chains = []
    for chain in chains:
        if len(chain.chain) >= __min_chain_size:
            long_chains.append(chain)

    return long_chains


# Default color is red
def make_image_with_bounding_boxes(img, chains: List[Chain], color=(0, 0, 255)):
    img_drawn = copy.deepcopy(img)
    for chain in chains:
        # Bounding-box top-left clockwise
        bb = chain.get_bounding_box()
        top_left = (bb[0][1], bb[0][0])
        bottom_right = (bb[2][1], bb[2][0])
        cv2.rectangle(img_drawn, top_left, bottom_right, color)

    return img_drawn
