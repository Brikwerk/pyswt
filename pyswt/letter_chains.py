import math

from .connected_component import ConnectedComponentData
from typing import List


# Produce the final set of letter chains and get their bounding boxes
def run(cc_data_filtered: List[ConnectedComponentData]):
    chains = populate_pairs(cc_data_filtered)


# Check each pair of connected components and produce a tuple of sufficicently close letter candidates
def populate_pairs(cc_data_filtered: List[ConnectedComponentData]):
    # Only need to check each pair of elements once
    chains = []
    for i in range(len(cc_data_filtered)):
        for j in range(i + 1, range(len(cc_data_filtered))):
            # If the two components are close enough together, add them together in a chain
            if is_within_relative_distance(cc_data_filtered[i], cc_data_filtered[j]):
                chains.append(Chain(cc_data_filtered[i], cc_data_filtered[j]))

    return chains


def is_within_relative_distance(cc_1: ConnectedComponentData, cc_2: ConnectedComponentData, max_dist_multiplier=3):
    # Computing distance from bottom_corner right corner to bottom_corner left because this should not change much
    # Euclidean distance
    dist = math.sqrt((cc_2.row_max - cc_1.row_max) ** 2 + (cc_2.col_min - cc_1.col_max) ** 2)
    largest_width = max(cc_1.col_max - cc_1.col_min, cc_2.col_max - cc_2.col_min)
    return dist <= largest_width * 3


class Chain:

    def __init__(self, min_row: int, max_row: int, min_col: int, max_col: int):
        self.chain = []

        self.min_row = min_row
        self.max_row = max_row
        self.min_col = min_col
        self.max_col = max_col

    def __init__(self, cc_1: ConnectedComponentData, cc_2: ConnectedComponentData):
        # Stores the connected components that are a part of this chain
        self.chain = [cc_1, cc_2]

        self.min_row = min(cc_1.min_row, cc_2.min_row)
        self.max_row = max(cc_1.max_row, cc_2.max_row)
        self.min_col = min(cc_1.min_col, cc_2.min_col)
        self.max_col = max(cc_1.max_col, cc_2.max_col)


def merge_chains(c1: Chain, c2: Chain):
    min_row = min(c1.min_row, c2.min_row)
    max_row = max(c1.max_row, c2.max_row)
    min_col = min(c1.min_col, c2.min_col)
    max_col = max(c1.max_col, c2.max_col)

    merged_chain = Chain(min_row, max_row, min_col, max_col)
    merged_chain.chain = list(set().union(c1.chain, c2.chain))

    return merged_chain
