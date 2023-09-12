"""
    Rank Discounted Cumulative Gain and Loss algorithm for testing unranked search results against a ranked list.
    Copyright (C) 2023 Healthee

    Yoav.Vollansky (@t) healthee.co

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
from logger.logger import JSONLogger

json_logger = JSONLogger(__name__)


def invert_rank(rank, max_rank):
    """
    Invert the relevance score so that a higher score indicates higher relevance.

    Parameters:
    rel_score: int
        The original relevance score.
    max_rel_score: int
        The maximum possible relevance score.

    Note:
        Ranks are assumed to be 1-indexed.

    Returns:
    int: The inverted relevance score.
    """
    return max_rank + 1 - rank


def build_assumed_ranges_mapping(true_relevance_set):
    """
    Builds an assumed ranges mapping based on the true relevance set.

    This function takes a dictionary, true_relevance_set, as input and sorts it
    based on the values. It then constructs a mapping of ranges where the same
    values occur consecutively in the sorted dictionary. Each range represents
    a group of keys with the same value.

    Args:
        true_relevance_set (dict): A dictionary representing the true relevance
        set, where keys are items and values are their corresponding relevance
        values.

    Returns:
        dict: A dictionary mapping the relevance values to ranges, where each
        range is represented by a tuple (start, end). The start and end indices
        indicate the positions of the keys in the sorted dictionary.

    Example:
        true_relevance_set =
        {'item1': 3, 'item2': 1, 'item3': 2, 'item4': 3, 'item5': 2}

        range_mapping = build_assumed_renages_mapping(true_relevance_set)
        print(range_mapping)

        Output:
        {1: (0, 0), 2: (1, 3), 3: (4, 4)}

    Note:
        The range_mapping dictionary represents the assumed ranges for each
        relevance value in the true_relevance_set. In the example above, the
        relevance value 1 has a range from index 0 to index 0 (indicating that
        there is one item with relevance 1). The relevance value 2 has a range
        from index 1 to index 3 (indicating that there are three items with
        relevance 2), and the relevance value 3 has a range from index 4 to
        index 4 (indicating that there is one item with relevance 3).

    Another note:
        the ranks are 1-indexed, the ranges are 0-indexed.
    """

    # Sort the dictionary based on its values
    sorted_dict = dict(sorted(true_relevance_set.items(),
                              key=lambda item: item[1]))

    range_mapping = {}
    for idx, key in enumerate(sorted_dict):
        value = sorted_dict[key]

        # If the value is not already in the mapping, add it
        if value not in range_mapping:
            range_mapping[value] = {'start': idx, 'end': idx}

        # If the value is already in the mapping, update the 'end' index
        else:
            range_mapping[value]['end'] = idx

    # Convert ranges from dictionary to tuple
    for value in range_mapping:
        range_mapping[value] = (range_mapping[value]['start'],
                                range_mapping[value]['end'])

    return range_mapping


def get_rank_for_index(index_in_list, range_mapping):
    """
    Retrieves the relevance rank corresponding to a given index number based
    on its position in the sorted relevance set.

    Args:
        index_in_list (int): An integer representing the position of an item
            in the sorted relevance set.
        range_mapping (dict): A dictionary mapping the relevance scores to
            ranges, where each range is represented by a tuple (start, end).
            The start and end indices indicate the positions of the keys
            in the sorted dictionary.

    Returns:
        int or None: The relevance score corresponding to the index number.
            If the input number does not fall within any range, it returns None.

    Example:
        range_mapping = {1: (0, 0), 2: (1, 2), 3: (3, 4)}

        relevance = get_relevance_for_range(2, range_mapping)
        print(relevance)  # Output: 2a

        relevance = get_relevance_for_range(5, range_mapping)
        print(relevance)  # Output: None
    """
    for key, value_range in range_mapping.items():
        if value_range[0] <= index_in_list <= value_range[1]:
            return key
    return 0


def calculate_gain(true_rank, assumed_rank):
    den = max(true_rank, assumed_rank)
    num = min(true_rank, assumed_rank)
    gain = (2 ** num - 1) / np.log2(den + 2)
    return gain


def rdcgl(search_results,
          true_relevance_set,
          k=None,
          fp_penalty=1,
          fn_penalty=1,
          invert=True,
          punish_max=False):
    """
    The "RaDiCuGLoss" Algorithm
    Computes the Rank Discounted Cumulative Gain and Loss (RDCGL) at a specified rank
    position for a given set of search results.

    The function assesses the quality of a ranking system by measuring the
    usefulness of the search results, while also considering the impact of false
    positives and false negatives. It provides options to invert the relevance
    scores and choose the penalty calculation method based on the maximum
    possible rank or the rank of the last seen relevant item.

    Parameters
    ----------
    search_results : list
        The list of items ranked by a search algorithm or recommendation system.
    true_relevance_set : dict
        A dictionary of items where keys are items and values are their
        respective relevance ranking.
    k : int, optional
        The rank position at which to stop computing RDCG. If None, the function
        computes the RDCG over all items. Defaults to None.
    fp_penalty : float, optional
        The penalty factor to be applied for each false positive (irrelevant
        item returned). Defaults to 0, i.e., no penalty.
    fn_penalty : float, optional
        The penalty factor to be applied for each false negative
        (relevant item not returned). Defaults to 0, i.e., no penalty.
    invert : bool, optional
        If True, the relevance ranking scores are inverted (high scores become
        low and vice versa). This is useful when low rank values represent high
        relevance (e.g., rank 1 is more relevant than rank 5). Defaults to True.
    punish_max : bool, optional
        If True, the penalty for false positives/negatives is based on the
        maximum possible rank; if False, it is based on the rank of the last
        seen relevant item for false positives, or the rank of the unreturned
        relevant item for false negatives. Defaults to False.

    Returns
    -------
    rdcg : float
        The Rank Discounted Cumulative Gain score at position k for the given
        ranked list of items.

    Notes
    -----
    The function prints intermediate steps and information to stdout for
    debugging purposes.

    The function relies on a helper function 'build_assumed_renages_mapping'
    (not shown in this snippet) which is expected to build a mapping of assumed
    ranges.
    """

    print('\n', '=' * 25, 'Calculating RaDiCuGLoss', '=' * 25)
    print(f'\nSearch results: {search_results}')
    rdcgl = 0  # init score
    if k is None:
        k = len(search_results)
    max_rank = max(true_relevance_set.values())
    print(f'\nMax rank={max_rank}')
    last_seen_rank = 1  # initializing with the best possible rank

    assumed_ranges = build_assumed_ranges_mapping(true_relevance_set)
    print(f'assumed_ranges: {assumed_ranges}')

    # iterating over the search results
    print('\nCheking for correct rankings, ranking inconsistencies and false positives:')
    for i, item in enumerate(search_results[:k]):
        if item in true_relevance_set:  # good results gain score according to their relevancy ranking
            true_rank = true_relevance_set[item]
            assumed_rank = get_rank_for_index(i, assumed_ranges)
            print(f'Result #{i + 1}: {str(item).ljust(10)} original rank={true_rank}\t ', end='')
            if invert:
                true_rank = invert_rank(true_rank, max_rank)
                assumed_rank = invert_rank(assumed_rank, max_rank)

            print(f'true rank={true_rank}\t assumed rank={assumed_rank}', end='\t\t')
            # assert get_rank_for_index(i, assumed_ranges) == true_rank if i <= len(true_relevance_set) else None

            gain = calculate_gain(true_rank, assumed_rank)
            print(f'Gain: {gain:.3f}')
            rdcgl += gain

            last_seen_rank = true_rank
        else:  # punishing infiltrators
            if fp_penalty == 0:
                print(f'Result #{i + 1}: {str(item).ljust(10)} not in true set, but no penalty requested.')
                continue
            if fp_penalty > 0:
                if punish_max:
                    penalty = fp_penalty / np.log2(
                        max_rank + 1)
                else:
                    penalty = fp_penalty / np.log2(
                        last_seen_rank + 2)
                    # TODO: VERIFY THE last_seen_rank MECHANISM

                print(f'Result #{i + 1}: {str(item).ljust(10)} not in true set.\t Penalty: -{penalty:.3f}', end='\t')
                if punish_max:
                    print('(max)')
                else:
                    print()
                rdcgl -= penalty
    print('Done.')

    # iterating over the true ranking set - this can only punish, not gain
    if fn_penalty > 0:
        print('\nChecking for false negatives:')
        for i, (item, true_rank) in enumerate(true_relevance_set.items()):
            if invert:
                true_rank = invert_rank(true_rank, max_rank)
            if item not in search_results:
                if punish_max:
                    penalty = fn_penalty
                else:
                    penalty = fn_penalty / np.log2(max_rank + 2 - true_rank)

                print(
                    f'True item: {str(item).ljust(10)} true_rank={true_rank}\t not in results set.\t Penalty: -{penalty:.3f}',
                    end='\t')
                if punish_max:
                    print('(max)')
                else:
                    print()
                rdcgl -= penalty
        print('Done.')
    json_logger.info(f'RDCGL: {rdcgl:.3f}')
    print(f'\nRDCGL: {rdcgl:.3f}')
    return rdcgl


def nrdcgl(search_results,
           true_relevance_set,
           k=None,
           fp_penalty=1,
           fn_penalty=1,
           invert=True,
           punish_max=False):
    """
    Normalized RaDiCuGLoss
    """
    json_logger.info(f'Calculating NRDCGL for search results: {search_results}')
    json_logger.info(f'and true relevance set: {true_relevance_set}')

    print('\n** Building perfect search results list for normalization. **')
    perfect_results = sorted(true_relevance_set,
                             key=true_relevance_set.get,
                             reverse=not invert)
    rdcgl_max = rdcgl(perfect_results,  # no penalty for ideal list
                      true_relevance_set)
    print('\n** Done calculating max result according to a perfect search result list **')
    if not rdcgl_max:
        return 0.
    rdcgl_pred = rdcgl(search_results,
                       true_relevance_set,
                       k,
                       fp_penalty,
                       fn_penalty,
                       invert=invert,
                       punish_max=punish_max)
    _nrdcgl = rdcgl_pred / rdcgl_max
    json_logger.info(f'NRDCGL: {_nrdcgl:.3f}')
    print(f'NRDCGL: {_nrdcgl:.3f}')
    return _nrdcgl


def pnrdcgl(search_results,
            true_relevance_set,
            k=None,
            fp_penalty=1,
            fn_penalty=1,
            invert=True,
            punish_max=False):
    """
    Positive Normalized RaDiCuGLoss (to keep scores between 0.0 and 1.0)
    """
    _nrdcgl = nrdcgl(search_results,
                     true_relevance_set,
                     k=k,
                     fp_penalty=fp_penalty,
                     fn_penalty=fn_penalty,
                     invert=invert,
                     punish_max=punish_max)

    _pnrdcgl = max(_nrdcgl, 0)
    json_logger.info(f'PNRDCGL: {_pnrdcgl:.3f}')
    return _pnrdcgl


def main():
    search_results = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    true_relevance_set = {'a': 1, 'b': 2, 'c': 2, 'd': 3, 'e': 3}
    k = 5
    fp_penalty = 1
    fn_penalty = 1
    invert = True
    punish_max = False
    nrdcgl(search_results, true_relevance_set, k, fp_penalty, fn_penalty, invert, punish_max)


if __name__ == '__main__':
    main()
