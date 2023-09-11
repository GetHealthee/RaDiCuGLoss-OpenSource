# RaDiCuGLoss

The RaDiCuGLoss **(Rank Discounted Cumulative Gain and Loss)** project provides a Python implementation of a search result evaluation algorithm. It is based on a development of a variation of the traditional Discounted Cumulative Gain (DCG) method with extensions to account for ranking inconsistencies and false positives or negatives.

The aim is to provide a more accurate measure of the effectiveness of a search algorithm by taking into account not just the relevance of the search results, but also their ordering and possible omissions or inaccuracies.

*Inspiration and foundation for RaDiCuGLoss: https://torchmetrics.readthedocs.io/en/v0.8.1/retrieval/normalized_dcg.html*

## Algorithm

The RaDiCuGLoss algorithm works as follows:

1. Given a list of search results, it checks for correct rankings, ranking inconsistencies, and false positives. It then calculates the gain for each result using the formula `(2**true_rank - 1) / np.log2(assumed_rank + 1)`. This formula gives a higher score to items that are ranked higher and penalizes items that are ranked lower than they should be.

2. The algorithm then checks for false negatives, i.e., relevant items that were not included in the search results. These are penalized by subtracting a certain value from the total gain.

3. The total gain is then normalized by dividing it by the maximum possible gain, which is calculated based on a perfect ordering of the search results.

## Usage

Here is an example of how to use the `nrdcgl` function from the RaDiCuGLoss module:

```python

# Example search results and true relevance set
search_results = ['Elad', 'Uzi', 'Gold', 'Namer', 'Yoav', 'GT']
true_relevance_set = {'Elad': 3, 'Yoav': 2, 'Gold': 2, 'Namer': 2, 'Uzi': 1, 'GT': 1}

# Calculate NRDCGL
nrdcgl(search_results, true_relevance_set, k=None, fp_penalty=1, fn_penalty=1, invert=True, punish_max=False)

```
Note that `nrdcgl` calculates the normalized score. If you want the absolute score for some reason, use `rdcgl`. 

### Parameters

The `rdcgl` function has the following parameters:

- **search_results**: A list of items representing the search results. Each item in the list is expected to be a string representing the item's identifier.

- **true_relevance_set**: A dictionary representing the true relevance of the items. The keys of the dictionary are the item identifiers, and the values are the true ranks of the items.

- **k**: An integer representing the cutoff for calculating the RaDiCuGLoss. If `k` is None, the function considers all items in the search results.

- **fp_penalty**: A float value that indicates the penalty for false positives. This is the ratio of the penalty to the gain of the incorrectly ranked item. The default value is 1.0.

- **fn_penalty**: A float value that indicates the penalty for false negatives. This is the ratio of the penalty to the gain of the missing item. The default value is 1.0.

- **invert**: A boolean that indicates whether to invert the ranks or not. If set to True, the function assumes that a smaller rank value is better. If set to False, it assumes that a larger rank value is better. The default value is True.

- **punish_max**: A boolean that indicates whether to penalize the items that have the maximum rank value. If set to True, the function applies the false positive penalty to the items with the maximum rank. If set to False, it does not apply any penalty. The default value is False.

## Algorithm Concepts

The development of the RaDiCuGLoss algorithm involves a series of steps and helper functions. Here's an overview of the process and the functions involved:

1. **Inverting Ranks**: When the `invert` parameter is set to True, the function treats a smaller rank value as better. For instance, in a search result, the item at the top (rank 1) is considered better than the item at the bottom (rank n). To accomplish this, we use the `invert_rank` helper function, which simply inverts the ranks by subtracting each rank from the maximum rank plus 1.

2. **Assumed Ranks Mapping**: After inverting the ranks, the next step is to compute the assumed ranks of the items in the search results. For this, we use the `build_assumed_ranges_mapping` function. It generates a mapping between the original rank (index in the search results) and the assumed rank. This is a key part of the RaDiCuGLoss computation as it identifies rank inconsistencies in the search results.

3. **Calculating Gain**: Once we have the assumed ranks, we calculate the gain of each item using the `calculate_gain` function. This function uses both the true and assumed ranks to compute the gain. The gain computation involves a modified version of the Discounted Cumulative Gain formula, where a higher ranking consistency is rewarded more than a lower one.

4. **False Positives and Negatives**: The RaDiCuGLoss algorithm also takes into account false positives and negatives in the search results. For false positives (items in the search results that are not in the true relevance set), it applies a penalty proportional to the `fp_penalty` parameter. Similarly, for false negatives (items in the true relevance set that are not in the search results), it applies a penalty proportional to the `fn_penalty` parameter.

5. **Normalization**: Finally, the raw RaDiCuGLoss value is normalized by comparing it with the maximum possible RaDiCuGLoss value (obtained by sorting the true relevance set perfectly). The function for this is `nrdcgl`, which includes the normalization process.


