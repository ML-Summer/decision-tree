from math import log2
from pandas import Series


def labelOccurrenceRatios(labels: Series | list) -> dict:
    """
    Calculates labels' occurrence ratio to the whole list size.
    # Input
    - `labels` - a `pandas.Series` or list object that contains labels to calculate occurrence ratios from.
    # Output
    A dictionary containing pairs `label:occurence_ratio`
    # Example
    - input: [1, 1, 0]
    - output: {1: 0.6666..., 0: 0.3333...}
    """
    ratios = {label: 0 for label in labels}
    amount_of_all_labels = len(labels)
    for label in labels:
        ratios[label] = ratios[label] + 1
    for label in ratios.keys():
        ratios[label] /= amount_of_all_labels
    return ratios


def entropy(labels: Series | list) -> float:
    """
    Calculates entropy(information gain) of provided feature vector.
    # Parameter
    - `features` - a `pandas.Series` or list object that contains labels.
    # Output
    An entropy value in float format.
    """
    ratios = labelOccurrenceRatios(labels)
    entropy = 0.0
    for ratio in ratios.values():
        entropy += ratio * log2(ratio)
    entropy *= -1
    entropy_rounded = round(entropy, 4)
    return entropy_rounded
