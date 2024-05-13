import pandas as pd
import itertools as it


def remove_consecutive_duplicates(arr):
    """
    Remove consecutive duplicate values from an array.

    Parameters:
        arr (pd.Series): Input array.

    Returns:
        pd.Series: Series without consecutive duplicate values.
    """
    arr_series = pd.Series(arr)
    unique_values = arr_series[arr_series.diff() != 0].reset_index(drop=True)
    return unique_values


def count_permutation(ts: pd.Series, window: int) -> list:
    """
    Count permutation in time series
    Parameters:
        ts(pd.Series): time series
        window(int): window for walking through a time series for which the type of permutation is located
    Returns:
        count_p(list): vector with a number of permutations of different types. The length of the vector is equal to the factorial of the window value
    """
    r = list(range(window))
    permutation_list = list(it.permutations(r))
    count_p = [0] * len(permutation_list)
    real_ts = remove_consecutive_duplicates(ts.copy())
    for i in range(real_ts.size - window + 1):
        slice_df = real_ts.copy().iloc[i:i + window]
        sort_slice = slice_df.copy().sort_values()
        for j in range(window):
            slice_df[slice_df[slice_df == sort_slice.iloc[j]].first_valid_index()] = j
        count_p[permutation_list.index(tuple(slice_df.tolist()))] += 1
    return count_p


def permutation_series(ts: pd.Series, window: int) -> pd.Series:
    """
        Calculation of a series of permutations
        Parameters:
            ts(pd.Series): time series
            window(int): window for walking through a time series for which the type of permutation is located
        Returns:
            pseries(pd.Series): Series of permutations
        """
    r = list(range(window))
    permutation_list = list(it.permutations(r))
    pseries = []
    real_ts = remove_consecutive_duplicates(ts.copy())
    for i in range(real_ts.size - window + 1):
        slice_df = real_ts.copy().iloc[i:i + window]
        sort_slice = slice_df.copy().sort_values()
        for j in range(window):
            slice_df[slice_df[slice_df == sort_slice.iloc[j]].first_valid_index()] = j
        pseries.append(permutation_list.index(tuple(slice_df.tolist())))
    return pd.Series(pseries)
