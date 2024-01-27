import pandas as pd
import numpy as np
import statsmodels.api as sm
import datetime as dt
import itertools as it
import scipy as sp


def generate_ts(date_start: str = "29/06/2023 00:00:00", date_end: str = "30/07/2023 00:00:00", alpha: float = 0.6,
                beta: float = 0.4, max_kf: float = 5, trend: float = 4, season: float = 0):
    """
    Generate time series with autocorrelation, trend and seasons dependency.
    Parameters:
        date_start(str): first data in time series in 'DD/MM/YYYY HH:MM:SS' format
        date_end(str): last data in time series in 'DD/MM/YYYY HH:MM:SS' format
        alpha(float): correlation coefficient with previous value
        beta(float): correlation coefficient with previous last value
        max_kf(float): maximum absolute value of the additional noise component with uniform distribution
        trend(float): absolute additional value for introducing a trend component
        season(float): absolute addition value for elements that are one week apart
    Returns:
        time_series(np.Series): generated time series
    """
    time_series = pd.date_range(start=date_start, end=date_end, freq='H').to_series()
    time_series.index = time_series
    time_series[0] = 5000.0
    time_series[1] = 5015.0
    date = 2
    for i in time_series[2:]:
        time_series[i] = (alpha * time_series[i - dt.timedelta(1 / 24)] + beta * time_series[
            i - dt.timedelta(2 / 24)]) + np.random.uniform(-max_kf, max_kf) + trend
        if date % (7 * 24) == 0:
            time_series[i] += season
        if date % (7 * 24) == 1:
            time_series[i] -= season
        date += 1
    return time_series


def random_walk(date_start: str = "29/06/2023 00:00:00", date_end: str = "30/07/2023 00:00:00", noise: float = 10,
                noise_dist: str = 'normal'):
    """
    Generate time series as random walk.
    Parameters:
        date_start(str): first data in time series in 'DD/MM/YYYY HH:MM:SS' format
        date_end(str): last data in time series in 'DD/MM/YYYY HH:MM:SS' format
        noise(float): noise component
        noise_dist(str): distribution of noise. 'normal': np.random.normal(0,noise). 'uniform': np.random.uniform(-noise, noise)
    Returns:
        time_series(np.Series): generated time series
    """
    time_series = pd.date_range(start=date_start, end=date_end, freq='H').to_series()
    time_series.index = time_series
    time_series[date_start] = 5000.0
    if noise_dist != 'uniform':
        for i in time_series[1:]:
            time_series[i] = time_series[i - dt.timedelta(1 / 24)] + np.random.normal(0, noise)
    else:
        for i in time_series[1:]:
            time_series[i] = time_series[i - dt.timedelta(1 / 24)] + np.random.uniform(-noise, noise)
    return time_series


def count_permutation(ts: pd.Series, window: int):
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
    for i in range(ts.size - window + 1):
        slice_df = ts.copy().iloc[i:i + window]
        sort_slice = slice_df.copy().sort_values()
        for j in range(window):
            slice_df[slice_df[slice_df == sort_slice.iloc[j]].first_valid_index()] = j
        count_p[permutation_list.index(tuple(slice_df.tolist()))] += 1
    return count_p


def entropy(permutations: list):
    """
    Count permutation entropy of time series.
    Parameters:
        permutations(list): vector with a number of permutations of different types
    Returns:
        ent(float): permutation entropy
    """
    allperm = sum(permutations)
    ent = 0
    for i in range(len(permutations)):
        probability = permutations[i] / allperm
        if probability != 0:
            ent -= probability * np.log2(probability)
    return ent


def equals_test(n1: int, n2: int, n: int):
    """
    Test of two value of permutations equally likely
    Parameters:
        n1(int): number of objects with the first property
        n2(int): number of objects with the second property
        n(int): number of all objects in sample
    Returns:
        z, p_value((float,float)): statistics and p_value of test
    """
    w1 = n1 / n
    w2 = n2 / n
    p = (n1 + n2) / (2 * n)
    z = 0
    if p != 0 and p != 1:
        z = (w1 - w2) / (p * (1 - p) * (2 / n)) ** 0.5
    mod_z = abs(z)
    p_value = 2 * (1 - sp.stats.norm.cdf(mod_z))
    return z, p_value


def multiply_equals_test(permutations: list, alpha: float = 0.05, method: str = "hs"):
    """
    Symmetry category test with multiple hypothesis of equals tests
    Parameters:
        permutations(list): vector with a number of permutations of different types
        alpha(float): overall significance level
        method(str): method to multipletest
    Returns:
        list of bool(true for hypothesis that can be rejected for given alpha), p-values corrected for multiple tests, corrected alpha for Sidak method, corrected alpha for Bonferroni method
    """
    p_list = []
    for pair_index in range(len(permutations) // 2):
        statistic, p_value = equals_test(permutations[pair_index], permutations[-1 - pair_index],
                                         sum(permutations))
        p_list.append(p_value)
    return sm.stats.multipletests(p_list, alpha, method)
