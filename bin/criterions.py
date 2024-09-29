import scipy as sp
import statsmodels.api as sm
import numpy as np
import pandas as pd
from permutation import count_permutation, count_shapes


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
    if n == 0:
        n = 1
        n1 = 1
        n2 = 1
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
            bonferroni : one-step correctio
            sidak : one-step correction
            holm-sidak : step down method using Sidak adjustments
            holm : step-down method using Bonferroni adjustments
            simes-hochberg : step-up method (independent)
            hommel : closed method based on Simes tests (non-negative)
            fdr_bh : Benjamini/Hochberg (non-negative)
            fdr_by : Benjamini/Yekutieli (negative)
            fdr_tsbh : two stage fdr correction (non-negative)
            fdr_tsbky : two stage fdr correction (non-negative)
    Returns:
        list of bool(true for hypothesis that can be rejected for given alpha), p-values corrected for multiple tests, corrected alpha for Sidak method, corrected alpha for Bonferroni method
    """
    p_list = []
    for pair_index in range(len(permutations) // 2):
        statistic, p_value = equals_test(permutations[pair_index], permutations[-1 - pair_index],
                                         sum(permutations))
        p_list.append(p_value)
    return sm.stats.multipletests(p_list, alpha, method)


def double_equals_test(time_series: pd.Series, alpha: float = 0.01, method: str = 'hs'):
    """
    Double symmetry category test with multiple hypothesis of equals tests (window=3,4)
    Args:
        time_series: Time series
        alpha: overall significance level
        method: method to multipletest

    Returns: True if time series have symmetrical permutation distribution, False otherwise

    """
    p_ts3 = count_permutation(time_series, 3)
    result3 = multiply_equals_test(p_ts3, alpha=alpha, method=method)
    p_ts4 = count_permutation(time_series, 4)
    result4 = multiply_equals_test(p_ts4, alpha=alpha, method=method)
    if bool(sum(result3[0])) == False and bool(sum(result4[0])) == False:
        return True
    else:
        return False


def chi_square_test(data, alpha=0.05):
    """
    A function to test a sample for discrete uniform distribution using the chi-square test.

    Args:
        data (list): The frequency of each value.
        alpha (float): Significance level (default 0.05).

    Returns:
        bool: True, if the null hypothesis of uniform distribution is rejected, False otherwise.
    """

    n_samples = len(data)
    expected_frequencies = np.ones(n_samples) * (sum(data) / n_samples)
    res = sp.stats.chi2_contingency([data, expected_frequencies])
    chi2_stat, p_value = res.statistic, res.pvalue
    df = len(expected_frequencies) - 1
    critical_value = sp.stats.chi2.ppf(alpha, df)

    if chi2_stat > critical_value:
        return False, p_value, chi2_stat, critical_value
    else:
        return True, p_value, chi2_stat, critical_value


def double_chi2_test(time_series: pd.Series, alpha: float = 0.95):
    """
    Double test a sample for discrete uniform distribution using the chi-square test. (window=4,5)
    Args:
        time_series: Time series
        alpha: overall significance level

    Returns: True if time series have uniform shape distribution, False otherwise

    """
    s_ts4 = count_shapes(time_series, 4)
    result4 = chi_square_test(s_ts4, alpha=alpha)
    s_ts5 = count_shapes(time_series, 5)
    result5 = chi_square_test(s_ts5, alpha=alpha)
    if result4[0] == True and result5[0] == True:
        return True
    else:
        return False
