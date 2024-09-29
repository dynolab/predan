import numpy as np
import itertools as it

import pandas as pd

from permutation import permutation_series, count_permutation, count_shapes, shape_series
from markov_permutation import compute_transition_matrix


def entropy(freq: list, relative: bool = True) -> float:
    """
    Count Shannon entropy.
    Parameters:
        freq(list): vector with a frequency of different types
        relative(bool): scaling entropy to interval from 0 to 1
    Returns:
        ent(float): Shannon entropy
    """
    allfreq = sum(freq)
    ent = 0
    if allfreq != 0:
        for i in range(len(freq)):
            probability = freq[i] / allfreq
            if probability != 0:
                ent -= probability * np.log2(probability)
        if relative:
            ent = ent / np.log2(len(freq))
        return ent
    else:
        return 0


def stationary_entropy(series: pd.Series, relative: bool = True) -> float:
    """
    Count Shannon entropy of stationary distribution a Markov chain
    Args:
        series: series of quantization time series
        relative: scaling entropy to interval from 0 to 1

    Returns: Shannon entropy

    """
    transition_matrix = compute_transition_matrix(series)

    matrix_dot = transition_matrix.copy()
    for _ in range(6):
        matrix_dot = np.dot(matrix_dot, matrix_dot)

    stationary_distribution = np.mean(matrix_dot, axis=0)

    # Compute stationary distribution
    # TODO разобраться с нормальным вычислением. С помощью собственного вектора возникает проблема, когда 1 - собственное число кратности больше одного. С помощью решения системы уравнения stat_dist * P = stat_dist возникает проблема детерминантна, равного нулю. Пока что вычисляем приближенно за 50 шагов.
    # eigenvalues, eigenvectors = np.linalg.eig(transition_probabilities.T)
    # stationary_vector = np.real(eigenvectors[:, np.isclose(eigenvalues, 1)])

    # n = len(transition_probabilities)
    # A = np.append(transpose(transition_probabilities)-identity(n), [[1 for _ in range(n)]], axis=0)
    # b = transpose(np.array([0 for _ in range(n)]+[1]))
    # stationary_distribution = np.linalg.solve(transpose(A).dot(A), transpose(A).dot(b))

    # Normalize the stationary distribution
    stationary_distribution /= stationary_distribution.sum()

    # Compute entropy rate of vectors stationary_distribution
    ent_rate = -np.sum(np.nan_to_num(stationary_distribution * np.log2(stationary_distribution)))
    if relative:
        ent_rate = ent_rate / np.log2(len(stationary_distribution))
    return ent_rate


def entropy_rate(series: pd.Series, num_transitions: int = 0, relative: bool = True):
    """
    Compute the entropy rate  of a Markov chain
    Args:
        series: series of quantization time series
        num_transitions: number of degree of transition matrix
        relative: scaling entropy to interval from 0 to 1

    Returns: entropy rate

    """
    transition_matrix = compute_transition_matrix(series)

    matrix_dot = transition_matrix.copy()
    for _ in range(num_transitions):
        matrix_dot = np.dot(matrix_dot, transition_matrix)
    transition_probabilities = matrix_dot.copy()
    for _ in range(5):
        matrix_dot = np.dot(matrix_dot, matrix_dot)
    stationary_distribution = np.mean(matrix_dot, axis=0)

    # Normalize the stationary distribution
    stationary_distribution /= stationary_distribution.sum()

    # Compute entropy rate
    ent_rate = -np.sum(np.nan_to_num(
        stationary_distribution * np.sum(np.nan_to_num(transition_probabilities * np.log2(transition_probabilities)),
                                         axis=0)))
    if relative:
        ent_rate = ent_rate / np.log2(len(stationary_distribution))
    return ent_rate


def sample_entropy(timeseries_data: list, window_size: int, r: float):
    """
    Count sample entropy of time series.
    :param timeseries_data: time series
    :param window_size: window for walking through a time series
    :param r: tolerance interval
    :return: sample entropy
    """

    def construct_templates(ts_data: list, m: int = 2):
        num_windows = len(ts_data) - m + 1
        return [ts_data[x:x + m] for x in range(0, num_windows)]

    def is_match(template_1: list, template_2: list, r1: float):
        return all([abs(x - y) < r1 for (x, y) in zip(template_1, template_2)])

    def get_matches(templates: list, r1: float):
        return len(list(filter(lambda x: is_match(x[0], x[1], r1), it.combinations(templates, 2))))

    B = get_matches(construct_templates(timeseries_data, window_size), r)
    A = get_matches(construct_templates(timeseries_data, window_size + 1), r)
    return -np.log2(A / B)


def permutation_stationary_entropy(time_series: pd.Series, window: int, relative: bool = True) -> float:
    """
    Compute the permutation entropy of stationary distribution a Markov chain of permutation
    :param time_series: Time series
    :param window: window for walking through a time series
    :param relative: scaling entropy to interval from 0 to 1
    :return: entropy
    """
    permutation = permutation_series(time_series, window)
    return stationary_entropy(permutation, relative=relative)


def permutation_entropy_rate(time_series: pd.Series, window: int, num_transitions: int = 0,
                             relative: bool = True) -> float:
    """
    Compute the entropy rate  of a Markov chain of permutation
    Args:
        time_series: Time series
        window: window for walking through a time series
        num_transitions: number of degree of transition matrix
        relative: scaling entropy to interval from 0 to 1

    Returns:

    """
    permutation = permutation_series(time_series, window)
    return entropy_rate(permutation, num_transitions=num_transitions, relative=relative)


def permutation_entropy(time_series: pd.Series, window: int, relative: bool = True) -> float:
    perm_ts = count_permutation(time_series, window)
    return entropy(perm_ts, relative=relative)


def shape_entropy(time_series: pd.Series, window: int, relative: bool = True) -> float:
    shape_ts = count_shapes(time_series, window)
    return entropy(shape_ts, relative=relative)


def shape_stationary_entropy(time_series: pd.Series, window: int, relative: bool = True) -> float:
    """
    Compute the shape entropy of stationary distribution a Markov chain of shapes
    :param time_series: Time series
    :param window: window for walking through a time series
    :param relative: scaling entropy to interval from 0 to 1
    :return: entropy
    """
    shape = shape_series(time_series, window)
    return stationary_entropy(shape, relative=relative)


def shape_entropy_rate(time_series: pd.Series, window: int, num_transitions: int = 0,
                       relative: bool = True) -> float:
    """
    Compute the entropy rate  of a Markov chain of shapes
    Args:
        time_series: Time series
        window: window for walking through a time series
        num_transitions: number of degree of transition matrix
        relative: scaling entropy to interval from 0 to 1

    Returns: entropy rate

    """
    shape = shape_series(time_series, window)
    return entropy_rate(shape, num_transitions, relative=relative)
