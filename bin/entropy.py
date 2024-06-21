import numpy as np
import itertools as it

import pandas as pd

from permutation import permutation_series, count_permutation
from markov_permutation import compute_transition_matrix


def entropy(permutations: list) -> float:
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


def entropy_rate(transition_matrix):
    """
    Compute the entropy rate  of a Markov chain of permutation and entropy of stationary distribution of permutation states.

    Parameters:
        transition_matrix (numpy.ndarray): Transition matrix of the Markov chain.

    Returns:
        float, float: Entropy rate and Markov permutation entropy.
    """

    # Compute stationary distribution
    # TODO разобраться с нормальным вычислением. С помощью собственного вектора возникает проблема, когда 1 - собственное число кратности больше одного. С помощью решения системы уравнения stat_dist * P = stat_dist возникает проблема детерминантна, равного нулю. Пока что вычисляем приближенно за 50 шагов.
    # eigenvalues, eigenvectors = np.linalg.eig(transition_probabilities.T)
    # stationary_vector = np.real(eigenvectors[:, np.isclose(eigenvalues, 1)])

    # n = len(transition_probabilities)
    # A = np.append(transpose(transition_probabilities)-identity(n), [[1 for _ in range(n)]], axis=0)
    # b = transpose(np.array([0 for _ in range(n)]+[1]))
    # stationary_distribution = np.linalg.solve(transpose(A).dot(A), transpose(A).dot(b))

    matrix_dot = transition_matrix.copy()
    for _ in range(6):
        matrix_dot = np.dot(matrix_dot, matrix_dot)
    stationary_distribution = np.mean(matrix_dot, axis=0)

    # Normalize the stationary distribution
    stationary_distribution /= stationary_distribution.sum()

    # Compute entropy rate of vectors stationary_distribution
    ent_rate = -np.sum(np.nan_to_num(stationary_distribution * np.log2(stationary_distribution)))

    # Compute entropy rate of transition matrix
    tr_ent_rate = -np.sum(np.nan_to_num(
        stationary_distribution * np.sum(np.nan_to_num(transition_matrix * np.log2(transition_matrix)),
                                         axis=0)))

    return ent_rate, tr_ent_rate


def permutation_stationary_entropy(time_series: pd.Series, window: int) -> float:
    """
    Compute the entropy rate  of a Markov chain of permutation
    :param time_series: Time series
    :param window: window for walking through a time series
    :return: entropy
    """
    permutation = permutation_series(time_series, window)
    transition_matrix = compute_transition_matrix(permutation)

    matrix_dot = transition_matrix.copy()
    for _ in range(6):
        matrix_dot = np.dot(matrix_dot, matrix_dot)

    stationary_distribution = np.mean(matrix_dot, axis=0)

    # Normalize the stationary distribution
    stationary_distribution /= stationary_distribution.sum()

    # Compute entropy rate of vectors stationary_distribution
    ent_rate = -np.sum(np.nan_to_num(stationary_distribution * np.log2(stationary_distribution)))
    return ent_rate


def permutation_entropy_rate(time_series: pd.Series, window: int, num_transitions: int = 0) -> float:
    permutation = permutation_series(time_series, window)
    transition_matrix = compute_transition_matrix(permutation)

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
    tr_ent_rate = -np.sum(np.nan_to_num(
        stationary_distribution * np.sum(np.nan_to_num(transition_probabilities * np.log2(transition_probabilities)),
                                         axis=0)))
    return tr_ent_rate


def permutation_entropy(time_series: pd, window: int):
    perm_ts = count_permutation(time_series, window)
    return entropy(perm_ts)
