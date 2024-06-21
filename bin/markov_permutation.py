from collections import defaultdict
import numpy as np
import pandas as pd


def compute_transition_matrix(sequence: pd.Series) -> np.ndarray:
    """
    Compute the transition matrix of Markov chain of permutation
    :param sequence(pd.Series): series of permutations
    :return: transition matrix(np.ndarray)
    """
    states = sequence.unique()
    states.sort()
    transitions = defaultdict(lambda: defaultdict(int))

    for i in range(len(sequence) - 1):
        current_state = sequence[i]
        next_state = sequence[i + 1]
        transitions[current_state][next_state] += 1

    transition_matrix = np.zeros((len(states), len(states)))
    for i, state in enumerate(states):
        total_transitions = sum(transitions[state].values())
        # Заполню одинаковыми значениями
        if total_transitions == 0:
            transition_matrix[i] = [1 / len(states)] * len(states)
        else:
            for j, next_state in enumerate(states):
                transition_count = transitions[state][next_state]

                transition_matrix[i][j] = transition_count / total_transitions

    return transition_matrix
