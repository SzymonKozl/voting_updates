from typing import Optional

import numpy as np


EPSILON = 1e-6


def calculate_mes(valuations: np.ndarray, costs: Optional[np.ndarray] = None, budgets: Optional[np.ndarray] = None) -> np.ndarray:
    M = valuations.shape[0]
    N = valuations.shape[1]
    costs = costs if costs is not None else np.ones(M)
    budgets = budgets if budgets is not None else np.ones(N) * M / (2 * N)
    assert M == costs.shape[0]
    assert N == budgets.shape[0]

    selected = np.zeros(M)
    feasible_mask = np.ones(M, dtype=bool)
    np.seterr(divide='ignore') # we might have 0 utilities and it is ok
    while True:
        T = budgets / valuations[feasible_mask]
        sigma = np.argsort(T, axis=1)
        T_prime = np.take_along_axis(T, sigma, axis=1)
        inf_mask = np.isinf(T_prime)
        T_prime[inf_mask] = 0.
        B_prime = np.take_along_axis(budgets[None, :].repeat(T_prime.shape[0], axis=0), sigma, axis=1)
        B_prime[inf_mask] = 0.
        P = np.pad(np.cumsum(B_prime, axis=1), ((0, 0), (1, 0)), 'constant')[:,:-1]
        U_prime = np.take_along_axis(valuations[feasible_mask], sigma, axis=1)
        S = np.cumsum(U_prime[:, ::-1], axis=1)[:, ::-1]
        a = S * T_prime
        b = a + P
        if (np.max(b, axis=1) < costs[feasible_mask] - EPSILON).all():
            break
        indexes = np.argmax(b >= costs[feasible_mask][:, None] - EPSILON, axis=1)
        tmp = np.arange(len(indexes)), indexes
        rhos = (1 - P[tmp]) / S[tmp]
        mask = b[tmp] >= 1 - EPSILON
        chosen = np.argmin(rhos[mask])
        chosen = np.where(mask)[0][chosen]
        rho = rhos[chosen]

        budgets[sigma[chosen][:indexes[chosen]]] = 0
        budgets[sigma[chosen][indexes[chosen]:]] -= rho * valuations[feasible_mask][chosen, sigma[chosen][indexes[chosen]:]]
        chosen_adj = np.where(feasible_mask)[0][chosen]
        selected[chosen_adj] = 1
        feasible_mask[[chosen_adj, (chosen_adj + M // 2) % M]] = False
        if not sum(feasible_mask):
            break
    return selected


if __name__ == '__main__':
    valuations = np.array([
        [1, 2, 3],
        [4, 0, 6],
        [0, 1, 3],
        [0, 0, 2]
    ])
    print(calculate_mes(valuations))