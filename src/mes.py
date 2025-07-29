from typing import Optional, Callable

import numpy as np
import time
import tqdm


EPSILON = 1e-6


def config_numpy():
    np.seterr(divide='ignore', invalid='ignore')


def binary_decsisions_feasible_updater(feasible_mask: np.ndarray, chosen: int):
    num_projs = feasible_mask.shape[0] // 2
    feasible_mask[[chosen, (chosen + num_projs) % feasible_mask.shape[0]]] = False


def dummy_feasible_updater(feasible_mask: np.ndarray, chosen: int):
    pass


def calculate_mes(
        valuations: np.ndarray,
        costs: Optional[np.ndarray] = None,
        budgets: Optional[np.ndarray] = None,
        safe: bool = True,
        feasible_updater: Optional[Callable[[np.ndarray, int], None]] = None,
        verbose: bool = False,
) -> np.ndarray:
    if verbose:
        start = time.time_ns()
    M = valuations.shape[0]
    N = valuations.shape[1]
    costs = costs if costs is not None else np.ones(M)
    budgets = budgets if budgets is not None else np.ones(N) * M / (2 * N)
    if safe:
        assert M == costs.shape[0]
        assert N == budgets.shape[0]
        assert (valuations >= 0).all()
        assert (costs > 0).all()
        assert (budgets > 0).all()

    selected = np.zeros(M)
    feasible_mask = np.ones(M, dtype=bool)
    while True:
        T = budgets / valuations[feasible_mask]
        T[np.isnan(T)] = 0
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
        rhos = (costs[feasible_mask] - P[tmp]) / S[tmp]
        mask = b[tmp] >= costs[feasible_mask] - EPSILON
        if not mask.any():
            break
        chosen = np.argmin(rhos[mask])
        chosen = np.where(mask)[0][chosen]
        rho = rhos[chosen]

        budgets[sigma[chosen][:indexes[chosen]]] = 0
        budgets[sigma[chosen][indexes[chosen]:]] -= rho * valuations[feasible_mask][chosen, sigma[chosen][indexes[chosen]:]]
        chosen_adj = np.where(feasible_mask)[0][chosen]
        selected[chosen_adj] = 1
        feasible_mask[chosen_adj] = False
        if feasible_updater is not None:
            feasible_updater(feasible_mask, chosen_adj)
        if not sum(feasible_mask):
            break
    if verbose:
        print(f"computed mes over {M=}, {N=} in {(time.time_ns() - start) / 1_000_000} ms")
    return selected


def mes_chunked(
        chunk_size: int,
        valuations: np.ndarray,
        costs: Optional[np.ndarray] = None,
        budgets: Optional[np.ndarray] = None,
        safe: bool = True,
        feasible_updater: Optional[Callable[[np.ndarray, int], None]] = None,
        verbose: bool = False,
) -> np.ndarray:
    if verbose:
        start = time.time_ns()
    M = valuations.shape[0]
    N = valuations.shape[1]
    if costs is None:
        costs = np.ones(M)
    if budgets is None:
        budgets = np.ones(N) * M / (2 * N)
    if M < chunk_size * 2:
        return calculate_mes(valuations, costs, budgets, safe, feasible_updater)
    splits = range(chunk_size, M - chunk_size, chunk_size) # last chunk will be bigger
    if not len(splits):
        return calculate_mes(valuations, costs, budgets, safe, feasible_updater)
    res = []
    prev_sep = 0
    for sep in splits:
        valuations_chunk = valuations[prev_sep:sep]
        costs_chunk = costs[prev_sep:sep]
        budgets_chunk = budgets / M * (sep - prev_sep)
        res.append(calculate_mes(valuations_chunk, costs_chunk, budgets_chunk, safe, feasible_updater))
    res = np.concatenate(res)
    if verbose:
        print(f"computed mes over {M=}, {N=} in {(time.time_ns() - start) / 1_000_000} ms")
    return res

if __name__ == '__main__':
    config_numpy()
    valuations = np.array([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 1],
        [0, 2, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])
    print(calculate_mes(valuations, feasible_updater=binary_decsisions_feasible_updater))