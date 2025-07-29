import numpy as np
import tqdm

from src.mes import calculate_mes, config_numpy, binary_decsisions_feasible_updater, mes_chunked
from time import time_ns
import pandas as pd


def run_mes(voters_no, candidates_no):
    utils = np.random.randn(candidates_no, voters_no)
    msk = utils < 0
    N = utils.shape[0]
    M = utils.shape[1]
    res = np.zeros(shape=(N, M * 2))
    res[:, :M] += utils * np.logical_not(msk)
    res[:, M:] += utils * -1. * msk
    start = time_ns()
    _ = calculate_mes(res, feasible_updater=binary_decsisions_feasible_updater, safe=False, verbose=False)
    end = time_ns()
    return (end - start) / 1_000_000


def run_mes_chunked(voters_no, candidates_no):
    utils = np.random.randn(candidates_no, voters_no)
    msk = utils < 0
    N = utils.shape[0]
    M = utils.shape[1]
    res = np.zeros(shape=(N, M * 2))
    res[:, :M] += utils * np.logical_not(msk)
    res[:, M:] += utils * -1. * msk
    start = time_ns()
    _ = mes_chunked(2048, res, feasible_updater=binary_decsisions_feasible_updater, safe=False, verbose=False)
    end = time_ns()
    return (end - start) / 1_000_000


BATCH_SIZES = [16]
CANDIDATE_NUMS = [10000]
REPEATS = 1

if __name__ == '__main__':
    config_numpy()
    results = {(bs, cn): [] for bs in BATCH_SIZES for cn in CANDIDATE_NUMS}
    for bs in BATCH_SIZES:
        for cn in CANDIDATE_NUMS:
            print(bs, cn)
            for _ in range(REPEATS):
                results[(bs, cn)].append(run_mes(bs, cn))
    string = pd.DataFrame(results).describe()
    print(string)
    """string.to_csv('mes_benchmark.csv')
    for bs in BATCH_SIZES:
        for cn in CANDIDATE_NUMS:
            print(bs, cn)
            for _ in tqdm.tqdm(range(REPEATS)):
                results[(bs, cn)].append(run_mes_chunked(bs, cn))
    string = pd.DataFrame(results).describe()
    print(string)
    string.to_csv('mes_chunked_benchmark.csv')"""