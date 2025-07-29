import unittest
import random
import numpy as np
import pytest

from ext.mes_mikolaj.model import *
import sys
sys.path.append('../ext/mes_mikolaj')
from ext.mes_mikolaj.rules import equal_shares

from src.mes import calculate_mes

@pytest.fixture
def funds():
    voters_no = random.randint(1, 20)
    funds = np.random.rand(voters_no)
    budget_total = int(funds.sum())
    to_cut = float(funds.sum()) - budget_total
    funds -= to_cut / voters_no
    funds += 1 / voters_no
    return funds


@pytest.fixture
def costs(funds):
    voters_no = len(funds)
    candidate_no = random.randint(1, voters_no)
    costs = np.random.randint(1, 5, candidate_no)
    return costs


@pytest.fixture
def utils(funds, costs):
    voters_no = len(funds)
    candidate_no = len(costs)
    zero_util_mask = np.random.randint(0, 2, size=(voters_no, candidate_no))
    utils = np.random.rand(voters_no, candidate_no)
    utils *= zero_util_mask
    for i in range(candidate_no):
        if utils[:, i].sum() == 0:
            utils[:, i] += 0.01
    return utils


@pytest.mark.repeat(10)
def test_mes_1(funds, costs, utils):
    voters_no = len(funds)
    candidate_no = len(costs)
    budget_total = int(funds.sum())
    voters = set()
    candidates = set()
    for i in range(voters_no):
        v = Voter(str(i), funds=float(funds[i]))
        voters.add(v)
    for j in range(candidate_no):
        c = Candidate(str(j), cost=int(costs[j]))
        candidates.add(c)
    election = Election(
        voters=voters,
        profile={
            c: {
                v: float(utils[i, j])
                for i, v in enumerate(voters) if utils[i, j] > 0
            }
            for j, c in enumerate(candidates)
        },
        budget=budget_total,
    )
    result_true = {c.id for c in equal_shares(election)}
    result_npy = calculate_mes(utils.transpose(), costs, funds)
    result_npy = np.where(result_npy)[0]
    result_npy = {str(c_id) for c_id in result_npy}
    assert(result_npy == result_true)


def test_mes_2():
    voters_no = 2
    candidate_no = 2
    budget_total = 1
    funds = np.array([0.08433372 + 0.01, 0.91566628 + 0.01])
    costs = np.array([2, 1])
    utils = np.array([[0., 0.1], [0.10503397, 0.01]])
    voters = set()
    candidates = set()
    for i in range(voters_no):
        v = Voter(str(i), funds=float(funds[i]))
        voters.add(v)
    for j in range(candidate_no):
        c = Candidate(str(j), cost=int(costs[j]))
        candidates.add(c)
    election = Election(
        voters=voters,
        profile={
            c: {
                v: float(utils[i, j])
                for i, v in enumerate(voters) if utils[i, j] > 0
            }
            for j, c in enumerate(candidates)
        },
        budget=budget_total,
    )
    result_true = {c.id for c in equal_shares(election)}
    result_npy = calculate_mes(utils.transpose(), costs, funds)
    result_npy = np.where(result_npy)[0]
    result_npy = {str(c_id) for c_id in result_npy}
    assert (result_npy == result_true)


def test_lackner_skowron_2023():
    costs = np.ones(7)
    funds = np.ones(12) * 4 / 12
    utils = np.array([
        [1, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
    ])
    assert set(np.where(calculate_mes(utils.transpose(), costs, funds))[0]) == {0}


if __name__ == '__main__':
    unittest.main()
