import unittest
import random
import numpy as np

from ext.mes_mikolaj.model import *
import sys
sys.path.append('../ext/mes_mikolaj')
from ext.mes_mikolaj.rules import equal_shares

from src.mes import calculate_mes


class MesRandomizedTest(unittest.TestCase):
    def test_mes_1(self):
        voters_no = random.randint(1, 20)
        candidate_no = random.randint(1, voters_no)
        zero_util_mask = np.random.randint(0, 1, size=(voters_no, candidate_no))
        utils = np.random.rand(voters_no, candidate_no)
        utils *= zero_util_mask
        utils += np.diag(np.ones_like(utils))
        costs = np.random.randint(1, 5, candidate_no)
        funds = np.random.rand(voters_no)
        budget_total = int(funds.sum())
        to_cut = float(funds.sum()) - budget_total
        funds -= to_cut / voters_no
        funds += 1 / voters_no
        budget_total += 1
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


if __name__ == '__main__':
    unittest.main()
