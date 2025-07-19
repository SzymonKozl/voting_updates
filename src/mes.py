from typing import Optional

import numpy as np


def calculate_mes(valuations: np.ndarray, costs: Optional[np.ndarray] = None, budgets: Optional[np.ndarray] = None) -> np.ndarray:
    costs = costs if costs is not None else np.ones(valuations.shape[0])
    budgets = budgets if budgets is not None else np.ones(valuations.shape[1]) * valuations.shape[0] / (2 * valuations.shape[1])
    assert valuations.shape[0] == costs.shape[0]
    assert valuations.shape[1] == budgets.shape[0]

    selected = np.zeros(valuations.shape[0])
    feasible_mask = np.ones(valuations.shape[0], dtype=bool)
    while True:
        
    return selected