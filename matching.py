"""
Filename: matching.py

Author: Daisuke Oyama

Matching algorithms.

"""
import numpy as np
from numba import jit


@jit
def deferred_acceptance(prop_prefs, resp_prefs):
    """
    Compute a stable matching by the deferred acceptance (Gale-Shapley)
    algorithm.

    Parameters
    ----------
    prop_prefs : array_like(int, ndim=2)
        Array of shape (m, n+1) containing the proposers' preference
        orders as rows, where m is the number of proposers and n is that
        of the respondants. prop_prefs[i, j] is the j-th preferred
        respondant for the i-th proposer, where "respondant n"
        represents "being single".

    resp_prefs : array_like(int, ndim=2)
        Array of shape (n, m+1) containing the respondants' preference
        orders as rows. resp_prefs[j, i] is the i-th preferred proposer
        for the j-th respondant, where "proposer m" represents
        "being single".

    Returns
    -------
    prop_matches : ndarray(int, ndim=1)
        Array of length m, where prop_matches[i] is the respondant who
        proposer i is matched with.

    resp_matches : ndarray(int, ndim=1)
        Array of length n, where resp_matches[j] is the proposer who
        respondant j is matched with.

    """
    prop_prefs = np.asarray(prop_prefs)
    resp_prefs = np.asarray(resp_prefs)
    num_props, num_resps = prop_prefs.shape[0], resp_prefs.shape[0]

    if not (prop_prefs.shape == (num_props, num_resps+1) and
            resp_prefs.shape == (num_resps, num_props+1)):
        raise ValueError('shapes of the input arrays do not match')

    # Convert preference orders to rankings
    resp_ranks = np.empty((num_resps, num_props+1), dtype=int)
    prefs2ranks(resp_prefs, out=resp_ranks)

    # IDs representing unmatched
    prop_unmatched, resp_unmatched = num_resps, num_props

    is_single_prop = np.ones(num_props, dtype=bool)

    # Next resp to propose to
    next_resp = np.zeros(num_props, dtype=int)

    # Prop currently matched with
    current_prop = np.ones(num_resps, dtype=int) * resp_unmatched

    while(is_single_prop.sum() > 0):
        for p in range(num_props):
            if is_single_prop[p]:
                r = prop_prefs[p, next_resp[p]]  # p proposes to r
                if r == prop_unmatched:
                    is_single_prop[p] = False
                elif resp_ranks[r, p] < resp_ranks[r, current_prop[r]]:
                    if current_prop[r] != resp_unmatched:
                        is_single_prop[current_prop[r]] = True
                    current_prop[r] = p
                    is_single_prop[p] = False
                next_resp[p] += 1

    prop_matches = prop_prefs[np.arange(num_props), next_resp-1]
    resp_matches = current_prop

    return prop_matches, resp_matches


@jit(nopython=True)
def prefs2ranks(prefs, out):
    m, n = prefs.shape
    for i in range(m):
        for j in range(n):
            out[i, prefs[i, j]] = j
