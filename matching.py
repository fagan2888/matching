"""
Filename: matching.py

Author: Daisuke Oyama

Matching algorithms.

"""
import numpy as np
from numba import jit


@jit
def deferred_acceptance(prop_prefs, resp_prefs, caps=None):
    """
    Compute a stable matching by the deferred acceptance (Gale-Shapley)
    algorithm. Support both one-to-one (marriage) and many-to-one
    (college admission) matchings.

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
        "being single" (or "vacancy" in the context of college
        admissions).

    caps : array_like(int, ndim=1), optional(default=None)
        Array of shape (n,) containing the respondants' capacities. If
        None, the capacities are all regarded as one (i.e., the matching
        is one-to-one).

    Returns
    -------
    prop_matches : ndarray(int, ndim=1)
        Array of length m representing the matches for the proposals,
        where prop_matches[i] is the respondant who proposer i is
        matched with.

    resp_matches : ndarray(int, ndim=1)
        Array of length n representing the matches for the respondants:
        if caps=None, resp_matches[j] is the proposer who respondant j
        is matched with; if caps is specified, the proposers who
        respondant j is matched with are contined in
        resp_matches[indptr[j]:indptr[j+1]].

    indptr : ndarray(int, ndim=1)
        Returned only when caps is specified. Contains index pointers
        for resp_matches.

    """
    prop_prefs = np.asarray(prop_prefs)
    resp_prefs = np.asarray(resp_prefs)
    num_props, num_resps = prop_prefs.shape[0], resp_prefs.shape[0]

    if not (prop_prefs.shape == (num_props, num_resps+1) and
            resp_prefs.shape == (num_resps, num_props+1)):
        raise ValueError('shapes of preferences arrays do not match')

    if (caps is not None) and (len(caps) != num_resps):
        raise ValueError('length of caps must be equal to that of resp_prefs')

    # Convert preference orders to rankings
    resp_ranks = np.empty((num_resps, num_props+1), dtype=int)
    prefs2ranks(resp_prefs, out=resp_ranks)

    # IDs representing unmatched
    prop_unmatched, resp_unmatched = num_resps, num_props

    is_single_prop = np.ones(num_props, dtype=bool)

    # Next resp to propose to
    next_resp = np.zeros(num_props, dtype=int)

    # Set up index pointers
    if caps is not None:  # Many-to-one
        indptr = np.empty(num_resps+1, dtype=int)
        indptr[0] = 0
        np.cumsum(caps, out=indptr[1:])
    else:  # One-to-one
        indptr = np.arange(num_resps+1)

    num_caps = indptr[-1]

    # Prop currently matched with
    current_prop = np.ones(num_caps, dtype=int) * resp_unmatched

    # Numbers of occupied seats
    nums_occupied = np.zeros(num_resps, dtype=int)

    # Main loop
    while(is_single_prop.sum() > 0):
        for p in range(num_props):
            if is_single_prop[p]:
                r = prop_prefs[p, next_resp[p]]  # p proposes to r

                # Prefers to be unmatched
                if r == prop_unmatched:
                    is_single_prop[p] = False

                # Unacceptable for r
                elif resp_ranks[r, p] > resp_ranks[r, resp_unmatched]:
                    pass

                # Some seats vacant
                elif nums_occupied[r] < indptr[r+1] - indptr[r]:
                    current_prop[indptr[r]+nums_occupied[r]] = p
                    is_single_prop[p] = False
                    nums_occupied[r] += 1

                # All seats occupied
                else:
                    # Find the least preferred among the currently accepted
                    least_ptr = indptr[r]
                    least = current_prop[least_ptr]
                    for i in range(indptr[r]+1, indptr[r+1]):
                        compared = current_prop[i]
                        if resp_ranks[r, least] < resp_ranks[r, compared]:
                            least_ptr = i
                            least = compared

                    if resp_ranks[r, p] < resp_ranks[r, least]:
                        current_prop[least_ptr] = p
                        is_single_prop[p] = False
                        is_single_prop[least] = True

                next_resp[p] += 1

    prop_matches = prop_prefs[np.arange(num_props), next_resp-1]
    resp_matches = current_prop

    if caps is None:
        return prop_matches, resp_matches
    else:
        return prop_matches, resp_matches, indptr


@jit(nopython=True)
def prefs2ranks(prefs, out):
    m, n = prefs.shape
    for i in range(m):
        for j in range(n):
            out[i, prefs[i, j]] = j
