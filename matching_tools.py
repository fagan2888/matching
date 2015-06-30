"""
Filename: matching_tools.py

Author: Daisuke Oyama

Tools for matching algorithms.

"""
import numpy as np


def random_prefs(m, n, allow_unmatched=True, return_caps=False):
    """
    Generate random preference order lists for two groups, say, m males
    and n females.

    Each male has a preference order over femals [0, ..., n-1] and
    "unmatched" which is represented by n, while each female has a
    preference order over males [0, ..., m-1] and "unmatched" which is
    represented by m.

    return_caps should be set True in the context of college admissions,
    in which case "males" and "females" should be read as "students" and
    "colleges", respectively, where each college has its capacity.

    Parameters
    ----------
    m : scalar(int)
        Number of males.

    n : scalar(int)
        Number of femals.

    allow_unmatched : bool, optional(default=True)
        If False, return preference order lists of males and females
        where n and m are always placed in the last place, repectively
        (i.e., "unmatched" is least preferred by every individual).

    return_caps : bool, optional(default=False)
        If True, caps is also returned.

    Returns
    -------
    m_prefs : ndarray(int, ndim=2)
        Array of shape (m, n+1), where each row contains a random
        permutation of 0, ..., n-1, n.

    f_prefs : ndarray(int, ndim=2)
        Array of shape (n, m+1), where each row contains a random
        permutation of 0, ..., m-1, m.

    caps : ndarray(int, ndim=1)
        Array of shape (m,) containing each female's (or college's)
        capacity. Returned only when return_caps is True.

    Examples
    --------
    >>> m_prefs, f_prefs = random_prefs(4, 3)
    >>> m_prefs
    array([[0, 3, 1, 2],
           [1, 2, 3, 0],
           [1, 3, 0, 2],
           [1, 0, 3, 2]])
    >>> f_prefs
    array([[2, 4, 0, 1, 3],
           [1, 3, 4, 2, 0],
           [3, 2, 4, 0, 1]])

    >>> m_prefs, f_prefs = random_prefs(4, 3, allow_unmatched=False)
    >>> m_prefs
    array([[2, 0, 1, 3],
           [0, 1, 2, 3],
           [2, 0, 1, 3],
           [2, 1, 0, 3]])
    >>> f_prefs
    array([[2, 3, 0, 1, 4],
           [1, 0, 2, 3, 4],
           [1, 3, 0, 2, 4]])

    >>> s_prefs, c_prefs, caps = random_prefs(4, 3, return_caps=True)
    >>> s_prefs
    array([[0, 1, 2, 3],
           [0, 3, 2, 1],
           [2, 3, 0, 1],
           [0, 2, 3, 1]])
    >>> c_prefs
    array([[3, 0, 4, 1, 2],
           [3, 1, 2, 0, 4],
           [1, 3, 2, 0, 4]])
    >>> caps
    array([2, 3, 1])

    """
    m_prefs = _random_prefs(m, n, allow_unmatched=allow_unmatched,
                            return_caps=False)
    if not return_caps:
        f_prefs = _random_prefs(n, m, allow_unmatched=allow_unmatched,
                                return_caps=False)
        return m_prefs, f_prefs
    else:
        f_prefs, caps = _random_prefs(n, m, allow_unmatched=allow_unmatched,
                                      return_caps=True)
        return m_prefs, f_prefs, caps


def _random_prefs(m, n, allow_unmatched, return_caps):
    unmatched = n

    prefs = np.tile(np.arange(n+1), (m, 1))
    for i in range(m):
        np.random.shuffle(prefs[i, :-1])

    if allow_unmatched:
        unmatched_rankings = np.random.randint(1, n+1, size=m)
        swapped = prefs[np.arange(m), unmatched_rankings]
        prefs[:, -1] = swapped
        prefs[np.arange(m), unmatched_rankings] = unmatched
    elif return_caps:
        unmatched_rankings = np.ones(m, dtype=int) * n

    if return_caps:
        u = np.random.random_sample(size=m)
        caps = np.floor(unmatched_rankings*u + 1).astype(int)

    if return_caps:
        return prefs, caps
    else:
        return prefs
