"""
Filename: matching_tools.py

Author: Daisuke Oyama

Tools for matching algorithms.

"""
import numpy as np


def random_prefs(m, n, allow_unmatched=True):
    """
    Generate random preference order lists for two groups, say, m males
    and n females.

    Each male has a preference order over femals [0, ..., n-1] and
    "unmatched" which is represented by n, while each female has a
    preference order over males [0, ..., m-1] and "unmatched" which is
    represented by m.

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

    Returns
    -------
    m_prefs : ndarray(int, ndim=2)
        Array of shape (m, n+1), where each row contains a random
        permutation of 0, ..., n-1, n.

    f_prefs : ndarray(int, ndim=2)
        Array of shape (n, m+1), where each row contains a random
        permutation of 0, ..., m-1, m.

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

    """
    m_prefs = np.tile(np.arange(n+1), (m, 1))
    f_prefs = np.tile(np.arange(m+1), (n, 1))

    stop = None if allow_unmatched else -1

    for i in range(m):
        np.random.shuffle(m_prefs[i, :stop])
    for j in range(n):
        np.random.shuffle(f_prefs[j, :stop])

    return m_prefs, f_prefs
