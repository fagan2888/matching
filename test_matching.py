"""
Tests for matching algorithms.

"""
import numpy as np
from numpy.testing import assert_array_equal

from matching import deferred_acceptance


class TestDeferredAcceptance:

    def setUp(self):
        '''Setup preference order lists'''
        # Males' preference orders over females [0, 1, 2] and unmatched
        m_unmatched = 3
        self.m_prefs = [[0, 1, 2, m_unmatched],
                        [2, 0, 1, m_unmatched],
                        [1, 2, 0, m_unmatched],
                        [2, 0, 1, m_unmatched]]
        # Females' preference orders over males [0, 1, 2, 3] and unmatched
        f_unmatched = 4
        self.f_prefs = [[2, 0, 1, 3, f_unmatched],
                        [0, 1, 2, 3, f_unmatched],
                        [2, f_unmatched, 1, 0, 3]]

        # Unique stable matching
        self.m_matched = [0, 1, 2, m_unmatched]
        self.f_matched = [0, 1, 2]

    def test_male_proposal(self):
        m_matched_computed, f_matched_computed = \
            deferred_acceptance(self.m_prefs, self.f_prefs)
        assert_array_equal(m_matched_computed, self.m_matched)
        assert_array_equal(f_matched_computed, self.f_matched)

    def test_female_proposal(self):
        f_matched_computed, m_matched_computed = \
            deferred_acceptance(self.f_prefs, self.m_prefs)
        assert_array_equal(m_matched_computed, self.m_matched)
        assert_array_equal(f_matched_computed, self.f_matched)


class TestDeferredAcceptanceManyToOneCap1:

    def setUp(self):
        '''Setup preference order lists'''
        # Males' preference orders over females [0, 1, 2] and unmatched
        m_unmatched = 3
        self.m_prefs = [[0, 1, 2, m_unmatched],
                        [2, 0, 1, m_unmatched],
                        [1, 2, 0, m_unmatched],
                        [2, 0, 1, m_unmatched]]
        # Females' preference orders over males [0, 1, 2, 3] and unmatched
        f_unmatched = 4
        self.f_prefs = [[2, 0, 1, 3, f_unmatched],
                        [0, 1, 2, 3, f_unmatched],
                        [2, f_unmatched, 1, 0, 3]]

        # Capacities for females
        self.caps = [1, 1, 1]
        self.indptr = [0, 1, 2, 3]

        # Unique stable matching
        self.m_matched = [0, 1, 2, m_unmatched]
        self.f_matched = [0, 1, 2]

    def test_male_proposal(self):
        m_matched_computed, f_matched_computed, indptr_computed = \
            deferred_acceptance(self.m_prefs, self.f_prefs, self.caps)
        assert_array_equal(m_matched_computed, self.m_matched)
        assert_array_equal(f_matched_computed, self.f_matched)
        assert_array_equal(indptr_computed, self.indptr)


class TestDeferredAcceptanceManyToOne:

    def setUp(self):
        '''Setup preference order lists'''
        # From http://www.columbia.edu/~js1353/pubs/qst-many-to-one.pdf
        # Originally from Gusfield and Irving (1989, Section 1.6.5)
        self.num_s, self.num_c = 11, 5

        # Students' preference orders over colleges 0, ..., 4 and unmatched
        s_unmatched = self.num_c
        self.s_prefs = [[2, 0, 4, 3, s_unmatched, 1],
                        [0, 2, 3, 1, 4, s_unmatched],
                        [3, 4, 2, 0, 1, s_unmatched],
                        [2, 3, 0, 4, s_unmatched, 1],
                        [0, 3, 1, s_unmatched, 2, 4],
                        [3, 2, 1, 0, 4, s_unmatched],
                        [1, 4, 0, 2, s_unmatched, 3],
                        [0, 2, 1, 4, 3, s_unmatched],
                        [3, 0, 4, s_unmatched, 1, 2],
                        [2, 0, 4, 1, 3, s_unmatched],
                        [4, 3, 0, 2, 1, s_unmatched]]
        # Colleges' preference orders over students 0, ..., 10 and unmatched
        c_unmatched = self.num_s
        self.c_prefs = [[2, 6, 8, 10, 4, 3, 9, 7, 5, 0, 1, c_unmatched],
                        [4, 6, 9, 5, 7, 1, 2, 10, c_unmatched, 0, 3, 8],
                        [10, 5, 7, 2, 1, 3, 6, 0, 9, c_unmatched, 4, 8],
                        [9, 0, 1, 10, 3, 8, 4, 2, 5, 7, c_unmatched, 6],
                        [1, 3, 9, 6, 5, 0, 7, 2, 10, 8, c_unmatched, 4]]

        # Capacities for females
        self.caps = [4, 1, 3, 2, 1]
        self.indptr = [0, 4, 5, 8, 10, 11]

        # Unique stable matching
        self.s_matched = [2, 0, 3, 2, 0, 2, 1, 0, 3, 0, 4]
        self.c_matched = [1, 4, 7, 9, 6, 0, 3, 5, 2, 8, 10]

    def test_student_proposal(self):
        s_matched_computed, c_matched_computed, indptr_computed = \
            deferred_acceptance(self.s_prefs, self.c_prefs, self.caps)
        assert_array_equal(s_matched_computed, self.s_matched)
        assert_array_equal(indptr_computed, self.indptr)

        # Sort c_matched_computed for each college
        c_matched_sorted = np.array(c_matched_computed, copy=True)
        for j in range(self.num_c):
            c_matched_sorted[self.indptr[j]:self.indptr[j+1]].sort()
        assert_array_equal(c_matched_sorted, self.c_matched)


if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
