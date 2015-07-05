"""
Tests for matching tools.

"""
import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import eq_, ok_

from matching_tools import random_prefs


class TestRandomPrefsOneToOne:

    def setUp(self):
        self.nums = (8, 6)
        self.unmatched_IDs = self.nums[::-1]
        self.prefs_arrays = random_prefs(*self.nums, allow_unmatched=False)
        self.prefs_arrays_allowed = \
            random_prefs(*self.nums, allow_unmatched=True)

    def test_shapes(self):
        prefs_arrays_all = self.prefs_arrays + self.prefs_arrays_allowed
        for i, prefs_array in enumerate(prefs_arrays_all):
            assert_array_equal(prefs_array.shape,
                               (self.nums[i % 2], self.nums[(i+1) % 2] + 1))

    def test_permutation(self):
        prefs_arrays_all = self.prefs_arrays + self.prefs_arrays_allowed
        sorted_arrays = [
            np.tile(np.arange(self.nums[(i+1) % 2] + 1), (self.nums[i % 2], 1))
            for i in range(2)
        ]
        for i, prefs_array in enumerate(prefs_arrays_all):
            assert_array_equal(np.sort(prefs_array, axis=-1),
                               sorted_arrays[i % 2])

    def test_unmatched_not_allowed(self):
        for prefs, unmatched in zip(self.prefs_arrays, self.unmatched_IDs):
            ok_(np.all(prefs[:, -1] == unmatched))

    def test_unmatched_not_most_preferred(self):
        for prefs, unmatched in zip(self.prefs_arrays_allowed,
                                    self.unmatched_IDs):
            ok_(np.all(prefs[:, 0] != unmatched))


class TestRandomPrefsManyToOne:

    def setUp(self):
        self.nums = (8, 6)
        self.unmatched_IDs = self.nums[::-1]
        self.s_prefs, self.c_prefs, self.caps = \
            random_prefs(*self.nums, allow_unmatched=False, return_caps=True)
        self.s_prefs_allowed, self.c_prefs_allowed, self.caps_allowed = \
            random_prefs(*self.nums, allow_unmatched=True, return_caps=True)

    def test_shapes(self):
        prefs_arrays_all = (self.s_prefs, self.c_prefs,
                            self.s_prefs_allowed, self.c_prefs_allowed)
        for i, prefs_array in enumerate(prefs_arrays_all):
            assert_array_equal(prefs_array.shape,
                               (self.nums[i % 2], self.nums[(i+1) % 2] + 1))
        for caps in (self.caps, self.caps_allowed):
            eq_(len(caps), self.nums[1])

    def test_permutation(self):
        prefs_arrays_all = (self.s_prefs, self.c_prefs,
                            self.s_prefs_allowed, self.c_prefs_allowed)
        sorted_arrays = [
            np.tile(np.arange(self.nums[(i+1) % 2] + 1), (self.nums[i % 2], 1))
            for i in range(2)
        ]
        for i, prefs_array in enumerate(prefs_arrays_all):
            assert_array_equal(np.sort(prefs_array, axis=-1),
                               sorted_arrays[i % 2])

    def test_unmatched_not_allowed(self):
        prefs_arrays = (self.s_prefs, self.c_prefs)
        for prefs, unmatched in zip(prefs_arrays, self.unmatched_IDs):
            ok_(np.all(prefs[:, -1] == unmatched))

    def test_unmatched_not_most_preferred(self):
        prefs_arrays = (self.s_prefs_allowed, self.c_prefs_allowed)
        for prefs, unmatched in zip(prefs_arrays, self.unmatched_IDs):
            ok_(np.all(prefs[:, 0] != unmatched))

    def test_caps_positive(self):
        ok_(np.all(self.caps > 0))

    def test_caps_not_exceed_rankings_of_unmatched(self):
        rankings_unmatched = np.where(self.c_prefs == self.unmatched_IDs[1])[1]
        ok_(np.all(self.caps <= rankings_unmatched))


if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
