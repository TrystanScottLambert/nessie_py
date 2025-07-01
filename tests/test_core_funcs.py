"""
Testing the core nessie functions in the core_funcs module.
"""

import unittest
import numpy as np

from core_funcs import _group_graph, _find_groups


class TestGroupGraph(unittest.TestCase):
    """
    Test functions for the _group_graph function.
    """

    def test_two_disconneted_graphs(self):
        """
        Simple case with two groups, one with three members and one with two members.
        """
        links = {"i": np.array([1, 1, 2, 4]), "j": np.array([2, 3, 3, 5])}
        groups = _group_graph(links)

        correct_galaxy_ids = [1, 2, 3, 4, 5]
        correct_group_ids = [1, 1, 1, 2, 2]

        for res, ans in zip(groups["galaxy_id"], correct_galaxy_ids):
            self.assertEqual(res, ans)

        for res, ans in zip(groups["group_id"], correct_group_ids):
            self.assertEqual(res, ans)


class TestFindGroups(unittest.TestCase):
    """
    Test functions for the _find_groups function.
    """

    def test_two_groups(self):
        """
        Simple tests with two groups, 3 members and 2 members and one isolated.
        """
        ra = np.array([120, 120, 120, 0, 0, 180])
        dec = np.array([0, 0, 0, -50, -50, 50])
        comoving_distance = np.array([20, 20, 20, 100, 100, 400])
        los_link_lengths = np.array([1, 1, 1, 1, 1, 1])
        pos_link_lengths = np.array([1, 1, 1, 1, 1, 1])

        correct_galaxy_ids = np.array([1, 2, 3, 4, 5])
        correct_group_ids = np.array([1, 1, 1, 2, 2])

        result = _find_groups(
            ra, dec, comoving_distance, pos_link_lengths, los_link_lengths
        )
        for res, ans in zip(result["galaxy_id"], correct_galaxy_ids):
            self.assertEqual(res, ans)

        for res, ans in zip(result["group_id"], correct_group_ids):
            self.assertEqual(res, ans)


if __name__ == "__main__":
    unittest.main()
