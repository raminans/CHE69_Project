#!/usr/bin/env python3
"""
Unit and regression test for the kabsch_proj package.
"""

# Import package, test suite, and other packages as needed
import sys
import numpy as np
sys.path.append('../kabsch_proj')
import unittest
from contextlib import contextmanager
from io import StringIO

import kabsch_algorithm


class MyTest(unittest.TestCase):

    def test_parse_xyz(self):

        self.assertEqual(
            kabsch_algorithm.parse_xyz('../../mol1.xyz').shape[-1], 3)
        with self.assertRaises(FileNotFoundError):
            kabsch_algorithm.parse_xyz('Null')

    def test_centriod(self):

        self.assertEqual(kabsch_algorithm.centroid(np.ones(10)), 1)
        self.assertEqual(kabsch_algorithm.centroid(np.zeros(10)), 0)
        with self.assertRaises(ValueError):
            kabsch_algorithm.centroid([])

    def test_rmsd(self):

        with self.assertRaises(AssertionError) as context:
            kabsch_algorithm.rmsd(np.random.random((10,3)), np.random.random((10,4)))
        self.assertTrue('Dimensions are not equal!' in str(context.exception))
        test_mol1 = np.array([[1, 0, 0]] * 10)
        self.assertEqual(kabsch_algorithm.rmsd(test_mol1, test_mol1), 0)
        test_mol2 = np.array([[0, 1, 0]] * 10)
        self.assertAlmostEqual(kabsch_algorithm.rmsd(test_mol1, test_mol2), np.sqrt(2))

    def test_parse_arguments(self):

        test_argv = ['--mol1=mol1_address', '--mol2=mol2_address',
                     '--normal=True']
        args = kabsch_algorithm.parse_arguments(test_argv)
        self.assertEqual(args.mol1, 'mol1_address')
        self.assertEqual(args.mol2, 'mol2_address')
        self.assertTrue(args.normal)
        self.assertFalse(args.rotation)

    def test_kabsch(self):

        test_mol1 = np.array([[1, 0, 0]] * 10)
        self.assertAlmostEqual(kabsch_algorithm.kabsch(test_mol1, test_mol1)[0], 0)

    def test_main(self):

        test_argv = ['--mol1=../../mol1.xyz', '--mol2=../../mol2.xyz']
        args = kabsch_algorithm.parse_arguments(test_argv)
        self.assertAlmostEqual(kabsch_algorithm.main(args)[0],
                               0.029961368078169674)





# Utility functions

# From http://schinckel.net/2013/04/15/capture-and-test-sys.stdout-sys.stderr-in-unittest.testcase/
# @contextmanager
# def capture_stdout(command, *args, **kwargs):
#     # pycharm doesn't know six very well, so ignore the false warning
#     # noinspection PyCallingNonCallable
#     out, sys.stdout = sys.stdout, StringIO()
#     command(*args, **kwargs)
#     sys.stdout.seek(0)
#     yield sys.stdout.read()
#     sys.stdout = out

if __name__ == '__main__':

    unittest.main()