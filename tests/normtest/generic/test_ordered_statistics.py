"""Tests if  ``ordered_statistics`` is working as expected

--------------------------------------------------------------------------------
Command to run at the prompt:
    python -m unittest -v tests/normtest/generic/test_ordered_statistics.py
    or
    python -m unittest -b tests/normtest/generic/test_ordered_statistics.py

--------------------------------------------------------------------------------
"""

### GENERAL IMPORTS ###
import os
import unittest
import numpy as np
import random

### FUNCTION IMPORT ###
from normtest.normtest import ordered_statistics

os.system('cls')



class Test_ordered_statistics(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.n = random.randrange(4,200)
        methods = ["blom", "blom2", "blom3", "filliben"]
        cls.method = random.sample(methods, 1)[0]

    def test_outputs(self):
        result = ordered_statistics(self.n, self.method)
        self.assertIsInstance(result, np.ndarray, msg=f"not a float when method={self.method} and n={self.n}")

    def test_safe(self):
        result = ordered_statistics(self.n, self.method, safe=True)
        self.assertIsInstance(result, np.ndarray, msg=f"not a float when method={self.method} and n={self.n}")

    def test_blom_odd(self):
        n = 9
        expected = np.array([0.067568, 0.175676, 0.283784, 0.391892, 0.500000, 0.608108, 0.716216, 0.824324, 0.932432])
        result = ordered_statistics(n=n, method="blom")
        for pair in zip(result, expected):
            self.assertAlmostEqual(pair[0], pair[1], places=5, msg=f"wrong statisitc order for blom")       

    def test_blom_even(self):
        n = 10
        expected = np.array([0.060976, 0.158537, 0.256098, 0.353659, 0.451220, 0.548780, 0.646341, 0.743902, 0.841463, 0.939024])
        result = ordered_statistics(n=n, method="blom")
        for pair in zip(result, expected):
            self.assertAlmostEqual(pair[0], pair[1], places=5, msg=f"wrong statisitc order for blom")                            


    def test_blom2_odd(self):
        n = 9
        expected = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        result = ordered_statistics(n=n, method="blom2")
        for pair in zip(result, expected):
            self.assertAlmostEqual(pair[0], pair[1], places=5, msg=f"wrong statisitc order for blom2")    

    def test_blom2_even(self):
        n = 10
        expected = np.array([0.090909, 0.181818, 0.272727, 0.363636, 0.454545, 0.545455, 0.636364, 0.727273, 0.818182, 0.909091])
        result = ordered_statistics(n=n, method="blom2")
        for pair in zip(result, expected):
            self.assertAlmostEqual(pair[0], pair[1], places=5, msg=f"wrong statisitc order for blom2")  

    def test_blom3_odd(self):
        n = 9
        expected = np.array([0.055556, 0.166667, 0.277778, 0.388889, 0.500000, 0.611111, 0.722222, 0.833333, 0.944444])
        result = ordered_statistics(n=n, method="blom3")
        for pair in zip(result, expected):
            self.assertAlmostEqual(pair[0], pair[1], places=5, msg=f"wrong statisitc order for blom3")                                        

    def test_blom3_even(self):
        n = 10
        expected = np.array([0.050000, 0.150000, 0.250000, 0.350000, 0.450000, 0.550000, 0.650000, 0.750000, 0.850000, 0.950000])
        result = ordered_statistics(n=n, method="blom3")
        for pair in zip(result, expected):
            self.assertAlmostEqual(pair[0], pair[1], places=5, msg=f"wrong statisitc order for blom3")  

    def test_wrong_method(self):

        with self.assertRaises(ValueError, msg=f"Does not raised ValueError when method={None} and n={self.n}"):
            result = ordered_statistics(self.n, None, safe=True)    

    def test_n_not_int(self):
        n_values = [5.1, "5", [5], (6,), ]
        for n in n_values:
            with self.assertRaises(TypeError, msg=f"Does not raised ValueError when method={self.method} and n={n}"):
                result = ordered_statistics(n, self.method, safe=True)     


    def test_small_n(self):
        n_values = [-5, 0, 3]
        for n in n_values:
            with self.assertRaises(ValueError, msg=f"Does not raised ValueError when method={self.method} and n={n}"):
                result = ordered_statistics(n, self.method, safe=True)            