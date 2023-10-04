"""Tests if  ``normal_distribution_plot`` is working as expected

--------------------------------------------------------------------------------
Command to run at the prompt:
    python -m unittest -v tests/normtest/generic/test_normal_distribution_plot.py
    or
    python -m unittest -b tests/normtest/generic/test_normal_distribution_plot.py

--------------------------------------------------------------------------------
"""

### GENERAL IMPORTS ###
import os
import unittest
import numpy as np
from matplotlib.axes import SubplotBase
import matplotlib.pyplot as plt
from pathlib import Path

### FUNCTION IMPORT ###
from normtest.normtest import normal_distribution_plot
from tests.functions_to_test import functions

os.system('cls')

class Test_normal_distribution_plot(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        fig, cls.axes = plt.subplots()
        cls.n_rep = 30


    def test_outputs(self):

        result = normal_distribution_plot(self.axes, self.n_rep)
        self.assertIsInstance(result, SubplotBase, msg="not a SubplotBase")
        plt.close()

    def test_safe(self):

        result = normal_distribution_plot(self.axes, self.n_rep, safe=True)
        self.assertIsInstance(result, SubplotBase, msg="not a SubplotBase")
        plt.close()


    def test_basic_plot(self):
        
        fig1_base_path = Path("tests/normtest/generic/figs_normal_distribution_plot/normal_distribution_plot_42.png")

        fig, ax = plt.subplots(figsize=(6,4))
        result = normal_distribution_plot(ax, 30, seed=42)
        fig1_file = Path("tests/normtest/generic/figs_normal_distribution_plot/fig1_test.png")
        plt.savefig(fig1_file)
        plt.close()

        self.assertTrue(functions.validate_file_contents(fig1_base_path, fig1_file), msg="figures does not match")
        fig1_file.unlink()


 