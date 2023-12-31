"""Tests if  ``rj_dist_plot`` is working as expected

--------------------------------------------------------------------------------
Command to run at the prompt:
    python -m unittest -v tests/normtest/ryan-joiner/test_rj_dist_plot.py
    or
    python -m unittest -b tests/normtest/ryan-joiner/test_rj_dist_plot.py

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
from normtest.normtest import rj_dist_plot
from tests.functions_to_test import functions

os.system('cls')

class Test_rj_dist_plot(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        fig, cls.axes = plt.subplots()
        cls.x_data = np.array([148, 148, 154, 158, 158, 160, 161, 162, 166,  170, 182, 195, 210])


    def test_outputs(self):

        result = rj_dist_plot(self.axes, self.x_data)
        self.assertIsInstance(result, SubplotBase, msg="not a SubplotBase")
        plt.close()

    def test_safe(self):

        result = rj_dist_plot(self.axes, self.x_data, safe=True)
        self.assertIsInstance(result, SubplotBase, msg="not a SubplotBase")
        plt.close()        



    def test_basic_plot(self):
        
        fig1_base_path = Path("tests/normtest/ryan-joiner/figs_rj_dist_plot/fig1.png")

        fig, ax = plt.subplots()
        result = rj_dist_plot(ax, self.x_data)
        fig1_file = Path("tests/normtest/ryan-joiner/figs_rj_dist_plot/fig1_test.png")
        plt.savefig(fig1_file)
        plt.close()

        self.assertTrue(functions.validate_file_contents(fig1_base_path, fig1_file), msg="figures does not match")
        fig1_file.unlink()


    def test_plot_deleted_true(self):
        x = np.array([43.5, 125.1, 166.3, 38.1, 116.7, 25.4, 40, 38.1, 253.7, 81.1, 96.7])
        
        fig2_base_path = Path("tests/normtest/ryan-joiner/figs_rj_dist_plot/fig2.png")

        fig, ax = plt.subplots()
        result = rj_dist_plot(ax, x, deleted=True)
        fig2_file = Path("tests/normtest/ryan-joiner/figs_rj_dist_plot/fig2_test.png")
        plt.savefig(fig2_file)
        plt.close()

        self.assertTrue(functions.validate_file_contents(fig2_base_path, fig2_file), msg="figures does not match")
        fig2_file.unlink()        