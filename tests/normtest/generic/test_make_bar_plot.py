"""Tests if  ``make_bar_plot`` is working as expected

--------------------------------------------------------------------------------
Command to run at the prompt:
    python -m unittest -v tests/normtest/generic/test_make_bar_plot.py
    or
    python -m unittest -b tests/normtest/generic/test_make_bar_plot.py

--------------------------------------------------------------------------------
"""

### GENERAL IMPORTS ###
import os
import unittest
import numpy as np
from matplotlib.axes import SubplotBase
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

### FUNCTION IMPORT ###
from normtest.normtest import make_bar_plot
from normtest import normtest as nm
from tests.functions_to_test import functions

os.system('cls')

class Test_make_bar_plot(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        
        cls.n_samples = 10
        seeds = np.arange(1,cls.n_samples+1)
        rng = np.random.default_rng(42)
        rng.shuffle(seeds)
        n_rep = np.arange(4,31)
        alphas = [0.1, 0.05, 0.01]
        seed_values = []
        alpha_list = []
        blom_1 = []
        blom_2 = []
        blom_3 = []
        n_rep_list = []

        for seed in seeds:
            rng = np.random.default_rng(seed=seed)
            normal_data = rng.normal(loc=0, scale=1.0, size=max(n_rep))

            for n in n_rep:
                for alpha in alphas:
                    n_rep_list.append(n)
                    seed_values.append(seed)
                    if alpha == 0.1:
                        alpha_list.append("0.10")
                    elif alpha == 0.05:
                        alpha_list.append("0.05")
                    else:
                        alpha_list.append("0.01")

                    result = nm.ryan_joiner(x_data=normal_data[:n], method="blom", alpha=alpha)
                    if result.statistic < result.critical:
                        blom_1.append(False)
                        
                    else:
                        blom_1.append(True)

                    result = nm.ryan_joiner(x_data=normal_data[:n], method="blom2", alpha=alpha)
                    if result.statistic < result.critical:
                        blom_2.append(False)
                        
                    else:
                        blom_2.append(True)                

                    result = nm.ryan_joiner(x_data=normal_data[:n], method="blom3", alpha=alpha)
                    if result.statistic < result.critical:
                        blom_3.append(False)
                        
                    else:
                        blom_3.append(True)                                
                

        cls.df_data = pd.DataFrame({
            "Alpha": alpha_list,
            "n amostral": n_rep_list,
            "blom": blom_1,
            "blom2": blom_2,
            "blom3": blom_3,       

        })



    def test_outputs(self):
        fig, ax = plt.subplots()
        result = make_bar_plot(ax, self.df_data, self.n_samples)
        self.assertIsInstance(result[0], SubplotBase, msg="not a SubplotBase")
        self.assertIsInstance(result[1], pd.DataFrame, msg="not a dataframe")
        plt.close()

    def test_safe(self):
        fig, ax = plt.subplots()
        result = make_bar_plot(ax, self.df_data, self.n_samples, safe=True)
        self.assertIsInstance(result[0], SubplotBase, msg="not a SubplotBase")
        self.assertIsInstance(result[1], pd.DataFrame, msg="not a dataframe")
        plt.close()


    def test_basic_plot(self):
        
        fig1_base_path = Path("tests/normtest/generic/figs_make_bar_plot/make_bar_plot_42.png")

        fig, ax = plt.subplots(figsize=(10,4)) 
        result = make_bar_plot(ax, self.df_data, self.n_samples, normal=False)
        fig1_file = Path("tests/normtest/generic/figs_make_bar_plot/fig1_test.png")
        plt.savefig(fig1_file, bbox_inches='tight')
        plt.close()

        self.assertTrue(functions.validate_file_contents(fig1_base_path, fig1_file), msg="figures does not match")
        fig1_file.unlink()


 