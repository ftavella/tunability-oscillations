"""
Unit tests for the bio_networks module
"""

import os
import sys
import pytest
import numpy as np
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(ROOT_DIR)
from simulation_tools import bio_networks as bn  # noqa : E402


class TestBioNetworks():

    def test_goodwin_gamma_terms(self):
        connectivity = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]])
        term_0 = "+ G[0,2]*(1.0 - x[0])*x[2]**N[0,2]/"
        term_0 += "(x[2]**N[0,2] + K[0,2]**N[0,2]) "
        term_1 = "+ G[1,0]*(1.0 - x[1])*x[0]**N[1,0]/"
        term_1 += "(x[0]**N[1,0] + K[1,0]**N[1,0]) "
        term_2 = "- G[2,1]*x[2]*x[1]**N[2,1]/"
        term_2 += "(x[1]**N[2,1] + K[2,1]**N[2,1]) "
        terms = [term_0, term_1, term_2]
        assert terms == bn.create_gamma_terms(connectivity)

    def test_input_empty_array(self):
        with pytest.raises(ValueError):
            bn.create_gamma_terms(np.array([]))

    def test_input_not_array(self):
        with pytest.raises(TypeError):
            bn.create_gamma_terms("")

    def test_input_incorrect_values(self):
        connectivity = np.array([[0, 0, 2], [1, 0, 0], [0, -1, 0]])
        with pytest.raises(ValueError):
            bn.create_gamma_terms(connectivity)

    def test_input_incorrect_dimension(self):
        connectivity = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0], [1, 0, 0]])
        with pytest.raises(ValueError):
            bn.create_gamma_terms(connectivity)

    def test_network_initialization(self):
        connectivity = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]])
        net = bn.Network(connectivity)
        assert np.all(net.connectivity == connectivity)
