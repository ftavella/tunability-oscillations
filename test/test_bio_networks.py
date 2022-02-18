"""
Unit tests for the bio_networks module
"""

import os
import sys
import pytest
import copy
import numpy as np
from scipy.integrate import solve_ivp
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(ROOT_DIR)
from simulation_tools import bio_networks as bn  # noqa : E402


class TestBioNetworks():

    def test_empty_connectivity_as_input(self):
        connectivity = np.array([])
        with pytest.raises(ValueError):
            _ = bn.Network(connectivity, {})

    def test_str_connectivity_as_input(self):
        connectivity = ""
        with pytest.raises(TypeError):
            _ = bn.Network(connectivity, {})

    def test_incorrect_values_in_connectivty(self):
        connectivity = np.array([[0, 0, 2], [1, 0, 0], [0, -1, 0]])
        with pytest.raises(ValueError):
            _ = bn.Network(connectivity, {})

    def test_incorrect_dimension_in_connectivity(self):
        connectivity = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0], [1, 0, 0]])
        with pytest.raises(ValueError):
            _ = bn.Network(connectivity, {})

    def test_network_has_connectivity(self):
        connectivity = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]])
        p_dict = {
            "A": np.array([0.00004564, 0.16342066, 0.70351496]),
            "B": np.array([56.67356362, 50.87885942, 30.31904319]),
            "G": np.array([[0, 0, 1657.39796806],
                          [6988.78292809, 0, 0],
                          [0, 7461.31128709, 0]]),
            "K": np.array([[0, 0, 0.11196140],
                          [0.50912113, 0, 0],
                          [0, 0.33210760, 0]]),
            "N": np.array([[0, 0, 3.67709117],
                          [7.17307667, 0, 0],
                          [0, 6.80130072, 0]])
        }
        network = bn.Network(connectivity, p_dict)
        assert np.all(network.connectivity == connectivity)

    def test_network_creates_appropriate_equations(self):
        connectivity = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]])
        p_dict = {
            "A": np.array([0.00004564, 0.16342066, 0.70351496]),
            "B": np.array([56.67356362, 50.87885942, 30.31904319]),
            "G": np.array([[0, 0, 1657.39796806],
                          [6988.78292809, 0, 0],
                          [0, 7461.31128709, 0]]),
            "K": np.array([[0, 0, 0.11196140],
                          [0.50912113, 0, 0],
                          [0, 0.33210760, 0]]),
            "N": np.array([[0, 0, 3.67709117],
                          [7.17307667, 0, 0],
                          [0, 6.80130072, 0]])
        }
        network = bn.Network(connectivity, p_dict)

        def expected_equations(t, x, A, B, G, K, N):
            dx0dt = B[0]*(A[0] - x[0])
            dx1dt = B[1]*(A[1] - x[1])
            dx2dt = B[2]*(A[2] - x[2])
            # Interactions
            dx0dt += + G[0, 2] * \
                (1.0 - x[0]) * x[2] ** N[0, 2] / \
                (x[2] ** N[0, 2] + K[0, 2] ** N[0, 2])
            dx1dt += + G[1, 0] * \
                (1.0 - x[1]) * x[0] ** N[1, 0] / \
                (x[0] ** N[1, 0] + K[1, 0] ** N[1, 0])
            dx2dt += - G[2, 1] * \
                x[2] * x[1] ** N[2, 1] / \
                (x[1] ** N[2, 1] + K[2, 1] ** N[2, 1])
            return np.array([dx0dt, dx1dt, dx2dt])

        expected_solution = solve_ivp(expected_equations, [0, 1], [0, 0, 0],
                                      method="LSODA",
                                      args=(network.parameters))
        network_solution = solve_ivp(network.equations, [0, 1], [0, 0, 0],
                                     method="LSODA",
                                     args=(network.parameters))

        assert np.all(expected_solution.y[0, :] == network_solution.y[0, :])
        assert np.all(expected_solution.y[1, :] == network_solution.y[1, :])
        assert np.all(expected_solution.y[2, :] == network_solution.y[2, :])

    def test_parameters_have_wrong_dimension(self):
        connectivity = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]])
        p_dict = {
            "A": np.array([0.00004564, 0.16342066, 0.70351496]),
            "B": np.array([56.67356362, 50.87885942, 30.31904319]),
            "G": np.array([[0, 0, 1657.39796806],
                          [6988.78292809, 0, 0],
                          [0, 7461.31128709, 0]]),
            "K": np.array([[0, 0, 0.11196140],
                          [0.50912113, 0, 0],
                          [0, 0.33210760, 0]]),
            "N": np.array([[0, 0, 3.67709117],
                          [7.17307667, 0, 0],
                          [0, 6.80130072, 0]])
        }
        for key in ["A", "B", "G", "K", "N"]:
            p_dict_new = copy.deepcopy(p_dict)
            p_dict_new[key] = np.concatenate((p_dict[key], p_dict[key]))
            with pytest.raises(ValueError):
                _ = bn.Network(connectivity, p_dict_new)

    def test_parameters_have_wrong_keys(self):
        connectivity = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]])
        more_params = {
            "A": np.array([0.00004564, 0.16342066, 0.70351496]),
            "B": np.array([56.67356362, 50.87885942, 30.31904319]),
            "G": np.array([[0, 0, 1657.39796806],
                          [6988.78292809, 0, 0],
                          [0, 7461.31128709, 0]]),
            "K": np.array([[0, 0, 0.11196140],
                          [0.50912113, 0, 0],
                          [0, 0.33210760, 0]]),
            "N": np.array([[0, 0, 3.67709117],
                          [7.17307667, 0, 0],
                          [0, 6.80130072, 0]]),
            "H": np.array([1.0, 1.0, 1.0])
        }
        less_params = {
            "A": np.array([0.00004564, 0.16342066, 0.70351496]),
            "G": np.array([[0, 0, 1657.39796806],
                          [6988.78292809, 0, 0],
                          [0, 7461.31128709, 0]]),
            "K": np.array([[0, 0, 0.11196140],
                          [0.50912113, 0, 0],
                          [0, 0.33210760, 0]]),
            "N": np.array([[0, 0, 3.67709117],
                          [7.17307667, 0, 0],
                          [0, 6.80130072, 0]]),
        }
        with pytest.raises(ValueError):
            _ = bn.Network(connectivity, more_params)

        with pytest.raises(ValueError):
            _ = bn.Network(connectivity, less_params)