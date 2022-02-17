"""
Unit tests for the detection module
"""

import os
import sys
import numpy as np
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(ROOT_DIR)
from simulation_tools import detection  # noqa : E402
from simulation_tools import bio_networks as bn  # noqa : E402


class TestDetection:

    def test_oscillation_detection(self):
        connectivity = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]])
        params = {
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
        B_ref = params["B"][0]
        params["B"] = params["B"]/B_ref
        params["G"] = params["G"]/B_ref

        hyperparams = {
            "vel_threshold": 1e-3,
            "solve_method": "LSODA",
            "rtol": 1e-5,
            "atol": 1e-7,
            "find_peaks_tpoints": 2000,
            "peak_dist": 10,
            "peak_prom": 1e-3,
            "num_peak_threshold": 5,
            "equil_time": 10,
            "T_mult": 10,
        }
        network = bn.Network(connectivity, params)
        detector = detection.OscillationDetector(hyperparams)
        network = detector.analyze(network)

    """
    def test_true_steady_state(self):
        connectivity = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]])
        p_dict = {
            "A": np.array([0.01, 0.5, 0.01]),
            "B": np.array([10.0, 50.0, 30.0]),
            "G": np.array([[0, 0, 1.0],
                          [68.0, 0, 0],
                          [0, 761.0, 0]]),
            "K": np.array([[0, 0, 0.1],
                          [0.5, 0, 0],
                          [0, 0.3, 0]]),
            "N": np.array([[0, 0, 1.0],
                          [1.0, 0, 0],
                          [0, 1.0, 0]])
        }
        network = bn.Network(connectivity, p_dict)
        network_solution = solve_ivp(network.equations, [0, 1], [0, 0, 0],
                                     method="LSODA",
                                     args=(network.parameters))
        end_point = network_solution.y[:, -1]
        assert detection.is_steady_state(network, end_point, 1e-3)

    def test_false_steady_state(self):
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
        network_solution = solve_ivp(network.equations, [0, 1], [0, 0, 0],
                                     method="LSODA",
                                     args=(network.parameters))
        end_point = network_solution.y[:, -1]
        assert ~detection.is_steady_state(network, end_point, 1e-3)
    """
