"""
Unit tests for the detection module
"""

import os
import sys
import pytest
import numpy as np
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(ROOT_DIR)
from simulation_tools import detection  # noqa : E402
from simulation_tools import bio_networks as bn  # noqa : E402


class TestDetection:

    def test_detection_per_amp_endpoint(self):
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
        # Dimensionless time transformation
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
        T_guess = 1.0
        y0 = [0.0, 0.0, 0.0]
        network = detector.analyze(network, y0, T_guess)

        assert network.features['status'] == "oscillatory"
        assert np.abs(network.features['per'][0] == 1.277) < 1e-3
        assert np.abs(network.features['per'][1] == 1.277) < 1e-3
        assert np.abs(network.features['per'][2] == 1.277) < 1e-3
        assert np.abs(network.features['amp'][0] == 0.0932) < 1e-4
        assert np.abs(network.features['amp'][1] == 0.0437) < 1e-4
        assert np.abs(network.features['amp'][2] == 0.0248) < 1e-4
        assert np.abs(network.features['end_point'][0] - 0.219) < 1e-3
        assert np.abs(network.features['end_point'][1] - 0.258) < 1e-3
        assert np.abs(network.features['end_point'][2] - 0.0230) < 1e-4

    def test_true_steady_state(self):
        connectivity = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]])
        params = {
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
        T_guess = 1.0
        y0 = [0.0, 0.0, 0.0]
        network = detector.analyze(network, y0, T_guess)

        assert network.features['status'] == "steady_state"
        assert np.isnan(network.features['per'][0])
        assert np.isnan(network.features['per'][1])
        assert np.isnan(network.features['per'][2])
        assert np.isnan(network.features['amp'][0])
        assert np.isnan(network.features['amp'][1])
        assert np.isnan(network.features['amp'][2])
        assert np.abs(network.features['end_point'][0] - 0.0105) < 1e-4
        assert np.abs(network.features['end_point'][1] - 0.513) < 1e-3
        assert np.abs(network.features['end_point'][2] - 0.000587) < 1e-6

    def test_not_enough_peaks(self):
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
        # Dimensionless time transformation
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
            "num_peak_threshold": 20,  # NOTICE change here
            "equil_time": 10,
            "T_mult": 10,
        }
        network = bn.Network(connectivity, params)
        detector = detection.OscillationDetector(hyperparams)
        T_guess = 1.0
        y0 = [0.0, 0.0, 0.0]
        network = detector.analyze(network, y0, T_guess)

        status = 'not_enough_peaks max-0,min-0,max-1,min-1,max-2,min-2'
        assert network.features['status'] == status

    def test_hparams_incomplete(self):
        less_hyperparams = {
            "vel_threshold": 1e-3,
            "rtol": 1e-5,
            "atol": 1e-7,
            "find_peaks_tpoints": 2000,
            "peak_dist": 10,
            "peak_prom": 1e-3,
            "num_peak_threshold": 20,
            "equil_time": 10,
            "T_mult": 10,
        }

        more_hyperparams = {
            "vel_threshold": 1e-3,
            "solve_method": "LSODA",
            "rtol": 1e-5,
            "atol": 1e-7,
            "find_peaks_tpoints": 2000,
            "peak_dist": 10,
            "peak_prom": 1e-3,
            "num_peak_threshold": 20,
            "equil_time": 10,
            "T_mult": 10,
            "equations": "123"
        }

        with pytest.raises(ValueError):
            _ = detection.OscillationDetector(less_hyperparams)

        with pytest.raises(ValueError):
            _ = detection.OscillationDetector(more_hyperparams)
