"""
Unit tests for the detection library
"""

import os
import sys
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(ROOT_DIR)
from simulation_tools import detection

class TestDetection:

    def test_simulate_network(self):
        def model(t, y):
            return 0
        assert 1 == detection.simulate_network(model).success