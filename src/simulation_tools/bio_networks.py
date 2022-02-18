"""
Module defining the classes and methods for representing a biological network
of interacting nodes.

Classes
-------

Network
    A class used to represent a biochemical network with defined parameters
"""

from typing import Callable
import numpy as np
from numba import njit


class Network:
    """A class used to represent a biochemical network with defined parameters

    Attributes
    ----------
    connectivity : ndarray
        Two-dimensional array specifying the interaction among nodes. Only
        0, 1, and -1 are allowed as interaction types
    equations : function
        Function containing ODE equations that is suitable for simulation with
        scipy's solve_ivp module. The equations are defined
    nodes : int
        Number of nodes of the network. Inferred from connectivity
    edges : list
        List of edges. Edges are represented as 2D tuples that contain the
        indexes where connectivity is different from 0.
    parameters : list
        A list containing all parameter arrays. The list is created from
        param_dict (see __init__).
    features : dict
        Dictionary with the properties of the network. Properties are status
        (not_simulated, steady_state, not_enough_peaks, oscillatory), per
        (period), amp (amplitude), and end_point (final state of the
        simulation). Per, amp, and end_point are stored for each node.

    Methods
    -------
    __edge_list_from_connectivity()
        Create a list of edges from the connectivity matrix.
    __create_gamma_terms()
        Creates strings representing the interaction terms of the network
    __create_function_str()
        Creates a string defining a python function. The function represents
        an ODE model the network.
    __create_equations():
        Defines a python function encoding the ODE equations of the model.
    """

    def __init__(self, connectivity: np.ndarray, param_dict: dict):
        """
        Parameters
        ----------
        connectivity : ndarray
            Two-dimensional array specifying the interaction among nodes. Only
            0, 1, and -1 are allowed as interaction types
        param_dict : dict
            Dictionary specifying the parameters of the model. The only allowed
            keys are A, B, G, K, N. the corresponding values should have the
            correct dimension. Arrays for A (alpha) and B (beta) should have a
            length that matches the number of nodes in the network. Arrays for
            G, K, and N (gamma, kappa, eta) should have the same shape as the
            connectivity.

        Raises
        ------
        TypeError
            If connectivity is not a numpy ndarray

        ValueError
            If connectivity doesn't have the right dimension or
            values at each entry. Also ValueError is raised if parameter
            arrays don't have the correct dimensions
        """

        if not isinstance(connectivity, np.ndarray):
            raise TypeError("Input should be a numpy array")
        if len(connectivity.shape) != 2:
            raise ValueError("Connectivity matrix should have dimension 2")
        if connectivity.shape[0] != connectivity.shape[1]:
            raise ValueError("Dimensions of connectivity matrix should have \
                            the same length")
        if np.any(~np.isin(np.unique(connectivity), [-1, 0, 1])):
            raise ValueError("Input should only contain 0, 1, -1")
        param_keys = ["A", "B", "G", "K", "N"]
        if np.any(~np.isin(list(param_dict.keys()), param_keys)) or \
           np.any(~np.isin(param_keys, list(param_dict.keys()))):
            raise ValueError("Parameter dict can only have keys named: \
                             A, B, G, K, N")

        self.connectivity = connectivity
        self.equations = self.__create_equations()
        self.nodes = np.shape(connectivity)[0]
        self.edges = self.__edge_list_from_connectivity()

        for key, value in param_dict.items():
            if key == "A" or key == "B":
                if np.shape(value)[0] != self.nodes:
                    raise ValueError(f"Parameter array for {key} should have \
                                     the same length as the number of rows in \
                                     the connectivity matrix")
            elif key == "G" or key == "K" or key == "N":
                if np.shape(value) != np.shape(connectivity):
                    raise ValueError("Parameter array for {key} should have \
                                     the same dimension as the connectivity")

        self.parameters = [param_dict[key] for key in
                           ["A", "B", "G", "K", "N"]]

        self.features = {
            "status": "not_simulated",
            "per": [np.NaN for _ in range(self.nodes)],
            "per_cv": [np.NaN for _ in range(self.nodes)],
            "amp": [np.NaN for _ in range(self.nodes)],
            "amp_cv": [np.NaN for _ in range(self.nodes)],
            "end_point": [np.NaN for _ in range(self.nodes)],
        }

    def __edge_list_from_connectivity(self) -> list:
        """Create a list of edges from the connectivity matrix. Edges are
        represented as tuples containing the index of the maxtrix where
        connectivity is different from 0

        Returns
        -------
        list
            A list of tuples. Each tuple represents the two dimensional index
            where the connectivity matrix is different from 0.
        """

        edges = []
        for idx in range(self.nodes):
            for jdx in range(self.nodes):
                if self.connectivity[idx, jdx] != 0:
                    edges.append((idx, jdx))
        return edges

    def __create_gamma_terms(self) -> list:
        """Creates strings representing the interaction terms of the network
        based on its connectivity.

        Returns
        -------
        list
            A list of strings containing the string that encodes for the
            interaction terms of each node. The dimension matches the number of
            rows of connectivity
        """

        # Write terms
        terms = ["" for _ in range(self.connectivity.shape[0])]
        for idx, row in enumerate(self.connectivity):
            terms_in_row = ""
            for jdx, value in enumerate(row):
                sign = ""
                subs = ""
                if value == 0:
                    continue
                else:
                    enz = f"x[{jdx}]"
                    pair = f"[{idx},{jdx}]"
                    if value == 1:
                        sign = "+"
                        subs = f"(1.0 - x[{idx}])"
                    elif value == -1:
                        sign = "-"
                        subs = f"x[{idx}]"
                # Write term and add it to the terms for this node
                new_term = f"{sign} G{pair}*{subs}*{enz}**N{pair}/"
                new_term += f"({enz}**N{pair} + K{pair}**N{pair}) "
                terms_in_row += new_term
            terms[idx] = terms_in_row
        return terms

    def __create_function_str(self) -> str:
        """Creates a string defining a python function with an ODE model
        following the connectivity of the network.

        Returns
        -------
        str
            A string defining the ODE equations as a python function.
        """

        G_terms = self.__create_gamma_terms()
        fn_template = "def equations_f(t, x, A, B, G, K, N):"
        return_template = "\n\treturn np.array(["
        for idx, terms in enumerate(G_terms):
            fn_template += f"\n\tdx{idx}dt ="
            fn_template += f" B[{idx}]*(A[{idx}] - x[{idx}]) {terms}"
            return_template += f"dx{idx}dt,"
        fn_template += return_template[:-1] + "])"
        fn_template += "\nequations = njit(equations_f)"
        return fn_template

    def __create_equations(self) -> Callable:
        """Defines a python function encoding the ODE equations of the model
        based on the connectivity of the network.

        Returns
        -------
        function
            A function with the signature: equations(t, x, A, B, G, K, N).
            Connectivity defines the terms included in each equation.
        """

        fn_str = self.__create_function_str()
        loc = {}
        exec(fn_str, globals(), loc)
        equations = loc['equations']
        return equations
