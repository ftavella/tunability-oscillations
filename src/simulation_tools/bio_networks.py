"""
- A brief description of the module and its purpose
- A list of any classes, exception, functions, and any other objects exported
  by the module
"""

from typing import Callable
import numpy as np


class Network():
    """A class used to represent a biochemical network with defined parameters

    Attributes
    ----------
    connectivity : ndarray
        Two-dimensional array specifying the interaction among nodes. Only
        0, 1, and -1 are allowed as interaction types
    equations : function
        Function containing ODE equations that is suitable for simulation with
        scipy's solve_ivp module. The equations are defined

    Methods
    -------
    __create_gamma_terms()
        Creates strings representing the interaction terms of the network
    __create_function_str()
        Creates a string defining a python function. The function represents
        an ODE model the network.
    __create_equations():
        Defines a python function encoding the ODE equations of the model.
    """

    def __init__(self, connectivity: np.ndarray):
        """
        Parameters
        ----------
        connectivity : ndarray
            Two-dimensional array specifying the interaction among nodes. Only
            0, 1, and -1 are allowed as interaction types

        Raises
        ------
        TypeError
            If connectivity is not a numpy ndarray

        ValueError
            If connectivity doesn't have the right dimension or
            values at each entry
        """

        if not isinstance(connectivity, np.ndarray):
            raise TypeError("Input should be a numpy array")
        if len(connectivity.shape) != 2:
            raise ValueError("Connectivity matrix should have dimension 2")
        elif connectivity.shape[0] != connectivity.shape[1]:
            raise ValueError("Dimensions of connectivity matrix should have \
                            the same length")
        elif np.any(~np.isin(np.unique(connectivity), [-1, 0, 1])):
            raise ValueError("Input should only contain 0, 1, -1")

        self.connectivity = connectivity
        self.equations = self.__create_equations()

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
        fn_template = "def equations(t, x, A, B, G, K, N):"
        return_template = "\n\treturn np.array(["
        for idx, terms in enumerate(G_terms):
            fn_template += f"\n\tdx{idx}dt ="
            fn_template += f" B[{idx}]*(A[{idx}] - x[{idx}]) {terms}"
            return_template += f"dx{idx}dt,"
        fn_template += return_template[:-1] + "])"
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
