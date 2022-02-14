"""
- A brief description of the module and its purpose
- A list of any classes, exception, functions, and any other objects exported
  by the module
"""

import numpy as np


def create_gamma_terms(connectivity: np.ndarray) -> list:
    """Create interaction terms based on the connectivity of the network

    ...

    Parameters
    ----------
    connectivity : ndarray
        Two-dimensional array specifying the interaction among nodes. Only
        0, 1, and -1 are allowed as interaction types

    Raises
    ------
    TypeError
        If connectivity is not an array

    ValueError
        If connectivity doesn't have the right dimension or
        values at each entry

    Returns
    -------
    list
        A list of strings containing the string that encodes for the
        interaction terms of each node. The dimension matches the number of
        rows of connectivity
    """

    if not isinstance(connectivity, np.ndarray):
        raise TypeError("Input should be a numpy array")
    if len(connectivity.shape) != 2:
        raise ValueError("Connectivity matrix should have dimension 2")
    elif connectivity.shape[0] != connectivity.shape[1]:
        raise ValueError("Dimensions of connectivity matrix should have \
                          the same length")
    elif np.any(~np.isin(np.unique(connectivity), [-1, 0, 1])):
        raise ValueError("Connectivity matrix should contain only 0, 1, -1")
    # Write terms
    terms = ["" for _ in range(connectivity.shape[0])]
    for idx, row in enumerate(connectivity):
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


class Network():
    """A class used to represent a biochemical network with defined parameters

    ...

    Attributes
    ----------
    nodes : int
        Number of nodes in the network.
    edges : int
        Number of interactions in the network.
    period : float
        If parameters are oscillatory stores the period.
        If not oscillatory, gets a value of NaN.
    amplitude : ndarray
        If parameters are oscillatory stores the amplitude of each node.
        If not oscillatory, each element of the gets a value of NaN.

    Methods
    -------
    says(sound=None)
        Prints the animals name and what sound it makesSummary
    """

    def __init__(self, connectivity):
        """
        Parameters
        ----------
        connectivity : ndarray
            Numpy array specifying positive or negative interactions between
            nodes. Self-loops are specified don its diagonal.
        """

        self.connectivity = connectivity
