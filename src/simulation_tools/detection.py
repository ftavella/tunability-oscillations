"""
Module for the detection of oscillations in networks
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.integrate import solve_ivp


class Hyperparameters:
    """Hyperparameters of the detection of oscillations

    Attributes
    ----------
    vel_threshold : float
        Threshold for the norm of velocity to determine whether a system has
        reached a steady state or not
    solve_method : str
        String defining which method to use in solve_ivp (see scipy's 
        solve_ivp documentation for options)
    rtol : float
        Relative tolerance for solve_ivp's solution
    atol : float
        Absolute tolerance for solve_ivp's solution
    find_peaks_tpoints : int
        Number of timepoints to interpolate solve_ivp's solution. The
        interpolated solution is used for peak detection
    peak_dist : int
        Minimum number of samples between two detected peaks
    peak_prom : float
        Minimum peak prominence that is detected
    num_peak_threshold : int
        Minimum number of peaks that have to be detected to consider a solution
        oscillatory
    equil_time : float
        Equilibration time that the network is simulated before acquiring
        samples for peak detection
    T_mult : float
        Multiplicative factor by which multiplied the period guess is
        multiplied. Usually T_mult should be larger than num_peak_threshold
    """

    def __init__(self, hparam_dict):
        """
        Parameters
        ----------
        hparam_dict : dict
            Dictionary specifying the hyperparameters for the detection of
            oscillations.
        """
 
        # Check that hparam_dict has the correct attributes
        required_hparams = ["vel_threshold", "solve_method", "rtol", "atol",
                            "find_peaks_tpoints", "peak_dist", "peak_prom",
                            "num_peak_threshold", "equil_time", "T_mult",]
        if np.any(~np.isin(list(hparam_dict.keys()), required_hparams)) or \
           np.any(~np.isin(required_hparams, list(hparam_dict.keys()))):
            raise ValueError(f"Hyperparameter dict should contain the \
                             following keys: {required_hparams}")

        for key, value in hparam_dict.items():
            setattr(self, key, value)


class OscillationDetector:
    """Class containing the methods for detecting oscillations in networks.

    Attributes
    ----------
    hparams : Hyperparameters
        Object containing the hyperparameters for the detection of oscillations

    Methods
    -------
    __is_steady_state(network, end_point)
        Checks if the magnitude of the model's velocity vector at end_point is
        less than a threshold
    __calculate_features(network, y0, T_guess)
        Calculates the period, amplitude, status, and endpoint of a network
    analyze(network, T_guess)
    """

    def __init__(self, hparams_dict):
        self.hparams = Hyperparameters(hparams_dict)

    def __is_steady_state(self, network, end_point):
        """Checks if the magnitude of the model's velocity vector at end_point
        is less than a threshold. If that's the case, the end_point is
        considered to be a steady state

        Parameters
        ----------
        network : Network
            Object specifying the network under consideration
        end_point : ndarray
            Array with a value for each node

        Returns
        -------
        bool
            True if velocity magnitude is less than the threshold
        """

        dxdt = network.equations(0, end_point, *network.parameters)
        velocity = np.linalg.norm(dxdt)
        return velocity < self.hparams.vel_threshold

    def __calculate_features(self, network, y0, T_guess):
        """Calculates the period, amplitude, status, and endpoint of a network
        and sets the network's features attribute.
        
        Network is simulated and oscillations are analyzed using peak 
        detection. If not enough peaks are detected the status is set to
        'not_enough_peaks' plus information that tells where not enough peaks
        were found. For example if not enough maximum peaks were found for
        the first variable of the equation, the final status reads
        'not_enough_peaks max-0'. The same is reported for minimum peaks using
        the notation 'min-variable'. If enough peaks are detected the status
        is set to 'oscillatory' and period, amplitude, endpoint and variability
        are calculated and returned inside the features of the network.

        Parameters
        ----------
        network : Network
            The network to be analyzed
        y0 : ndarray
            Initial condition for the simulation
        T_guess : float
            Guess for the period to be used as reference for total simulation
            time

        Returns
        -------
        network : Network
            Network with updated features
        """
        # Simulate system from y0_guess
        t_stop = self.hparams.T_mult * T_guess
        solution = solve_ivp(network.equations, [0, t_stop], y0,
                             args=(network.parameters), dense_output=True,
                             method=self.hparams.solve_method,
                             rtol=self.hparams.rtol,
                             atol=self.hparams.atol)

        # Check if enough peaks are found
        time = np.linspace(0, t_stop, self.hparams.find_peaks_tpoints)
        max_pks = [None for _ in range(network.nodes)]
        min_pks = [None for _ in range(network.nodes)]

        status = ""
        for var in range(network.nodes):
            conc = solution.sol(time)[var]
            max_pks[var], _ = find_peaks(conc,
                                         distance=self.hparams.peak_dist,
                                         prominence=self.hparams.peak_prom)
            min_pks[var], _ = find_peaks(-conc,
                                         distance=self.hparams.peak_dist,
                                         prominence=self.hparams.peak_prom)
            if len(max_pks[var]) < self.hparams.num_peak_threshold:
                status += f"max-{var},"
            if len(min_pks[var]) < self.hparams.num_peak_threshold:
                status += f"min-{var},"

        """
        # DEBUG
        import matplotlib.pyplot as plt
        plt.plot(solution.t, solution.y[0,:], '-o')
        plt.vlines(time[max_pks[0]], np.min(solution.y[0,:]),
                   np.max(solution.y[0,:]))
        plt.vlines(time[min_pks[0]], np.min(solution.y[0,:]),
                   np.max(solution.y[0,:]))
        plt.plot(solution.t, solution.y[1,:], '-o')
        plt.plot(solution.t, solution.y[2,:], '-o')
        plt.show()
        """

        if len(status) > 0:
            status = "not_enough_peaks " + status[:-1]
            network.features["status"] = status
            network.features["end_point"] = solution.y[:, -1]
            return network

        elif len(status) == 0:
            # Enough peaks found, network is oscillatory, find per and amp
            for var in range(network.nodes):
                # Period
                time_pks = time[max_pks[var]]
                per_vals = np.diff(time_pks)[1:]  # Don't use first difference
                per = np.mean(per_vals)
                # Amplitude
                conc = solution.sol(time)[var]
                if len(max_pks[var]) == len(min_pks[var]):
                    amp_vals = conc[max_pks[var]] - conc[min_pks[var]]
                    amp = np.mean(amp_vals)
                elif len(max_pks[var]) > len(min_pks[var]):
                    d = len(max_pks[var]) - len(min_pks[var])  # Difference
                    amp_vals = conc[max_pks[var][d:]] - conc[min_pks[var]]
                    amp = np.mean(amp_vals)
                else:
                    d = len(min_pks[var]) - len(max_pks[var])  # Difference
                    amp_vals = conc[max_pks[var]] - conc[min_pks[var][d:]]
                    amp = np.mean(amp_vals)

                # Period and amplitude variability
                per_cv = np.std(per_vals)/np.mean(per_vals)
                amp_cv = np.std(amp_vals)/np.mean(amp_vals)

                # Store calculation results
                network.features["per"][var] = per
                network.features["per_cv"][var] = per_cv
                network.features["amp"][var] = amp
                network.features["amp_cv"][var] = amp_cv

            network.features["status"] = "oscillatory"
            network.features["end_point"] = solution.y[:, -1]
            return network

    def analyze(self, network, y0, T_guess):
        """Analyzes whether a network reaches a steady state or oscillations.

        The network is equilibrated first and then check if it has reached a
        steady state. In such case, the status is set to 'steady_state'. If not
        oscillatory features are calculated

        Parameters
        ----------
        network : Network
            The network to be analyzed
        y0 : ndarray
            Initial condition for the simulation
        T_guess : float
            Guess for the period to be used as reference for total simulation
            time


        Returns
        -------
        network : Network
            Network with updated features
        """

        # Equilibrate
        t_stop = self.hparams.equil_time
        solution = solve_ivp(network.equations, [0, t_stop], y0,
                             args=(network.parameters), dense_output=True,
                             method=self.hparams.solve_method,
                             rtol=self.hparams.rtol,
                             atol=self.hparams.atol)
        end_point = solution.y[:, -1]
        # Check if steady state was reached
        if self.__is_steady_state(network, end_point):
            network.features["status"] = "steady_state"
            network.features["end_point"] = end_point
            return network
        else:
            # Oscillatory, calculate features
            return self.__calculate_features(network, end_point, T_guess)
