
"""
This module contains the functors which determine the current at the time t.

NOTE: the two current models used in this pset do not depend explicitly on
time, so they will not actually use the variable t.  However, since in many
cases the current will be dependent on time, the functors will include the
time t for generality.  There is an example of such a time-dependent current
in the functor Isine.
"""

import numpy as np


class Iconst():
    """Constant current functor"""

    def __init__(self, Iamp):
        """
        Args:
            Iamp: value of constant current
        """
        self.Iamp = Iamp

    def __call__(self, t):
        return self.Iamp


class Isine():
    """Sinusoidal current"""

    def __init__(self, Iamp, Iperiod):
        """
        Args:
            Iamp: amplitude of current
            Iperiod: time period of current
        """
        self.Iamp = Iamp
        self.Iperiod = Iperiod

    def __call__(self, t):
        return self.Iamp * np.sin(2 * np.pi * t / self.Iperiod)


class Iconst_noise():
    """Current with Gaussian white noise"""

    def __init__(self, Iamp, Isig):
        """
        Args:
            Iamp: mean value of current
            Isig: standard deviation of Gaussian white noise
        """
        self.Iamp = Iamp
        self.Isig = Isig
        self.rng = np.random.default_rng()

    def __call__(self, t):
        #### BEGIN SOLUTION #####
        return self.rng.normal(self.Iamp, self.Isig)
