
"""
Model neuron dynamics using the FitzHugh-Nagumo equations

     dV/dt = 1/tauV * (F(V) - W) + I(t)
     dW/dt = 1/tauW * (alpha * V - W)

where:
        V  = neuron voltage state
        W  = recovery state
      F(V) = V * (V - Vs) * (1 - V)
        Vs = voltage at which F(V) switch signs
      I(t) = current input (stimulus)
      tauV = voltage response timescale
      tauW = voltage response timescale
     alpha = response factor of W to V
"""

import numpy as np
from IVPlib import IVP
import IVPlib as IVPlib


class NeuronIVP(IVP):

    def evalf(self, u, t):
        """
        Calculates right-hand side in the  model

        Args:
            u (NumPy ndarray): current state
            t (float): current time

        Returns:
            f (NumPy ndarray): current right-hand side
        """
        Vs    = self.get_p('Vs')
        tauV  = self.get_p('tauV')
        tauW  = self.get_p('tauW')
        alpha = self.get_p('alpha')
        I     = self.get_p('I')

        V = u[0]
        W = u[1]

        fV = 1 / tauV * (V * (Vs - V) * (V - 1) - W) + I(t)
        fW = 1 / tauW * (alpha * V - W)

        return np.array([fV, fW])


def solve_neuron(uI, tF, dt, p):
    neuron_IVP = NeuronIVP(uI, 0.0, tF, p)
    t, u = IVPlib.solve(neuron_IVP, dt, IVPlib.step_RK4)
    V = u[:, 0]
    W = u[:, 1]
    return t, V, W
