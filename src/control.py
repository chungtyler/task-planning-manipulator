import numpy as np

class Impedance:
    def __init__(self, K_d, D_d):
        self.K_d = K_d
        self.D_d = D_d

    # Calculate the torque commands for a given desired position and velocity
    def compute(self, e, e_dot):
        W = self.K_d @ e + self.D_d @ e_dot
        return W

class VariableImpedance(Impedance):
    def __init__(self, K_d, D_d, e):
        super().__init__(K_d, D_d)
        self.e = e

    def ramp():
        pass

    def step():
        pass

    def update_K_d(self, K_d):
        self.K_d = K_d
        self.D_d = 2 * np.sqrt(K_d)