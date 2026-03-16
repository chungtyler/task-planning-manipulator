import numpy as np

class Impedance:
    def __init__(self, K_d, D_d):
        self.K_d = K_d
        self.D_d = D_d

    # Calculate the force-torque [Fx, Fy, Fz, tau_x, tau_y, tau_z] on the end effector for a given desired position and velocity
    def compute(self, e, e_dot):
        W = self.K_d @ e + self.D_d @ e_dot
        return W

# TODO Finish controller implementation
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