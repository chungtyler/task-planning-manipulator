import numpy as np

class Impedance:
    def __init__(self, manipulator, K_d, D_d):
        self.manipulator = manipulator
        self.K_d = K_d
        self.D_d = D_d

    # Calculate the torque commands for a given desired position and velocity
    def compute_torque(self, jacobian_p, input_d):
        x_d = input_d.get('x_d')
        x_d_dot = input_d.get('x_d_dot', np.zeros(3))

        # Get linear position and velocity of the joint
        x = self.manipulator.get_ee_position()
        x_dot = self.manipulator.get_ee_velocity(jacobian_p)

        # Error terms
        e = x_d - x
        e_dot = x_d_dot - x_dot

        # Calculate the external force and convert to joint torques
        F_ext = self.K_d @ e + self.D_d @ e_dot
        tau_d = jacobian_p.T @ F_ext + self.manipulator.d.qfrc_bias # Add centrifugal, coriolis, and gravity compensation
        return tau_d
    
class VariableImpednace(Impedance):
    def __init__(self, manipulator, K_d, D_d):
        super().__init__(manipulator, K_d, D_d)

    def compute_torque(self, input_d):
        x_d = input_d.get('x_d')
        x_d_dot = input_d.get('x_d_dot', np.zeros(3))
        pass
