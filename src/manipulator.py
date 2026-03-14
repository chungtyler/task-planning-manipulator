import numpy as np

class Manipulator:
    def __init__(self, mujoco, m, d, ee_joint_name):
        self.mujoco = mujoco
        self.m = m
        self.d = d
        self.ee_joint_name = ee_joint_name
        self.ee_joint_id = m.body(ee_joint_name).id

        self.controller = None
        self.potential_field = None

    # Calculate the geometric jacobian with respect to the joint COM
    def compute_ee_jacobian(self):
        jacobian_p = np.zeros((3, self.m.nv))
        jacobian_r = np.zeros((3, self.m.nv))
        self.mujoco.mj_jacBodyCom(self.m, self.d, jacobian_p, jacobian_r, self.ee_joint_id)
        return jacobian_p, jacobian_r
    
    # Calculate the geometric jacobian with respect to the joint COM
    def compute_joint_jacobian(self, joint_id):
        jacobian_p = np.zeros((3, self.m.nv))
        jacobian_r = np.zeros((3, self.m.nv))
        self.mujoco.mj_jacBodyCom(self.m, self.d, jacobian_p, jacobian_r, joint_id)
        return jacobian_p, jacobian_r
    
    # Get the joint position [x, y, z]
    def get_ee_position(self):
        x = self.d.xpos[self.ee_joint_id] # Joint position from forward kinematics
        return x
    
    # Get the joint velocity [x_dot, y_dot, z_dot]
    def get_ee_velocity(self, jacobian_p):
        x_dot = jacobian_p @ self.d.qvel # Joint velocity from jacobian
        return x_dot
    
    # Calculate the inverse kinematics
    def compute_IK(self, ee_target, alpha=0.2):
        error = ee_target - self.get_ee_position()
        jacobian_p, _ = self.compute_jacobian()
        q_dot = jacobian_p * error

        q_target_pos = self.d.qpos + q_dot * alpha
        return q_target_pos
    
    # Setup controller reference
    def set_controller(self, controller):
        self.controller = controller

    # Setup potential field reference
    def set_potential_field(self, potential_field):
        self.potential_field = potential_field
    
    # Compute joint torques based on desired inputs
    def step(self, input_d):
        if not self.controller:
            print("Error no controller is set!")

        jacobian_ee_p, _ = self.compute_ee_jacobian()

        tau_d = self.controller.compute_torque(jacobian_ee_p, input_d)

        F_potential = np.zeros(3)
        if self.potential_field:
            q = self.d.qpos
            F_potential = self.potential_field.compute_F(q)
            tau_pf = np.zeros_like(q)
            for joint_id in range(q.shape[0]):
                jacobian_p, _ = self.compute_joint_jacobian(joint_id)
                tau_pf += jacobian_p.T @ F_potential[joint_id]

        self.d.ctrl = tau_d + tau_pf

