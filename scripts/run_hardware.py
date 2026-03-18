import numpy as np
import pinocchio as pin
from unitree_arm_interface import UnitreeArm

from src.manipulator import Manipulator
from src.control import Impedance
from src.planning import PotentialField
from src.utils import get_yaml_content

# Scene and robot paths
PATH_TO_SCENE = 'env/scene.xml'
PATH_TO_ROBOT = 'models/z1.urdf'

# Create unitree arm
arm = UnitreeArm()

# Create manipulator
manipulator = Manipulator(PATH_TO_ROBOT)
tau_limit = np.full(manipulator.model.nv, 30.0)

# Create controller
K_d_translation = np.array([1.0, 1.0, 1.0]) * 250
K_d_rotation = np.array([1.0, 1.0, 1.0]) * 0
K_d = np.diag(np.concatenate([K_d_translation, K_d_rotation]))
D_d = 2 * np.sqrt(K_d)
controller = Impedance(K_d, D_d)

# Define desired states as pose (4x4 homogenous transformation) and twist (6 DOF vector)
R_d = np.eye(3)
p_d = np.array([0.3, 0.0, 0.3])

pose_d = pin.SE3(R_d, p_d)
twist_d = np.array([0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0])

# Setup potential field
objects_data = get_yaml_content('config/potential_field.yaml')
potential_field = PotentialField(objects_data)

# End effector frame
frame_name = "joint6"

# Control loop logic
def control_loop(q, q_dot, pose_d, twist_d):
    # Update pinocchio data object
    manipulator.update(q, q_dot)

    # Compute potential field
    F_p = potential_field.compute_F(q)
    jacobians = manipulator.get_joint_jacobians()
    tau_p = jacobians.T @ F_p

    # Compute error terms [translations orientations]
    e, e_dot = manipulator.get_frame_error(pose_d, twist_d, frame_name)

    # Compute control command 6D wrench vector [force, torque]
    jacobian = manipulator.get_joint_jacobian(frame_name)
    tau_nle = manipulator.get_non_linear_effects() # Coriolis + centrifugal + gravity

    W_d = controller.compute(e, e_dot)
    tau_c = jacobian.T @ W_d + tau_p + tau_nle

    # Send control command
    return np.clip(tau_c, -tau_limit, tau_limit)

# Run control loop
running = True
while running:
    try:
        tau_c = control_loop(arm.q, arm.qd, pose_d, twist_d)
        arm.sendRecv(tau_c)
    except KeyboardInterrupt:
        print("Stop Initiated!")
        arm.setFsm(arm.ARMFSMSTATE.PASSIVE)
        arm.sendRecv()
        running = False
        break

# Return back to home with safety block
try:
    print("Homing back to start")
    arm.setFsm(arm.ARMFSMSTATE.ACTIVE)
    arm.backToStart()
except KeyboardInterrupt:
    print("Homing interrupted")

finally:
    print("Setting robot to passive mode!")
    arm.setFsm(arm.ARMFSMSTATE.PASSIVE)
    arm.sendRecv()