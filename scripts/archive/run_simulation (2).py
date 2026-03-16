import mujoco
import mujoco.viewer
import pinocchio as pin
import time
import numpy as np

from src.manipulator import Manipulator
from src.control import Impedance
from src.planning import PotentialField
from src.utils import create_marker, get_yaml_content

# Scene and robot paths
PATH_TO_SCENE = 'env/scene.xml'
PATH_TO_ROBOT = 'models/z1.urdf'

# Create mujoco objects
model = mujoco.MjModel.from_xml_path(PATH_TO_SCENE)
data = mujoco.MjData(model)

# Create manipulator
manipulator = Manipulator(PATH_TO_ROBOT)
tau_limit = np.full(manipulator.model.nv, 30.0)

# Create controller
K_d_translation = np.array([1.0, 1.0, 1.0]) * 0
K_d_rotation = np.array([1.0, 1.0, 1.0]) * 10
K_d = np.diag(np.concatenate([K_d_translation, K_d_rotation]))
D_d = 2 * np.sqrt(K_d)
controller = Impedance(K_d, D_d)

# Define desired states as pose (homogenous transformation) and twist (6 DOF vector)
R_d = pin.utils.rotate('y', np.pi/2)
p_d = np.array([0.3, 0.0, 0.3])

pose_d = pin.SE3(R_d, p_d)
twist_d = np.array([0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0])

# Define simulation time limit
time_limit = 3000

# End effector frame
frame_name = "joint6"

# Provide a random setpoint when the EE is within epsilon of current setpoint
def random_setpoint(epsilon=0.1):
    global p_d, pose_d
    p = manipulator.get_frame_pose(frame_name).translation
    if np.linalg.norm(p_d - p) < epsilon:
        p_d = np.array([np.random.uniform(0, 0.5), np.random.uniform(-0.5, 0.5), np.random.uniform(0.1, 0.3)])
    pose_d = pin.SE3(R_d, p_d)

# Control loop logic
def control_loop(q, q_dot):
    global pose_d, twist_d
    manipulator.update(q, q_dot) # Update pinocchio data

    # Compute error terms [translations orientations]
    e, e_dot = manipulator.get_frame_error(pose_d, twist_d, frame_name)

    # Compute potential field
    joint_cartesian_positions = np.array([manipulator.data.oMi[joint_id].translation for joint_id in range(1, len(manipulator.model.joints))])
    F_p = potential_field.compute_F(joint_cartesian_positions)
    jacobians = manipulator.get_joint_jacobians()
    tau_p = jacobians.T @ F_p.reshape(-1)

    # Compute control command 6D vector [force, torque]
    jacobian = manipulator.get_joint_jacobian(frame_name)
    tau_nle = manipulator.get_non_linear_effects()

    W_d = controller.compute(e, e_dot)
    tau_c = jacobian.T @ W_d + tau_nle

    # Send control command
    return np.clip(tau_c, -tau_limit, tau_limit)

# Run mujoco simulator
with mujoco.viewer.launch_passive(model, data) as viewer:
    marker = create_marker(viewer, mujoco)
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY

    start = time.time()
    while viewer.is_running() and time.time() - start < time_limit:
        step_start = time.time()
        
        ### Control Loop ###
        random_setpoint()
        tau_c = control_loop(data.qpos, data.qvel)
        data.ctrl[:] = tau_c
        #####################

        mujoco.mj_step(model, data) # Step simulation

        with viewer.lock():
            marker.pos[:] = pose_d.translation # Display marker at position setpoint
            viewer.user_scn_updated = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2) # Show contact points every 2 seconds

        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)