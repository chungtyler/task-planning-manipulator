import mujoco
import mujoco.viewer
import pinocchio as pin
import numpy as np
import time

from src.manipulator import Manipulator
from src.control import Impedance
from src.planning import PotentialField
from src.utils import StateLogger, create_marker, get_yaml_content

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
K_d_translation = np.array([1.0, 1.0, 1.0]) * 30000 # 30000
K_d_rotation = np.array([1.0, 1.0, 1.0]) * 50
K_d = np.diag(np.concatenate([K_d_translation, K_d_rotation]))
D_d = 2 * np.sqrt(K_d)
controller = Impedance(K_d, D_d)

# Define desired states as pose (4x4 homogenous transformation) and twist (6 DOF vector)
R_d = pin.utils.rotate('y', np.pi/2) #np.eye(3)
p_d = np.array([0.4, 0.2, 0.3])

pose_d = pin.SE3(R_d, p_d)
twist_d = np.array([0.0, 0.0, 0.0, 
                    0.0, 0.0, 0.0])

# Create state data logging object
state_logger = StateLogger(state_types=['time', 'position', 'force'],
                           units=['s', 'm', 'N'])

# Define simulation time limit
time_limit = 1500

# End effector frame
frame_name = "joint6"

# # Generate potential field
# objects_data = get_yaml_content('config/potential_field.yaml')
# potential_field = PotentialField(objects_data)

def compute_Rd_pointing_down():
    # Desired tool x-axis in world frame (pointing down)
    x_d = np.array([0.0, 0.0, -1.0])

    # Reference vector to define yaw (world y-axis)
    y_ref = np.array([0.0, 1.0, 0.0])

    # Compute orthonormal basis
    z_d = np.cross(x_d, y_ref)
    z_d /= np.linalg.norm(z_d)

    y_d = np.cross(z_d, x_d)

    # Assemble rotation matrix (world <- tool)
    R_d = np.column_stack((x_d, y_d, z_d))
    return R_d

# Control loop logic
def control_loop(q, q_dot, pose_d, twist_d):
    # Update pinocchio data object
    manipulator.update(q, q_dot)

    # Compute error terms position, velocity
    e, e_dot = manipulator.get_frame_error(pose_d, twist_d, frame_name)
    #print(e)

    # # Compute potential field
    # joint_cartesian_positions = np.array([manipulator.data.oMi[joint_id].translation for joint_id in range(1, len(manipulator.model.joints))])
    # F_p = potential_field.compute_F(joint_cartesian_positions)
    # jacobians = manipulator.get_joint_jacobians()
    # tau_p = jacobians.T @ F_p.reshape(-1)

    # Compute control command 6D wrench vector [force, torque]
    jacobian = manipulator.get_joint_jacobian(frame_name)
    tau_nle = manipulator.get_non_linear_effects() # Coriolis + centrifugal + gravity

    W_d = controller.compute(e, e_dot)
    tau_c = jacobian.T @ W_d + tau_nle #+ tau_p

    # Send control command
    return np.clip(tau_c, -tau_limit, tau_limit)

# Run mujoco simulator
with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
    marker = create_marker(viewer, mujoco) # Show desired position
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY # Show frames
    site_id = model.joint(frame_name).bodyid # End effector body id

    #time.sleep(5) # To allow for recording setup

    start = time.time()
    while viewer.is_running() and time.time() - start < time_limit:
        step_start = time.time()

        ### Control Loop ###
        t = time.time() - start
        manipulator.update(data.qpos, data.qvel)
        # pose_d.rotation = compute_Rd_pointing_down()
        # pose_d.translation = np.array([0.3, 0.0, 0.1*np.sin(t)+0.2])
        # pose_d.translation = np.array([0.1*np.cos(t)+0.3, 0.0, 0.1*np.sin(t)+0.2])
        # pose_d.translation = np.array([0.3, 0.2*np.sin(t), 0.2])
        # pose_d.translation = np.array([0.1*np.cos(t)+0.3, 0.1*np.sin(t), 0.1*np.sin(t)+0.2])
        # tau_c = control_loop(data.qpos, data.qvel, pose_d, twist_d)
        p = manipulator.get_IK_step(frame_name, pose_d, data.qpos)
        data.ctrl[:] = p
        #####################

        mujoco.mj_step(model, data) # Step simulation
        mujoco.mj_rnePostConstraint(model, data) # Update external force readings

        # Log state data
        target_data = {
            'time': time.time() - start,
            'position': pose_d.translation.copy(),
            'force': []
        }

        state_data = {
            'time': time.time() - start,
            'position': data.xpos[site_id].copy(), #data.site_xpos[site_id].copy(),
            'force': data.cfrc_ext[site_id, 3:].copy()
        }
        state_logger.log(state_data, target_data)

        with viewer.lock():
            marker.pos[:] = pose_d.translation # Display marker at setpoint
            viewer.user_scn_updated = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2) # Show contact points every 2 seconds

        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = model.opt.timestep - (time.time() - step_start)

        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

# Show plots
state_logger.plot()