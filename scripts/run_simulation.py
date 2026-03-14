import mujoco
import mujoco.viewer
import time
import numpy as np

from src.manipulator import Manipulator
from src.control import Impedance
from src.planning import PotentialField
from src.utils import create_marker, get_yaml_content

m = mujoco.MjModel.from_xml_path('env/scene.xml')
d = mujoco.MjData(m)

# Create manipulator
manipulator = Manipulator(mujoco, m, d, ee_joint_name='gripperMover')

# Create controller
K_d = np.diag([1, 1, 1]) * 100
D_d = 2 * np.sqrt(K_d)
controller = Impedance(manipulator, K_d, D_d)
manipulator.set_controller(controller)

# Generate potential field
objects_data = get_yaml_content('config/potential_field.yaml')
potential_field = PotentialField(objects_data)
manipulator.set_potential_field(potential_field)

# Define desired inputs
input_d = {
    'x_d': np.array([0.4, 0.0, 0.5]),
    'x_d_dot': np.array([0.0, 0.0, 0.0])
}

# Provide a random setpoint when the EE is within epsilon of current setpoint
def random_setpoint(epsilon=0.1):
    x = manipulator.get_ee_position()
    if np.linalg.norm(x - input_d['x_d']) < epsilon:
        input_d['x_d'] = np.array([np.random.uniform(0, 0.5), np.random.uniform(0, 0.5), np.random.uniform(0, 0.5)])

with mujoco.viewer.launch_passive(m, d) as viewer:
    marker = create_marker(viewer, mujoco)

    start = time.time()
    while viewer.is_running():
        step_start = time.time()

        ### Control Logic ###
        #random_setpoint()

        manipulator.step(input_d) # Compute and send torque commands
        #####################

        mujoco.mj_step(m, d) # Step simulation

        with viewer.lock():
            marker.pos[:] = input_d['x_d'] # Display marker at position setpoint
            viewer.user_scn_updated = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2) # Show contact points every 2 seconds

        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)