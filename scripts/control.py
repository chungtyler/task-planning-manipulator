import time
import numpy as np

import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path('env/scene.xml')
d = mujoco.MjData(m)

K_d = np.diag([1, 1, 1]) * 50
D_d = 2 * np.sqrt(K_d)

def create_marker():
  scene = viewer.user_scn
  num_geoms = scene.ngeom
  geom = scene.geoms[num_geoms]
  mujoco.mjv_initGeom(
      geom,
      mujoco.mjtGeom.mjGEOM_SPHERE,
      np.array([0.05, 0.05, 0.05]),  # size
      np.array([0, 0, 0], dtype=np.float64),  # pos
      np.eye(3).flatten(),  # rotation matrix (identity)
      np.array([1, 0, 0.9, 1], dtype=np.float32)  # rgba
    )
  scene.ngeom += 1
  return geom

def IK(ee_target_pos, joint_name='gripperMover'):
  joint_id = m.body(joint_name).id
  ee_pos = d.xpos[joint_id]
  jac = np.zeros((3, m.nv))
  mujoco.mj_jacBodyCom(m, d, jac, None, joint_id)
  jac_pinv = np.linalg.pinv(jac)
  qdot = jac_pinv @ (ee_target_pos - ee_pos)
  q_target_pos = d.qpos + qdot * 0.2
  return q_target_pos 

def get_jacobian(joint_name):
  joint_id = m.body(joint_name).id
  jac_p = np.zeros((3, m.nv))
  mujoco.mj_jacBodyCom(m, d, jac_p, None, joint_id)
  return jac_p

p_0 = 0.2
k_rep = 1000
def potential_field(x):
  floor = np.array([0, 0, 1])
  p_q = floor @ x

  F = 0
  if p_q <= p_0:
    F = 0.5 * k_rep * ((1/p_q) - (1/p_0)) ** 2
    print(F)
  return F * floor


def impedance_control(x_d, x_d_dot, joint_name='gripperMover'):
  joint_id = m.body(joint_name).id
  jac_p = get_jacobian(joint_name)
  x = d.xpos[joint_id]
  x_dot = jac_p @ d.qvel
  F_ext = (K_d @ -(x - x_d) + D_d @ -(x_dot - x_d_dot)) 
  tau_d = jac_p.T @ (F_ext + potential_field(x)) + d.qfrc_bias
  return tau_d

with mujoco.viewer.launch_passive(m, d) as viewer:
  marker = create_marker()

  #x_d = np.array([0.5, 0, 0.4])
  x_d_dot = [0, 0, 0]

  start = time.time()
  while viewer.is_running():
    step_start = time.time()

    ### Add Logic Here ###
    joint_id = m.body('gripperMover').id
    x = d.xpos[joint_id]
    x_d = x
    x_d_dot = [0, 0, 0]
    d.ctrl = impedance_control(x_d, x_d_dot)

    # if np.linalg.norm(x - x_d) < 0.1:
    #   x_d = np.array([np.random.uniform(0, 0.5), np.random.uniform(0, 0.5), np.random.uniform(0, 0.5)])
 
    #####################

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(m, d)

    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      marker.pos[:] = x_d
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)
      viewer.user_scn_updated = True

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)