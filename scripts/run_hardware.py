"""
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

# Load URDF of Z1 arm
robot = RobotWrapper.BuildFromURDF("z1_arm.urdf", package_dirs=["path_to_urdf_folder"])

q = pin.neutral(robot.model)  # neutral joint positions
ee_frame = robot.model.getFrameId("ee_link")

# Compute Jacobian at the end-effector
J = pin.computeFrameJacobian(robot.model, robot.data, q, ee_frame)
"""

# from unitree_arm_interface import UnitreeArm
# arm = UnitreeArm()

# arm.kp = [10]*6
# arm.kd = [0.5]*6

# arm.q = [0,0,0,0,0,0]
# arm.qd = [0,0,0,0,0,0]
# arm.tau_f = [0,0,0,0,,0]
# arm.gripper_pos = 0 # 1 closed
# arm.gripper_vel = 0
# arm.sendRecv()
# time.sleep(0.002)
