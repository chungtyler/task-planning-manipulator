import pinocchio as pin
import numpy as np

class Manipulator:
    def __init__(self, PATH_TO_ROBOT):
        self.model = pin.buildModelFromUrdf(PATH_TO_ROBOT)
        self.data = self.model.createData()
    
    # Get the pose of the frame in w.r.t world coordinates
    def get_frame_pose(self, frame_name):
        frame_id = self.model.getFrameId(frame_name)
        frame_pose = self.data.oMf[frame_id]
        return frame_pose

    # Get the twist of the frame w.r.t world coordinates
    def get_frame_twist(self, frame_name):
        frame_id = self.model.getFrameId(frame_name)
        frame_twist = pin.getFrameVelocity(self.model, self.data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return frame_twist
    
    # Get the joint jacobian
    def get_joint_jacobian(self, joint_name):
        joint_id = self.model.getJointId(joint_name)
        jacobian = pin.getJointJacobian(self.model, self.data, joint_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return jacobian
    
    # Get all the joint jacobians and stack
    def get_joint_jacobians(self):
        jacobians = []
        for frame_id in range(self.model.nframes):
            jacobian = pin.getJointJacobian(self.model, self.data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            jacobians.append(jacobian)

        jacobians = np.vstack(jacobians)
        return jacobians
    
    # Get non linear effects (corilos, centrifugal, gravity) for the joint torques
    def get_non_linear_effects(self):
        return self.data.nle
    
    def get_gravity_effects(self):
        return self.data.g
    
    # Get mass matrix
    def get_mass_matrix(self):
        return self.data.M
    
    # Compute the pose error using the homogenous transformation matrices
    def compute_pose_error(self, pose_d, frame_name):
        pose = self.get_frame_pose(frame_name)

        # Translation error
        e_translation = pose_d.translation - pose.translation

        # Orientation error
        R_relative = pose.rotation.T @ pose_d.rotation
        e_rotation = pose.rotation @ pin.log3(R_relative)

        # 6D Error vector
        e = np.concatenate([e_translation, e_rotation])
        return e
    
    # Get error between target and states
    def get_frame_error(self, pose_d, twist_d, frame_name):
        twist = self.get_frame_twist(frame_name)

        # Compute pose error and twist error
        e = self.compute_pose_error(pose_d, frame_name)
        e_dot = twist_d - twist.vector

        # Return as 6 DOF error vectors
        return e, e_dot
    
    # Compute the mass matrix, non-linear effects (coriolis, centrifugal, gravity), and jacobians
    def update(self, q, q_dot):
        pin.computeAllTerms(self.model, self.data, q, q_dot)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
