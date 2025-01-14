import os
import time

import numpy as np
# import pybullet as pb
from pybullet_robot.bullet_robot import BulletRobot

import logging
from ..allegro_hand.allegro_hand_config import ROBOT_CONFIG

description_path = os.path.dirname(
    os.path.abspath(__file__)) + "/models/allegro_hand_description_left.urdf"
# print description_path

class AllegroHand(BulletRobot):

    """
    Bullet simulation interface for the Allegro Hand robot

    Available methods (for usage, see documentation at function definition):
        - exec_position_cmd
        - exec_position_cmd_delta
        - move_to_joint_position
        - move_to_joint_pos_delta
        - exec_velocity_cmd
        - exec_torque_cmd
        - inverse_kinematics
        - untuck
        - tuck
        - q_mean
        - state
        - angles
        - n_joints
        - joint_limits
        - joint_names

        - jacobian*
        - joint_velocities*
        - joint_efforts*
        - ee_pose*
        - ee_velocity*
        - inertia*
        - inverse_kinematics*
        - joint_ids*
        - get_link_pose*
        - get_link_velocity*
        - get_joint_state*
        - set_joint_angles*
        - get_movable_joints*
        - get_all_joints*
        - get_joint_by_name*
        - set_default_pos_ori*
        - set_pos_ori*
        - set_ctrl_mode*

        *These methods can be accessed using the self._bullet_robot object from this class.
         Documentation for these methods in BulletRobot class. Refer bullet_robot.py       


    """

    def __init__(self, pb_client, tip_pose=None, robot_description=description_path, config=ROBOT_CONFIG, uid=None, *args, **kwargs):
        """
        :param robot_description: path to description file (urdf, .bullet, etc.)
        :param config: optional config file for specifying robot information 
        :param uid: optional server id of bullet 

        :type robot_description: str
        :type config: dict
        :type uid: int
        """
        self._pb = pb_client
        self._ready = False

        self._joint_names = ['joint_%s.0' % (s,) for s in range(0, 16)]

        BulletRobot.__init__(self, self._pb, robot_description, config=config, uid=uid, **kwargs)
        self._ee_link_offset = config["ee_link_offset"]
        all_joint_dict = self.get_joint_dict()

        self.ref_q = config["ref_q"]
        self._joint_ids = [all_joint_dict[joint_name]
                           for joint_name in self._joint_names]

        if tip_pose is None:
            self._tuck = self.ref_q.tolist()
        else:
            self._tuck = self.inverse_kinematics(tip_pose)[0]
        self._untuck = self._tuck

        lower_limits = self.get_joint_limits()['lower'][self._joint_ids]
        upper_limits = self.get_joint_limits()['upper'][self._joint_ids]

        self._jnt_limits = [{'lower': x[0], 'upper': x[1]}
                            for x in zip(lower_limits, upper_limits)]

        self.set_joint_angles(self._tuck)
        self._ready = True

    def reset(self):
        self.set_joint_angles(self._tuck)

    def exec_position_cmd(self, cmd):
        """
        Execute position command. Use for position controlling.

        :param cmd: joint position values
        :type cmd: [float] len: self._nu

        """
        self.set_joint_positions(cmd, self._joint_ids)

    def exec_position_cmd_delta(self, cmd):
        """
        Execute position command by specifying difference from current positions. Use for position controlling.

        :param cmd: joint position delta values
        :type cmd: [float] len: self._nu

        """
        self.set_joint_positions(self.angles() + cmd, self._joint_ids)

    def move_to_joint_position(self, cmd):
        """
        Same as exec_position_cmd. (Left here for maintaining structure of PandaArm class from panda_robot package)

        :param cmd: joint position values
        :type cmd: [float] len: self._nu

        """
        self.exec_position_cmd(cmd)

    def move_to_joint_pos_delta(self, cmd):
        """
        Same as exec_position_cmd_delta. (Left here for maintaining structure of PandaArm class from panda_robot package)

        :param cmd: joint position delta values
        :type cmd: [float] len: self._nu

        """
        self.exec_position_cmd_delta(cmd)

    def exec_velocity_cmd(self, cmd):
        """
        Execute velocity command. Use for velocity controlling.

        :param cmd: joint velocity values
        :type cmd: [float] len: self._nu

        """
        self.set_joint_velocities(cmd, self._joint_ids)

    def exec_torque_cmd(self, cmd):
        """
        Execute torque command. Use for torque controlling.

        :param cmd: joint torque values
        :type cmd: [float] len: self._nu

        """
        self.set_joint_torques(cmd, self._joint_ids)

    def position_ik(self, position, orientation=None):
        """
        :return: Joint positions for given end-effector pose obtained using bullet IK.
        :rtype: np.ndarray

        :param position: target end-effector position (X,Y,Z) in world frame
        :param orientation: target end-effector orientation in quaternion format (w, x, y , z) in world frame

        :type position: [float] * 3
        :type orientation: [float] * 4

        """
        return self.inverse_kinematics(position, orientation)[0]

    def set_sampling_rate(self, sampling_rate=100):
        """
        (Does Nothing. Left here for maintaining structure of PandaArm class from panda_robot package)
        """
        pass

    def untuck(self):
        """
        Send robot to tuck position.
        """
        self.exec_position_cmd(self._untuck)

    def tuck(self):
        """
        Send robot to tuck position.
        """
        self.exec_position_cmd(self._tuck)

    def joint_limits(self):
        """
        :return: Joint limits
        :rtype: dict {'lower': ndarray, 'upper': ndarray}
        """
        return self._jnt_limits

    def joint_names(self):
        """
        :return: Name of all joints
        :rtype: [str] * self._nq
        """
        return self._joint_names
    
    def jacobian(self, joint_angles=None):
        """
        :return: Jacobian matrix for provided joint configuration
        :rtype: ndarray (shape: 6x7)

        :param joint_angles: Optional parameter. If different than None, 
                             then returned jacobian will be evaluated at    
                             given joint_angles. Otherwise the returned 
                             jacobian will be evaluated at current robot 
                             joint angles.

        :type joint_angles: [float] * len(self.get_movable_joints)    
        """

        if joint_angles is None:
            joint_angles = self.angles()

        jacobians = []
        for i in range(len(self._ee_link_idx)):
            linear_jacobian, _ = self._pb.calculateJacobian(bodyUniqueId=self._id,
                                                                 linkIndex=self._ee_link_idx[i],
                                                                 localPosition=self._ee_link_offset[i],
                                                                 objPositions=joint_angles.tolist(),
                                                                 objVelocities=np.zeros(
                                                                     self.n_joints()).tolist(),
                                                                 objAccelerations=np.zeros(self.n_joints()).tolist(), physicsClientId=self._uid)
            jacobians.append(linear_jacobian)

        jacobian = np.vstack(jacobians)

        return jacobian
    
    def ee_pose(self):
        """

        :return: end-effector pose of this robot in the format (position,orientation)
        .. note: orientation is a quaternion following Hamilton convention, i.e. (w, x, y, z)
        """
        ee_poses = []
        ee_orns = []
        for i in range(len(self._ee_link_idx)):
            raw_pose = self.get_link_pose(link_id=self._ee_link_idx[i], link_offset=self._ee_link_offset[i])
            ee_poses.append(raw_pose[0])
            ee_orns.append(raw_pose[1])
        return np.hstack(ee_poses), np.hstack(ee_orns)

    def ee_velocity(self):
        ee_linvel = []
        ee_angvel = []
        for i in range(len(self._ee_link_idx)):
            raw_vel = self.get_link_velocity(link_id = self._ee_link_idx[i],
                                             link_offset=self._ee_link_offset[i])
            ee_linvel.append(raw_vel[0])
            ee_angvel.append(raw_vel[1])
        return np.hstack(ee_linvel), np.hstack(ee_angvel)

    def n_ee(self):
        return len(self._ee_link_idx)

    def inverse_kinematics(self,positions):
        solution = None
        solution = self._pb.calculateInverseDynamics2(self._id,
                                                      self._ee_link_idx,
                                                      positions,
                                                      physicsClientId=self._uid)
        return np.array(solution), solution is None
    
    def set_tip_friction(self, friction):
        for eid in self._ee_link_idx:
            self._pb.changeDynamics(self._id, eid, lateralFriction=friction)

    def gravity_compensation(self):
        """
        :return: Joint torques for gravity compensation
        :rtype: np.ndarray
        """
        return self._pb.calculateInverseDynamics(self._id, self.angles().tolist(), np.zeros(self.n_joints()).tolist(),
                                                 np.zeros(self.n_joints()).tolist())

    @staticmethod
    def load_robot_models():
        """
        Add the robot's URDF models to discoverable path for robot.
        """
        import os
        BulletRobot.add_to_models_path(os.path.dirname(
            os.path.abspath(__file__)) + "/models")


if __name__ == '__main__':

    p = AllegroHand(realtime_sim=True)
    # pass
