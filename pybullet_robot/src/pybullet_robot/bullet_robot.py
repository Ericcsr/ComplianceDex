import pybullet as pb
import numpy as np
import quaternion


class BulletRobot(object):
    def __init__(self, pb_client, description_path, basePosition=[0.0, 0.0, 0.04],
                 uid=None, config=None, realtime_sim=False, all_joint_names=None):
        """
        :param description_path: path to description file (urdf, .bullet, etc.)
        :param config: optional config file for specifying robot information 
        :param uid: optional server id of bullet 

        :type description_path: str
        :type config: dict
        :type uid: int
        """
        self._pb = pb_client
        if uid is None:
            uid = self._pb.connect(self._pb.GUI)
        #     uid = self._pb.connect(self._pb.SHARED_MEMORY)
        #     if uid < 0:

        self._uid = uid
        self._pb.resetSimulation(physicsClientId=self._uid)

        extension = description_path.split('.')[-1]
        if extension == "urdf":
            robot_id = self._pb.loadURDF(
                description_path, useFixedBase=True, basePosition=basePosition,
                physicsClientId=self._uid, flags=self._pb.URDF_MERGE_FIXED_LINKS)
        elif extension == 'sdf':
            robot_id = self._pb.loadSDF(
                description_path, useFixedBase=True, physicsClientId=self._uid)
        elif extension == 'bullet':
            robot_id = self._pb.loadBullet(
                description_path, useFixedBase=True, physicsClientId=self._uid)
        else:
            robot_id = self._pb.loadMJCF(
                description_path, useFixedBase=True, physicsClientId=self._uid)

        self._id = robot_id

        self._pb.setGravity(0.0, 0.0, 0.0, physicsClientId=self._uid)
        if realtime_sim:
            self._pb.setRealTimeSimulation(1, physicsClientId=self._uid)
            self._pb.setTimeStep(0.01, physicsClientId=self._uid)
        self._rt_sim = realtime_sim

        self._all_joints = np.array(
            range(self._pb.getNumJoints(self._id, physicsClientId=self._uid)))

        self._movable_joints = self.get_movable_joints()

        self._nu = len(self._movable_joints)
        self._nq = self._nu

        joint_information = self.get_joint_info()

        if all_joint_names is None:
            self._all_joint_names = [info['jointName'].decode(
                "utf-8") for info in joint_information]
        else:
            self._all_joint_names = all_joint_names

        self._all_joint_dict = dict(
            zip(self._all_joint_names, self._all_joints))

        if config is not None and config['ee_link_name'] is not None:
            self._ee_link_name = config['ee_link_name']
            self._ee_link_idx = config['ee_link_idx']
        else:
            self._ee_link_idx, self._ee_link_name = self._use_last_defined_link()

        self._joint_limits = self.get_joint_limits()

        self._ft_joints = [self._all_joints[-1]]

        # by default, set FT sensor at last fixed joint
        self.set_ft_sensor_at(self._ft_joints[0])

    def step_sim(self):
        self._pb.stepSimulation(self._uid)

    def step_if_not_rtsim(self):
        if not self._rt_sim:
            self.step_sim()

    def set_ft_sensor_at(self, joint_id, enable=True):
        if joint_id in self._ft_joints and not enable:
            self._ft_joints.remove(joint_id)
        elif joint_id not in self._ft_joints and enable:
            self._ft_joints.append(joint_id)
        print ("FT sensor at joint", joint_id)
        self._pb.enableJointForceTorqueSensor(self._id, joint_id, enable, self._uid)

    def __del__(self):
        self._pb.disconnect(self._uid)

    def _use_last_defined_link(self):
        joint_information = self._pb.getJointInfo(
            self._id, self._all_joints[-1], physicsClientId=self._uid)
        return joint_information[-1]+1, joint_information[-5]

    def state(self):
        """
        :return: Current robot state, as a dictionary, containing 
                joint positions, velocities, efforts, zero jacobian,
                joint space inertia tensor, end-effector position, 
                end-effector orientation, end-effector velocity (linear and angular),
                end-effector force, end-effector torque
        :rtype: dict: {'position': np.ndarray,
                       'velocity': np.ndarray,
                       'effort': np.ndarray,
                       'jacobian': np.ndarray,
                       'inertia': np.ndarray,
                       'ee_point': np.ndarray,
                       'ee_ori': np.ndarray,
                       'ee_vel': np.ndarray,
                       'ee_omg': np.ndarray,
                       'tip_state'['force']: np.ndarray,
                       'tip_state'['torque']: np.ndarray,
                       }
        """

        joint_angles = self.angles()
        joint_velocities = self.joint_velocities()
        joint_efforts = self.joint_efforts()

        state = {}
        state['position'] = joint_angles
        state['velocity'] = joint_velocities
        state['effort'] = joint_efforts
        state['jacobian'] = self.jacobian(None)
        state['inertia'] = self.inertia(None)

        state['ee_point'], state['ee_ori'] = self.ee_pose()

        state['ee_vel'], state['ee_omg'] = self.ee_velocity()

        tip_state = {}
        ft_joint_state = self._pb.getJointState(self._id, max(
            self._ft_joints), physicsClientId=self._uid)
        ft = np.asarray(ft_joint_state[2])

        tip_state['force'] = ft[:3]
        tip_state['torque'] = ft[3:]

        state['tip_state'] = tip_state

        return state

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

        linear_jacobian, angular_jacobian = self._pb.calculateJacobian(bodyUniqueId=self._id,
                                                                 linkIndex=self._ee_link_idx,
                                                                 localPosition=[
                                                                     0.0, 0.0, 0.0],
                                                                 objPositions=joint_angles.tolist(),
                                                                 objVelocities=np.zeros(
                                                                     self.n_joints()).tolist(),
                                                                 objAccelerations=np.zeros(self.n_joints()).tolist(), physicsClientId=self._uid)

        jacobian = np.vstack(
            [np.array(linear_jacobian), np.array(angular_jacobian)])

        return jacobian

    def ee_pose(self):
        """

        :return: end-effector pose of this robot in the format (position,orientation)
        .. note: orientation is a quaternion following Hamilton convention, i.e. (w, x, y, z)
        """
        return self.get_link_pose(link_id=self._ee_link_idx)

    def ee_velocity(self):
        """

        :return: end-effector velocity, which includes linear and angular velocities, i.e. (v,omega)
        :rtype: np.ndarray
        """

        return self.get_link_velocity(link_id=self._ee_link_idx)
    
    def get_link_state(self, link_idx, as_tuple=False):
        """
        returns orientation in bullet format quaternion [x,y,z,w]
        """

        link_state = self._pb.getLinkState(self._id, link_idx, computeLinkVelocity = 1, physicsClientId=self._uid)

        if not as_tuple:
            ee_pos = np.asarray(link_state[0])
            ee_ori = np.asarray(link_state[1])
            ee_vel = np.asarray(link_state[2])
            ee_omg = np.asarray(link_state[3])
        else:
            ee_pos = link_state[0]
            ee_ori = link_state[1]
            ee_vel = link_state[2]
            ee_omg = link_state[3]

        return ee_pos, ee_ori, ee_vel, ee_omg

    def get_ee_wrench(self, local=False, verbose=False):
        '''
        :param local: if True, computes reaction forces in local sensor frame, else in base frame of robot
        :type local: bool
        :return: End effector forces and torques. Returns [fx, fy, fz, tx, ty, tz]
        :rtype: np.ndarray
        '''

        _, _, jnt_reaction_force, _ = self.get_joint_state(self._ft_joints[-1])
        
        if not local:
            jnt_reaction_force = np.asarray(jnt_reaction_force)
            ee_pos, ee_ori = self.get_link_pose(self._ft_joints[-1])
            rot_mat = quaternion.as_rotation_matrix(ee_ori)
            f = np.dot(rot_mat,np.asarray([-jnt_reaction_force[0], -jnt_reaction_force[1], -jnt_reaction_force[2]]))
            t = np.dot(rot_mat,np.asarray([-jnt_reaction_force[0+3], -jnt_reaction_force[1+3], -jnt_reaction_force[2+3]]))
            jnt_reaction_force = np.append(f,t).flatten()

        return jnt_reaction_force

    def inertia(self, joint_angles=None):
        """

        :param joint_angles: optional parameter, if not None, then returned inertia is evaluated at given joint_angles.
            Otherwise, returned inertia tensor is evaluated at current joint angles.
        :return: Joint space inertia tensor
        """

        if joint_angles is None:
            joint_angles = self.angles()

        inertia_tensor = np.array(self._pb.calculateMassMatrix(
            self._id, joint_angles.tolist()))

        return inertia_tensor

    def inverse_kinematics(self, position, orientation=None):
        """

        :param position: target position
        :param orientation: target orientation in quaternion format (w, x, y , z)
        :return: joint positions that take the end effector to the desired target position and/or orientation, and success status (solution_found) of IK operation.
        """

        solution = None
        if orientation is None:

            solution = self._pb.calculateInverseKinematics(self._id,
                                                     self._ee_link_idx,
                                                     targetPosition=position, physicsClientId=self._uid)
        else:

            orientation = [orientation[1], orientation[2],
                           orientation[3], orientation[0]]
            solution = self._pb.calculateInverseKinematics(self._id,
                                                     self._ee_link_idx,
                                                     targetPosition=position,
                                                     targetOrientation=orientation, physicsClientId=self._uid)

        return np.array(solution), solution is None

    def q_mean(self):
        """
        :return: Mean joint positions.
        :rtype: [float] * self._nq
        """
        return self._joint_limits['mean']

    def joint_names(self):
        """
        :return: Joint names.
        :rtype: [str] * self._nq
        """
        return self._all_joint_names

    def joint_ids(self):
        """
        :return: Get joint ids (bullet id) for all robots.
        :rtype: [int] * self._nq
        """
        return self._all_joints

    def angles(self):
        """
        :return: Current joint positions.
        :rtype: [float] * self._nq
        """
        return self.get_joint_state()[0]

    def joint_velocities(self):
        """
        :return: Current velocities of all joints (movable, non-movable).
        :rtype: [float] * self._nq
        """
        return self.get_joint_state()[1]

    def joint_efforts(self):
        """
        :return: Current efforts of all joints (movable, non-movable).
        :rtype: [float] * self._nq
        """
        return self.get_joint_state()[3]

    def n_cmd(self):
        """
        :return: Number of motors (controls)
        :rtype: float
        """
        return self._nu

    def n_joints(self):
        """
        :return: Number of joints
        :rtype: float
        """
        return self._nq

    def set_joint_velocities(self, cmd, joints=None):
        """
        Set motor velocities. Use for velocity controlling.

        :param cmd: joint velocity values
        :type cmd: [float] * self._nu

        """
        if joints is None:
            joints = self._movable_joints

        self._pb.setJointMotorControlArray(
            self._id, joints, controlMode=self._pb.VELOCITY_CONTROL, targetVelocities=cmd, physicsClientId=self._uid)

    def set_joint_torques(self, cmd, joints=None):
        """
        Set motor torques. Use for velocity controlling.

        :param cmd: joint torque values
        :type cmd: [float] * self._nu

        """

        if joints is None:
            joints = self._movable_joints

        self._pb.setJointMotorControlArray(
            self._id, joints, controlMode=self._pb.TORQUE_CONTROL, forces=cmd, physicsClientId=self._uid)

    def set_joint_positions_delta(self, cmd, joints=None, forces=None):
        """
        Execute position command by specifying target joint velocities. (?!)

        :param cmd: joint velocity values
        :type cmd: [float] * self._nu

        """
        if joints is None:
            joints = self._movable_joints

        if forces is None:
            forces = np.ones(len(joints)) * 1.5

        self._pb.setJointMotorControlArray(self._id, joints,
                                     controlMode=self._pb.POSITION_CONTROL,
                                     targetVelocities=cmd,
                                     forces=forces, physicsClientId=self._uid)

    def set_joint_positions(self, cmd, joints=None, forces=None):
        """
        Set target joint positions. Use for position controlling.

        :param cmd: joint position values
        :type cmd: [float] * self._nu

        """
        vels = [0.0005 for n in range(len(cmd))]
        self._pb.setJointMotorControlArray(self._id, joints, controlMode=self._pb.POSITION_CONTROL,
                                     targetPositions=cmd, targetVelocities=vels, physicsClientId=self._uid)

    def get_joint_info(self):
        """
        :return: JointInfo() method return values from pybullet            
        :rtype: [dict] * self._nq
        """
        attribute_list = ['jointIndex', 'jointName', 'jointType',
                          'qIndex', 'uIndex', 'flags',
                          'jointDamping', 'jointFriction', 'jointLowerLimit',
                          'jointUpperLimit', 'jointMaxForce', 'jointMaxVelocity', 'linkName',
                          'jointAxis', 'parentFramePos', 'parentFrameOrn', 'parentIndex']

        joint_information = []
        for idx in self._all_joints:
            info = self._pb.getJointInfo(self._uid, idx, physicsClientId=self._uid)
            joint_information.append(dict(zip(attribute_list, info)))

        return joint_information

    def get_link_velocity(self, link_id, link_offset):
        q_vel = self.joint_velocities()
        J_lin, J_ang = self._pb.calculateJacobian(self._id, 
                                                  linkIndex=link_id, 
                                                  localPosition=link_offset,
                                                  objPositions=self.angles().tolist(),
                                                  objVelocities=self.joint_velocities().tolist(),
                                                  objAccelerations=[0.0] * self.n_joints())
        linvel = J_lin @ q_vel
        angvel = J_ang @ q_vel
        return linvel, angvel
    
    def get_link_pose(self, link_id=-3, link_offset=None):
        """
        :return: Pose of link (Cartesian positionof center of mass, 
                            Cartesian orientation of center of mass in quaternion [x,y,z,w]) 
        :rtype: [np.ndarray, np.quaternion]

        :param link_id: optional parameter to specify the link id. If not provided,
                        will return pose of end-effector
        :type link_id: int
        """
        if link_id == -3:
            self._ee_link_idx
        offset = [0, 0, 0] if link_offset is None else link_offset

        link_state = self._pb.getLinkState(
            self._id, link_id, physicsClientId=self._uid)
        ori = np.quaternion(link_state[1][3], link_state[1][0], link_state[1][1],
                            link_state[1][2])  # hamilton convention
        pos = np.asarray(link_state[0]) + quaternion.rotate_vectors(ori, offset)
        return pos, ori

    def get_joint_state(self, joint_id=None):
        """
        :return: joint positions, velocity, reaction forces, joint efforts as given from
                bullet physics
        :rtype: [np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        """
        if joint_id is None:
            joint_angles = []
            joint_velocities = []
            joint_reaction_forces = []
            joint_efforts = []

            for idx in self._movable_joints:
                joint_state = self._pb.getJointState(
                    self._id, idx, physicsClientId=self._uid)

                joint_angles.append(joint_state[0])

                joint_velocities.append(joint_state[1])

                joint_reaction_forces.append(joint_state[2])

                joint_efforts.append(joint_state[3])

            return np.array(joint_angles), np.array(joint_velocities), np.array(joint_reaction_forces), np.array(
                joint_efforts)

        else:
            jnt_state = self._pb.getJointState(
                self._id, joint_id, physicsClientId=self._uid)

            jnt_poss = jnt_state[0]

            jnt_vels = jnt_state[1]

            jnt_reaction_forces = jnt_state[2]

            jnt_applied_torques = jnt_state[3]

            return jnt_poss, jnt_vels, np.array(jnt_reaction_forces), jnt_applied_torques

    def set_joint_angles(self, joint_angles, joint_indices=None):
        """
        Set joint positions. Note: will hard reset the joints, no controllers used.

        :param cmd: joint position values
        :type cmd: [float] * self._nu

        """

        if joint_indices is None:
            joint_indices = self._movable_joints

        for i, jnt_idx in enumerate(joint_indices):
            self._pb.resetJointState(self._id, jnt_idx,
                               joint_angles[i], physicsClientId=self._uid)

    def get_movable_joints(self):
        """
        :return: Ids of all movable joints.
        :rtype: np.ndarray (shape: (self._nu,))

        """
        movable_joints = []
        for i in self._all_joints:
            joint_info = self._pb.getJointInfo(
                self._id, i, physicsClientId=self._uid)
            q_index = joint_info[3]
            if q_index > -1:
                movable_joints.append(i)

        return np.array(movable_joints)

    def get_all_joints(self):
        """
        :return: Ids of all joints.
        :rtype: np.ndarray (shape: (self._nq,))

        """
        return self._all_joints

    def get_joint_dict(self):
        return self._all_joint_dict

    def get_joint_limits(self):
        """
        :return: Joint limits, mean positions, range
        :rtype: dict {'lower': [float], 'upper': [float], 'mean': [float], 'range':[float]}
        """

        lower_lim = np.zeros(self.n_joints())

        upper_lim = np.zeros(self.n_joints())

        mean_ = np.zeros(self.n_joints())

        range_ = np.zeros(self.n_joints())

        for k, idx in enumerate(self._movable_joints):
            lower_lim[k] = self._pb.getJointInfo(
                self._id, idx, physicsClientId=self._uid)[8]

            upper_lim[k] = self._pb.getJointInfo(
                self._id, idx, physicsClientId=self._uid)[9]

            mean_[k] = 0.5 * (lower_lim[k] + upper_lim[k])

            range_[k] = (upper_lim[k] - lower_lim[k])

        return {'lower': lower_lim, 'upper': upper_lim, 'mean': mean_, 'range': range_}

    def get_joint_by_name(self, joint_name):
        """
        :return: Joint ID of given joint
        :rtype: int

        :param joint_name: name of joint
        :type joint_name: str
        """
        if joint_name in self._all_joint_dict:
            return self._all_joint_dict[joint_name]
        else:
            raise Exception("Joint name does not exist!")

    def configure_default_pos(self, pos, ori):
        self._default_pos = pos
        self._default_ori = ori
        self.set_default_pos_ori()

    def set_default_pos_ori(self):
        self.set_pos_ori(self._default_pos, self._default_ori)

    def set_pos_ori(self, pos, ori):
        """
        Set robot position (base of robot)

        :param pos: position of CoM of base
        :param ori: orientation of CoM of base (quaternion [x,y,z,w])

        """
        self._pb.resetBasePositionAndOrientation(
            self._id, pos, ori, physicsClientId=self._uid)

    def set_ctrl_mode(self, ctrl_type='pos'):
        """
        Use to disable the default position_control mode.

        :param ctrl_type: type of controller to use (give any string except 'pos' to enable
                            velocity and torque control)
        :type ctrl_type: str

        """

        angles = self.angles()

        for k, jnt_index in enumerate(self._movable_joints):
            self._pb.resetJointState(self._id, jnt_index,
                               angles[k], physicsClientId=self._uid)

        if ctrl_type == 'pos':

            self._pb.setJointMotorControlArray(self._id, self._movable_joints, self._pb.POSITION_CONTROL,
                                         targetPosition=angles, forces=[500]*len(self._movable_joints), physicsClientId=self._uid)

        else: # disable default velocity controller
            self._pb.setJointMotorControlArray(self._id, self._movable_joints, self._pb.VELOCITY_CONTROL,
                                         forces=[0.0]*len(self._movable_joints), physicsClientId=self._uid)

    def triangle_mesh(self):

        visual_shape_data = self._pb.getVisualShapeData(
            self._id, physicsClientId=self._uid)

        triangle_mesh = [None]*len(visual_shape_data)

        for i, data in enumerate(visual_shape_data):
            link_index = data[1]
            geometry_type = data[2]
            dimensions = data[3]
            mesh_asset_file = data[4]
            local_pos = data[5]
            local_ori = data[6]
            colour = data[7]

            link_state = self._pb.getLinkState(
                self._id, link_index, physicsClientId=self._uid)
            pos = link_state[0]
            ori = link_state[1]
            loc_inertial_pos = link_state[2]
            loc_inertial_ori = link_state[3]

            inv_loc_inertial_pos, inv_loc_inertial_ori = self._pb.invertTransform(loc_inertial_pos,
                                                                            loc_inertial_ori, physicsClientId=self._uid)
            tp, to = self._pb.multiplyTransforms(inv_loc_inertial_pos,
                                           inv_loc_inertial_ori,
                                           pos, ori, physicsClientId=self._uid)

            global_pos, global_ori = self._pb.multiplyTransforms(
                tp, to, local_pos, local_ori, physicsClientId=self._uid)

            triangle_mesh[i] = (np.asarray(global_pos),)

        return triangle_mesh

    @staticmethod
    def add_to_models_path(path):
        """
        Add the specified directory (absolute) to Pybullet's searchpath for easily adding models from the path.

        :param path: the absolute path to the directory
        :type path: str
        """
        import os
        if os.path.isdir(path):
            pb.setAdditionalSearchPath(path)
            print ("Info: Added {} to Pybullet path.".format(path))
        else:
            print ("Error adding to Pybullet path! {} not a directory.".format(path))
