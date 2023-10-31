import pybullet as pb
import numpy as np
import torch
import rigidBodySento as rb
from pybullet_robot.robots import LeapHand
from pybullet_robot.controllers import OSImpedanceController
import time

class TipValidator:
    def __init__(self, pb_client, object_urdf, init_object_pose, num_fingers=4,friction=0.5):
        self._pb = pb_client
        self.oid = pb.loadURDF(object_urdf, basePosition=init_object_pose[0:3], baseOrientation=init_object_pose[3:7])
        self.floor_id = pb.loadURDF("assets/plane.urdf", basePosition=[0,0,-0.1], baseOrientation=[0,0,0,1], useFixedBase=True)
        self._pb.changeDynamics(self.oid, -1, lateralFriction=friction)
        self.num_fingers = num_fingers
        self.f_ids = []
        for i in range(self.num_fingers):
            f_id = pb.loadURDF("assets/tip.urdf", basePosition=[0,0,0], baseOrientation=[0,0,0,1], useFixedBase=True)
            self.f_ids.append(f_id)
            # Change tip to position control
            self._pb.setJointMotorControlArray(f_id, range(3),pb.VELOCITY_CONTROL, 
                                               targetVelocities=[0.0, 0.0, 0.0], forces=[0.0, 0.0, 0.0])
            self._pb.changeDynamics(f_id, 2, lateralFriction=friction)

    def set_tip_pose(self, tip_pose):
        """
        tip_pose: [num_tips, 3]
        """
        assert tip_pose.shape[0] == self.num_fingers
        for i in range(self.num_fingers):
            for j in range(3):
                self._pb.resetJointState(self.f_ids[i], j, targetValue=tip_pose[i,j])

    def set_object_pose(self, object_pose):
        self._pb.resetBasePositionAndOrientation(self.oid, object_pose[0:3], object_pose[3:7])

    def setCompliance(self, kP, kD):
        """
        kP: [num_tips]
        kD: [num_tips]
        """
        self.kP = kP
        self.kD = kD

    def get_tip_pose(self):
        tip_pose = np.zeros([self.num_fingers, 3])
        tip_vel = np.zeros([self.num_fingers, 3])
        for i in range(self.num_fingers):
            for j in range(3):
                state = self._pb.getJointState(self.f_ids[i], j)
                tip_pose[i,j] = state[0]
                tip_vel[i,j] = state[1]
        return tip_pose, tip_vel

    def control_finger(self, ref_pose):
        """
        ref_pose: [num_tips, 3]
        """
        tip_pose, tip_vel = self.get_tip_pose()
        for i in range(self.num_fingers):
            self._pb.setJointMotorControlArray(self.f_ids[i], range(3), 
                                               pb.TORQUE_CONTROL, 
                                               forces = self.kP[i] * (ref_pose[i] - tip_pose[i]) - self.kD[i] * tip_vel[i])
            
    def execute_grasp(self, tip_pose, ref_pose, object_pose, kP, kD, max_steps=2000):
        """
        tip_pose: [num_tips, 3] Should be in contact
        ref_pose: [num_tips, 3] Should be inside the object
        object_pose: [7]
        kP: [num_tips]
        kD: [num_tips]
        """
        self.set_tip_pose(tip_pose)
        self.set_object_pose(object_pose)
        self.setCompliance(kP, kD)
        self._pb.setGravity(0,0,-1.0)
        input("Press Enter to continue...")
        for i in range(max_steps):
            if i == 400:
                self._pb.setGravity(0.0, 0.0, -3.0)
                self._pb.removeBody(self.floor_id)
            self.control_finger(ref_pose)
            self._pb.stepSimulation()
            time.sleep(0.001)
        return self._pb.getBasePositionAndOrientation(self.oid)
    

class LeapHandValidator:
    def __init__(self, pb_client, object_urdf, init_object_pose, uid, friction=0.5, visualize_tip=True):
        self._pb = pb_client
        self.robot = LeapHand(self._pb,uid=uid)
        self.oid = self._pb.loadURDF(object_urdf, basePosition=init_object_pose[0:3], baseOrientation=init_object_pose[3:7])
        self.floor_id = self._pb.loadURDF("assets/plane.urdf", basePosition=[0,0,-0.0325], baseOrientation=[0,0,0,1], useFixedBase=True)
        self._pb.changeDynamics(self.oid, -1, lateralFriction=friction)
        # create visualization tools
        self.visualize_tip = visualize_tip
        if visualize_tip:
            self.tips = []
            color_code = [[1,0,0,0.7],
                        [0,1,0,0.7],
                        [0,0,1,0.7],
                        [1,1,1,0.7]]
            for i in range(4):
                tip = rb.create_primitive_shape(self._pb, 0, pb.GEOM_SPHERE, [0.015], color=color_code[i], collidable=False)
                self.tips.append(tip)
        self.robot.set_tip_friction(friction)
        self.controller = OSImpedanceController(self.robot)

    def set_tip_pose(self, tip_pose):
        joint_pose, flag = self.robot.inverse_kinematics(tip_pose)
        if flag is not None:
            self.controller.update_goal(tip_pose)
        else:
            print("Failed to move the joint, IK not feasible")

    def set_object_pose(self, object_pose):
        self._pb.resetBasePositionAndOrientation(self.oid, object_pose[0:3], object_pose[3:7])

    def setCompliance(self, kP, kD):
        """
        kP: [num_tips]
        kD: [num_tips]
        """
        self.kP = kP
        self.kD = kD
        self.controller.update_compliance(self.kP.flatten(), self.kD.flatten())

    def control_finger(self, ref_pose):
        self.controller.update_goal(ref_pose.flatten(),stepPhysics=True)
        if self.visualize_tip:
            for i in range(4):
                self._pb.resetBasePositionAndOrientation(self.tips[i], ref_pose[i], [0,0,0,1])

    def execute_grasp(self, tip_pose, ref_pose, object_pose, kP, kD, max_steps=10000):
        """
        tip_pose: [num_tips, 3] Should be in contact
        ref_pose: [num_tips, 3] Should be inside the object
        object_pose: [7]
        kP: [num_tips]
        kD: [num_tips]
        """
        self.set_object_pose(object_pose)
        self.setCompliance(kP, kD)
        self.robot.configure_default_pos([-0.02,0.035, 0.09], [0, 0, 0, 1])
        #self.set_tip_pose(tip_pose)
        # Each fingertip reach pre-grasp pose
        for i in range(200):
            self.control_finger(tip_pose)
            time.sleep(0.001)
        input("Press Enter to continue...")
        for i in range(max_steps):
            if i == 800:
                self._pb.setGravity(0.0, 0.0, -3.0)
                self._pb.removeBody(self.floor_id)
            self.control_finger(ref_pose)
            #print(self.robot.ee_pose())
            #self._pb.stepSimulation()
            time.sleep(0.001)
        return self._pb.getBasePositionAndOrientation(self.oid)

if __name__ == "__main__":
    import time
    c = pb.connect(pb.GUI)
    pb.setTimeStep(0.001)
    pb.setGravity(0,0,-1.0)
    object_urdf = "assets/cube_visualization.urdf"
    validator = LeapHandValidator(pb, object_urdf, [0,0,0,0,0,0,1], uid=c)
    finger_pose = np.array([[0.02, 0.04, 0.0], 
                            [0.04, 0.0, 0.0],
                            [0.02, -0.04, 0.0],
                            [-0.04,0.0, 0.0]])
    
    target_pose = np.array([[0.02, 0.02, 0.0],
                            [0.02, 0.0, 0.0], 
                            [0.02, -0.02, 0.0],
                            [-0.02, 0.0, 0.0]])
    kp = np.array([[50.0,50.0,50.0],
                   [50.0,50.0,50.0],
                   [50.0,50.0,50.0],
                   [50.0,50.0,50.0]]) * 1.5
    kd = np.sqrt(kp) * 0.6
    validator.execute_grasp(finger_pose,target_pose,[0,0,0,0,0,0,1],kp,kd)





    
        