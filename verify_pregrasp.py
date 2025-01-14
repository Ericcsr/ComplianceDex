import pybullet as pb
import numpy as np
import torch
import rigidBodySento as rb
from pybullet_robot.robots import LeapHand, AllegroHand
from pybullet_robot.controllers import OSImpedanceController
import time

object_dict = {
    "box": "assets/cube_visualization.urdf",
    "banana": "assets/banana/banana.urdf",
    "hammer": "assets/hammer/hammer.urdf",
    "lego": "assets/lego/lego.urdf",
    "mug": "assets/mug/mug.urdf",
    "mug2": "assets/mug2/mug2.urdf",
    "coffeebottle": "assets/coffeebottle/coffeebottle.urdf",
}

hand_dict = {"leap":LeapHand, "allegro":AllegroHand}

class HandValidator:
    def __init__(self, pb_client, hand_model, object_urdf, init_object_pose, init_robot_pos, uid, friction=2.0, visualize_tip=True, floor_offset=0.0):
        self._pb = pb_client
        self.robot = hand_dict[hand_model](self._pb,uid=uid)
        self.base_position = init_object_pose[0:3]
        self.base_orientation = init_object_pose[3:7]
        self.init_robot_pose = init_robot_pos
        self.oid = self._pb.loadURDF(object_urdf, basePosition=init_object_pose[:3])
        self.floor_offset = floor_offset
        self.floor_id = self._pb.loadURDF("assets/plane.urdf", basePosition=[0,0,0.0], baseOrientation=[0,0,0,1], useFixedBase=True)
        self._pb.changeDynamics(self.oid, -1, lateralFriction=friction)
        self._pb.changeDynamics(self.floor_id, -1, lateralFriction=2.0)
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
        self.robot.configure_default_pos([self.init_robot_pose[0], 
                                          self.init_robot_pose[1], 
                                          self.init_robot_pose[2]], [0, 0, 0, 1]) # -0.02

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

    def reset(self):
        self.robot.reset()
        self.floor_id = self._pb.loadURDF("assets/plane.urdf", 
                                          basePosition=[0,0,-0.025+self.floor_offset], 
                                          baseOrientation=[0,0,0,1], useFixedBase=True)
        self._pb.changeDynamics(self.floor_id, -1, lateralFriction=2.0)

    def execute_grasp(self, tip_pose, ref_pose, object_pose, kP, kD, max_steps=2000,visualize=True,pause=True):
        """
        tip_pose: [num_tips, 3] Should be in contact
        ref_pose: [num_tips, 3] Should be inside the object
        object_pose: [7]
        kP: [num_tips]
        kD: [num_tips]
        """
        self.set_object_pose(object_pose)
        self.setCompliance(kP, kD)
        self.robot.configure_default_pos([self.init_robot_pose[0], 
                                          self.init_robot_pose[1], 
                                          self.init_robot_pose[2]], [0, 0, 0, 1]) 
        pb.resetBasePositionAndOrientation(self.oid, self.base_position, self.base_orientation)
        #self.set_tip_pose(tip_pose)
        # Each fingertip reach pre-grasp pose
        if pause:
            input("Press Enter to continue...")
        for i in range(100):
            pb.stepSimulation()
            if visualize:
                time.sleep(0.001)
        for i in range(300):
            self.control_finger(tip_pose)
            if visualize:
                time.sleep(0.001)
        if pause:
            input("Press Enter to continue...")
        for i in range(max_steps):
            if i == 800:
                self._pb.setGravity(0.0, 0.0, -3.0)
                self._pb.removeBody(self.floor_id)
            self.control_finger(ref_pose)
            #print(self.robot.ee_pose())
            #self._pb.stepSimulation()
            if visualize:
                time.sleep(0.001)
        pos, orn = self._pb.getBasePositionAndOrientation(self.oid)
        self.reset()
        return pos, orn

if __name__ == "__main__":
    import time
    from argparse import ArgumentParser
    import json
    parser = ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--hand", type=str, default="allegro") # ["leap", "allegro"]
    parser.add_argument("--use_config", action="store_true", default=False)
    parser.add_argument("--wrist_x", type=float, default=0.0)
    parser.add_argument("--wrist_y", type=float, default=0.0)
    parser.add_argument("--wrist_z", type=float, default=0.0)
    parser.add_argument("--floor_offset", type=float, default=0.0)
    args = parser.parse_args()

    if args.use_config:
        config = json.load(open(f"assets/{args.exp_name}/config.json"))
        args.wrist_x = config["wrist_x"]
        args.wrist_y = config["wrist_y"]
        args.wrist_z = config["wrist_z"]
        args.floor_offset = config["floor_offset"]
    c = pb.connect(pb.GUI)
    pb.setTimeStep(0.001)
    pb.setGravity(0,0,-1.0)
    object_urdf = object_dict[args.exp_name]
    finger_pose = np.load(f"data/contact_{args.exp_name}.npy")
    target_pose = np.load(f"data/target_{args.exp_name}.npy")
    print(finger_pose.shape, target_pose.shape)
    #finger_pose -= np.array([args.wrist_x,args.wrist_y,args.wrist_z])
    #target_pose -= np.array([args.wrist_x,args.wrist_y,args.wrist_z])
    # center = target_pose.sum(axis=0) / 4
    # finger_pose = finger_pose - center
    kp = np.load(f"data/compliance_{args.exp_name}.npy").repeat(3).reshape(-1,3)
    kd = np.sqrt(kp) * 0.8
    validator = HandValidator(pb, args.hand, object_urdf, [0.0,0.0,args.floor_offset,0,0,0,1],[args.wrist_x, args.wrist_y, args.wrist_z], uid=c, floor_offset=args.floor_offset)
    print("Finger:",args.floor_offset)
    validator.execute_grasp(finger_pose,target_pose,[0,0,args.floor_offset,0,0,0,1],kp,kd) # Should set floor offset as positive
    # original floor offset: -0.025 + args.floor_offset
        