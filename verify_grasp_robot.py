import rospy
import numpy as np
import tf
from scipy.spatial.transform import Rotation as R
from argparse import ArgumentParser
import json
from allegro_hand_kdl.srv import PoseGoalRequest, PoseGoal
from allegro_hand_kdl.srv import GainParamRequest, GainParam
import pybullet as pb


#RIGHT_HAND_ORDER = [0,1,2,3]
#LEFT_HAND_ORDER = [2,1,0,3]
HAND = "left"

IDLE_WRIST_POSE = [0.0, 0.0, 0.5, 0.0,0.0, 0.0]
IDLE_WRIST_JOINTS = [-np.pi/6, np.pi/6, 0.0, -np.pi/2,0,np.pi/3,-np.pi/2-np.pi/6]
# IDLE_WRIST_JOINTS = [0.0] * 7

def map_to_palm(world_poses, palm_pose):
    """
    world_poses: (N, 3) array of poses in world frame.
    palm_pose: (6) position of pal in world frame and euler angles of palm frame.
    """
    palm_pos = palm_pose[:3]
    palm_rot = palm_pose[3:]
    palm_rot_mat = R.from_euler("xyz", palm_rot).as_matrix().T
    palm_poses = (palm_rot_mat @ (world_poses - palm_pos).T).T
    palm_poses = palm_poses[:,[2,1,0]] * np.array([-1,1,1])
    return palm_poses

def set_joint_angles(robot_id, joint_angles):
    """
    Set the joint angles of the robot.
    """
    for i, angle in enumerate(joint_angles):
        pb.resetJointState(robot_id, i, angle)

def get_joint_angles(robot_id):
    """
    Get the joint angles of the robot.
    """
    joint_angles = []
    for i in range(7):
        joint_angles.append(pb.getJointState(robot_id, i)[0])
    return joint_angles

def solve_joint_angle(robot_id, ee_pose):
    """
    Solve for the joint angles of the robot given the end-effector pose.
    """
    old_joint_pose = get_joint_angles(robot_id)
    set_joint_angles(robot_id, IDLE_WRIST_JOINTS)
    joint_pose = pb.calculateInverseKinematics(robot_id,
                                              7,
                                              ee_pose[:3],
                                              R.from_euler("XYZ",ee_pose[3:]).as_quat().tolist(),
                                              jointDamping=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                              solver=0,
                                              maxNumIterations=100,
                                              residualThreshold=.01)
    set_joint_angles(robot_id, old_joint_pose)
    return joint_pose

def get_ee_pose(robot_id, joints):
    """
    Get the end-effector pose of the robot.
    """
    old_joint_pose = get_joint_angles(robot_id)
    set_joint_angles(robot_id, joints)
    state = pb.getLinkState(robot_id, 8)
    set_joint_angles(robot_id, old_joint_pose)
    pos, ori =state[0], state[1]
    return pos, ori

def control_arm(r, pose):
    joints = solve_joint_angle(r, pose)
    set_joint_angles(r, joints)
    input("Execute?")
    wrist_req = PoseGoalRequest()
    wrist_req.pose = joints
    result = arm_client(wrist_req)
    if not result.success:
        print("Kuka cannot reach desired arm position!")
        exit(1)
    hand_pos, hand_ori = get_ee_pose(r, joints)
    br.sendTransform(hand_pos, hand_ori, rospy.Time.now(), "hand_root", "world")
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--use_config", action="store_true", default=False)
    parser.add_argument("--wrist_x", type=float, default=0.0)
    parser.add_argument("--wrist_y", type=float, default=0.0)
    parser.add_argument("--wrist_z", type=float, default=0.0)
    parser.add_argument("--floor_offset", type=float, default=0.0)
    parser.add_argument("--no_robot", action="store_true", default=False)

    args = parser.parse_args()

    c = pb.connect(pb.GUI)
    r = pb.loadURDF("assets/kuka_allegro/model.urdf", 
                    basePosition = [-0.14134081,  0.50142033, -0.15],
                    baseOrientation = [0, 0, -0.3826834, 0.9238795],
                    useFixedBase=True)


    if args.use_config:
        config = json.load(open(f"assets/{args.exp_name}/config.json"))
        args.wrist_x = config["wrist_x"]
        args.wrist_y = config["wrist_y"]
        args.wrist_z = config["wrist_z"]
        args.floor_offset = config["floor_offset"]
        wrist_pose = [args.wrist_x, args.wrist_y, args.wrist_z, 0.0, 0.0, 0.0]
    else:
        wrist_pose = np.load(f"data/wrist_{args.exp_name}.npy")

    finger_pose = np.load(f"data/contact_{args.exp_name}.npy")
    target_pose = np.load(f"data/target_{args.exp_name}.npy")
    compliance = np.load(f"data/compliance_{args.exp_name}.npy").flatten()
    # if HAND == "left":
    #     finger_pose[0,1] *= -1
    #     finger_pose[2,1] *= -1
    #     finger_pose[3,1] *= -1

    finger_pose = map_to_palm(finger_pose, np.array(wrist_pose)).flatten()
    print(finger_pose)
    target_pose = map_to_palm(target_pose, np.array(wrist_pose)).flatten()

    if args.no_robot:
        exit(0)
    rospy.init_node("verify_grasp")
    arm_client = rospy.ServiceProxy("kuka_joint_service", PoseGoal)
    hand_client = rospy.ServiceProxy("desired_cartesian_pose", PoseGoal)
    hand_gain_client = rospy.ServiceProxy("desired_pd_gain", GainParam)
    br = tf.TransformBroadcaster()
    #hand_joint_client = rospy.Service("desired_joint_pose", PoseGoal)
    arm_client.wait_for_service()
    hand_client.wait_for_service()
    hand_gain_client.wait_for_service()

    req = GainParamRequest()
    req.kp = np.array([200,200,200,200.0]).tolist()
    req.kd = (0.8 * np.sqrt(compliance)).tolist()
    res = hand_gain_client(req)
    print(res.success)

    # move the hand to idle pose in joint space.
    idle_pose_req = PoseGoalRequest()
    idle_pose_req.pose = [0.06, -0.06, 0.0825, 0.06, 0.0, 0.0825, 0.06, 0.06, 0.0825, 0.08, -0.0071, -0.06]
    result = hand_client(idle_pose_req)
    if not result.success:
        print("allegro hand fail to initialize")
        exit(1)
    print("Compliance:", compliance)
    # Should know wrist position and orientation

    control_arm(r,IDLE_WRIST_POSE)
    

    control_arm(r, wrist_pose)

    input("Press to send request")
    req = PoseGoalRequest()
    # may be some sign need to invert...
    req.pose = finger_pose.tolist()
    res = hand_client(req)
    print(res.success)

    req = GainParamRequest()
    compliance = compliance * 1.5
    req.kp = compliance.tolist()
    req.kd = (0.8 * np.sqrt(compliance)).tolist()
    res = hand_gain_client(req)
    print(res.success)

    input("Press to send request")
    print("Target pose:", target_pose)
    req = PoseGoalRequest()
    req.pose = target_pose.tolist()
    res = hand_client(req)
    print(res.success)

    # Lift the hand up to verify grasp
    control_arm(r, IDLE_WRIST_POSE)