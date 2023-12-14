import rospy
import numpy as np
from scipy.spatial.transform import Rotation as R
from argparse import ArgumentParser
import json
from allegro_hand_kdl.srv import PoseGoalRequest, PoseGoal
from allegro_hand_kdl.srv import GainParamRequest, GainParam

#RIGHT_HAND_ORDER = [0,1,2,3]
#LEFT_HAND_ORDER = [2,1,0,3]
HAND = "left"
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

    if args.use_config:
        config = json.load(open(f"assets/{args.exp_name}/config.json"))
        args.wrist_x = config["wrist_x"]
        args.wrist_y = config["wrist_y"]
        args.wrist_z = config["wrist_z"]
        args.floor_offset = config["floor_offset"]
        wrist_pose = [args.wrist_x, args.wrist_y, args.wrist_z, 0.0, 0.0, 0.0]

    finger_pose = np.load(f"data/contact_{args.exp_name}.npy")
    target_pose = np.load(f"data/target_{args.exp_name}.npy")
    compliance = np.load(f"data/compliance_{args.exp_name}.npy").flatten()
    if HAND == "left":
        finger_pose[0,1] *= -1
        finger_pose[2,1] *= -1
        finger_pose[3,1] *= -1

    finger_pose = map_to_palm(finger_pose, np.array(wrist_pose)).flatten()
    print(finger_pose)
    target_pose = map_to_palm(target_pose, np.array(wrist_pose)).flatten()

    if args.no_robot:
        exit(0)
    arm_client = rospy.ServiceProxy("kuka_pose_service", PoseGoal)
    hand_client = rospy.ServiceProxy("desired_cartesian_pose", PoseGoal)
    hand_gain_client = rospy.ServiceProxy("desired_pd_gain", GainParam)
    #hand_joint_client = rospy.Service("desired_joint_pose", PoseGoal)
    arm_client.wait_for_service()
    hand_client.wait_for_service()
    hand_gain_client.wait_for_service()

    req = GainParamRequest()
    req.kp = np.array([100,100,100,100.0]).tolist()
    req.kd = (0.8 * np.sqrt(compliance)).tolist()
    res = hand_gain_client(req)
    print(res.success)

    # move the hand to idle pose in joint space.
    idle_pose_req = PoseGoalRequest()
    idle_pose_req.pose = [0.06, -0.06, 0.0825, 0.06, 0.0, 0.0825, 0.06, 0.06, 0.0825, 0.08, -0.0071, -0.05]
    result = hand_client(idle_pose_req)
    if not result.success:
        print("allegro hand fail to initialize")
        exit(1)
    print("Compliance:", compliance)
    # Should know wrist position and orientation
    wrist_req = PoseGoalRequest()
    wrist_req.pose = wrist_pose
    result = arm_client(wrist_req)
    if not result.success:
        print("Kuka cannot reach desired arm position!")
        exit(1)

    input("Press to send request")
    req = PoseGoalRequest()
    # may be some sign need to invert...
    req.pose = finger_pose.tolist()
    res = hand_client(req)
    print(res.success)

    req = GainParamRequest()
    compliance = compliance * 7
    req.kp = compliance.tolist()
    req.kd = (0.8 * np.sqrt(compliance)).tolist()
    res = hand_gain_client(req)
    print(res.success)

    input("Press to send request")
    req = PoseGoalRequest()
    req.pose = target_pose.tolist()
    res = hand_client(req)
    print(res.success)

    # Lift the hand up to verify grasp
    wrist_req = PoseGoalRequest()
    wrist_req.pose = [args.wrist_x, args.wrist_y, args.wrist_z+0.15, 0.0, 0.0, 0.0]
    result = arm_client(wrist_req)
    if not result.success:
        print("Kuka cannot reach desired arm position!")
        exit(1)