import rospy
import numpy as np
from argparse import ArgumentParser
import json
from allegro_hand_kdl.srv import PoseGoalRequest, PoseGoal
from allegro_hand_kdl.srv import GainRequestRequest, GainRequest


def map_to_palm(world_poses):
    pass
    # Should return pose w.r.t. palm frame.



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
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

    finger_pose = np.load(f"data/contact_{args.exp_name}.npy")
    target_pose = np.load(f"data/target_{args.exp_name}.npy")
    finger_pose = map_to_palm(finger_pose)
    target_pose = map_to_palm(target_pose)

    client = rospy.ServiceProxy("desired_cartesian_pose", PoseGoal)
    client.wait_for_service(10)

    input("Press to send request")
    req = PoseGoalRequest()
    req.pose = [0.08, 0.08, 0.0625,0.08, 0.0, 0.0625,0.08, -0.08, 0.0625,0.08, 0.0571, -0.03]
    res = client(req)
    print(res.success)

    input("Press to send request")
    req = PoseGoalRequest()
    req.pose = [0.06, 0.08, 0.1,0.06, 0.0, 0.1,0.06, -0.08, 0.1,0.08, 0.0571, -0.0]
    res = client(req)
    print(res.success)
