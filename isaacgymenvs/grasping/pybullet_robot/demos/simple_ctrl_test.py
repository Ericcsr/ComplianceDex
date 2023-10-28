import numpy as np
import pybullet as pb
from pybullet_robot.worlds import SimpleWorld, add_PyB_models_to_path
from pybullet_robot.robots import LeapHand
from pybullet_robot.controllers import OSImpedanceController
import time

def create_primitive_shape(pb, mass, shape, dim, color=(0.6, 0, 0, 1), 
                           collidable=True, init_xyz=(0, 0, 0),
                           init_quat=(0, 0, 0, 1)):
    # shape: p.GEOM_SPHERE or p.GEOM_BOX or p.GEOM_CYLINDER
    # dim: halfExtents (vec3) for box, (radius, length)vec2 for cylinder, (radius) for sphere
    # init_xyz vec3 being initial obj location, init_quat being initial obj orientation
    visual_shape_id = None
    collision_shape_id = -1
    if shape == pb.GEOM_BOX:
        visual_shape_id = pb.createVisualShape(shapeType=shape, halfExtents=dim, rgbaColor=color)
        if collidable:
            collision_shape_id = pb.createCollisionShape(shapeType=shape, halfExtents=dim)
    elif shape == pb.GEOM_CYLINDER:
        visual_shape_id = pb.createVisualShape(shape, dim[0], [1, 1, 1], dim[1], rgbaColor=color)
        if collidable:
            collision_shape_id = pb.createCollisionShape(shape, dim[0], [1, 1, 1], dim[1])
    elif shape == pb.GEOM_SPHERE:
        visual_shape_id = pb.createVisualShape(shape, radius=dim[0], rgbaColor=color)
        if collidable:
            collision_shape_id = pb.createCollisionShape(shape, radius=dim[0])

    sid = pb.createMultiBody(baseMass=mass, baseInertialFramePosition=[0, 0, 0],
                             baseCollisionShapeIndex=collision_shape_id,
                             baseVisualShapeIndex=visual_shape_id,
                             basePosition=init_xyz, baseOrientation=init_quat)
    return sid

if __name__ == "__main__":
    robot = LeapHand(pb)

    add_PyB_models_to_path()
    pb.setTimeStep(0.001)
    plane = pb.loadURDF('plane.urdf', basePosition=[0, 0, -0.5])
    # table = pb.loadURDF('table/table.urdf',
    #                     useFixedBase=True, globalScaling=0.5)
    cube = pb.loadURDF('cube_small.urdf', useFixedBase=True, basePosition=[0, 0, -0.5], globalScaling=1.)
    # pb.resetBasePositionAndOrientation(
    #     table, [0.4, 0., 0.0], [0, 0, -0.707, 0.707])
    goal_pos = np.array([[0.02, 0.04, 0.0], 
                         [0.04, 0.0, 0.0],
                         [0.02, -0.04, 0.0],
                         [-0.04, 0.0, 0.0]])
    tips = []
    for i in range(4):
        tid = create_primitive_shape(pb, 0.0, pb.GEOM_SPHERE, [0.02], collidable=False,init_xyz=goal_pos[i])

    goal_pos = goal_pos.flatten()
    objects = {'plane': plane}
               #'table': table}

    world = SimpleWorld(robot, objects)

    slow_rate = 1000.

    init_pos, goal_ori = world.robot.ee_pose()
    

    controller = OSImpedanceController(robot)

    print("started")

    z_traj = np.linspace(init_pos, goal_pos, 2000)

    input("Press Enter to continue...")
    controller.start_controller_thread()

    i = 0
    while True:
        now = time.time()

        ee_pos, _ = world.robot.ee_pose()
        wrench = world.robot.get_ee_wrench(local=False)
        
        if i < len(z_traj):
            goal_pos = z_traj[i]

        controller.update_goal(goal_pos, goal_ori)

        elapsed = time.time() - now
        sleep_time = (1./slow_rate) - elapsed
        if sleep_time > 0.0:
            time.sleep(sleep_time)

        i+=1
