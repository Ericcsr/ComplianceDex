import yaml
import pybullet as pb
import rigidBodySento as rb

c = pb.connect(pb.GUI)

configs = yaml.safe_load(open("assets/iiwa14.yml","r"))

robot = pb.loadURDF("assets/kuka_allegro/model.urdf", useFixedBase=True)

link_names = {"lbr_iiwa_link_0": -1}
for i in range(pb.getNumJoints(robot)):
    link_names[pb.getJointInfo(robot, i)[12].decode("utf-8")] = i

color_codes = [[1,0,0,0.7],[0,1,0,0.7]]

for i, link in enumerate(configs["collision_spheres"].keys()):
    if link not in link_names:
        continue
    link_id = link_names[link]
    link_pos, link_ori = rb.get_link_com_xyz_orn(pb, robot, link_id)
    for sphere in configs["collision_spheres"][link]:
        s = rb.create_primitive_shape(pb, 0.0, shape=pb.GEOM_SPHERE, dim=(sphere["radius"],), collidable=False, color=color_codes[i%2])
        # Place the sphere relative to the link
        world_coord = list(pb.multiplyTransforms(link_pos, link_ori, sphere["center"], [0,0,0,1])[0])
        world_coord[1] += 0.
        pb.resetBasePositionAndOrientation(s, world_coord, [0,0,0,1])


input()
