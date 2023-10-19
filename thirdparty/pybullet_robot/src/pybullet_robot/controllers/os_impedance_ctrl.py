import numpy as np
from .os_controller import OSControllerBase
from .ctrl_config import OSImpConfig
phi = np.pi/6

class OSImpedanceController(OSControllerBase):

    def __init__(self, robot, config=OSImpConfig, **kwargs):
        self.ref_q = config['ref_q']
        OSControllerBase.__init__(self, robot=robot, config=config, **kwargs)

    def update_goal(self, goal_pos, goal_ori):
        self._mutex.acquire()
        self._goal_pos = np.asarray(goal_pos).reshape(-1,1)
        self._goal_ori = np.asarray(goal_ori)
        self._mutex.release()

    def _compute_cmd(self):
        """
        Actual control loop. Uses goal pose from the feedback thread
        and current robot states from the subscribed messages to compute
        task-space force, and then the corresponding joint torques.
        """
        curr_pos, curr_ori = self._robot.ee_pose()

        delta_pos = self._goal_pos - curr_pos.reshape(-1, 1)
        # print goal_pos, curr_pos

        curr_vel, curr_omg = self._robot.ee_velocity()
        # print self._goal_pos, curr_pos
        # Desired task-space force using PD law
        F = self._P_pos.dot(delta_pos) - self._D_pos.dot(curr_vel.reshape(-1,1))

        error = np.asarray([np.linalg.norm(delta_pos)])

        J = self._robot.jacobian()
        tau = np.zeros(self._robot.n_joints(), dtype=np.float32)
        for i in range(self._robot.n_ee()):
            J_sub = J[i*3:(i+1)*3,i*4:(i+1)*4]
            tau[i*4:(i+1)*4] = np.dot(J_sub.T, F[i*3:(i+1)*3]).flatten()
            Jpinv_sub = np.linalg.pinv(J_sub)
            tau[i*4:(i+1)*4] += (np.eye(4) - J_sub.T @ Jpinv_sub.T)@ (1.0 * (self.ref_q[i*4:(i+1)*4] - self._robot.angles()[i*4:(i+1)*4]))

        # joint torques to be commanded
        return tau, error

    def _initialise_goal(self):
        self.update_goal(self._robot.ee_pose()[0],self._robot.ee_pose()[1])


