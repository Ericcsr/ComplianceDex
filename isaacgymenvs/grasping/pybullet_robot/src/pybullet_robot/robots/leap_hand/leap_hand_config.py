import numpy as np

ROBOT_CONFIG = {
    'ee_link_idx': [3,7,11,15],
    'ee_link_offset': np.array([[0.0, -0.045, 0.015],
                                [0.0, -0.045, 0.015],
                                [0.0, -0.045, 0.015],
                                [0.0, -0.055, -0.015]]),
    'ee_link_name': ['fingertip','fingertip_2','fingertip_3','thumb_fingertip'],
    'ref_q': np.array([np.pi/6, -np.pi/6, np.pi/6, np.pi/6,
                       np.pi/6, 0.0     , np.pi/6, np.pi/6,
                       np.pi/6, np.pi/6 , np.pi/6, np.pi/6,
                       np.pi/3, np.pi/4 , np.pi/6, np.pi/6]),
}