import numpy as np

ROBOT_CONFIG = {
    'ee_link_idx': [3,7,11,15],
    'ee_link_offset': np.array([[0.0, 0.0, 0.0267],
                                [0.0, 0.0, 0.0267],
                                [0.0, 0.0, 0.0267],
                                [0.0, 0.0, 0.0423]]),
    'ee_link_name': ['link_3.0_tip','link_7.0_tip','link_11.0_tip','link_15.0_tip']
}