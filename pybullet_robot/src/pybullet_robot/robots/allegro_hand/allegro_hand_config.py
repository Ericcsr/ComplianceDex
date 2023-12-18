import numpy as np

ROBOT_CONFIG = {
    'ee_link_idx': [3,7,11,15],
    'ee_link_offset': np.array([[0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0]]),

    'ee_link_name': ['link_3.0_tip','link_7.0_tip','link_11.0_tip','link_15.0_tip'],

    'ref_q': np.array([0.0, np.pi/9, np.pi/9, np.pi/6, 
                0.0, np.pi/9, np.pi/9, np.pi/6, 
                0.0, np.pi/9, np.pi/9, np.pi/6, 
                2 * np.pi/6, np.pi/9, np.pi/6, np.pi/6]),

    'collision_links':['link_3.0_tip','link_7.0_tip','link_11.0_tip','link_15.0_tip'],

    'collision_offsets':np.array([[0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0]]),

    'collision_pairs':[[0, 1], [0, 2], [0, 3],
                      [1,2], [1,3],
                      [2,3]]

}