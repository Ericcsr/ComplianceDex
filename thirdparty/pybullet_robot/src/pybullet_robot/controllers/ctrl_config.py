import numpy as np

KP_P = np.asarray([400., 400., 400.]*4) # 6000
KP_O = np.asarray([0., 0., 0.]) # 300
phi = np.pi/6
OSImpConfig = {
    'P_pos': KP_P,
    'D_pos': 3*np.sqrt(KP_P),
    'P_ori': KP_O,
    'D_ori': np.asarray([0.0,0.0,0.0]),
    'error_thresh': np.asarray([0.005, 0.005]),
    'start_err': np.asarray([200., 200.]),
    'ref_q': np.array([0.0, phi, phi, phi,
                       0.0, phi, phi, phi,
                       0.0, phi, phi, phi,
                       2*phi, phi, phi, phi])
}
