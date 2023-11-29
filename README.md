# Compliance Dexterous grasping under shape uncertainty

## File structure
```
root directory
  ├── assets  
  │   └── // folders for different YCB and real scan object
  ├── data  
  │   └── // folders for data after optimization
  ├── gpis_states  
  │   └── // state data for restoring and visualizing gaussian process implicit surface
  ├── pybullet_robot  
  │   ├── // fingertip impedance and null space controller for leap hand and allegro hand
  |   └── src
  |      └── // source code for robot and controller definition
  ├── thirdparty  
  │   ├── // dependent third-party package
  |   └── TorchSDF
  |      └── // Compute SDF of mesh in pytorch
  |   └── differentiable-robot-model
  |      └── // Differentiable forward kinematics model
  ├── 3dplot.py // Visualizing GPIS intersection and its uncertainty
  ├── concatenate_pcd.py // Merging generated point cloud with observed pointcloud
  ├── gpis.py // Definition for Gaussian process implicit surface
  ├── optimize_pregrasp.py // Running probabilistic pregrasp optimization
  ├── rigidBodySento.py  // Some useful tools for creating and handling Pybullet rigid objects
  ├── train_gpis_completion.py  // Training GPIS from observed + completed point clouds
  └── verify_pregrasp.py  // verifying pregrasp by simulation
```