1 - 1_dataset_transformation.py

Matches each depth map with its corresponding color image by finding the closest timestamps.

2 - 2_trajectory_reconstruction.py

Reconstructs the camera trajectory using stereo calibration between consecutive stereo image pairs.
Feature correspondences between the two images are established with SIFT.

3 - 3_trajectory_isolation.py

Filters the trajectory to include only the RGB images that have an associated depth map.

4 - 4_scene_reconstruction.py

Reconstructs the 3D scene for the first five frames by aligning them using:

a) The previously computed camera trajectory, and

b) ICP (Iterative Closest Point), which automatically estimates the transformation matrix needed to merge the scenes.