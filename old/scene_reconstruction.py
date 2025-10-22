import open3d as o3d
import numpy as np
import glob

# Datasets
color_files = sorted(glob.glob('raw/test/sync_color_image/*.png'))
depth_files = sorted(glob.glob('raw/test/sync_depth_image/*.png'))

# Intrinsics
fx, fy, cx, cy = 306.0, 306.1, 318.5, 201.4
intrinsics = o3d.camera.PinholeCameraIntrinsic(640, 400, fx, fy, cx, cy)

# ICP threshold
threshold = 0.01

# Generate point cloud from RGB and depth image pair
def create_point_cloud(color_file, depth_file):
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.io.read_image(color_file),
        o3d.io.read_image(depth_file),
        convert_rgb_to_intensity=False
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
    #pcd = pcd.voxel_down_sample(voxel_size=0.05)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
    )
    return pcd


# Initialize point cloud (first point cloud)
merged_pt_cloud = create_point_cloud(color_files[0], depth_files[0])
global_transform = np.eye(4)
previous_pt_cloud = merged_pt_cloud

# Loop through all remaining images
for i in range(1, len(color_files)):
    # Create next point cloud
    new_pt_cloud = create_point_cloud(color_files[i], depth_files[i])

    # ICP alignment with merged dataset
    icp_registration = o3d.pipelines.registration.registration_icp(
        previous_pt_cloud, new_pt_cloud, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    # Apply global transform to new point cloud
    global_transform = global_transform @ icp_registration.transformation
    new_pt_cloud.transform(global_transform)

    # Merge point clouds
    merged_pt_cloud += new_pt_cloud
    #merged_pt_cloud = merged_pt_cloud.voxel_down_sample(voxel_size=0.1)
    merged_pt_cloud, ind = merged_pt_cloud.remove_statistical_outlier(
        nb_neighbors=100,
        std_ratio=2.0
    )
    previous_pt_cloud = new_pt_cloud
    print(f"Image {i+1}/{len(color_files)} processed")

# Final visualization
o3d.visualization.draw_geometries([merged_pt_cloud])