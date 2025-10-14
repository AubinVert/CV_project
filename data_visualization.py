import open3d as o3d
import numpy as np

# Charger deux paires RGB-D
rgbd1 = o3d.geometry.RGBDImage.create_from_color_and_depth(
    o3d.io.read_image('raw/test/camera_color_image_raw/camera_color_image_1727164782247779444.png'),
    o3d.io.read_image('raw/test/camera_depth_image_raw/camera_depth_image_1727164782243982104.png'),
    convert_rgb_to_intensity=False
)

rgbd2 = o3d.geometry.RGBDImage.create_from_color_and_depth(
    o3d.io.read_image('raw/test/camera_color_image_raw/camera_color_image_1727164782274488837.png'),
    o3d.io.read_image('raw/test/camera_depth_image_raw/camera_depth_image_1727164782668126160.png'),
    convert_rgb_to_intensity=False
)



# Intrinsics
fx, fy, cx, cy = 306.0, 306.1, 318.5, 201.4
intrinsics = o3d.camera.PinholeCameraIntrinsic(640, 400, fx, fy, cx, cy)

# Convertir en point clouds
pcd1 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd1, intrinsics)
pcd2 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd2, intrinsics)

# Calcul des normales pour les deux nuages
pcd1.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
)
pcd2.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
)

# Alignement par ICP
threshold = 0.02  # mètre
trans_init = np.eye(4)
reg_icp = o3d.pipelines.registration.registration_icp(
    pcd2, pcd1, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPlane()
)

# Appliquer la transformation
pcd2.transform(reg_icp.transformation)

# Fusionner les deux nuages
merged = pcd1 + pcd2
o3d.visualization.draw_geometries([merged])