import open3d as o3d
import numpy as np
import glob

# Liste des fichiers images
color_files = sorted(glob.glob('raw/test/camera_color_image_raw/*.png'))
depth_files = sorted(glob.glob('raw/test/camera_depth_image_raw/*.png'))

# Intrinsics
fx, fy, cx, cy = 306.0, 306.1, 318.5, 201.4
intrinsics = o3d.camera.PinholeCameraIntrinsic(640, 400, fx, fy, cx, cy)

# Paramètres ICP
threshold = 0.02


# Fonction pour créer un nuage à partir d'une image RGB-D
def create_pcd(color_file, depth_file):
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.io.read_image(color_file),
        o3d.io.read_image(depth_file),
        convert_rgb_to_intensity=False
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
    )
    return pcd


# Initialisation avec la première image
merged_pcd = create_pcd(color_files[0], depth_files[0])
trans_global = np.eye(4)

# Boucle sur les images suivantes
for i in range(1, len(color_files)):
    pcd_next = create_pcd(color_files[i], depth_files[i])

    # ICP par rapport au nuage précédent (ou au nuage fusionné)
    reg_icp = o3d.pipelines.registration.registration_icp(
        pcd_next, merged_pcd, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    # Appliquer la transformation globale
    trans_global = reg_icp.transformation @ trans_global
    pcd_next.transform(trans_global)

    # Fusionner
    merged_pcd += pcd_next

# Affichage final
o3d.visualization.draw_geometries([merged_pcd])