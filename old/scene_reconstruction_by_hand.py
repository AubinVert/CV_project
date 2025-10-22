import cv2
import numpy as np
import open3d as o3d
from pathlib import Path
from tqdm import tqdm

# =========================
# CONFIG
# =========================
IMG_DIR = Path("../raw/test/camera_left_ir_image_raw")
fx, fy, cx, cy = 306.0, 306.1, 318.5, 201.4
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])

# Load images
images = sorted(list(IMG_DIR.glob("*.png")))
imgs = [cv2.imread(str(im), cv2.IMREAD_GRAYSCALE) for im in images]

# =========================
# SIFT DETECTION
# =========================
sift = cv2.SIFT_create()
keypoints, descriptors = [], []

print(">> Detecting features...")
for img in tqdm(imgs):
    kp, des = sift.detectAndCompute(img, None)
    keypoints.append(kp)
    descriptors.append(des)

# =========================
# MATCHING BETWEEN NEIGHBORING PAIRS
# =========================
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches_list = []

print(">> Matching neighbors (i, i+1)...")
for i in tqdm(range(len(imgs) - 1)):
    matches = bf.match(descriptors[i], descriptors[i + 1])
    matches = sorted(matches, key=lambda x: x.distance)
    matches_list.append(matches)

# =========================
# INITIAL POSE (image 0)
# =========================
camera_poses = {0: (np.eye(3), np.zeros((3, 1)))}
points3D = []

print(">> Incremental reconstruction...")
for i, matches in enumerate(tqdm(matches_list)):
    kp1, kp2 = keypoints[i], keypoints[i + 1]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Essential matrix
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

    # Absolute pose
    R_prev, t_prev = camera_poses[i]
    R_abs = R @ R_prev
    t_abs = t_prev + R_prev.T @ t  # approximation
    camera_poses[i + 1] = (R_abs, t_abs)

    # Triangulation
    proj1 = K @ np.hstack((R_prev, t_prev))
    proj2 = K @ np.hstack((R_abs, t_abs))
    pts_4d = cv2.triangulatePoints(proj1, proj2, pts1.T, pts2.T)
    pts_3d = (pts_4d[:3] / pts_4d[3]).T
    points3D.append(pts_3d)

# Merge all points
points3D = np.vstack(points3D)

# =========================
# OPEN3D VISUALIZATION
# =========================
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points3D)
o3d.visualization.draw_geometries([pcd])