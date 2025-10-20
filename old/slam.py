# slam_pathlib.py
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
from collections import deque

# ---------------- CONFIG ----------------
# Dossier contenant tes images
dataset_dir = Path("../raw/test/camera_color_image_raw")
# Seuils
MIN_MATCH_COUNT = 30
KEYFRAME_MIN_INLIERS = 10
KEYFRAME_MIN_TRANSLATION = 1e-6
MAX_RECENT_KF = 5
KEYFRAME_MIN_ROTATION = 0.01

# Caméra (remplace par tes valeurs)
fx, fy, cx, cy = 306.0, 306.1, 318.5, 201.4
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]], dtype=float)

# ---------------- UTIL ----------------
def load_images(dataset_dir):
    image_paths = sorted([p for p in dataset_dir.glob("*.png")])
    if not image_paths:
        raise FileNotFoundError(f"Aucune image trouvée dans {dataset_dir}")
    images = [cv2.imread(str(p)) for p in image_paths]
    for p,img in zip(image_paths, images):
        if img is None:
            raise ValueError(f"Impossible de charger l'image : {p}")
    return images, image_paths

def rt_to_extrinsic(R, t):
    extr = np.eye(4)
    extr[:3,:3] = R
    extr[:3,3:] = t.reshape(3,1)
    return extr

# ---------------- FEATURE DETECTION / MATCHING ----------------
sift = cv2.SIFT_create()

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)


def detect_and_compute(img):
    kp, des = sift.detectAndCompute(img, None)
    return kp, des

def match_descriptors(des1, des2, ratio=0.75):
    if des1 is None or des2 is None:
        return []

    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < ratio * n.distance]
    return good

# ---------------- POSE ESTIMATION ----------------
def estimate_relative_pose(kp1, kp2, matches, K):
    if len(matches) < 8:
        return None
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        return None
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
    inlier1 = pts1[mask_pose.ravel()>0]
    inlier2 = pts2[mask_pose.ravel()>0]
    return R, t, inlier1, inlier2, mask_pose

# ---------------- TRIANGULATION ----------------
def triangulate_pair(R1, t1, R2, t2, pts1, pts2, K):
    P1 = K @ np.hstack((R1, t1))
    P2 = K @ np.hstack((R2, t2))
    pts4 = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3 = (pts4[:3] / pts4[3]).T
    return pts3

# ---------------- POSE GRAPH ----------------
def create_o3d_pose_graph():
    return o3d.pipelines.registration.PoseGraph()

# ---------------- KEYFRAME CLASS ----------------
class Keyframe:
    def __init__(self, img, path, kp, des, pose, id):
        self.img = img
        self.path = path
        self.kp = kp
        self.des = des
        self.pose = pose
        self.id = id

# ---------------- MONOCULAR SLAM ----------------
def monocular_slam(imgs, paths, K):
    n = len(imgs)
    keyframes = []
    all_points = []
    all_colors = []
    pose_graph = create_o3d_pose_graph()
    pose_id = 0

    # première keyframe
    kp0, des0 = detect_and_compute(imgs[0])
    kf0 = Keyframe(imgs[0], paths[0], kp0, des0, rt_to_extrinsic(np.eye(3), np.zeros((3,1))), pose_id)
    keyframes.append(kf0)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(kf0.pose))
    pose_id += 1
    prev_kp, prev_des, prev_img = kp0, des0, imgs[0]

    recent_kf_indices = deque([0], maxlen=MAX_RECENT_KF)

    for i in range(1, n):
        img = imgs[i]
        kp, des = detect_and_compute(img)
        matches = match_descriptors(prev_des, des)
        print(f"[{i}] Matches = {len(matches)}")
        if len(matches) < MIN_MATCH_COUNT:
            prev_kp, prev_des, prev_img = kp, des, img
            continue
        est = estimate_relative_pose(prev_kp, kp, matches, K)
        if est is None:
            prev_kp, prev_des, prev_img = kp, des, img
            continue
        R_rel, t_rel, in1, in2, mask_pose = est
        prev_R = keyframes[-1].pose[:3,:3]
        prev_t = keyframes[-1].pose[:3,3].reshape(3,1)
        new_R = prev_R @ R_rel
        new_t = prev_t + prev_R @ t_rel
        curr_pose = rt_to_extrinsic(new_R, new_t)
        num_inliers = in1.shape[0]
        print(f"[{i}] Inliers = {num_inliers}")
        baseline = np.linalg.norm(t_rel)
        print(f"[{i}] baseline = {baseline}")
        rot_angle = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))
        print(f"Rotation angle = {rot_angle}")
        is_keyframe = (num_inliers > KEYFRAME_MIN_INLIERS) and (baseline > KEYFRAME_MIN_TRANSLATION or rot_angle > KEYFRAME_MIN_ROTATION)
        if is_keyframe:
            kf = Keyframe(img, paths[i], kp, des, curr_pose, pose_id)
            keyframes.append(kf)
            pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(kf.pose))
            T_last = keyframes[-2].pose
            T_curr = curr_pose
            T_ij = np.linalg.inv(T_last) @ T_curr
            information = np.identity(6)
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    keyframes[-2].id, kf.id, T_ij, information, uncertain=False
                )
            )
            # Triangulation sparse entre les keyframes
            matches_kf = match_descriptors(keyframes[-2].des, kf.des)
            if len(matches_kf) >= 8:
                pts1 = np.float32([keyframes[-2].kp[m.queryIdx].pt for m in matches_kf])
                pts2 = np.float32([kf.kp[m.trainIdx].pt for m in matches_kf])
                R1 = np.linalg.inv(keyframes[-2].pose)[:3,:3]
                t1 = np.linalg.inv(keyframes[-2].pose)[:3,3].reshape(3,1)
                R2 = np.linalg.inv(kf.pose)[:3,:3]
                t2 = np.linalg.inv(kf.pose)[:3,3].reshape(3,1)
                pts3d = triangulate_pair(R1,t1,R2,t2,pts1,pts2,K)
                for p in pts3d:
                    uv,_ = cv2.projectPoints(p.reshape(1,3), np.zeros(3), np.zeros(3), K, None)
                    u,v = int(uv[0,0,0]), int(uv[0,0,1])
                    h,w = img.shape[:2]
                    if 0<=v<h and 0<=u<w:
                        col = img[v,u]/255.0
                    else:
                        col = np.array([0.5,0.5,0.5])
                    all_points.append(p)
                    all_colors.append(col)
            pose_id += 1
            recent_kf_indices.append(kf.id)
        prev_kp, prev_des, prev_img = kp, des, img

    pts_np = np.vstack(all_points) if all_points else np.zeros((0,3))
    cols_np = np.vstack(all_colors) if all_colors else np.zeros((0,3))
    return keyframes, pts_np, cols_np, pose_graph

# ---------------- OPTIMISATION DU GRAPHE ----------------
def optimize_pose_graph(pose_graph):
    option = o3d.pipelines.registration.GlobalOptimizationOption(max_correspondence_distance=0.1,
                                                                  edge_prune_threshold=0.25,
                                                                  reference_node=0)
    method = o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt()
    criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
    o3d.pipelines.registration.global_optimization(pose_graph, method, criteria, option)
    return pose_graph

# ---------------- VISUALISATION ----------------
def show_point_cloud(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd, _ = pcd.remove_radius_outlier(15, 1)
    o3d.visualization.draw_geometries([pcd])

# ---------------- MAIN ----------------
if __name__ == "__main__":
    imgs, paths = load_images(dataset_dir)
    print(f"{len(imgs)} images chargées ✅")
    kfs, pts, cols, pg = monocular_slam(imgs, paths, K)
    print("Keyframes:", len(kfs), "Points:", pts.shape)
    pg_opt = optimize_pose_graph(pg)
    show_point_cloud(pts, cols)