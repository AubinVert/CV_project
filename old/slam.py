from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
from collections import deque

# ---------------- CONFIG ----------------
# Dataset directory
dataset_dir = Path("../raw/test/camera_color_image_raw")

# Thresholds
MIN_MATCH_COUNT = 30
KEYFRAME_MIN_INLIERS = 10
KEYFRAME_MIN_TRANSLATION = 1e-6
MAX_RECENT_KF = 5
KEYFRAME_MIN_ROTATION = 0.01

# Camera intrinsics
fx, fy, cx, cy = 306.0, 306.1, 318.5, 201.4
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]], dtype=float)

# ---------------- UTILS ----------------
def load_images(dataset_dir):
    image_paths = sorted([p for p in dataset_dir.glob("*.png")])
    if not image_paths:
        raise FileNotFoundError(f"No images found in {dataset_dir}")
    images = [cv2.imread(str(p)) for p in image_paths]
    for p,img in zip(image_paths, images):
        if img is None:
            raise ValueError(f"Failed to load image: {p}")
    return images, image_paths

def rt_to_extrinsic(R, t):
    extr = np.eye(4)
    extr[:3,:3] = R
    extr[:3,3:] = t.reshape(3,1)
    return extr

# ---------------- FEATURE DETECTION / MATCHING ----------------
sift = cv2.SIFT_create()

matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)


def detect_and_compute(img):
    key_points, descriptors = sift.detectAndCompute(img, None)
    return key_points, descriptors

def match_descriptors(des1, des2, ratio=0.75):
    if des1 is None or des2 is None:
        return []

    matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < ratio * n.distance]
    return good_matches

# ---------------- POSE ESTIMATION ----------------
def estimate_relative_pose(key_points_1, key_points_2, matches, K):
    if len(matches) < 8:
        return None

    pts1 = np.float32([key_points_1[m.queryIdx].pt for m in matches]) # m.queryIdx = ids from matched point in first image
    pts2 = np.float32([key_points_2[m.trainIdx].pt for m in matches]) # m.trainIdx = ids from matched point in second image

    # Essential matrix estimation with ransac method
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        return None

    # Find R, t using E, K and the points
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

    # Filter point that are really corresponding (mask_pose.ravel()==1)
    inlier1 = pts1[mask_pose.ravel()>0]
    inlier2 = pts2[mask_pose.ravel()>0]
    return R, t, inlier1, inlier2, mask_pose

# ---------------- TRIANGULATION ----------------
def triangulate_pair(R1, t1, R2, t2, pts1, pts2, K):
    # Calculate projetction matrices
    P1 = K @ np.hstack((R1, t1)) # P = K.[R|t]
    P2 = K @ np.hstack((R2, t2)) # P = K.[R|t]

    # Triangulation
    pts4 = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)

    # Going from homogeneous coordinate frame to 3D space frame
    pts3 = (pts4[:3] / pts4[3]).T

    return pts3

# ---------------- POSE GRAPH ----------------
def create_o3d_pose_graph():
    return o3d.pipelines.registration.PoseGraph()

# ---------------- KEYFRAME CLASS ----------------
class Keyframe:
    def __init__(self, img, path, key_points, descriptors, pose, id):
        self.img = img
        self.path = path
        self.key_points = key_points
        self.descriptors = descriptors
        self.pose = pose
        self.id = id

# ---------------- SLAM BASED ALGORITHM ----------------
def slam_based_algorithm(imgs, paths, K):
    n = len(imgs)
    keyframes = []
    all_points = []
    all_colors = []
    pose_graph = create_o3d_pose_graph()
    pose_id = 0

    # First keyframe
    key_point0, descriptor0 = detect_and_compute(imgs[0])
    kf0 = Keyframe(imgs[0], paths[0], key_point0, descriptor0, rt_to_extrinsic(np.eye(3), np.zeros((3,1))), pose_id)
    keyframes.append(kf0)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(kf0.pose))
    pose_id += 1
    prev_key_point, prev_descriptor, prev_img = key_point0, descriptor0, imgs[0]

    recent_kf_indices = deque([0], maxlen=MAX_RECENT_KF)

    # For each image :
    # - Compute sift feature matching
    # - Match descriptors with those from previous image
    # - Estimate relative pose
    # - Compute absolute pose
    # - Determine if current image can be seen as a keyframe (for further calculation)
    # - If key frame :
    #       - Add frame to key frames list
    #       - Add pose to the pose graph
    #       - Compute triangulation

    for i in range(1, n):
        img = imgs[i]

        # Sift feature detection
        key_point, descriptor = detect_and_compute(img)

        # Feature matching
        matches = match_descriptors(prev_descriptor, descriptor)
        print(f"[{i}] Matches = {len(matches)}")
        if len(matches) < MIN_MATCH_COUNT:
            prev_key_point, prev_descriptor, prev_img = key_point, descriptor, img
            continue

        # Relative pose estimation
        est = estimate_relative_pose(prev_key_point, key_point, matches, K)
        if est is None:
            prev_key_point, prev_descriptor, prev_img = key_point, descriptor, img
            continue

        # Compute absolute pose
        relative_R, relative_t, inliers1, inliers2, mask_pose = est
        prev_R = keyframes[-1].pose[:3,:3]
        prev_t = keyframes[-1].pose[:3,3].reshape(3,1)
        new_R = prev_R @ relative_R
        new_t = prev_t + prev_R @ relative_t
        curr_pose = rt_to_extrinsic(new_R, new_t)

        # Compute some coefficient to know if image is a key frame
        num_inliers = inliers1.shape[0]
        print(f"[{i}] Inliers = {num_inliers}")
        baseline = np.linalg.norm(relative_t)
        print(f"[{i}] baseline = {baseline}")
        rot_angle = np.arccos(np.clip((np.trace(relative_R) - 1) / 2, -1, 1))
        print(f"Rotation angle = {rot_angle}")

        # Determine if current image can be seen as a keyframe (for further calculation)
        is_keyframe = (num_inliers > KEYFRAME_MIN_INLIERS) and (baseline > KEYFRAME_MIN_TRANSLATION or rot_angle > KEYFRAME_MIN_ROTATION)

        # If current frame is a key frame
        if is_keyframe:
            # Append it the the key frame list
            current_keyframe = Keyframe(img, paths[i], key_point, descriptor, curr_pose, pose_id)
            keyframes.append(current_keyframe)

            # Add pose to pose graph
            pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(current_keyframe.pose))
            T_last = keyframes[-2].pose
            T_curr = curr_pose
            T_ij = np.linalg.inv(T_last) @ T_curr
            information = np.identity(6)
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    keyframes[-2].id, current_keyframe.id, T_ij, information, uncertain=False
                )
            )

            # Match descriptors from previous and current keyframe
            matches_kf = match_descriptors(keyframes[-2].descriptor, current_keyframe.descriptor)
            if len(matches_kf) >= 8:
                # Get point from both image
                pts1 = np.float32([keyframes[-2].key_point[m.queryIdx].pt for m in matches_kf]) # m.queryIdx = ids from matched point in previous image
                pts2 = np.float32([current_keyframe.key_point[m.trainIdx].pt for m in matches_kf]) # m.trainIdx = ids from matched point current image

                # Compute Rotation and translation matrix for each image
                # All matrices are inverted because cv2 triangulation function need to have them going from the image
                # to the world coordinate frame
                R1 = np.linalg.inv(keyframes[-2].pose)[:3,:3]
                t1 = np.linalg.inv(keyframes[-2].pose)[:3,3].reshape(3,1)
                R2 = np.linalg.inv(current_keyframe.pose)[:3,:3]
                t2 = np.linalg.inv(current_keyframe.pose)[:3,3].reshape(3,1)

                # Triangulation : results are given in the absolute coordinate frame
                pts3d = triangulate_pair(R1,t1,R2,t2,pts1,pts2,K)

                # Find color for each point
                for p in pts3d:
                    # Find point in the image coordinate frame (in pixels)
                    uv,_ = cv2.projectPoints(p.reshape(1,3), np.zeros(3), np.zeros(3), K, None)
                    u,v = int(uv[0,0,0]), int(uv[0,0,1])

                    # Find point color
                    h,w = img.shape[:2]
                    if 0<=v<h and 0<=u<w:
                        col = img[v,u]/255.0
                    else:
                        col = np.array([0.5,0.5,0.5])

                    # Append 3D point and its color
                    all_points.append(p)
                    all_colors.append(col)


            pose_id += 1
            recent_kf_indices.append(current_keyframe.id)

        prev_key_point, prev_descriptor, prev_img = key_point, descriptor, img

    # Reformat datas to numpy vertical stack vector before returning
    pts_np = np.vstack(all_points) if all_points else np.zeros((0,3))
    cols_np = np.vstack(all_colors) if all_colors else np.zeros((0,3))
    return keyframes, pts_np, cols_np, pose_graph

# ---------------- POSE GRAPH OPTIMIZATION ----------------
def optimize_pose_graph(pose_graph):
    """ Optimization of the pose graph """
    option = o3d.pipelines.registration.GlobalOptimizationOption(max_correspondence_distance=0.1,
                                                                  edge_prune_threshold=0.25,
                                                                  reference_node=0)
    method = o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt()
    criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
    o3d.pipelines.registration.global_optimization(pose_graph, method, criteria, option)
    return pose_graph

# ---------------- VISUALIZATION ----------------
def show_point_cloud(points, colors):
    pt_cld = o3d.geometry.PointCloud()
    pt_cld.points = o3d.utility.Vector3dVector(points)
    pt_cld.colors = o3d.utility.Vector3dVector(colors)
    pt_cld, _ = pt_cld.remove_radius_outlier(15, 1)
    o3d.visualization.draw_geometries([pt_cld])

# ---------------- MAIN ----------------
if __name__ == "__main__":
    imgs, paths = load_images(dataset_dir)
    print(f"{len(imgs)} images loaded")
    kfs, pts, cols, pg = slam_based_algorithm(imgs, paths, K)
    print("Keyframes:", len(kfs), "Points:", pts.shape)
    pg_opt = optimize_pose_graph(pg)
    show_point_cloud(pts, cols)
