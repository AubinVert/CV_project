import os
import shutil

cam1_dir = "../raw/test/camera_right_ir_image_raw"
cam2_dir = "../raw/test/camera_left_ir_image_raw"
out_cam1_dir = "sync_right_ir"
out_cam2_dir = "sync_left_ir"

# New directories
os.makedirs(out_cam1_dir, exist_ok=True)
os.makedirs(out_cam2_dir, exist_ok=True)


def extract_timestamp(filename):
    """Timestamp extraction from the names"""
    name, _ = os.path.splitext(filename)
    try:
        return float(name.split('_')[-1])
    except ValueError:
        raise ValueError(f"No timestamp in the file : {filename}")


# Ordered list of names
cam1_files = sorted(os.listdir(cam1_dir))
cam2_files = sorted(os.listdir(cam2_dir))

# Extracting timestamp for each file in both lists
cam1_times = [extract_timestamp(f) for f in cam1_files]
cam2_times = [extract_timestamp(f) for f in cam2_files]

import os
import shutil

cam1_dir = "../raw/test/camera_right_ir_image_raw"
cam2_dir = "../raw/test/camera_left_ir_image_raw"
out_cam1_dir = "sync_right_ir"
out_cam2_dir = "sync_left_ir"

# Create new directories
os.makedirs(out_cam1_dir, exist_ok=True)
os.makedirs(out_cam2_dir, exist_ok=True)


def extract_timestamp(filename):
    """Extract timestamp from filename."""
    name, _ = os.path.splitext(filename)
    try:
        return float(name.split('_')[-1])
    except ValueError:
        raise ValueError(f"No timestamp found in file: {filename}")


# Get sorted file lists
cam1_files = sorted(os.listdir(cam1_dir))
cam2_files = sorted(os.listdir(cam2_dir))

# Extract timestamps for each file in both lists
cam1_times = [extract_timestamp(f) for f in cam1_files]
cam2_times = [extract_timestamp(f) for f in cam2_files]

# Cam2 folder has fewer images, so for each cam2 image, find the closest
# timestamp match in the cam1 folder
for cam2_file, t2 in zip(cam2_files, cam2_times):
    # Find the closest timestamp
    closest_idx = min(range(len(cam1_times)), key=lambda i: abs(cam1_times[i] - t2))
    closest_cam1_file = cam1_files[closest_idx]

    # Copy the matched pair to new directories
    shutil.copy(os.path.join(cam1_dir, closest_cam1_file),
                os.path.join(out_cam1_dir, closest_cam1_file))
    shutil.copy(os.path.join(cam2_dir, cam2_file),
                os.path.join(out_cam2_dir, cam2_file))

print("Synchronization complete.")
print(f"{len(os.listdir(out_cam1_dir))} image pairs generated.")