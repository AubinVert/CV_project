import os
import shutil

# === À MODIFIER ===
cam1_dir = "raw/test/camera_color_image_raw"
cam2_dir = "raw/test/camera_depth_image_raw"
out_cam1_dir = "sync_color_image"
out_cam2_dir = "sync_depth_image"

# Crée les dossiers de sortie s’ils n’existent pas
os.makedirs(out_cam1_dir, exist_ok=True)
os.makedirs(out_cam2_dir, exist_ok=True)


def extract_timestamp(filename):
    """Extrait le timestamp depuis le nom de fichier (sans extension)."""
    name, _ = os.path.splitext(filename)
    try:
        return float(name.split('_')[-1])
    except ValueError:
        raise ValueError(f"Impossible d'extraire un timestamp du nom {filename}")


# Liste des fichiers triés
cam1_files = sorted(os.listdir(cam1_dir))
cam2_files = sorted(os.listdir(cam2_dir))

cam1_times = [extract_timestamp(f) for f in cam1_files]
cam2_times = [extract_timestamp(f) for f in cam2_files]

for cam2_file, t2 in zip(cam2_files, cam2_times):
    # Cherche dans cam1 le timestamp le plus proche
    closest_idx = min(range(len(cam1_times)), key=lambda i: abs(cam1_times[i] - t2))
    closest_cam1_file = cam1_files[closest_idx]

    # Copie les deux fichiers dans les dossiers synchronisés
    shutil.copy(os.path.join(cam1_dir, closest_cam1_file),
                os.path.join(out_cam1_dir, closest_cam1_file))
    shutil.copy(os.path.join(cam2_dir, cam2_file),
                os.path.join(out_cam2_dir, cam2_file))

print("Synchronisation terminée.")
print(f"{len(os.listdir(out_cam1_dir))} paires d’images générées.")