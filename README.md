# README: Processing Orbec Camera Data from ROS1 Rosbags

## Overview
This document provides details on the dataset generated from processing ROS1 rosbags containing Orbec camera feeds, depth images, and 3D colored point clouds. The extracted data is organized into specific folders, allowing for easy access and analysis.

## Dataset Structure
The processed data is organized into the following directory structure:


### Description of Files
- **Camera Info Files**:
  - **Location**: `camera_color_camera_info/`, `camera_depth_camera_info/`
  - **Format**: `.txt`
  - **Content**: These files contain calibration information for the RGB and depth cameras, such as intrinsic parameters and distortion coefficients.

- **Color Images**:
  - **Location**: `camera_color_image_raw/`
  - **Format**: `.png`
  - **Content**: These are the raw RGB images captured by the Orbec camera. Each image file is timestamped for reference.

- **Depth Images**:
  - **Location**: `camera_depth_image_raw/`
  - **Format**: `.png`
  - **Content**: These are the raw depth images captured by the camera, representing depth information in a visual format.

- **Left Infrared Images**:
  - **Location**: `camera_left_ir_image_raw/`
  - **Format**: `.png`
  - **Content**: These are the raw infrared images captured by the left IR camera. Each image file is timestamped for reference.

- **Right Infrared Images**:
  - **Location**: `camera_right_ir_image_raw/`
  - **Format**: `.png`
  - **Content**: These are the raw infrared images captured by the right IR camera. Each image file is timestamped for reference.

- **Colored Point Clouds**:
  - **Location**: `depth_registered_colored_pointclouds/`
  - **Format**: `.pcd`
  - **Content**: These files contain 3D colored point clouds generated from the registered depth data. Each point in the cloud is associated with its RGB color, allowing for a detailed representation of the environment.

## Usage
The extracted images and point clouds can be used for various applications, including:

- **3D Reconstruction**: Analyzing the point clouds for creating 3D models of the captured environment.
- **Robotic Navigation**: Utilizing depth information for obstacle avoidance and environment mapping.
- **Machine Learning**: Training algorithms for tasks such as object recognition and scene understanding.

## Data Format
- **PNG**: The color, depth, and infrared images are stored in PNG format, ensuring lossless quality for image processing tasks.
- **PCD**: The point cloud files are stored in the PCD format, which is commonly used for storing 3D point cloud data.


## Environnement setup commands
### Windows
- python -m venv .venv
- .venv\Scripts\activate.bat
- pip install -r requirements.txt
### Linux/MacOs
- python3 -m venv .venv
- source .venv/bin/activate
- pip install -r requirements.txt