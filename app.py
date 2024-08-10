from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import transform
from scipy.interpolate import griddata
import os

app = Flask(__name__)

# Path to save uploaded files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Save uploaded files
    optical_file = request.files['optical']
    radar_file = request.files['radar']
    lidar_file = request.files['lidar']

    optical_path = os.path.join(UPLOAD_FOLDER, 'optical_image.jpg')
    radar_path = os.path.join(UPLOAD_FOLDER, 'radar_image.jpg')
    lidar_path = os.path.join(UPLOAD_FOLDER, 'lidar_data.txt')

    optical_file.save(optical_path)
    radar_file.save(radar_path)
    lidar_file.save(lidar_path)

    # Load images and LiDAR data
    optical_img = cv2.imread(optical_path, cv2.IMREAD_GRAYSCALE)
    radar_img = cv2.imread(radar_path, cv2.IMREAD_GRAYSCALE)
    lidar_data = np.loadtxt(lidar_path)
    
    # Resize radar image to match optical image dimensions
    radar_img = cv2.resize(radar_img, (optical_img.shape[1], optical_img.shape[0]))

    # Process LiDAR data
    x, y, z = lidar_data[:, 0], lidar_data[:, 1], lidar_data[:, 2]
    x_grid, y_grid = np.linspace(min(x), max(x), 100), np.linspace(min(y), max(y), 100)
    x_grid, y_grid = np.meshgrid(x_grid, y_grid)
    z_grid = griddata((x, y), z, (x_grid, y_grid), method='cubic')

    z_grid_resized = cv2.resize(z_grid, (optical_img.shape[1], optical_img.shape[0]))
    z_grid_normalized = cv2.normalize(z_grid_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Image Registration: Align Radar Image to Optical Image
    aligned_radar_img = align_images(radar_img, optical_img)

    # Generate and save visualization plots
    save_plots(optical_img, aligned_radar_img, z_grid_normalized)

    return redirect(url_for('results'))

@app.route('/results')
def results():
    return render_template('results.html')

def align_images(src, target):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(src, None)
    kp2, des2 = orb.detectAndCompute(target, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
    tgt_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    H, _ = cv2.findHomography(src_pts, tgt_pts, cv2.RANSAC)
    aligned_src = cv2.warpPerspective(src, H, (target.shape[1], target.shape[0]))
    
    return aligned_src

def save_plots(optical_img, aligned_radar_img, z_grid_normalized):
    plt.figure(figsize=(18, 6))

    # Optical Image
    plt.subplot(1, 3, 1)
    plt.imshow(optical_img, cmap='gray')
    plt.title('Optical Image')

    # Aligned Radar Image
    plt.subplot(1, 3, 2)
    plt.imshow(aligned_radar_img, cmap='gray')
    plt.title('Aligned Radar Image')

    # DEM Image
    plt.subplot(1, 3, 3)
    plt.imshow(z_grid_normalized, cmap='terrain')
    plt.title('Digital Elevation Model (DEM)')

    plt.savefig(os.path.join(UPLOAD_FOLDER, 'result_plot.png'))

    # Combined Optical and DEM
    plt.figure(figsize=(12, 12))
    plt.imshow(z_grid_normalized, cmap='terrain', alpha=0.5)
    plt.imshow(optical_img, cmap='gray', alpha=0.5)
    plt.title('Combined Optical Image and DEM')
    plt.colorbar(label='Elevation (m)')
    plt.savefig(os.path.join(UPLOAD_FOLDER, 'combined_plot.png'))

if __name__ == '__main__':
    app.run(debug=True)
