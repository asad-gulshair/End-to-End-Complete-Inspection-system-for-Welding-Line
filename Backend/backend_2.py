from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import io
import pandas as pd
import json
from datetime import datetime
import os
from werkzeug.utils import secure_filename
import tempfile
import threading
import time
from math import hypot
from flask import Flask, Response, jsonify, request
import torch
import platform
import psutil
from pathlib import Path
from math import sqrt, atan2, degrees
from flask import send_from_directory
from math import hypot, sqrt, atan2, degrees
import pandas as pd
from scipy.optimize import least_squares
import gc
import math
import random
from scipy.spatial import distance
from auth_module import register_auth_routes

from flask.json.provider import DefaultJSONProvider

class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)


app = Flask(__name__)
register_auth_routes(app)
CORS(app)
app.json = NumpyJSONProvider(app) 


@app.route('/')
def serve_frontend():
    import os
    folder_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(folder_path, 'dash_3.html') # Corrected to dash.html

    if os.path.exists(file_path):
        return send_from_directory(folder_path, 'dash_3.html')
    else:
        return f"❌ File not found at: {file_path}", 404 

# Add to your Flask app configuration (after CORS(app)):
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max


# Configuration
UPLOAD_FOLDER = 'E:\Testing'
RESULTS_FOLDER = 'E:\Testing'
VIDEO_OUTPUT_FOLDER = 'E:\Testing\Videos'
WELD_MODEL_PATH = 'best (1).pt'
COATING_MODEL_PATH = 'coating_defects.pt'
FLANGE_MODEL_PATH = 'flange.pt'  # Add flange model path
FLANGE_CSV_PATH = 'flange_specifications11.csv'  # CSV with flange standards

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(VIDEO_OUTPUT_FOLDER, exist_ok=True)

# Detect device for GPU/CPU optimization
def get_optimal_device():
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"✅ GPU detected: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = 'cpu'
        print(f"⚠️ GPU not available, using CPU: {platform.processor()}")
        print(f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical")
    return device

device = get_optimal_device()

# Load YOLO models with optimizations
try:
    weld_model = YOLO(WELD_MODEL_PATH)
    weld_model.to(device)
    weld_model.fuse()
    print(f"Weld model loaded and optimized on {device}")
except Exception as e:
    print(f"Error loading weld model: {e}")
    weld_model = None

try:
    coating_model = YOLO(COATING_MODEL_PATH)
    coating_model.to(device)
    coating_model.fuse()
    print(f"Coating model loaded and optimized on {device}")
except Exception as e:
    print(f"Error loading coating model: {e}")
    coating_model = None

try:
    flange_model = YOLO(FLANGE_MODEL_PATH)
    flange_model.to(device)
    flange_model.fuse()
    print(f"Flange model loaded and optimized on {device}")
except Exception as e:
    print(f"Error loading flange model: {e}")
    flange_model = None

# Load flange specifications CSV
flange_specifications = None
try:
    if os.path.exists(FLANGE_CSV_PATH):
        flange_specifications = pd.read_csv(FLANGE_CSV_PATH)
        # Clean column names - remove spaces and standardize
        flange_specifications.columns = flange_specifications.columns.str.strip()
        print(f"Flange specifications loaded: {len(flange_specifications)} records")
        print(f"Columns: {list(flange_specifications.columns)}")
    else:
        print(f"Flange CSV not found at: {FLANGE_CSV_PATH}")
except Exception as e:
    print(f"Error loading flange specifications: {e}")

# Global storage for detection results
detection_results = []
session_id = None
video_writer = None
recording_active = False
live_recording_active = False
live_video_writer = None
live_recording_path = None


# Add memory monitoring
def check_memory_usage():
    """Check current memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    return memory_mb

def cleanup_memory():
    """Force garbage collection and clear caches"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


import math

def clean_json_data(obj):
    """Replace NaN and inf values with None for JSON serialization"""
    if isinstance(obj, dict):
        return {k: clean_json_data(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json_data(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif hasattr(obj, 'item'):  # numpy types
        val = obj.item()
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        return val
    return obj

class FlangeAnalyzer:
    def __init__(self, model, specifications_df):
        self.model = model
        self.specifications = specifications_df
        
        # ArUco marker specifications (USER CONFIRMED: 4.8 CM = 48 MM)
        self.MARKER_REAL_CM = 4.8
        self.MARKER_REAL_MM = 48.0
        self.CM_TO_INCH = 0.393701
        self.MM_TO_INCH = 0.0393701
        
        # Tolerances
        self.TOLERANCE_INCHES = 0.125
        self.TOLERANCE_MM = 3.175
        
        # Camera calibration
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibration_active = False
        self.calibration_file = 'camera_calibration.json'

        # Detection thresholds
        self.MIN_CONTOUR_AREA = 100
        self.MAX_CONTOUR_AREA = 500000
        self.CIRCULARITY_THRESHOLD = 0.7

        # ADD THESE RANSAC PARAMETERS
        self.RANSAC_ITERATIONS = 100
        self.RANSAC_INLIER_THRESHOLD = 5.0

        # NEW: Distance-based scaling calibration
        self.distance_calibration_profile = {
            300: 1.15,   # Close range: 15% correction
            500: 1.08,
            750: 1.00,   # Reference distance
            1000: 0.93,
            1500: 0.85   # Far range: -15% correction
        }
        
        # NEW: Adaptive measurement confidence
        self.measurement_confidence_zones = {
            'optimal': (400, 1000),    # Best accuracy zone
            'acceptable': (300, 1500),  # Acceptable with correction
            'poor': (0, 300, 1500, 3000)  # High uncertainty
        }
        
        # Homography for perspective correction
        self.perspective_matrix = None
        self.reprojection_error = None
        
        # NEW: Angle guidance thresholds
        self.PERPENDICULAR_ANGLE_THRESHOLD = 15  # degrees from 90°
        self.OPTIMAL_ANGLE_THRESHOLD = 5  # degrees for "perfect"
        
        # Load saved calibration if exists
        self.load_calibration()
        
        # ✅ STEP 1A: Distance correction profile
        self.distance_correction_profile = {

            250: 1.15,   # reduced from 1.35
            300: 1.10,
            350: 1.06,
            390: 1.04,
            400: 1.02,   # smoother transition near real distance range
            450: 1.00,
            500: 1.00,   # baseline
            550: 0.99,
            600: 0.98,
            700: 0.97,
            800: 0.96,
            900: 0.95,
            1000: 0.93,
            1200: 0.91,
            1500: 0.88,
            2000: 0.85
        }
        
        # ✅ STEP 1B: Measurement confidence zones
        self.confidence_zones = {
            'excellent': (300, 600),
            'good': (200,800),
            'fair': (150,1000),
            'acceptable': (100,1200),
            'poor': (50,1500)
        }
        
        # ✅ STEP 1C: Relaxed validation thresholds for field use
        self.field_mode = True  # Set to True for field deployment
        
        if self.field_mode:
            self.MIN_DISTANCE_MM = 200
            self.MAX_DISTANCE_MM = 1500
            self.MIN_MARKER_AREA = 2000
            self.MAX_TILT_ANGLE = 25
        else:
            self.MIN_DISTANCE_MM = 400
            self.MAX_DISTANCE_MM = 1000
            self.MIN_MARKER_AREA = 3000
            self.MAX_TILT_ANGLE = 20

        
    # ==================== CAMERA CALIBRATION ====================
    
    def save_calibration(self):
        """Save camera calibration to file"""
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            calibration_data = {
                'camera_matrix': self.camera_matrix.tolist(),
                'dist_coeffs': self.dist_coeffs.tolist(),
                'calibration_active': self.calibration_active
            }
            with open(self.calibration_file, 'w') as f:
                json.dump(calibration_data, f)
            print(f"✅ Camera calibration saved to {self.calibration_file}")
            
    def load_calibration(self):
        """Load camera calibration from file"""
        if os.path.exists(self.calibration_file):
            try:
                with open(self.calibration_file, 'r') as f:
                    data = json.load(f)
                self.camera_matrix = np.array(data['camera_matrix'], dtype=np.float32)
                self.dist_coeffs = np.array(data['dist_coeffs'], dtype=np.float32)
                self.calibration_active = data['calibration_active']
                print("✅ Camera calibration loaded from file")
                return True
            except Exception as e:
                print(f"❌ Error loading calibration: {e}")
        return False
    
    def calibrate_camera_from_checkerboard(self, images, checkerboard_size=(9, 6), square_size_mm=25.0):
        """
        Calibrate camera using checkerboard pattern images
        
        Args:
            images: List of BGR images containing checkerboard
            checkerboard_size: (width, height) of internal corners
            square_size_mm: Size of checkerboard squares in millimeters
        
        Returns:
            bool: True if calibration successful
        """
        try:
            # Prepare object points (3D points in real world space)
            objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
            objp *= square_size_mm  # Scale to real-world units
            
            objpoints = []  # 3D points in real world space
            imgpoints = []  # 2D points in image plane
            
            print(f"🔍 Processing {len(images)} calibration images...")
            
            for idx, img in enumerate(images):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Find checkerboard corners
                ret, corners = cv2.findChessboardCorners(
                    gray, checkerboard_size,
                    cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
                )
                
                if ret:
                    objpoints.append(objp)
                    
                    # Refine corner positions to sub-pixel accuracy
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    imgpoints.append(corners_refined)
                    
                    print(f"  ✅ Image {idx + 1}: Checkerboard found")
                else:
                    print(f"  ❌ Image {idx + 1}: Checkerboard not found")
            
            if len(objpoints) < 10:
                print(f"❌ Insufficient valid images ({len(objpoints)}/10 minimum)")
                return False
            
            # Perform camera calibration
            print("📐 Computing camera calibration...")
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None
            )
            
            if ret:
                self.camera_matrix = camera_matrix
                self.dist_coeffs = dist_coeffs
                self.calibration_active = True
                
                # Calculate reprojection error
                mean_error = 0
                for i in range(len(objpoints)):
                    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], 
                                                     camera_matrix, dist_coeffs)
                    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                    mean_error += error
                
                mean_error /= len(objpoints)
                
                print("✅ Camera calibration successful!")
                print(f"   Reprojection error: {mean_error:.4f} pixels")
                print(f"   Camera Matrix:\n{camera_matrix}")
                print(f"   Distortion Coefficients: {dist_coeffs.ravel()}")
                
                # Save calibration
                self.save_calibration()
                return True
            
            print("❌ Camera calibration failed")
            return False
            
        except Exception as e:
            print(f"❌ Camera calibration error: {e}")
            import traceback
            traceback.print_exc()
            return False
    

    def auto_calibrate_from_aruco(self, image, marker_corners):
        """
        Advanced dynamic auto-calibration using ArUco marker.
        More accurate: compensates for FOV, tilt, and applies stability smoothing.
        """
        try:
            if marker_corners is None or len(marker_corners) != 4:
                print("⚠️ Invalid marker corners for auto-calibration.")
                return False

            h, w = image.shape[:2]
            cx, cy = w / 2.0, h / 2.0

            # --- 🧩 Compute marker metrics ---
            sides = [np.linalg.norm(marker_corners[i] - marker_corners[(i + 1) % 4]) for i in range(4)]
            marker_size_px = np.mean(sides)
            marker_area_px = cv2.contourArea(marker_corners.astype(np.float32))

            # --- 🧠 Automatic FOV-based focal guess ---
            CAMERA_FOV_DEG = 60.0  # default webcam FOV
            focal_guess_px = (w / 2) / np.tan(np.radians(CAMERA_FOV_DEG / 2))

            # --- 🔍 Estimate marker tilt (if not perfectly perpendicular) ---
            # Compute aspect ratio of detected marker bounding box
            edge_ratio = max(sides) / min(sides)
            tilt_angle_deg = np.degrees(np.arccos(1 / edge_ratio)) if edge_ratio > 1.0 else 0.0

            # Apply cosine correction to compensate tilt-induced shrinkage
            tilt_correction_factor = 1.0 / np.cos(np.radians(tilt_angle_deg))
            marker_size_px_corrected = marker_size_px * tilt_correction_factor

            # --- 📏 Estimate real-world distance (pinhole geometry) ---
            estimated_distance_mm = (self.MARKER_REAL_MM * focal_guess_px) / marker_size_px_corrected

            # --- 🔧 Compute refined focal length ---
            focal_length = (marker_size_px_corrected * estimated_distance_mm) / self.MARKER_REAL_MM

            # --- 🧮 Stabilize (smooth calibration over time) ---
            if hasattr(self, "_last_focal_length"):
                focal_length = 0.7 * self._last_focal_length + 0.3 * focal_length
            self._last_focal_length = focal_length

            # --- 🧩 Build camera matrix ---
            self.camera_matrix = np.array([
                [focal_length, 0, cx],
                [0, focal_length, cy],
                [0, 0, 1]
            ], dtype=np.float32)

            # --- Minimal distortion (flat lens assumption) ---
            self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)

            self.calibration_active = True

            # --- 🧾 Debug info ---
            print("✅ Auto-calibration from ArUco completed (enhanced)")
            print(f"   ▪ Estimated distance: {estimated_distance_mm:.1f} mm")
            print(f"   ▪ Focal length: {focal_length:.2f} px")
            print(f"   ▪ Marker size (px): {marker_size_px:.1f}")
            print(f"   ▪ Tilt correction: {tilt_correction_factor:.3f} (tilt: {tilt_angle_deg:.1f}°)")
            print(f"   ▪ FOV: {CAMERA_FOV_DEG:.1f}°")
            print(f"   ▪ Marker area: {marker_area_px:.1f} px²")
            print("------------------------------------------------------")

            return True

        except Exception as e:
            print(f"❌ Auto-calibration error: {e}")
            import traceback
            traceback.print_exc()
            return False




    
    def undistort_image(self, image):
        """Apply lens distortion correction to entire image"""
        if not self.calibration_active or self.camera_matrix is None:
            return image
        
        try:
            h, w = image.shape[:2]
            
            # Get optimal new camera matrix
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
            )
            
            # Undistort image
            undistorted = cv2.undistort(
                image, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix
            )
            
            # Crop to ROI if needed
            x, y, w, h = roi
            if w > 0 and h > 0:
                undistorted = undistorted[y:y+h, x:x+w]
            
            print("✅ Lens distortion correction applied")
            return undistorted
            
        except Exception as e:
            print(f"❌ Undistortion error: {e}")
            return image
    
    def undistort_points(self, points):
        """Undistort specific points for accurate measurements"""
        if not self.calibration_active or self.camera_matrix is None:
            return points
        
        try:
            points_reshaped = points.reshape(-1, 1, 2).astype(np.float32)
            
            # Undistort points
            undistorted_points = cv2.undistortPoints(
                points_reshaped, 
                self.camera_matrix, 
                self.dist_coeffs,
                None,
                self.camera_matrix  # Use same camera matrix for output
            )
            
            return undistorted_points.reshape(-1, 2)
            
        except Exception as e:
            print(f"❌ Point undistortion error: {e}")
            return points


    def calculate_camera_angle(self, marker_corners):
        """
        Calculate camera tilt angle from ArUco marker perspective
        Returns: (tilt_x, tilt_y, is_perpendicular, angle_status)
        """
        try:
            # Get the four corners of the marker
            corners = marker_corners.reshape(4, 2)
            
            # Calculate distances between corners
            top_width = np.linalg.norm(corners[1] - corners[0])
            bottom_width = np.linalg.norm(corners[2] - corners[3])
            left_height = np.linalg.norm(corners[3] - corners[0])
            right_height = np.linalg.norm(corners[2] - corners[1])
            
            # Calculate aspect ratios
            width_ratio = abs(top_width - bottom_width) / max(top_width, bottom_width)
            height_ratio = abs(left_height - right_height) / max(left_height, right_height)
            
            # Calculate average dimensions
            avg_width = (top_width + bottom_width) / 2
            avg_height = (left_height + right_height) / 2
            
            # Square aspect ratio check (should be close to 1.0 for perpendicular view)
            square_ratio = avg_width / avg_height if avg_height > 0 else 1.0
            square_deviation = abs(1.0 - square_ratio)
            
            # Calculate tilt angles
            # X-axis tilt (left-right lean)
            tilt_x = math.degrees(math.atan(width_ratio)) if width_ratio > 0 else 0
            
            # Y-axis tilt (top-bottom lean)
            tilt_y = math.degrees(math.atan(height_ratio)) if height_ratio > 0 else 0
            
            # Overall tilt magnitude
            total_tilt = math.sqrt(tilt_x**2 + tilt_y**2 + (square_deviation * 30)**2)
            
            # Determine status
            if total_tilt < self.OPTIMAL_ANGLE_THRESHOLD:
                status = "perfect"
                is_perpendicular = True
            elif total_tilt < self.PERPENDICULAR_ANGLE_THRESHOLD:
                status = "good"
                is_perpendicular = True
            else:
                status = "adjust"
                is_perpendicular = False
            
            return {
                'tilt_x': float(tilt_x),
                'tilt_y': float(tilt_y),
                'total_tilt': float(total_tilt),
                'is_perpendicular': is_perpendicular,
                'status': status,
                'square_ratio': float(square_ratio),
                'width_ratio': float(width_ratio),
                'height_ratio': float(height_ratio)
            }
            
        except Exception as e:
            print(f"❌ Angle calculation error: {e}")
            return {
                'tilt_x': 0,
                'tilt_y': 0,
                'total_tilt': 0,
                'is_perpendicular': False,
                'status': 'unknown',
                'square_ratio': 1.0,
                'width_ratio': 0,
                'height_ratio': 0
            }
    
    # ==================== ARUCO MARKER DETECTION ====================
    
    def detect_aruco_markers(self, gray, enable_auto_calibration=True):
        """
        Enhanced ArUco detection with multi-marker support
        
        Returns:
            tuple: (px_to_mm, all_marker_corners, avg_marker_angle)
        """
        if not hasattr(cv2, 'aruco'):
            print("❌ ArUco module not available")
            return None, None, 0
        
        try:
            # Image enhancement
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Sharpen
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            images_to_try = [gray, enhanced, sharpened]
            
            # Try multiple ArUco dictionaries
            aruco_dicts = [
                cv2.aruco.DICT_4X4_50, cv2.aruco.DICT_4X4_100, cv2.aruco.DICT_4X4_250,
                cv2.aruco.DICT_5X5_50, cv2.aruco.DICT_5X5_100, cv2.aruco.DICT_5X5_250,
                cv2.aruco.DICT_6X6_50, cv2.aruco.DICT_6X6_100, cv2.aruco.DICT_6X6_250,
            ]
            
            all_markers = []
            
            for img in images_to_try:
                for dict_id in aruco_dicts:
                    try:
                        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
                        
                        if hasattr(cv2.aruco, "DetectorParameters_create"):
                            aruco_params = cv2.aruco.DetectorParameters_create()
                        else:
                            aruco_params = cv2.aruco.DetectorParameters()
                        
                        # Detect markers
                        corners, ids, rejected = cv2.aruco.detectMarkers(
                            img, aruco_dict, parameters=aruco_params
                        )
                        
                        if ids is not None and len(corners) > 0:
                            for corner in corners:
                                marker_corners = corner[0]
                                
                                # Undistort marker corners if calibration available
                                if self.calibration_active:
                                    marker_corners = self.undistort_points(marker_corners)
                                
                                all_markers.append(marker_corners)
                            
                            # Auto-calibrate from first marker if needed
                            if enable_auto_calibration and not self.calibration_active:
                                full_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img
                                self.auto_calibrate_from_aruco(full_img, all_markers[0])
                            
                            # If we found markers, break
                            if len(all_markers) > 0:
                                break
                                
                    except Exception as e:
                        continue
                
                if len(all_markers) > 0:
                    break
            
            if len(all_markers) == 0:
                print("❌ No ArUco markers detected")
                return None, None, 0
            
            # Calculate scale factor from all detected markers
            scale_factors = []
            angles = []
            
            for marker_corners in all_markers:
                # Calculate all 6 distances (4 sides + 2 diagonals)
                side1 = np.linalg.norm(marker_corners[0] - marker_corners[1])
                side2 = np.linalg.norm(marker_corners[1] - marker_corners[2])
                side3 = np.linalg.norm(marker_corners[2] - marker_corners[3])
                side4 = np.linalg.norm(marker_corners[3] - marker_corners[0])
                diag1 = np.linalg.norm(marker_corners[0] - marker_corners[2])
                diag2 = np.linalg.norm(marker_corners[1] - marker_corners[3])
                
                # Average size (normalize diagonals by sqrt(2))
                avg_size_px = (side1 + side2 + side3 + side4 + 
                              (diag1 / np.sqrt(2)) + (diag2 / np.sqrt(2))) / 6
                
                # Calculate scale factor (mm per pixel)
                px_to_mm = self.MARKER_REAL_MM / avg_size_px
                scale_factors.append(px_to_mm)
                
                # Calculate marker angle
                dx = marker_corners[1][0] - marker_corners[0][0]
                dy = marker_corners[1][1] - marker_corners[0][1]
                angle = degrees(atan2(dy, dx))
                angles.append(angle)
            
            # Use median values for robustness
            final_px_to_mm = float(np.median(scale_factors))
            avg_angle = float(np.median(angles))
            
            print(f"✅ Detected {len(all_markers)} ArUco marker(s)")
            print(f"   Scale: {final_px_to_mm:.6f} mm/px")
            print(f"   Angle: {avg_angle:.2f}°")
            print(f"   Calibration: {'ACTIVE' if self.calibration_active else 'INACTIVE'}")
            
            return final_px_to_mm, all_markers, avg_angle
            
        except Exception as e:
            print(f"❌ ArUco detection error: {e}")
            import traceback
            traceback.print_exc()
            return None, None, 0
    
    def calculate_perspective_matrix(self, marker_corners):
        """
        Calculate homography matrix using ArUco marker corners
        
        Args:
            marker_corners: Array of 4 pixel coordinates [(x0, y0), ..., (x3, y3)]
        
        Returns:
            bool: True if successful
        """
        try:
            # Target points: orthogonal square in MM units
            marker_mm = self.MARKER_REAL_MM
            target_pts = np.array([
                [0, 0],
                [marker_mm, 0],
                [marker_mm, marker_mm],
                [0, marker_mm]
            ], dtype=np.float32)
            
            # Source points (distorted pixel coordinates)
            source_pts = marker_corners.astype(np.float32)
            
            # Calculate homography with RANSAC for robustness
            M, mask = cv2.findHomography(source_pts, target_pts, cv2.RANSAC, 5.0)
            
            if M is None:
                print("❌ Failed to compute homography matrix")
                return False
            
            self.perspective_matrix = M
            
            # Calculate reprojection error
            transformed_pts = cv2.perspectiveTransform(
                source_pts.reshape(-1, 1, 2), M
            ).reshape(-1, 2)
            
            self.reprojection_error = np.mean(np.linalg.norm(transformed_pts - target_pts, axis=1))
            
            print(f"✅ Homography matrix calculated")
            print(f"   Reprojection error: {self.reprojection_error:.4f} mm")
            
            if self.reprojection_error > 1.0:
                print("   ⚠️  WARNING: High reprojection error - measurements may be unreliable")
            
            return True
            
        except Exception as e:
            print(f"❌ Homography calculation error: {e}")
            self.perspective_matrix = None
            self.reprojection_error = None
            return False


    def estimate_distance_from_marker(self, marker_corners):
        """
        Estimate camera-to-flange distance using PnP algorithm
        
        Args:
            marker_corners: Array of 4 detected marker corners [[x,y], ...]
        
        Returns:
            dict: {
                'distance_mm': float,
                'rotation_x': float,  # Tilt around X-axis
                'rotation_y': float,  # Tilt around Y-axis
                'rotation_z': float,  # Rotation in-plane
                'quality': str,       # 'excellent', 'good', 'fair', 'poor'
                'warning': str        # User guidance
            }
        """
        if not self.calibration_active or self.camera_matrix is None:
            return {
                'distance_mm': None,
                'quality': 'unknown',
                'warning': 'Camera calibration required for distance estimation'
            }
        
        try:
            # Define 3D coordinates of ArUco marker corners (in millimeters)
            # Origin at top-left, marker lies in XY plane (Z=0)
            marker_3d = np.array([
                [0,                 0,                 0],  # Top-left
                [self.MARKER_REAL_MM, 0,                 0],  # Top-right
                [self.MARKER_REAL_MM, self.MARKER_REAL_MM, 0],  # Bottom-right
                [0,                 self.MARKER_REAL_MM, 0]   # Bottom-left
            ], dtype=np.float32)
            
            # 2D image points (detected corners)
            marker_2d = marker_corners.reshape(-1, 2).astype(np.float32)
            
            # Solve Perspective-n-Point problem
            success, rvec, tvec = cv2.solvePnP(
                marker_3d,
                marker_2d,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success:
                return {
                    'distance_mm': None,
                    'quality': 'poor',
                    'warning': 'Failed to estimate distance - marker detection unstable'
                }
            
            # Extract distance (Z-axis translation)
            distance_mm = float(tvec[2][0])
            
            # Convert rotation vector to Euler angles for interpretability
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            
            # Extract Euler angles (in degrees)
            # Using convention: Rotation order ZYX
            sy = np.sqrt(rotation_matrix[0,0]**2 + rotation_matrix[1,0]**2)
            
            if sy > 1e-6:  # Not a singularity
                x_rot = np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2])
                y_rot = np.arctan2(-rotation_matrix[2,0], sy)
                z_rot = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])
            else:  # Gimbal lock case
                x_rot = np.arctan2(-rotation_matrix[1,2], rotation_matrix[1,1])
                y_rot = np.arctan2(-rotation_matrix[2,0], sy)
                z_rot = 0
            
            # Convert to degrees
            rotation_x = np.degrees(x_rot)
            rotation_y = np.degrees(y_rot)
            rotation_z = np.degrees(z_rot)
            
            # Calculate reprojection error for quality assessment
            projected_points, _ = cv2.projectPoints(
                marker_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs
            )
            reprojection_error = np.mean(np.linalg.norm(
                marker_2d - projected_points.reshape(-1, 2), axis=1
            ))
            
            # Determine measurement quality based on distance and angle
            quality, warning = self._assess_measurement_quality(
                distance_mm, rotation_x, rotation_y, reprojection_error
            )
            
            print(f"📏 Distance Estimation:")
            print(f"   Distance: {distance_mm:.1f}mm")
            print(f"   Rotation: X={rotation_x:.1f}°, Y={rotation_y:.1f}°, Z={rotation_z:.1f}°")
            print(f"   Reprojection Error: {reprojection_error:.2f}px")
            print(f"   Quality: {quality}")
            
            return {
                'distance_mm': distance_mm,
                'rotation_x': float(rotation_x),
                'rotation_y': float(rotation_y),
                'rotation_z': float(rotation_z),
                'reprojection_error_px': float(reprojection_error),
                'quality': quality,
                'warning': warning
            }
            
        except Exception as e:
            print(f"❌ Distance estimation error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'distance_mm': None,
                'quality': 'error',
                'warning': f'Distance estimation failed: {str(e)}'
            }

    def _assess_measurement_quality(self, distance_mm, rot_x, rot_y, reproj_error):
        """
        Assess measurement quality based on geometric parameters
        
        Returns:
            tuple: (quality_level, user_warning)
        """
        warnings = []
        
        # Optimal distance range: 400-1000mm for most flanges
        if distance_mm < 150:
            warnings.append("📍 Too close - move back 10-20cm")
            quality = 'poor'
        elif distance_mm< 300:
            warnings.append("close_range use correction factor")
        elif distance_mm > 1500:
            warnings.append("📍 Too far - move closer for precision")
            quality = 'fair'
        elif 400 <= distance_mm <= 1000:
            quality = 'excellent'
        else:
            quality = 'good'
        
        # Check camera tilt (should be perpendicular)
        total_tilt = np.sqrt(rot_x**2 + rot_y**2)
        
        if total_tilt > 30:
            warnings.append("📐 Camera tilted >30° - adjust angle")
            quality = 'poor' if quality != 'poor' else quality
        elif total_tilt > 15:
            warnings.append("📐 Slight tilt detected - adjust for best accuracy")
            quality = 'fair' if quality == 'excellent' else quality
        
        # Check reprojection error (should be <2 pixels)
        if reproj_error > 3.0:
            warnings.append("⚠️ High detection error - ensure marker is visible")
            quality = 'poor'
        elif reproj_error > 1.5:
            quality = 'fair' if quality == 'excellent' else quality
        
        # Combine warnings
        warning_text = " | ".join(warnings) if warnings else "✅ Optimal measurement conditions"
        
        return quality, warning_text


    def get_distance_correction_factor(self, distance_mm):
        """
        Calculate adaptive correction factor using interpolation
        
        Args:
            distance_mm: Camera distance in millimeters
            
        Returns:
            float: Correction factor (1.0 = no correction)
        """
        if distance_mm is None:
            return 1.0
        
        # Get sorted distance points
        distances = sorted(self.distance_correction_profile.keys())
        factors = [self.distance_correction_profile[d] for d in distances]
        
        # Handle edge cases
        if distance_mm <= distances[0]:
            return factors[0]
        if distance_mm >= distances[-1]:
            return factors[-1]
        
        # Linear interpolation between two nearest points
        for i in range(len(distances) - 1):
            if distances[i] <= distance_mm <= distances[i + 1]:
                d1, d2 = distances[i], distances[i + 1]
                f1, f2 = factors[i], factors[i + 1]
                
                # Interpolation formula: factor = f1 + (distance - d1) * (f2 - f1) / (d2 - d1)
                t = (distance_mm - d1) / (d2 - d1)
                correction_factor = f1 + t * (f2 - f1)
                
                return correction_factor
        
        return 1.0

    def get_measurement_confidence(self, distance_mm):
        """
        Determine measurement confidence and expected error
        
        Returns:
            tuple: (confidence_level: str, expected_error_mm: float, color: str)
        """
        if distance_mm is None:
            return 'unknown', None, '#6b7280'
        
        if self.confidence_zones['excellent'][0] <= distance_mm <= self.confidence_zones['excellent'][1]:
            return 'excellent', 2.0, '#10b981'
        elif self.confidence_zones['good'][0] <= distance_mm <= self.confidence_zones['good'][1]:
            return 'good', 5.0, '#22c55e'
        elif self.confidence_zones['fair'][0] <= distance_mm <= self.confidence_zones['fair'][1]:
            return 'fair', 10.0, '#f59e0b'
        elif self.confidence_zones['acceptable'][0] <= distance_mm <= self.confidence_zones['acceptable'][1]:
            return 'acceptable', 15.0, '#fb923c'
        else:
            return 'poor', 25.0, '#ef4444'

    def apply_distance_compensation(self, measurements, distance_mm):
        """
        Apply distance-based corrections to all dimensional measurements
        
        Args:
            measurements: Dictionary of measurements
            distance_mm: Camera distance in millimeters
            
        Returns:
            dict: Corrected measurements with metadata
        """
        if distance_mm is None:
            measurements['distance_correction_applied'] = False
            return measurements
        
        # Get correction factor
        correction_factor = self.get_distance_correction_factor(distance_mm)
        
        # Get confidence level
        confidence_level, expected_error, confidence_color = self.get_measurement_confidence(distance_mm)
        
        # Store original values
        original_values = {
            'od': float(measurements.get('outer_dia', 0)),
            'id': float(measurements.get('inner_dia', 0)),
            'bolt_dia': float(measurements.get('bolt_dia', 0)),
            'pcd': float(measurements.get('pcd', 0))
        }
        
        # Apply correction to OD
        if measurements.get('outer_dia', 0) > 0:
            measurements['outer_dia'] = float(measurements['outer_dia']) * correction_factor
            measurements['od_mm'] = measurements['outer_dia']
            measurements['od_inch'] = measurements['outer_dia'] * 0.0393701
        
        # Apply correction to ID
        if measurements.get('inner_dia', 0) > 0:
            measurements['inner_dia'] = float(measurements['inner_dia']) * correction_factor
            measurements['id_mm'] = measurements['inner_dia']
            measurements['id_inch'] = measurements['inner_dia'] * 0.0393701
        
        # Apply correction to Bolt Diameter
        if measurements.get('bolt_dia', 0) > 0:
            measurements['bolt_dia'] = float(measurements['bolt_dia']) * correction_factor
            measurements['bolt_dia_mm'] = measurements['bolt_dia']
            measurements['bolt_dia_inch'] = measurements['bolt_dia'] * 0.0393701
        
        # Apply correction to PCD
        if measurements.get('pcd', 0) > 0:
            measurements['pcd'] = float(measurements['pcd']) * correction_factor
            measurements['pcd_mm'] = measurements['pcd']
            measurements['pcd_inch'] = measurements['pcd'] * 0.0393701
        
        # Add correction metadata
        measurements['distance_correction'] = {
            'applied': True,
            'factor': float(correction_factor),
            'distance_mm': float(distance_mm),
            'confidence_level': confidence_level,
            'confidence_color': confidence_color,
            'expected_error_mm': expected_error,
            'original_values': original_values,
            'corrected_values': {
                'od': float(measurements.get('outer_dia', 0)),
                'id': float(measurements.get('inner_dia', 0)),
                'bolt_dia': float(measurements.get('bolt_dia', 0)),
                'pcd': float(measurements.get('pcd', 0))
            }
        }
        
        # Console logging
        print(f"\n{'='*70}")
        print(f"🔧 AUTOMATIC DISTANCE COMPENSATION")
        print(f"{'='*70}")
        print(f"📏 Distance: {distance_mm:.1f}mm ({distance_mm/10:.1f}cm)")
        print(f"🎯 Correction Factor: {correction_factor:.4f} ({(correction_factor-1)*100:+.2f}%)")
        print(f"📊 Confidence: {confidence_level.upper()} (±{expected_error}mm)")
        print(f"")
        if original_values['od'] > 0:
            print(f"   OD: {original_values['od']:.2f} → {measurements['outer_dia']:.2f}mm")
        if original_values['id'] > 0:
            print(f"   ID: {original_values['id']:.2f} → {measurements['inner_dia']:.2f}mm")
        if original_values['bolt_dia'] > 0:
            print(f"   BD: {original_values['bolt_dia']:.2f} → {measurements['bolt_dia']:.2f}mm")
        if original_values['pcd'] > 0:
            print(f"   PCD: {original_values['pcd']:.2f} → {measurements['pcd']:.2f}mm")
        print(f"{'='*70}\n")
        
        return measurements


    
    def validate_measurement_consistency(self, distance_mm, tilt_angle, marker_area):
        """
        Enhanced validation with relaxed thresholds for field deployment
        
        Returns:
            tuple: (is_valid: bool, message: str)
        """
        warnings = []
        
        # 1. Distance range check (RELAXED for field use)
        if distance_mm is None:
            return False, "❌ Distance measurement failed"
        
        if distance_mm < self.MIN_DISTANCE_MM:
            return False, f"❌ Too close ({distance_mm:.0f}mm / {distance_mm/10:.1f}cm) - move back to {self.MIN_DISTANCE_MM/10:.0f}cm minimum"
        
        if distance_mm > self.MAX_DISTANCE_MM:
            return False, f"❌ Too far ({distance_mm:.0f}mm / {distance_mm/10:.1f}cm) - move closer to {self.MAX_DISTANCE_MM/10:.0f}cm maximum"
        
        # Add warnings for non-optimal distances
        if distance_mm < 400:
            warnings.append(f"⚠️ Close distance ({distance_mm:.0f}mm) - accuracy may be reduced")
        elif distance_mm > 1000:
            warnings.append(f"⚠️ Far distance ({distance_mm:.0f}mm) - accuracy may be reduced")
        
        # 2. Angle check (RELAXED)
        if tilt_angle is not None:
            if tilt_angle > self.MAX_TILT_ANGLE:
                return False, f"❌ Camera tilt too high: {tilt_angle:.1f}° (max {self.MAX_TILT_ANGLE}°)"
            elif tilt_angle > 15:
                warnings.append(f"⚠️ Camera tilt: {tilt_angle:.1f}° (optimal: <15°)")
        
        # 3. Marker quality check (RELAXED)
        if marker_area is not None:
            if marker_area < self.MIN_MARKER_AREA:
                return False, f"❌ Marker too small ({marker_area:.0f}px²) - move closer or use larger marker (minimum {self.MIN_MARKER_AREA}px²)"
            elif marker_area < 5000:
                warnings.append(f"⚠️ Marker area: {marker_area:.0f}px² (recommended: >5000px²)")
        
        # 4. Get confidence zone
        confidence, expected_error, _ = self.get_measurement_confidence(distance_mm)
        
        if confidence in ['excellent', 'good']:
            status_msg = f"✅ {confidence.capitalize()} conditions (±{expected_error}mm)"
        elif confidence in ['fair', 'acceptable']:
            status_msg = f"⚠️ {confidence.capitalize()} conditions (±{expected_error}mm)"
        else:
            status_msg = f"⚠️ Poor conditions (±{expected_error}mm)"
        
        # Combine messages
        if len(warnings) > 0:
            final_message = status_msg + " | " + " | ".join(warnings)
        else:
            final_message = status_msg
        
        return True, final_message


    
    def perspective_transform_points(self, points):
        """
        Transform pixel coordinates to orthogonal MM coordinates using homography
        
        Args:
            points: Array of pixel coordinates [[x1, y1], [x2, y2], ...]
        
        Returns:
            Array of transformed coordinates in MM
        """
        if self.perspective_matrix is None:
            return points
        
        try:
            points_reshaped = points.reshape(-1, 1, 2).astype(np.float32)
            transformed_points = cv2.perspectiveTransform(points_reshaped, self.perspective_matrix)
            return transformed_points.reshape(-1, 2)
        except Exception as e:
            print(f"❌ Point transformation error: {e}")
            return points
    
    # ==================== CONTOUR-BASED CIRCLE FITTING ====================

    def fit_circle_ransac(self, contour):
        """RANSAC wrapper for robust circle fitting (returns cx,cy,r) or None."""
        points = contour.reshape(-1, 2).astype(np.float32)
        n_points = len(points)
        if n_points < 6:
            return None

        best_inliers = 0
        best_params = None

        for _ in range(self.RANSAC_ITERATIONS):
            try:
                sample_idx = random.sample(range(n_points), 3)
            except ValueError:
                continue
            p0, p1, p2 = points[sample_idx]

            A = np.array([
                [p0[0], p0[1], 1.0],
                [p1[0], p1[1], 1.0],
                [p2[0], p2[1], 1.0]
            ])
            B = np.array([
                -(p0[0]**2 + p0[1]**2),
                -(p1[0]**2 + p1[1]**2),
                -(p2[0]**2 + p2[1]**2)
            ])
            try:
                sol = np.linalg.lstsq(A, B, rcond=None)[0]
                cx = -0.5 * sol[0]
                cy = -0.5 * sol[1]
                r = math.sqrt(cx**2 + cy**2 - sol[2])
            except Exception:
                (cx, cy), r = cv2.minEnclosingCircle(np.array([p0, p1, p2]))

            distances = np.abs(np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2) - r)
            inliers = np.sum(distances < self.RANSAC_INLIER_THRESHOLD)

            if inliers > best_inliers:
                best_inliers = int(inliers)
                best_params = (float(cx), float(cy), float(r))

        if best_params is None:
            return None

        inlier_ratio = best_inliers / float(n_points)
        if inlier_ratio < 0.6:
            return None

        distances = np.abs(np.sqrt((points[:, 0] - best_params[0])**2 + (points[:, 1] - best_params[1])**2) - best_params[2])
        inlier_points = points[distances < self.RANSAC_INLIER_THRESHOLD]
        
        if len(inlier_points) < 6:
            refined = self.fit_circle_least_squares(points)
        else:
            refined = self.fit_circle_least_squares(inlier_points)

        if refined:
            cx, cy, r = refined
            print(f"✅ RANSAC fit: {best_inliers}/{n_points} inliers, ratio={inlier_ratio:.2f}")
            return float(cx), float(cy), float(r)

        return None


    def fit_circle_to_contour(self, contour):
        """Use RANSAC + LS with better fallbacks"""
        # Try RANSAC first
        ransac_fit = self.fit_circle_ransac(contour)
        if ransac_fit is not None:
            cx, cy, r = ransac_fit
            # Validate the fit
            if r > 10 and r < 5000:  # Reasonable radius range
                return ransac_fit

        # Try least squares
        pts = contour.reshape(-1, 2).astype(float)
        ls_fit = self.fit_circle_least_squares(pts)
        if ls_fit is not None:
            cx, cy, r = ls_fit
            if r > 10 and r < 5000:
                return ls_fit

        # Final fallback
        try:
            (cx, cy), r = cv2.minEnclosingCircle(contour.astype(np.float32))
            return float(cx), float(cy), float(r)
        except Exception as e:
            print(f"⚠️ All circle fitting methods failed: {e}")
            return None



    def detect_circular_features(self, image, bbox):
        """
        Precise circular feature detection for ID and BOLT HOLES ONLY
        (OD should use bbox dimensions directly)
        """
        try:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Moderate padding for ID/bolts
            padding = 15
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.shape[1], x2 + padding)
            y2 = min(image.shape[0], y2 + padding)
            
            roi = image[y1:y2, x1:x2]
            
            if roi.size == 0:
                return None
            
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Preprocessing
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            
            # Edge detection
            v = np.median(enhanced)
            sigma = 0.33
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))
            edges = cv2.Canny(enhanced, lower, upper, apertureSize=5, L2gradient=True)
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                print("⚠️ No contours found, using bbox fallback")
                return self._fallback_measurement(x1, y1, x2, y2)

            # Find best circular contour
            best_contour = None
            best_score = 0
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                
                # Reasonable area range for ID/bolts
                if area < 500 or area > 200000:
                    continue
                    
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                    
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # Strict circularity check
                if circularity < 0.75:
                    continue
                
                # Check aspect ratio with ellipse fit
                if len(cnt) >= 5:
                    try:
                        ellipse = cv2.fitEllipse(cnt)
                        (_, _), (MA, ma), _ = ellipse
                        if MA > 0:
                            aspect_ratio = ma / MA
                            if 0.80 <= aspect_ratio <= 1.20:
                                score = circularity * aspect_ratio
                                if score > best_score:
                                    best_score = score
                                    best_contour = cnt
                    except:
                        continue
            
            if best_contour is None:
                print("⚠️ No valid circular contour, using bbox fallback")
                return self._fallback_measurement(x1, y1, x2, y2)
            
            # Measure the best contour
            leftmost = tuple(best_contour[best_contour[:, :, 0].argmin()][0])
            rightmost = tuple(best_contour[best_contour[:, :, 0].argmax()][0])
            topmost = tuple(best_contour[best_contour[:, :, 1].argmin()][0])
            bottommost = tuple(best_contour[best_contour[:, :, 1].argmax()][0])
            
            horizontal_diameter = np.linalg.norm(np.array(rightmost) - np.array(leftmost))
            vertical_diameter = np.linalg.norm(np.array(bottommost) - np.array(topmost))
            
            final_diameter = (horizontal_diameter + vertical_diameter) / 2
            
            # Center calculation
            cx_roi = (leftmost[0] + rightmost[0]) / 2
            cy_roi = (topmost[1] + bottommost[1]) / 2
            
            M = cv2.moments(best_contour)
            if M["m00"] != 0:
                cx_moments = M["m10"] / M["m00"]
                cy_moments = M["m01"] / M["m00"]
                cx_roi = (cx_roi + cx_moments) / 2
                cy_roi = (cy_roi + cy_moments) / 2
            
            # Apply lens distortion correction if available
            if self.calibration_active and self.camera_matrix is not None:
                center_undistorted = self.undistort_points(np.array([[cx_roi, cy_roi]]))
                cx_roi, cy_roi = center_undistorted[0]
            
            cx = float(cx_roi + x1)
            cy = float(cy_roi + y1)
            
            print(f"✅ Circle detected: {final_diameter:.2f}px")
            
            return (cx, cy, final_diameter)
        
        except Exception as e:
            print(f"❌ Detection error: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_measurement(x1, y1, x2, y2)


    def _fallback_measurement(self, x1, y1, x2, y2):
        """Fallback when no valid contours found"""
        width = x2 - x1
        height = y2 - y1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        diameter_px = (width + height) / 2
        print(f"⚠️ Using fallback measurement: {diameter_px:.2f}px")
        return (cx, cy, diameter_px)

    
    # ==================== MEASUREMENT EXTRACTION ====================

    def extract_measurements_from_segmentation(self, image, preferred_unit='inches'):
        """
        Extract flange measurements using YOLO + contour-based precision

        Args:
            image: BGR image
            preferred_unit: 'inches' or 'mm'

        Returns:
            tuple: (measurements_dict, annotated_image)
        """
        if self.model is None:
            return None, image

        # Step 1: Undistort image if calibration available
        working_image = self.undistort_image(image.copy())

        # Step 2: Run YOLO detection
        results = self.model.predict(working_image, conf=0.35, verbose=False)

        measurements = {
            'outer_dia': 0,
            'inner_dia': 0,
            'bolt_dia': 0,
            'bolt_count': 0,
            'pcd': 0,
            'unit': preferred_unit,
            'od_px': 0,
            'id_px': 0,
            'bolt_dia_px': 0
        }

        bolt_centers_px = []
        bolt_centers_real = []
        bolt_diameters_px = []
        bolt_diameters_real = []

        annotated_image = working_image.copy()

        # Step 3: Detect ArUco markers
        gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
        px_to_mm, marker_corners_list, marker_angle = self.detect_aruco_markers(
            gray, enable_auto_calibration=True
        )

        # ✅ DISTANCE ESTIMATION & VALIDATION
        distance_info = None
        validation_passed = True
        validation_message = "No scale reference"

        if marker_corners_list and len(marker_corners_list) > 0:
            # Get distance information
            distance_info = self.estimate_distance_from_marker(marker_corners_list[0])
            
            # Calculate marker area
            marker_area = cv2.contourArea(marker_corners_list[0].astype(np.float32))
            
            # Get angle information
            angle_info = self.calculate_camera_angle(marker_corners_list[0])
            
            # ✅ VALIDATE MEASUREMENT CONDITIONS
            if distance_info and distance_info.get('distance_mm'):
                validation_passed, validation_message = self.validate_measurement_consistency(
                    distance_mm=distance_info['distance_mm'],
                    tilt_angle=angle_info.get('total_tilt', 0),
                    marker_area=marker_area
                )
                
                print(f"\n{'='*60}")
                print(f"📊 MEASUREMENT VALIDATION:")
                print(f"   Distance: {distance_info['distance_mm']:.1f}mm ({distance_info['distance_mm']/10:.1f}cm)")
                print(f"   Tilt: {angle_info.get('total_tilt', 0):.1f}°")
                print(f"   Marker Area: {marker_area:.0f}px²")
                print(f"   Status: {'✅ PASSED' if validation_passed else '❌ FAILED'}")
                print(f"   Message: {validation_message}")
                print(f"{'='*60}\n")
                
                # ✅ REJECT MEASUREMENTS IF VALIDATION FAILS
                if not validation_passed:
                    measurements['validation_failed'] = True
                    measurements['validation_message'] = validation_message
                    measurements['distance_info'] = distance_info
                    return measurements, annotated_image
                
                # ✅ APPLY DISTANCE COMPENSATION (NEW!)
                # This happens AFTER validation passes
                if validation_passed and distance_info.get('distance_mm'):
                    measurements = self.apply_distance_compensation(
                        measurements, 
                        distance_info['distance_mm']
                    )

        # --- Existing code continues below (unchanged) ---

        if px_to_mm is None:
            print("⚠️ No scale reference found. Using pixel measurements only.")
            measurements['px_to_mm'] = None
        else:
            measurements['px_to_mm'] = px_to_mm
            measurements['marker_angle'] = marker_angle

            if marker_corners_list and len(marker_corners_list) > 0:
                self.calculate_perspective_matrix(marker_corners_list[0])

        if marker_corners_list:
            for marker_corners in marker_corners_list:
                cv2.polylines(annotated_image, [marker_corners.astype(int)], True, (0, 255, 255), 3)

            status_text = f"ArUco x{len(marker_corners_list)} ({marker_angle:.1f}°)"
            if self.perspective_matrix is not None:
                status_text += f" [Perspective OK]"
            if self.reprojection_error is not None:
                status_text += f" Err:{self.reprojection_error:.2f}mm"

            cv2.putText(annotated_image, status_text, tuple(marker_corners_list[0][0].astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Step 4 onward: your original YOLO + measurement logic
        # -----------------------------------------------------
        all_detections = []
        all_centers_px = []

        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes.data.cpu().numpy()
                for i, box in enumerate(boxes):
                    cls_id = int(box[5])
                    bbox = box[:4]
                    label = self.model.names[cls_id] if hasattr(self.model, 'names') else f"class_{cls_id}"
                    label_lower = label.lower().replace('_', '').replace('-', '')

                    if any(x in label_lower for x in ['outer', 'od', 'outerdia', 'outerdiam']):
                        cx_px = (bbox[0] + bbox[2]) / 2
                        cy_px = (bbox[1] + bbox[3]) / 2
                        diameter_px = ((bbox[2] - bbox[0]) + (bbox[3] - bbox[1])) / 2
                    else:
                        detection_result = self.detect_circular_features(working_image, bbox)
                        if detection_result is None:
                            continue
                        cx_px, cy_px, diameter_px = detection_result

                    all_centers_px.append((cx_px, cy_px))
                    all_detections.append({
                        'label': label,
                        'center_px': (cx_px, cy_px),
                        'diameter_px': diameter_px,
                        'bbox': bbox
                    })

        if self.perspective_matrix is not None and len(all_centers_px) > 0:
            all_centers_real = self.perspective_transform_points(np.array(all_centers_px))
            print(f"✅ Transformed {len(all_centers_px)} centers to real-world coordinates")
        else:
            all_centers_real = np.array(all_centers_px)
            print(f"⚠️ No perspective transformation - using pixel coordinates")

        for i, det in enumerate(all_detections):
            cx_px, cy_px = det['center_px']
            diameter_px = det['diameter_px']
            label = det['label']
            x1, y1, x2, y2 = det['bbox']
            cx_real, cy_real = all_centers_real[i]

            diameter_real = 0
            unit_symbol = 'px'

            if self.perspective_matrix is not None:
                corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                transformed_corners = self.perspective_transform_points(corners)
                width_mm = np.linalg.norm(transformed_corners[1] - transformed_corners[0])
                height_mm = np.linalg.norm(transformed_corners[2] - transformed_corners[1])
                diameter_real = (width_mm + height_mm) / 2
                if preferred_unit == 'inches':
                    diameter_real *= self.MM_TO_INCH
                    unit_symbol = '"'
                else:
                    unit_symbol = 'mm'
            elif px_to_mm is not None:
                diameter_real = diameter_px * px_to_mm
                if preferred_unit == 'inches':
                    diameter_real *= self.MM_TO_INCH
                    unit_symbol = '"'
                else:
                    unit_symbol = 'mm'
            else:
                diameter_real = diameter_px
                unit_symbol = 'px'

            label_lower = label.lower().replace('_', '').replace('-', '')

            if any(x in label_lower for x in ['outer', 'od', 'outerdia', 'outerdiam']):
                measurements['outer_dia'] = diameter_real
                measurements['od_px'] = diameter_px
                cv2.circle(annotated_image, (int(cx_px), int(cy_px)), int(diameter_px / 2), (0, 255, 0), 2)
                cv2.circle(annotated_image, (int(cx_px), int(cy_px)), 5, (0, 255, 0), -1)
                cv2.putText(annotated_image, f"OD: {diameter_real:.3f}{unit_symbol}",
                            (int(cx_px - 70), int(cy_px - diameter_px / 2 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            elif any(x in label_lower for x in ['inner', 'id', 'innerdia', 'innerdiam']):
                measurements['inner_dia'] = diameter_real
                measurements['id_px'] = diameter_px
                cv2.circle(annotated_image, (int(cx_px), int(cy_px)), int(diameter_px / 2), (255, 0, 0), 2)
                cv2.circle(annotated_image, (int(cx_px), int(cy_px)), 5, (255, 0, 0), -1)
                cv2.putText(annotated_image, f"ID: {diameter_real:.3f}{unit_symbol}",
                            (int(cx_px - 70), int(cy_px)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            elif any(x in label_lower for x in ['bolt', 'hole', 'bolthole', 'boltdia']):
                bolt_centers_px.append((cx_px, cy_px))
                if self.perspective_matrix is not None:
                    bolt_centers_real.append(all_centers_real[i])
                elif px_to_mm is not None:
                    bolt_centers_real.append((cx_px * px_to_mm, cy_px * px_to_mm))
                else:
                    bolt_centers_real.append((cx_px, cy_px))
                bolt_diameters_px.append(diameter_px)
                bolt_diameters_real.append(diameter_real)
                cv2.circle(annotated_image, (int(cx_px), int(cy_px)), int(diameter_px / 2), (0, 0, 255), 2)
                cv2.circle(annotated_image, (int(cx_px), int(cy_px)), 3, (0, 0, 255), -1)
                cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 1)
                cv2.putText(annotated_image, f"B{len(bolt_centers_px)}",
                            (int(cx_px - 15), int(cy_px - int(diameter_px / 2) - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Step 7–8 unchanged
        measurements['bolt_count'] = len(bolt_centers_px)
        if bolt_diameters_real:
            measurements['bolt_dia'] = float(np.mean(bolt_diameters_real))
            measurements['bolt_dia_px'] = float(np.mean(bolt_diameters_px))

        if len(bolt_centers_real) >= 2:
            pcd_value = self.calculate_pcd_advanced(
                np.array(bolt_centers_real), preferred_unit, annotated_image,
                is_real_coordinates=(self.perspective_matrix is not None or px_to_mm is not None)
            )
            measurements['pcd'] = pcd_value
            print(f"✅ PCD calculated: {pcd_value:.3f}")
        else:
            print("⚠️ Insufficient bolts for PCD calculation")

        # Conversions, quality, consistency, etc. (same as before)
        # --------------------------------------------------------
        if measurements['outer_dia'] > 0:
            if preferred_unit == 'inches':
                measurements['od_inch'] = measurements['outer_dia']
                measurements['od_mm'] = measurements['outer_dia'] * 25.4
            else:
                measurements['od_mm'] = measurements['outer_dia']
                measurements['od_inch'] = measurements['outer_dia'] / 25.4

        if measurements['inner_dia'] > 0:
            if preferred_unit == 'inches':
                measurements['id_inch'] = measurements['inner_dia']
                measurements['id_mm'] = measurements['inner_dia'] * 25.4
            else:
                measurements['id_mm'] = measurements['inner_dia']
                measurements['id_inch'] = measurements['inner_dia'] / 25.4

        if measurements['bolt_dia'] > 0:
            if preferred_unit == 'inches':
                measurements['bolt_dia_inch'] = measurements['bolt_dia']
                measurements['bolt_dia_mm'] = measurements['bolt_dia'] * 25.4
            else:
                measurements['bolt_dia_mm'] = measurements['bolt_dia']
                measurements['bolt_dia_inch'] = measurements['bolt_dia'] / 25.4

        if measurements['pcd'] > 0:
            if preferred_unit == 'inches':
                measurements['pcd_inch'] = measurements['pcd']
                measurements['pcd_mm'] = measurements['pcd'] * 25.4
            else:
                measurements['pcd_mm'] = measurements['pcd']
                measurements['pcd_inch'] = measurements['pcd'] / 25.4

        measurements['has_scale'] = self.perspective_matrix is not None or px_to_mm is not None
        measurements['calibration_active'] = self.calibration_active
        measurements['reprojection_error_mm'] = self.reprojection_error

        if self.reprojection_error is not None:
            if self.reprojection_error < 0.5:
                measurements['quality'] = 'excellent'
            elif self.reprojection_error < 1.0:
                measurements['quality'] = 'good'
            elif self.reprojection_error < 2.0:
                measurements['quality'] = 'fair'
            else:
                measurements['quality'] = 'poor'
        else:
            measurements['quality'] = 'unknown'

        if (
            'inner_dia' in measurements and 'outer_dia' in measurements and
            measurements['inner_dia'] > 0 and measurements['outer_dia'] > 0
        ):
            if measurements['inner_dia'] > measurements['outer_dia']:
                print("⚠️ Swapped OD/ID detected – correcting automatically")
                measurements['outer_dia'], measurements['inner_dia'] = measurements['inner_dia'], measurements['outer_dia']
                if 'od_px' in measurements and 'id_px' in measurements:
                    measurements['od_px'], measurements['id_px'] = measurements['id_px'], measurements['od_px']

        # ✅ Add validation info to final output
        measurements['validation_passed'] = validation_passed
        measurements['validation_message'] = validation_message
        measurements['distance_info'] = distance_info
        measurements['measurement_conditions'] = {
            'distance_mm': distance_info['distance_mm'] if distance_info else None,
            'quality': distance_info['quality'] if distance_info else 'unknown',
            'warning': distance_info['warning'] if distance_info else None
        }

        return measurements, annotated_image


        
      




    
    # ==================== IMPROVED PCD CALCULATION ====================


    def fit_circle_least_squares(self, points):
        """Geometric least squares circle fit (Taubin-like) — robust options."""
        # points: ndarray Nx2
        points = np.asarray(points, dtype=float)
        if points.shape[0] < 3:
            return None

        def circle_eq(params, x, y):
            cx, cy, r = params
            return (x - cx)**2 + (y - cy)**2 - r**2

        x, y = points[:, 0], points[:, 1]
        cx0, cy0 = np.mean(x), np.mean(y)
        r0 = np.mean(np.sqrt((x - cx0)**2 + (y - cy0)**2))
        initial_guess = [cx0, cy0, r0]

        try:
            res = least_squares(
                circle_eq,
                initial_guess,
                args=(x, y),
                method='trf',
                loss='soft_l1',
                ftol=1e-8,
                xtol=1e-8,
                verbose=0,
                max_nfev=2000
            )
            if res.success:
                cx, cy, r = res.x
                return float(cx), float(cy), float(abs(r))
        except Exception as e:
            print(f"⚠️ fit_circle_least_squares failed: {e}")

        return None
    
    

    def calculate_pcd_advanced(self, bolt_centers, preferred_unit, annotated_image, is_real_coordinates=False):
        """
        Advanced PCD calculation using best-fit circle

        Args:
            bolt_centers: Array of bolt center coordinates
            preferred_unit: 'inches' or 'mm'
            annotated_image: Image to draw on
            is_real_coordinates: True if coordinates are already in MM/inches

        Returns:
            float: PCD value in preferred unit
        """
        if len(bolt_centers) < 2:
            return 0

        bolt_centers = np.array(bolt_centers)
        pcd_real = 0

        print(f"\n📐 PCD Calculation Details:")
        print(f"   Input: {len(bolt_centers)} bolt centers")
        print(f"   Real coordinates: {is_real_coordinates}")
        print(f"   Unit: {preferred_unit}")

        if len(bolt_centers) >= 3:
            # ✅ Use least squares circle fitting
            try:
                cx, cy, radius = self.fit_circle_least_squares(bolt_centers)
                pcd_real = radius * 2

                print(f"   Best-fit circle:")
                print(f"      Center: ({cx:.2f}, {cy:.2f})")
                print(f"      Radius: {radius:.3f}")
                print(f"      PCD: {pcd_real:.3f}")

                # Draw the fitted circle (only if coordinates are in pixel space)
                if not is_real_coordinates:
                    cv2.circle(annotated_image, (int(cx), int(cy)), 5, (255, 255, 0), -1)
                    cv2.circle(annotated_image, (int(cx), int(cy)), int(radius), (255, 255, 0), 2)
                    cv2.putText(annotated_image, f"PCD Center", 
                               (int(cx-50), int(cy-radius-10)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            except Exception as e:
                print(f"   ⚠️ Best-fit failed: {e}, using centroid method")
                pattern_center = np.mean(bolt_centers, axis=0)
                distances = [np.linalg.norm(bolt - pattern_center) for bolt in bolt_centers]
                avg_radius = np.mean(distances)
                pcd_real = avg_radius * 2

        elif len(bolt_centers) == 2:
            # For 2 bolts, PCD is the distance between them
            pcd_real = np.linalg.norm(bolt_centers[0] - bolt_centers[1])
            print(f"   2-bolt PCD: {pcd_real:.3f}")

        # Verify with maximum opposing distance (for 4+ bolts)
        if len(bolt_centers) >= 4:
            max_distance_real = 0
            best_pair = None

            for i in range(len(bolt_centers)):
                for j in range(i+1, len(bolt_centers)):
                    distance = np.linalg.norm(bolt_centers[i] - bolt_centers[j])
                    if distance > max_distance_real:
                        max_distance_real = distance
                        best_pair = (i, j)

            print(f"   Max opposing distance: {max_distance_real:.3f}")

            # Use max distance if significantly larger
            if max_distance_real > pcd_real * 1.05:
                print(f"   ✅ Using max distance as PCD")
                pcd_real = max_distance_real

                # Draw max diameter line
                if not is_real_coordinates and best_pair:
                    pt1 = tuple(bolt_centers[best_pair[0]].astype(int))
                    pt2 = tuple(bolt_centers[best_pair[1]].astype(int))
                    cv2.line(annotated_image, pt1, pt2, (0, 255, 255), 4)

        # ✅ FIXED - Unit conversion logic
        if is_real_coordinates:
            # Coordinates are already in MM
            if preferred_unit == 'inches':
                pcd_real = pcd_real * self.MM_TO_INCH
                print(f"   Converted to inches: {pcd_real:.3f}\"")
        # else: coordinates are in pixels, no conversion needed

        print(f"   ✅ Final PCD: {pcd_real:.3f}")
        return float(pcd_real)







    
    # ==================== MAIN ANALYSIS FUNCTION ====================
    
    def analyze_flange_with_units(self, image, preferred_unit='inches'):
        """
        Main flange analysis function with comprehensive error reporting
        
        Args:
            image: BGR image
            preferred_unit: 'inches' or 'mm'
        
        Returns:
            tuple: (analysis_dict, annotated_image)
        """
        try:
            measurements, annotated_image = self.extract_measurements_from_segmentation(
                image, preferred_unit
            )
            
            if measurements is None:
                return None, image
            
            # Check compliance with specifications
            compliance = self.check_compliance_with_tolerance(measurements, preferred_unit)
            
            # Build comprehensive analysis result
            analysis_result = {
                'measurements': measurements,
                'compliance': compliance,
                'bolt_count': measurements['bolt_count'],
                'has_scale': measurements.get('px_to_mm') is not None or self.perspective_matrix is not None,
                'unit': preferred_unit,
                'calibration_active': self.calibration_active,
                'measurement_quality': measurements.get('quality', 'unknown'),
                'reprojection_error_mm': self.reprojection_error,
                'warnings': []
            }
            
            # Add warnings based on quality
            if self.reprojection_error is not None and self.reprojection_error > 1.0:
                analysis_result['warnings'].append(
                    f"High reprojection error ({self.reprojection_error:.2f}mm). "
                    "Measurements may be unreliable. Check marker placement and camera angle."
                )
            
            if not self.calibration_active:
                analysis_result['warnings'].append(
                    "Camera not calibrated. For best accuracy, perform checkerboard calibration."
                )
            
            if self.perspective_matrix is None and measurements.get('px_to_mm') is None:
                analysis_result['warnings'].append(
                    "No ArUco marker detected. Place a 4.8cm ArUco marker in the image for real-world measurements."
                )
            
            # Add measurement tolerances to result
            unit_symbol = '"' if preferred_unit == 'inches' else 'mm'
            tolerance_value = self.TOLERANCE_INCHES if preferred_unit == 'inches' else self.TOLERANCE_MM
            analysis_result['tolerance'] = f"±{tolerance_value}{unit_symbol}"
            
            return analysis_result, annotated_image
            
        except Exception as e:
            print(f"❌ Error in flange analysis: {e}")
            import traceback
            traceback.print_exc()
            return None, image
    
    def analyze_flange(self, image):
        """Backward compatibility wrapper"""
        return self.analyze_flange_with_units(image, 'inches')
    
    # ==================== COMPLIANCE CHECKING ====================
    
    def check_compliance_with_tolerance(self, measurements, unit):
        """
        Check ASME B16.5 compliance with proper tolerances
        
        Args:
            measurements: Measurements dictionary
            unit: 'inches' or 'mm'
        
        Returns:
            dict: Compliance results
        """
        if self.specifications is None or ('px_to_mm' not in measurements and self.perspective_matrix is None):
            return {
                'status': 'unknown', 
                'message': 'No specifications or scale reference available',
                'details': {}
            }
        
        measured_od = measurements.get('outer_dia', 0)
        measured_id = measurements.get('inner_dia', 0)
        bolt_count = measurements.get('bolt_count', 0)
        measured_bolt_dia = measurements.get('bolt_dia', 0)
        measured_pcd = measurements.get('pcd', 0)
        
        def get_tolerances(nps):
            """Get ASME B16.5 tolerances based on NPS"""
            if unit == 'inches':
                if nps <= 5:
                    od_tol = (0.08, -0.04)
                else:
                    od_tol = (0.16, -0.04)
                
                if nps <= 10:
                    id_tol = (0.04, -0.04)
                elif 12 <= nps <= 18:
                    id_tol = (0.06, -0.06)
                else:
                    id_tol = (0.12, -0.06)
                
                bolt_hole_tol = (0.03, -0.03)
                pcd_tol = (0.06, -0.06)
            else:  # mm
                if nps <= 5:
                    od_tol = (2.0, -1.0)
                else:
                    od_tol = (4.0, -1.0)
                
                if nps <= 10:
                    id_tol = (1.0, -1.0)
                elif 12 <= nps <= 18:
                    id_tol = (1.5, -1.5)
                else:
                    id_tol = (3.0, -1.5)
                
                bolt_hole_tol = (0.8, -0.8)
                pcd_tol = (1.5, -1.5)
            
            return {
                "od": od_tol, 
                "id": id_tol, 
                "bolt_hole": bolt_hole_tol,
                "pcd": pcd_tol
            }
        
        best_match = None
        min_score = float('inf')
        
        # Search for best matching specification
        for _, spec in self.specifications.iterrows():
            try:
                spec_nps = spec.get('NPS', spec.get('Nominal Pipe Size (NPS)', 'Unknown'))
                nps = float(spec_nps) if spec_nps not in [None, 'Unknown'] else 0
                
                spec_od = float(spec.get('OD', spec.get('Outer Diameter', 0)))
                spec_id = float(spec.get('ID', spec.get('Inner Diameter', 0)))
                spec_bolt_dia = float(spec.get('BD', spec.get('Bolt Diameter', 0)))
                spec_bolt_count = int(spec.get('Number of Bolts', spec.get('Bolts', 0)))
                spec_pcd = float(spec.get('PCD', spec.get('Bolt Circle Diameter', 0)))
                
                # Convert specs to measurement unit
                if unit == 'mm':
                    spec_od *= 25.4
                    spec_id *= 25.4 if spec_id > 0 else 0
                    spec_bolt_dia *= 25.4 if spec_bolt_dia > 0 else 0
                    spec_pcd *= 25.4 if spec_pcd > 0 else 0
                
                tol = get_tolerances(nps)
                
                # Calculate deviations
                od_dev = measured_od - spec_od
                id_dev = measured_id - spec_id if spec_id > 0 else 0
                bolt_dev = measured_bolt_dia - spec_bolt_dia if spec_bolt_dia > 0 else 0
                pcd_dev = measured_pcd - spec_pcd if spec_pcd > 0 and measured_pcd > 0 else 0
                
                # Check compliance
                od_ok = tol["od"][1] <= od_dev <= tol["od"][0]
                id_ok = (spec_id == 0) or (tol["id"][1] <= id_dev <= tol["id"][0])
                bolt_ok = (spec_bolt_dia == 0) or (tol["bolt_hole"][1] <= bolt_dev <= tol["bolt_hole"][0])
                bolt_count_ok = bolt_count == spec_bolt_count
                pcd_ok = (spec_pcd == 0) or (measured_pcd == 0) or (tol["pcd"][1] <= pcd_dev <= tol["pcd"][0])
                
                # Calculate matching score (lower is better)
                score = abs(od_dev) * 2.0 + abs(id_dev) * 1.5 + abs(bolt_dev) * 1.0 + abs(pcd_dev) * 1.0
                if not bolt_count_ok:
                    score += 50
                
                # Consider this spec if OD is reasonably close
                if abs(od_dev) <= max(abs(tol["od"][0]), abs(tol["od"][1])) * 2 and score < min_score:
                    min_score = score
                    best_match = {
                        'spec': spec.to_dict(),
                        'nps': str(spec_nps),
                        'od_compliant': od_ok,
                        'id_compliant': id_ok,
                        'bolt_count_compliant': bolt_count_ok,
                        'bolt_dia_compliant': bolt_ok,
                        'pcd_compliant': pcd_ok,
                        'deviations': {
                            'od': od_dev, 
                            'id': id_dev, 
                            'bolt_dia': bolt_dev,
                            'pcd': pcd_dev
                        },
                        'spec_values': {
                            'od': spec_od, 
                            'id': spec_id, 
                            'bolt_dia': spec_bolt_dia, 
                            'bolt_count': spec_bolt_count,
                            'pcd': spec_pcd
                        },
                        'tolerances_used': tol
                    }
                    
            except (ValueError, TypeError) as e:
                continue
        
        unit_symbol = '"' if unit == 'inches' else 'mm'
        
        if best_match:
            all_ok = (best_match['od_compliant'] and 
                     best_match['id_compliant'] and 
                     best_match['bolt_count_compliant'] and 
                     best_match['bolt_dia_compliant'] and
                     best_match['pcd_compliant'])
            
            status = 'pass' if all_ok else 'fail'
            msg = f"NPS {best_match['nps']} - {status.upper()}"
            
            return {
                'status': status,
                'message': msg,
                'specification': best_match['spec'],
                'deviations': best_match['deviations'],
                'spec_values': best_match['spec_values'],
                'tolerances_used': best_match['tolerances_used'],
                'unit': unit,
                'nps': best_match['nps'],
                'details': {
                    'od_status': 'PASS' if best_match['od_compliant'] else 'FAIL',
                    'id_status': 'PASS' if best_match['id_compliant'] else 'FAIL',
                    'bolt_count_status': 'PASS' if best_match['bolt_count_compliant'] else 'FAIL',
                    'bolt_dia_status': 'PASS' if best_match['bolt_dia_compliant'] else 'FAIL',
                    'pcd_status': 'PASS' if best_match['pcd_compliant'] else 'FAIL'
                }
            }
        
        return {
            'status': 'fail',
            'message': f'No matching specification found (OD {measured_od:.2f}{unit_symbol}, {bolt_count} bolts)',
            'unit': unit,
            'details': {}
        }

# ------------------- Annotations -------------------
    def draw_enhanced_annotations(self, image, outer, inner, bolts, measurements, marker_corners, compliance, unit):
        out = image.copy()
        unit_symbol = '"' if unit == 'inches' else 'mm'
        # Marker
        if marker_corners is not None:
            cv2.polylines(out, [marker_corners.astype(int)], True, (0, 255, 255), 3)
            cv2.putText(out, "ArUco Scale Marker", tuple(marker_corners[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        # OD
        x1, y1, x2, y2 = map(int, outer['box'])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(out, "OUTER DIAMETER", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # ID
        if inner:
            x1, y1, x2, y2 = map(int, inner['box'])
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.putText(out, "INNER DIAMETER", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        # Bolts
        for i, b in enumerate(bolts):
            x1, y1, x2, y2 = map(int, b['box'])
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 165, 255), 3)
            cv2.putText(out, f"BOLT-{i+1}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        # Measurements
        if measurements.get('has_scale'):
            measurement_text = (
                f"OD: {measurements.get('od_mm', 0):.3f}mm | "
                f"ID: {measurements.get('id_mm', 0):.3f}mm | "
                f"BD: {measurements.get('bolt_dia_mm', 0):.3f}mm | "
                f"PCD: {measurements.get('pcd_mm', 0):.3f}mm | "
                f"Bolts: {measurements['bolt_count']}"
            )
        else:
            measurement_text = (f"OD: {measurements.get('od_px', 0):.0f}px | "
                                f"ID: {measurements.get('id_px', 0):.0f}px | "
                                f"BD: {measurements.get('bolt_dia_px', 0):.0f}px | "
                                f"Bolts: {measurements['bolt_count']} | Scale: MISSING")
        
        text_size = cv2.getTextSize(measurement_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(out, (10, 10), (min(text_size[0] + 20, out.shape[1]-10), 55), (0, 0, 0), -1)
        cv2.putText(out, measurement_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Compliance
        status_color = (0, 255, 0) if compliance['status'] == 'pass' else (0, 0, 255) if compliance['status'] == 'fail' else (0, 255, 255)
        status_text = f"COMPLIANCE: {compliance['status'].upper()} - {compliance['message']}"
        
        # Check if compliance details exist and show tolerances
        if 'tolerances_used' in compliance:
            tol_mm = compliance['tolerances_used']['od']
            tolerance_text = f"OD Tol: +{tol_mm[0]:.2f}/-{abs(tol_mm[1]):.2f}mm"
            cv2.rectangle(out, (10, 65), (400, 95), (50, 50, 50), -1)
            cv2.putText(out, tolerance_text, (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            status_y = 125
        else:
            status_y = 95
            
        status_text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(out, (10, status_y-25), (min(status_text_size[0] + 20, out.shape[1]-10), status_y+5), (0, 0, 0), -1)
        cv2.putText(out, status_text[:100], (15, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        return out

# Corrected deduplicate_defects function with tighter parameters to reduce multiple counting
def deduplicate_defects(defects, time_window=5.0, location_threshold=50):
    """
    Enhanced deduplication to prevent showing same defect multiple times
    - time_window: seconds to consider for deduplication (increased for video)
    - location_threshold: pixel distance threshold for same location (tightened)
    """
    if not defects:
        return defects
    
    unique_defects = []
    
    for current_defect in defects:
        is_duplicate = False
        current_time = datetime.fromisoformat(current_defect['timestamp'])
        current_center = (current_defect['location']['center_x'], current_defect['location']['center_y'])
        
        for existing_defect in unique_defects:
            existing_time = datetime.fromisoformat(existing_defect['timestamp'])
            existing_center = (existing_defect['location']['center_x'], existing_defect['location']['center_y'])
            
            # Check if same type and within time window
            if (current_defect['type'] == existing_defect['type'] and 
                abs((current_time - existing_time).total_seconds()) <= time_window):
                
                # Check if locations are close (same defect in consecutive frames)
                distance = hypot(current_center[0] - existing_center[0], 
                               current_center[1] - existing_center[1])
                
                if distance <= location_threshold:
                    # Update existing with better data if current has higher confidence
                    if current_defect['confidence'] > existing_defect['confidence']:
                        existing_defect['confidence'] = current_defect['confidence']
                        existing_defect['location'] = current_defect['location']
                        existing_defect['timestamp'] = current_defect['timestamp']
                        
                        # Update frame snapshot if available
                        if 'frame_snapshot' in current_defect:
                            existing_defect['frame_snapshot'] = current_defect['frame_snapshot']
                    
                    # Track multiple detections
                    if 'detection_count' not in existing_defect:
                        existing_defect['detection_count'] = 1
                    existing_defect['detection_count'] += 1
                    
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            current_defect['detection_count'] = 1
            unique_defects.append(current_defect)
    
    return unique_defects




class DefectDetector:
    def __init__(self, weld_model, coating_model):
        self.weld_model = weld_model
        self.coating_model = coating_model
        self.device = device
        
        # Extended weld defect classes
        self.weld_classes = {
            0: 'Arc strike', 1: 'Burn Through', 2: 'Crack', 3: 'Excess-Reinforcement',
            4: 'Irregular', 5: 'Lack of Fusion', 6: 'Overlap', 7: 'Porosity',
            8: 'Spatter', 9: 'Undercut', 10: 'excess root penetration',
            11: 'mechanical-damage', 12: 'root concvity', 13: 'slag-inclusion'
        }
        
        # Coating defect classes
        self.coating_classes = {
            0: 'Blister', 1: 'Orange Peel', 2: 'Paint sagging', 3: 'Peel Off',
            4: 'Wrinkle', 5: 'chalking', 6: 'crack', 7: 'cratering', 8: 'rash'
        }
    
    def detect_defects(self, image, detection_type='weld', is_live=False, frame_number=0):
        """Enhanced defect detection with frame capture"""
        model = self.weld_model if detection_type == 'weld' else self.coating_model
        defect_classes = self.weld_classes if detection_type == 'weld' else self.coating_classes
        
        if model is None:
            return [], image, []
        
        try:
            # GPU/CPU optimized inference
            if is_live:
                results = model(image, imgsz=320, conf=0.3, iou=0.5, verbose=False, device=self.device)
            else:
                results = model(image, imgsz=640, conf=0.25, iou=0.45, verbose=False, device=self.device)
            
            defects = []
            annotated_image = image.copy()
            detection_boxes = []
            defect_frames = []  # Store individual defect frames
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        # Get coordinates and class
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        # Get defect name
                        defect_name = defect_classes.get(cls, f'unknown_{cls}')
                        
                        # Create individual defect frame
                        defect_frame = image.copy()
                        
                        # Define colors for different defect types
                        color_map = {
                            'Arc strike': (0, 255, 0), 'Burn Through': (255, 255, 255),
                            'Crack': (0, 0, 255), 'Excess-Reinforcement': (255, 0, 0),
                            'Irregular': (255, 255, 0), 'Lack of Fusion': (0, 255, 255),
                            'Overlap': (128, 0, 128), 'Porosity': (128, 255, 0),
                            'Spatter': (0, 128, 0), 'Undercut': (0, 128, 255),
                            'excess root penetration': (0, 128, 128), 'mechanical-damage': (0, 0, 255),
                            'root concvity': (128, 0, 255), 'slag-inclusion': (255, 0, 128),
                            # Coating colors
                            'Blister': (255, 0, 0), 'Orange Peel': (255, 165, 0),
                            'Paint sagging': (128, 0, 128), 'Peel Off': (255, 192, 203),
                            'Wrinkle': (0, 255, 255), 'chalking': (211, 211, 211),
                            'crack': (255, 0, 0), 'cratering': (105, 105, 105), 'rash': (255, 20, 147)
                        }
                        
                        color = color_map.get(defect_name, (0, 255, 0))
                        
                        # Draw on both annotated image and individual defect frame
                        for img in [annotated_image, defect_frame]:
                            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                            label = f"{defect_name.replace('_', ' ').title()} {conf:.2f}"
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                            cv2.rectangle(img, (int(x1), int(y1) - label_size[1] - 10), 
                                        (int(x1) + label_size[0], int(y1)), color, -1)
                            cv2.putText(img, label, (int(x1), int(y1-5)), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        
                        # Add frame info for video analysis
                        if frame_number > 0:
                            cv2.putText(defect_frame, f"Frame: {frame_number}", (10, 30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            cv2.putText(defect_frame, f"Time: {frame_number/30:.1f}s", (10, 60), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # Encode defect frame to base64
                        _, buffer = cv2.imencode('.jpg', defect_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        defect_frame_base64 = base64.b64encode(buffer).decode()
                        
                        # Create defect record with frame snapshot
                        defect = {
                            'id': f"{detection_type}_{frame_number}_{i}",
                            'type': defect_name,
                            'confidence': float(conf),
                            'location': {
                                'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2),
                                'center_x': int((x1 + x2) / 2), 'center_y': int((y1 + y2) / 2)
                            },
                            'detection_type': detection_type,
                            'timestamp': datetime.now().isoformat(),
                            'frame_number': frame_number,
                            'defect_frame': defect_frame_base64  # Individual defect frame
                        }
                        defects.append(defect)
                        
                        # For live detection, prepare bbox data
                        if is_live:
                            detection_boxes.append({
                                'id': defect['id'],
                                'type': defect_name,
                                'confidence': float(conf),
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'defect_frame': defect_frame_base64
                            })
            
            return defects, annotated_image, detection_boxes if is_live else defects
            
        except Exception as e:
            print(f"Detection error: {e}")
            return [], image, []

# Initialize detector and flange analyzer
detector = DefectDetector(weld_model, coating_model) if (weld_model or coating_model) else None
flange_analyzer = FlangeAnalyzer(flange_model, flange_specifications) if flange_model else None

def process_video_with_annotation(file, detection_type):
    """Enhanced video processing with frame-by-frame defect capture"""
    global detection_results, session_id
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            file.save(tmp_file.name)
            video_path = tmp_file.name
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            os.unlink(video_path)
            return jsonify({'error': 'Could not open video file'}), 400
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Create output video
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"analyzed_{detection_type}_{session_id}_{timestamp}.mp4"
        output_path = os.path.join(VIDEO_OUTPUT_FOLDER, output_filename)
        
        # Use H264 codec for web compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            cap.release()
            os.unlink(video_path)
            return jsonify({'error': 'Failed to initialize video writer'}), 500
        
        all_detections = []
        frame_count = 0
        
        # Process every frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            annotated_frame = frame.copy()
            current_time = frame_count / fps
            
            # Detect defects in current frame
            frame_defects = []
            if detector is not None and detection_type in ['weld', 'coating']:
                defects, temp_annotated_frame, _ = detector.detect_defects(
                    frame, detection_type, is_live=False, frame_number=frame_count
                )
                if defects:
                    frame_defects = defects
                    annotated_frame = temp_annotated_frame
                    
                    # Add video timing info to defects
                    for defect in frame_defects:
                        defect['video_timestamp'] = current_time
                        defect['from_video'] = True
                    
            elif flange_analyzer is not None and detection_type == 'flange':
                flange_result, annotated_frame = flange_analyzer.analyze_flange(frame)
                if flange_result:
                    frame_defects = [{
                        'id': f"flange_{frame_count}",
                        'type': f"Flange_{flange_result['compliance']['status']}",
                        'confidence': 1.0,
                        'location': {
                            'x1': 0, 'y1': 0, 'x2': frame.shape[1], 'y2': frame.shape[0],
                            'center_x': frame.shape[1]//2, 'center_y': frame.shape[0]//2
                        },
                        'detection_type': 'flange',
                        'timestamp': datetime.now().isoformat(),
                        'frame_number': frame_count,
                        'video_timestamp': current_time,
                        'measurements': flange_result['measurements'],
                        'compliance': flange_result['compliance'],
                        'from_video': True
                    }]
                    
                    # Add frame snapshot for flange
                    _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    frame_defects[0]['defect_frame'] = base64.b64encode(buffer).decode()
            
            # Write annotated frame to output video
            out.write(annotated_frame)
            all_detections.extend(frame_defects)
        
        # Release resources
        cap.release()
        out.release()
        os.unlink(video_path)
        
        # Deduplicate detections
        unique_detections = deduplicate_defects(all_detections, time_window=2.0, location_threshold=30)
        detection_results.extend(unique_detections)
        
        # Aggregate defects
        defect_summary = aggregate_defects(unique_detections)
        
        # Create video streaming URL
        video_stream_url = f"{request.host_url}api/stream_video/{output_filename}"
        video_download_url = f"{request.host_url}api/download_video?filename={output_filename}"
        
        return jsonify({
            'success': True,
            'video_info': {
                'duration': duration,
                'total_frames': frame_count,
                'fps': fps,
                'defect_density_per_minute': (len(unique_detections) / duration * 60) if duration > 0 else 0
            },
            'stream_video_url': video_stream_url,
            'download_video_url': video_download_url,
            'annotated_video': f"/videos/{output_filename}",
            'defect_summary': defect_summary,
            'total_defects': len(unique_detections),
            'defects': [{
                'id': defect['id'],
                'type': defect['type'],
                'confidence': defect['confidence'],
                'location': defect['location'],
                'detection_type': defect['detection_type'],
                'timestamp': defect['timestamp'],
                'frame_number': defect.get('frame_number', 0),
                'video_timestamp': defect.get('video_timestamp', 0),
                'defect_frame': defect.get('defect_frame', ''),
                'measurements': defect.get('measurements', {}),
                'compliance': defect.get('compliance', {})
            } for defect in unique_detections],
            'detection_type': detection_type,
            'session_id': session_id,
            'analysis_complete': True
        })
    
    except Exception as e:
        print(f"Video processing error: {e}")
        if 'video_path' in locals() and os.path.exists(video_path):
            os.unlink(video_path)
        return jsonify({'error': str(e)}), 500
    
    # below section need to change
    

@app.route('/api/download_defect_frame/<defect_id>')
def download_defect_frame(defect_id):
    """Download individual defect frame"""
    try:
        # Find defect in session results
        defect = None
        for result in detection_results:
            if result.get('id') == defect_id:
                defect = result
                break
        
        if not defect or 'defect_frame' not in defect:
            return jsonify({'error': 'Defect frame not found'}), 404
        
        # Decode base64 image
        image_data = base64.b64decode(defect['defect_frame'])
        
        # Create temporary file
        temp_filename = f"defect_{defect['type']}_{defect.get('frame_number', 0)}.jpg"
        
        return Response(
            image_data,
            mimetype='image/jpeg',
            headers={'Content-Disposition': f'attachment; filename="{temp_filename}"'}
        )
        
    except Exception as e:
        return jsonify({'error': f'Error downloading defect frame: {str(e)}'}), 500

@app.route('/api/videos/<filename>')
def serve_video(filename):
    """Serve video files directly"""
    try:
        file_path = os.path.join(VIDEO_OUTPUT_FOLDER, filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'Video file not found'}), 404
        return send_file(file_path)
    except Exception as e:
        return jsonify({'error': f'Error serving video: {str(e)}'}), 500
    
    ###above section need to chnage 
    
    
@app.route('/api/stream_video/<filename>')
def stream_video(filename):
    """Stream video file for web playback"""
    try:
        file_path = os.path.join(VIDEO_OUTPUT_FOLDER, filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'Video file not found'}), 404

        def generate():
            with open(file_path, 'rb') as f:
                while True:
                    data = f.read(1024 * 1024)  # 1MB chunks
                    if not data:
                        break
                    yield data

        return Response(generate(), mimetype='video/mp4')
    except Exception as e:
        return jsonify({'error': f'Error streaming video: {str(e)}'}), 500
    
@app.route('/api/download_video')
def download_video():
    """Download annotated video file"""
    filename = request.args.get('filename')
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    
    file_path = os.path.join(VIDEO_OUTPUT_FOLDER, filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(file_path, as_attachment=True, download_name=filename)


@app.route('/api/detect_flange', methods=['POST'])
def detect_flange():
    """Analyze flange dimensions and compliance"""
    global detection_results, session_id
    
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        if session_id is None:
            session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Read image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Analyze flange
        if flange_analyzer is None:
            return jsonify({'error': 'Flange analyzer not available'}), 500
        
        flange_result, annotated_image = flange_analyzer.analyze_flange(image)
        
        if flange_result is None:
            return jsonify({'error': 'No flange detected in image'}), 400
        
        # Save annotated image
        output_filename = f"flange_analysis_{session_id}_{datetime.now().strftime('%H%M%S')}.jpg"
        output_path = os.path.join(RESULTS_FOLDER, output_filename)
        cv2.imwrite(output_path, annotated_image)
        
        # Encode annotated image
        _, buffer = cv2.imencode('.jpg', annotated_image)
        img_base64 = base64.b64encode(buffer).decode()
        
        # Create defect record for consistency
        flange_defect = {
            'type': f"Flange_{flange_result['compliance']['status']}",
            'confidence': 1.0,
            'location': {
                'x1': 0, 'y1': 0, 
                'x2': image.shape[1], 'y2': image.shape[0],
                'center_x': image.shape[1]//2, 'center_y': image.shape[0]//2
            },
            'detection_type': 'flange',
            'timestamp': datetime.now().isoformat(),
            'measurements': flange_result['measurements'],
            'compliance': flange_result['compliance']
        }
        
        # Add to global results
        detection_results.append(flange_defect)
        
        return jsonify({
            'success': True,
            'flange_analysis': flange_result,
            'annotated_image': f"data:image/jpeg;base64,{img_base64}",
            'session_id': session_id,
            'analysis_complete': True
        })
        
    except Exception as e:
        print(f"Error in flange detection: {e}")
        return jsonify({'error': str(e)}), 500


##//////////////////////////////
# Updated API endpoint for flange detection with unit selection
# Add this enhanced error handling to your detect_flange_with_units endpoint

@app.route('/api/detect_flange_with_units', methods=['POST'])
def detect_flange_with_units():
    """Enhanced flange analysis with unit selection and better error handling"""
    global detection_results, session_id
    
    try:
        print("=== Flange Detection Request Started ===")
        
        if 'image' not in request.files:
            print("ERROR: No image provided")
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        if file.filename == '':
            print("ERROR: No image selected")
            return jsonify({'error': 'No image selected'}), 400
        
        # Get preferred unit (inches or mm)
        preferred_unit = request.form.get('unit', 'inches')
        print(f"Preferred unit: {preferred_unit}")
        
        if preferred_unit not in ['inches', 'mm']:
            preferred_unit = 'inches'
        
        if session_id is None:
            session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            print(f"Generated session ID: {session_id}")
        
        # Read image
        print("Reading image...")
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            print("ERROR: Invalid image format")
            return jsonify({'error': 'Invalid image format'}), 400
        
        print(f"Image loaded successfully: {image.shape}")
        
        # Check if flange analyzer is available
        if flange_analyzer is None:
            print("ERROR: Flange analyzer not available")
            return jsonify({'error': 'Flange analyzer not available'}), 500
        
        print("Starting flange analysis...")
        
        # Analyze flange with unit selection
        flange_result, annotated_image = flange_analyzer.analyze_flange_with_units(image, preferred_unit)
        
        if flange_result is None:
            print("ERROR: No flange detected in image")
            return jsonify({'error': 'No flange detected in image. Ensure the image clearly shows a flange.'}), 400
        
        print("Flange analysis completed successfully")
        print(f"Measurements found: {flange_result.get('measurements', {})}")
        print(f"Compliance: {flange_result.get('compliance', {})}")
        
        # Save annotated image
        output_filename = f"flange_analysis_{preferred_unit}_{session_id}_{datetime.now().strftime('%H%M%S')}.jpg"
        output_path = os.path.join(RESULTS_FOLDER, output_filename)
        
        try:
            cv2.imwrite(output_path, annotated_image)
            print(f"Annotated image saved: {output_path}")
        except Exception as e:
            print(f"Warning: Could not save annotated image: {e}")
        
        # Encode annotated image
        try:
            _, buffer = cv2.imencode('.jpg', annotated_image)
            img_base64 = base64.b64encode(buffer).decode()
            print("Image encoded successfully")
        except Exception as e:
            print(f"Error encoding image: {e}")
            img_base64 = ""
        
        # Create defect record for consistency
        flange_defect = {
            'type': f"Flange_{flange_result['compliance']['status']}",
            'confidence': 1.0,
            'location': {
                'x1': 0, 'y1': 0, 
                'x2': image.shape[1], 'y2': image.shape[0],
                'center_x': image.shape[1]//2, 'center_y': image.shape[0]//2
            },
            'detection_type': 'flange',
            'timestamp': datetime.now().isoformat(),
            'measurements': flange_result['measurements'],
            'compliance': flange_result['compliance'],
            'unit': preferred_unit
        }
        
        # Add to global results
        detection_results.append(flange_defect)
        
        # Prepare response with enhanced tolerance info
        tolerance_info = {
            'inches': f"±{flange_analyzer.TOLERANCE_INCHES}\"",
            'mm': f"±{flange_analyzer.TOLERANCE_MM}mm"
        }
        
        response_data = {
        'success': True,
        'flange_analysis': clean_json_data({
            'measurements': flange_result['measurements'],
            'compliance': flange_result['compliance'],
            'bolt_count': flange_result.get('bolt_count', 0),
            'has_scale': flange_result.get('has_scale', False),
            'unit': preferred_unit
        }),
        'annotated_image': f"data:image/jpeg;base64,{img_base64}" if img_base64 else None,
        'session_id': session_id,
        'analysis_complete': True,
        'unit': preferred_unit,
        'tolerance_info': {
            'inches': f"±{flange_analyzer.TOLERANCE_INCHES}\"",
            'mm': f"±{flange_analyzer.TOLERANCE_MM}mm"
        }
    }
    
        print("=== Flange Detection Response Prepared ===")
        print(f"Response keys: {list(response_data.keys())}")
    
        return jsonify(response_data)
        
    except Exception as e:
        print(f"=== ERROR in enhanced flange detection ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'error': f'Analysis failed: {str(e)}',
            'debug_info': {
                'error_type': type(e).__name__,
                'flange_analyzer_available': flange_analyzer is not None,
                'preferred_unit': preferred_unit if 'preferred_unit' in locals() else 'unknown'
            }
        }), 500
    
#/////////////////////////
def process_image_file(file, detection_type):
    """Process single image file with enhanced error handling"""
    global detection_results, session_id
    
    try:
        # Read image with size validation
        image_bytes = file.read()
        
        # Check if image data is valid
        if len(image_bytes) == 0:
            return jsonify({'error': 'Empty image file'}), 400
            
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image format or corrupted file'}), 400
        
        # Check image dimensions
        height, width = image.shape[:2]
        if height < 100 or width < 100:
            return jsonify({'error': 'Image too small (minimum 100x100 pixels)'}), 400
        if height > 4000 or width > 4000:
            # Resize large images
            scale = min(4000/height, 4000/width)
            new_height, new_width = int(height * scale), int(width * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            print(f"Resized image from {width}x{height} to {new_width}x{new_height}")
        
        # Handle different detection types
        if detection_type in ['weld', 'coating']:
            # Standard defect detection
            if detector is None:
                return jsonify({'error': 'Detection model not loaded'}), 500
            
            defects, annotated_image, _ = detector.detect_defects(image, detection_type, is_live=False)
            
            # Save annotated image
            output_filename = f"detected_{detection_type}_{session_id}_{datetime.now().strftime('%H%M%S')}.jpg"
            output_path = os.path.join(RESULTS_FOLDER, output_filename)
            
            # Use compression for large images
            compression_quality = 95 if image.size < 2000000 else 85
            cv2.imwrite(output_path, annotated_image, [cv2.IMWRITE_JPEG_QUALITY, compression_quality])
            
            # Add to global results
            detection_results.extend(defects)
            
            # Encode annotated image with error handling
            try:
                _, buffer = cv2.imencode('.jpg', annotated_image, [cv2.IMWRITE_JPEG_QUALITY, compression_quality])
                img_base64 = base64.b64encode(buffer).decode()
            except Exception as e:
                print(f"Error encoding image: {e}")
                img_base64 = ""
            
            # Aggregate defects for response
            defect_summary = aggregate_defects(defects)
            
            return jsonify({
                'success': True,
                'defects': defects,
                'defect_summary': defect_summary,
                'annotated_image': f"data:image/jpeg;base64,{img_base64}" if img_base64 else None,
                'total_defects': len(defects),
                'detection_type': detection_type,
                'session_id': session_id,
                'analysis_complete': True,
                'image_size': f"{image.shape[1]}x{image.shape[0]}"
            })
            
        elif detection_type == 'flange':
            # Flange dimension analysis
            if flange_analyzer is None:
                return jsonify({'error': 'Flange analyzer not available'}), 500
            
            flange_result, annotated_image = flange_analyzer.analyze_flange(image)
            
            if flange_result is None:
                return jsonify({'error': 'No flange detected in image. Ensure the image clearly shows a flange.'}), 400
            
            # Save annotated image
            output_filename = f"flange_analysis_{session_id}_{datetime.now().strftime('%H%M%S')}.jpg"
            output_path = os.path.join(RESULTS_FOLDER, output_filename)
            
            compression_quality = 95 if image.size < 2000000 else 85
            cv2.imwrite(output_path, annotated_image, [cv2.IMWRITE_JPEG_QUALITY, compression_quality])
            
            # Encode annotated image
            try:
                _, buffer = cv2.imencode('.jpg', annotated_image, [cv2.IMWRITE_JPEG_QUALITY, compression_quality])
                img_base64 = base64.b64encode(buffer).decode()
            except Exception as e:
                print(f"Error encoding image: {e}")
                img_base64 = ""
            
            # Create defect record for consistency
            flange_defect = {
                'type': f"Flange_{flange_result['compliance']['status']}",
                'confidence': 1.0,
                'location': {
                    'x1': 0, 'y1': 0,
                    'x2': image.shape[1], 'y2': image.shape[0],
                    'center_x': image.shape[1]//2, 'center_y': image.shape[0]//2
                },
                'detection_type': 'flange',
                'timestamp': datetime.now().isoformat(),
                'measurements': flange_result['measurements'],
                'compliance': flange_result['compliance']
            }
            
            # Add to global results
            detection_results.append(flange_defect)
            
            return jsonify({
                'success': True,
                'flange_analysis': flange_result,
                'annotated_image': f"data:image/jpeg;base64,{img_base64}" if img_base64 else None,
                'detection_type': detection_type,
                'session_id': session_id,
                'analysis_complete': True,
                'image_size': f"{image.shape[1]}x{image.shape[0]}"
            })
        
        else:
            return jsonify({'error': f'Unsupported detection type: {detection_type}'}), 400
            
    except cv2.error as e:
        print(f"OpenCV error: {e}")
        return jsonify({'error': 'Image processing failed. Please try a different image format.'}), 400
    except MemoryError:
        print("Memory error during image processing")
        return jsonify({'error': 'Image too large to process. Please use a smaller image.'}), 413
    except Exception as e:
        print(f"Unexpected error in process_image_file: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500
    


def process_video_file(file, detection_type):
    """Process video file - create annotated video output"""
    if detection_type in ['weld', 'coating', 'flange']:
        return process_video_with_annotation(file, detection_type)
    else:
        return jsonify({'error': f'Unsupported detection type for video: {detection_type}'}), 400

@app.route('/api/detect_images', methods=['POST'])
def detect_images():
    """Process multiple uploaded images for defect detection"""
    global detection_results, session_id
    
    try:
        files = request.files.getlist('images')
        if not files or len(files) == 0:
            return jsonify({'error': 'No images provided'}), 400

        # Get detection type from form data
        detection_type = request.form.get('detection_type', 'weld')
        
        # Generate session ID if not exists
        if session_id is None:
            session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        all_results = []
        total_defects = 0
        
        for i, file in enumerate(files):
            if file.filename == '':
                continue
                
            try:
                # Read image
                image_bytes = file.read()
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    continue
                
                # Process based on detection type
                if detection_type in ['weld', 'coating']:
                    if detector is None:
                        continue
                    
                    defects, annotated_image, _ = detector.detect_defects(image, detection_type, is_live=False)
                    
                elif detection_type == 'flange':
                    if flange_analyzer is None:
                        continue
                    
                    flange_result, annotated_image = flange_analyzer.analyze_flange(image)
                    
                    if flange_result is None:
                        defects = []
                    else:
                        # Convert flange result to defect format
                        defects = [{
                            'type': f"Flange_{flange_result['compliance']['status']}",
                            'confidence': 1.0,
                            'location': {
                                'x1': 0, 'y1': 0,
                                'x2': image.shape[1], 'y2': image.shape[0],
                                'center_x': image.shape[1]//2, 'center_y': image.shape[0]//2
                            },
                            'detection_type': 'flange',
                            'timestamp': datetime.now().isoformat(),
                            'measurements': flange_result['measurements'],
                            'compliance': flange_result['compliance']
                        }]
                else:
                    continue
                
                # Save annotated image
                output_filename = f"detected_{detection_type}_{session_id}_{i}_{datetime.now().strftime('%H%M%S')}.jpg"
                output_path = os.path.join(RESULTS_FOLDER, output_filename)
                cv2.imwrite(output_path, annotated_image)
                
                # FIXED: Encode annotated image for web display
                _, buffer = cv2.imencode('.jpg', annotated_image)
                img_base64 = base64.b64encode(buffer).decode()
                
                # Add to results
                result_data = {
                    'filename': file.filename,
                    'defects': defects,
                    'defect_count': len(defects),
                    'annotated_image': f"data:image/jpeg;base64,{img_base64}",  # FIXED: Added this line
                    'original_filename': file.filename,
                    'saved_path': output_path
                }
                
                # Add flange-specific data if applicable
                if detection_type == 'flange' and len(defects) > 0:
                    result_data['flange_analysis'] = defects[0].get('measurements')
                    result_data['compliance'] = defects[0].get('compliance')
                
                all_results.append(result_data)
                
                # Add to global results
                detection_results.extend(defects)
                total_defects += len(defects)
                
            except Exception as e:
                print(f"Error processing image {i}: {e}")
                continue
        
        # Aggregate defects for response
        defect_summary = aggregate_defects(detection_results)
        
        return jsonify({
            'success': True,
            'results': all_results,
            'total_images': len(all_results),
            'total_defects': total_defects,
            'defect_summary': defect_summary,
            'detection_type': detection_type,
            'session_id': session_id,
            'annotated_images_included': True  # FIXED: Flag to indicate images are included
        })
        
    except Exception as e:
        print(f"Error in detect_images: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect_image', methods=['POST'])
def detect_image():
    """Process uploaded image or video for defect detection"""
    global detection_results, session_id
    
    try:
        if 'image' not in request.files and 'video' not in request.files:
            return jsonify({'error': 'No image or video provided'}), 400

        file = request.files.get('image') or request.files.get('video')
        if file.filename == '':
            return jsonify({'error': 'No image or video selected'}), 400
        
        # Check file size (50MB limit)
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size > 50 * 1024 * 1024:  # 50MB
            return jsonify({'error': 'File too large. Maximum size is 50MB'}), 400
        
        # Get detection type from form data
        detection_type = request.form.get('detection_type', 'weld')
        
        # Generate session ID if not exists
        if session_id is None:
            session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Check if file is video or image
        is_video = file.content_type.startswith('video/') if file.content_type else file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))
        
        if is_video:
            # Handle video file with full annotation
            return process_video_file(file, detection_type)
        else:
            # Handle image file
            return process_image_file(file, detection_type)
        
    except MemoryError:
        print("Memory error occurred during image processing")
        return jsonify({'error': 'Image too large to process. Please use a smaller image.'}), 413
    except Exception as e:
        print(f"Error in detect_image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500


# Global variables for live recording
live_video_writer = None
live_recording_path = None
live_recording_active = False

@app.route('/api/detect_frame', methods=['POST'])
def detect_frame():
    """Enhanced live detection with automatic recording"""
    global detection_results, live_video_writer, live_recording_path, live_recording_active
    
    try:
        data = request.json
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        detection_type = data.get('detection_type', 'weld')
        is_recording_start = data.get('start_recording', False)
        
        # Decode base64 image
        try:
            image_data = data['image'].split(',')[1]
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            return jsonify({'error': f'Invalid image format: {str(e)}'}), 400
        
        if image is None:
            return jsonify({'error': 'Could not decode image'}), 400

        # Initialize recording if first frame
        if is_recording_start and not live_recording_active:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            live_recording_path = os.path.join(VIDEO_OUTPUT_FOLDER, f"live_{detection_type}_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            live_video_writer = cv2.VideoWriter(live_recording_path, fourcc, 15.0, (image.shape[1], image.shape[0]))
            live_recording_active = True

        # Process detection with frame capture
        current_defects = []
        detection_boxes = []
        
        if detection_type in ['weld', 'coating']:
            if detector is None:
                return jsonify({'error': 'Model not loaded'}), 500
            
            defects, annotated_image, detection_boxes = detector.detect_defects(
                image, detection_type, is_live=True, frame_number=0
            )
            
            for defect in defects:
                defect['live_detection'] = True
                current_defects.append(defect)
        
        elif detection_type == 'flange':
            if flange_analyzer is None:
                return jsonify({'error': 'Flange analyzer not available'}), 500
            
            flange_result, annotated_image = flange_analyzer.analyze_flange(image)
            
            if flange_result:
                # Create defect record with snapshot
                flange_defect = {
                    'id': f"flange_live_{int(datetime.now().timestamp())}",
                    'type': f"Flange_{flange_result['compliance']['status']}",
                    'confidence': 1.0,
                    'location': {
                        'x1': 0, 'y1': 0, 'x2': image.shape[1], 'y2': image.shape[0],
                        'center_x': image.shape[1]//2, 'center_y': image.shape[0]//2
                    },
                    'detection_type': 'flange',
                    'timestamp': datetime.now().isoformat(),
                    'measurements': flange_result['measurements'],
                    'compliance': flange_result['compliance'],
                    'live_detection': True
                }
                
                _, buffer = cv2.imencode('.jpg', annotated_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                flange_defect['defect_frame'] = base64.b64encode(buffer).decode()
                
                current_defects.append(flange_defect)
                detection_boxes = [{
                    'id': flange_defect['id'],
                    'type': flange_defect['type'],
                    'confidence': 1.0,
                    'bbox': [0, 0, image.shape[1], image.shape[0]],
                    'measurements': flange_result['measurements'],
                    'compliance': flange_result['compliance'],
                    'defect_frame': flange_defect['defect_frame']
                }]

        # Record frame to video if recording is active
        if live_recording_active and live_video_writer is not None:
            try:
                frame_to_record = annotated_image if len(current_defects) > 0 else image
                live_video_writer.write(frame_to_record)
            except Exception as e:
                print(f"Error writing video frame: {e}")

        # Add new detections to global results
        if current_defects:
            recent_results = detection_results[-50:] if len(detection_results) > 50 else detection_results
            combined_defects = recent_results + current_defects
            deduplicated = deduplicate_defects(combined_defects, time_window=1.0, location_threshold=40)
            
            new_detections = [d for d in deduplicated if d.get('live_detection') and d not in recent_results]
            detection_results.extend(new_detections)
            
            if len(detection_results) > 1000:
                detection_results = detection_results[-800:]

        return jsonify({
            'success': True,
            'detections': detection_boxes or [],
            'new_defects': current_defects,
            'frame_defects': len(detection_boxes),
            'detection_type': detection_type,
            'recording_active': live_recording_active,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error in detect_frame: {e}")
        return jsonify({'error': str(e)}), 500   

@app.route('/api/stop_live_recording', methods=['POST'])
def stop_live_recording():
    """Stop live recording and provide download link"""
    global live_video_writer, live_recording_path, live_recording_active
    
    try:
        if not live_recording_active:
            return jsonify({'error': 'No active recording'}), 400
        
        # Stop recording
        if live_video_writer is not None:
            live_video_writer.release()
            live_video_writer = None
        
        live_recording_active = False
        
        if live_recording_path and os.path.exists(live_recording_path):
            filename = os.path.basename(live_recording_path)
            download_url = f"{request.host_url}api/download_video?filename={filename}"
            stream_url = f"{request.host_url}api/videos/{filename}"
            
            return jsonify({
                'success': True,
                'message': 'Live recording stopped',
                'download_url': download_url,
                'stream_url': stream_url,
                'filename': filename
            })
        else:
            return jsonify({'error': 'Recording file not found'}), 404
        
    except Exception as e:
        print(f"Error stopping live recording: {e}")
        return jsonify({'error': str(e)}), 500   


@app.route('/api/start_recording', methods=['POST'])
def start_recording():
    """Start video recording for live detection"""
    global video_writer, recording_active, session_id
    
    try:
        if session_id is None:
            session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        detection_type = request.json.get('detection_type', 'weld') if request.json else 'weld'
        timestamp = datetime.now().strftime('%H%M%S')
        
        # Try different codec combinations
        codecs = [
            ('XVID', '.avi'),
            ('MP4V', '.mp4'),
            ('MJPG', '.avi')
        ]
        
        video_writer = None
        video_path = None
        
        for codec, ext in codecs:
            try:
                video_filename = f"live_detection_{detection_type}_{session_id}_{timestamp}{ext}"
                video_path = os.path.join(VIDEO_OUTPUT_FOLDER, video_filename)
                fourcc = cv2.VideoWriter_fourcc(*codec)
                video_writer = cv2.VideoWriter(video_path, fourcc, 15.0, (640, 480))
                
                if video_writer.isOpened():
                    print(f"Video recording started with {codec} codec")
                    break
                else:
                    video_writer.release()
                    video_writer = None
            except Exception as e:
                print(f"Failed to initialize {codec} codec: {e}")
                if video_writer:
                    video_writer.release()
                    video_writer = None
        
        if video_writer is None:
            return jsonify({'error': 'Failed to initialize video recording'}), 500
        
        recording_active = True
        
        return jsonify({
            'success': True,
            'message': 'Recording started',
            'video_path': video_path,
            'detection_type': detection_type,
            'session_id': session_id
        })
        
    except Exception as e:
        print(f"Error starting recording: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop_recording', methods=['POST'])
def stop_recording():
    """Stop video recording"""
    global video_writer, recording_active
    
    try:
        recording_active = False
        
        if video_writer is not None:
            video_writer.release()
            video_writer = None
        
        return jsonify({
            'success': True,
            'message': 'Recording stopped'
        })
        
    except Exception as e:
        print(f"Error stopping recording: {e}")
        return jsonify({'error': str(e)}), 500



@app.route('/api/calibrate_camera', methods=['POST'])
def calibrate_camera():
    """Calibrate camera using checkerboard pattern images"""
    try:
        files = request.files.getlist('images')
        
        if not files or len(files) < 10:
            return jsonify({
                'success': False,
                'error': 'Please upload at least 10 checkerboard images'
            }), 400
        
        print(f"📸 Received {len(files)} images for calibration")
        
        # Read all images
        images = []
        for file in files:
            image_bytes = file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is not None:
                images.append(image)
        
        if len(images) < 10:
            return jsonify({
                'success': False,
                'error': f'Only {len(images)} valid images found. Need at least 10.'
            }), 400
        
        # Perform calibration
        if flange_analyzer is None:
            return jsonify({
                'success': False,
                'error': 'Flange analyzer not available'
            }), 500
        
        # Calibrate (9x6 checkerboard, 25mm squares)
        success = flange_analyzer.calibrate_camera_from_checkerboard(
            images, 
            checkerboard_size=(9, 6),
            square_size_mm=25.0
        )
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Camera calibration successful',
                'reprojection_error': flange_analyzer.reprojection_error or 0.0,
                'calibration_active': flange_analyzer.calibration_active
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Calibration failed. Ensure checkerboard is clearly visible in all images.'
            }), 400
        
    except Exception as e:
        print(f"❌ Calibration error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Calibration failed: {str(e)}'
        }), 500




@app.route('/api/check_camera_angle', methods=['POST'])
def check_camera_angle():
    """Check if camera is perpendicular to flange using ArUco marker"""
    try:
        data = request.json
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode image
        try:
            image_data = data['image'].split(',')[1]
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            return jsonify({'error': f'Invalid image format: {str(e)}'}), 400
        
        if image is None:
            return jsonify({'error': 'Could not decode image'}), 400
        
        if flange_analyzer is None:
            return jsonify({'error': 'Flange analyzer not available'}), 500
        
        # Detect ArUco marker
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        px_to_mm, marker_corners_list, marker_angle = flange_analyzer.detect_aruco_markers(
            gray, enable_auto_calibration=False
        )
        
        if not marker_corners_list or len(marker_corners_list) == 0:
            return jsonify({
                'success': False,
                'message': '📍 Place 4.8cm ArUco marker in view',
                'angle_info': None
            })
        
        # Calculate camera angle
        angle_info = flange_analyzer.calculate_camera_angle(marker_corners_list[0])
        
        # Generate guidance message
        message = get_angle_guidance_message(angle_info)
        
        return jsonify({
            'success': True,
            'angle_info': angle_info,
            'message': message
        })
        
    except Exception as e:
        print(f"Error checking camera angle: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def get_angle_guidance_message(angle_info):
    """Generate user-friendly guidance message"""
    status = angle_info['status']
    tilt_x = angle_info['tilt_x']
    tilt_y = angle_info['tilt_y']
    
    if status == 'perfect':
        return "✅ Perfect! Camera is perpendicular. Take the shot now!"
    elif status == 'good':
        return "✅ Good angle! You can proceed with measurement."
    else:
        # Generate specific guidance
        guidance = []
        
        if abs(tilt_x) > 5:
            if tilt_x > 0:
                guidance.append("➡️ Move camera slightly to the LEFT")
            else:
                guidance.append("⬅️ Move camera slightly to the RIGHT")
        
        if abs(tilt_y) > 5:
            if tilt_y > 0:
                guidance.append("⬆️ Move camera slightly UP")
            else:
                guidance.append("⬇️ Move camera slightly DOWN")
        
        if angle_info['square_ratio'] < 0.9:
            guidance.append("🔍 Marker appears compressed - move closer or adjust distance")
        elif angle_info['square_ratio'] > 1.1:
            guidance.append("🔍 Marker appears stretched - adjust viewing angle")
        
        return " | ".join(guidance) if guidance else "⚠️ Adjust camera to be more perpendicular"






@app.route('/api/get_session_results', methods=['GET'])
def get_session_results():
    """Get all defects detected in current session"""
    global detection_results
    
    try:
        defect_summary = aggregate_defects(detection_results)
        
        # Separate results by type
        weld_results = [d for d in detection_results if d.get('detection_type') == 'weld']
        coating_results = [d for d in detection_results if d.get('detection_type') == 'coating']
        flange_results = [d for d in detection_results if d.get('detection_type') == 'flange']
        
        return jsonify({
            'success': True,
            'total_detections': len(detection_results),
            'weld_detections': len(weld_results),
            'coating_detections': len(coating_results),
            'flange_detections': len(flange_results),
            'defect_summary': defect_summary,
            'all_detections': detection_results[-100:],  # Return last 100 for performance
            'session_id': session_id
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export_csv', methods=['GET'])
def export_csv():
    """Export detection results to CSV"""
    global detection_results, session_id
    
    try:
        if not detection_results:
            return jsonify({'error': 'No detection results to export'}), 400
        
        detection_type = request.args.get('type', 'all')
        
        # Filter results based on type
        filtered_results = detection_results
        if detection_type in ['weld', 'coating', 'flange']:
            filtered_results = [d for d in detection_results if d.get('detection_type') == detection_type]
        
        # Create DataFrame
        df_data = []
        for defect in filtered_results:
            row_data = {
                'Detection_Type': defect.get('detection_type', 'unknown'),
                'Defect_Type': defect['type'],
                'Confidence': defect['confidence'],
                'Location_X': defect['location']['center_x'],
                'Location_Y': defect['location']['center_y'],
                'Bounding_Box': f"({defect['location']['x1']},{defect['location']['y1']},{defect['location']['x2']},{defect['location']['y2']})",
                'Timestamp': defect['timestamp']
            }
            
            # Add flange-specific data
            if 'measurements' in defect:
                measurements = defect['measurements']
                row_data.update({
                    'OD_Pixels': measurements.get('od_px', ''),
                    'ID_Pixels': measurements.get('id_px', ''),
                    'OD_Inches': measurements.get('od_inch', ''),
                    'ID_Inches': measurements.get('id_inch', ''),
                    'Bolt_Count': measurements.get('bolt_count', ''),
                    'Bolt_Diameter_Inches': measurements.get('bolt_dia_inch', ''),
                    'Has_Scale_Reference': measurements.get('px_to_cm') is not None
                })
            
            if 'compliance' in defect:
                compliance = defect['compliance']
                row_data.update({
                    'Compliance_Status': compliance.get('status', ''),
                    'Compliance_Message': compliance.get('message', '')
                })
            
            df_data.append(row_data)
        
        df = pd.DataFrame(df_data)
        
        # Add summary at the end
        summary_data = []
        defect_counts = aggregate_defects(filtered_results)
        
        summary_data.append({'Detection_Type': '--- SUMMARY ---'})
        for defect_type, count in defect_counts.items():
            summary_data.append({
                'Detection_Type': 'Summary',
                'Defect_Type': f'Total {defect_type}',
                'Confidence': count
            })
        
        summary_df = pd.DataFrame(summary_data)
        final_df = pd.concat([df, summary_df], ignore_index=True)
        
        # Save to temporary file
        filename = f'{detection_type}_defects_{session_id or "session"}.csv'
        filepath = os.path.join(RESULTS_FOLDER, filename)
        final_df.to_csv(filepath, index=False)
        
        return send_file(filepath, as_attachment=True, download_name=filename)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export_excel', methods=['GET'])
def export_excel():
    """Export detection results to Excel"""
    global detection_results, session_id
    
    try:
        if not detection_results:
            return jsonify({'error': 'No detection results to export'}), 400
        
        detection_type = request.args.get('type', 'all')
        
        # Filter results based on type
        filtered_results = detection_results
        if detection_type in ['weld', 'coating', 'flange']:
            filtered_results = [d for d in detection_results if d.get('detection_type') == detection_type]
        
        # Create DataFrame with enhanced columns for flange data
        df_data = []
        for defect in filtered_results:
            row_data = {
                'Detection Type': defect.get('detection_type', 'unknown'),
                'Defect Type': defect['type'],
                'Confidence': defect['confidence'],
                'Location X': defect['location']['center_x'],
                'Location Y': defect['location']['center_y'],
                'Bounding Box': f"({defect['location']['x1']},{defect['location']['y1']},{defect['location']['x2']},{defect['location']['y2']})",
                'Timestamp': defect['timestamp']
            }
            
            # Add flange-specific columns
            if 'measurements' in defect:
                measurements = defect['measurements']
                row_data.update({
                    'OD (pixels)': measurements.get('od_px', ''),
                    'ID (pixels)': measurements.get('id_px', ''),
                    'OD (inches)': measurements.get('od_inch', ''),
                    'ID (inches)': measurements.get('id_inch', ''),
                    'Bolt Count': measurements.get('bolt_count', ''),
                    'Bolt Diameter (inches)': measurements.get('bolt_dia_inch', ''),
                    'Scale Reference Available': 'Yes' if measurements.get('px_to_cm') else 'No'
                })
            
            if 'compliance' in defect:
                compliance = defect['compliance']
                row_data.update({
                    'Compliance Status': compliance.get('status', ''),
                    'Compliance Message': compliance.get('message', ''),
                    'Specification Match': 'Yes' if compliance.get('specification') else 'No'
                })
            
            df_data.append(row_data)
        
        df = pd.DataFrame(df_data)
        
        # Save to temporary file
        filename = f'{detection_type}_defects_{session_id or "session"}.xlsx'
        filepath = os.path.join(RESULTS_FOLDER, filename)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Detections', index=False)
            
            # Add summary sheet
            defect_counts = aggregate_defects(filtered_results)
            summary_df = pd.DataFrame(list(defect_counts.items()), columns=['Defect Type', 'Count'])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Add flange specifications sheet if available
            if flange_specifications is not None and detection_type in ['flange', 'all']:
                flange_specifications.to_excel(writer, sheet_name='Specifications', index=False)
        
        return send_file(filepath, as_attachment=True, download_name=filename)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500



######### this is the new function i am going to add 
@app.route('/api/export_pdf', methods=['GET'])
def export_pdf():
    """Export detection results to PDF report"""
    global detection_results, session_id
    
    try:
        if not detection_results:
            return jsonify({'error': 'No detection results to export'}), 400
        
        detection_type = request.args.get('type', 'all')
        
        # Filter results based on type
        filtered_results = detection_results
        if detection_type in ['weld', 'coating', 'flange']:
            filtered_results = [d for d in detection_results if d.get('detection_type') == detection_type]
        
        # Aggregate defects
        defect_counts = aggregate_defects(filtered_results)
        
        # Generate HTML report
        html_content = generate_pdf_html(filtered_results, defect_counts, detection_type, session_id)
        
        # Save HTML temporarily
        filename = f'{detection_type}_report_{session_id or "session"}.html'
        filepath = os.path.join(RESULTS_FOLDER, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return send_file(filepath, as_attachment=True, download_name=filename)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_pdf_html(filtered_results, defect_counts, detection_type, session_id):
    """Generate HTML content for PDF report"""
    
    # Generate defect rows
    defect_rows = ''
    for defect_type, count in defect_counts.items():
        # Determine severity
        severity_map = {
            'Crack': 'HIGH', 'Burn Through': 'HIGH', 'Lack of Fusion': 'HIGH',
            'Porosity': 'MEDIUM', 'Undercut': 'MEDIUM', 'Spatter': 'LOW'
        }
        severity = severity_map.get(defect_type, 'MEDIUM')
        
        severity_color = '#ef4444' if severity == 'HIGH' else '#f59e0b' if severity == 'MEDIUM' else '#10b981'
        
        defect_rows += f'''
        <tr>
            <td>{defect_type.replace("_", " ").title()}</td>
            <td style="text-align: center;">{count}</td>
            <td style="text-align: center;">
                <span style="background: {severity_color}; color: white; padding: 4px 12px; border-radius: 12px; font-size: 11px; font-weight: 600;">
                    {severity}
                </span>
            </td>
        </tr>
        '''
    
    if not defect_rows:
        defect_rows = '<tr><td colspan="3" style="text-align: center; padding: 2rem; color: #6b7280;">No defects detected</td></tr>'
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    html_template = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>RICI Industries - Defect Detection Report</title>
        <style>
            @page {{ size: A4; margin: 20mm; }}
            body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; color: #1f2937; line-height: 1.6; }}
            .header {{ text-align: center; margin-bottom: 40px; padding-bottom: 20px; border-bottom: 3px solid #2563eb; }}
            .company-logo {{ width: 80px; height: 80px; background: linear-gradient(135deg, #2563eb, #1a365d); 
                             border-radius: 12px; display: inline-flex; align-items: center; justify-content: center; 
                             color: white; font-weight: 700; font-size: 28px; margin-bottom: 15px; }}
            .company-name {{ font-size: 32px; font-weight: bold; color: #2563eb; margin: 10px 0; }}
            .report-title {{ font-size: 24px; margin: 15px 0; color: #374151; font-weight: 600; }}
            .report-subtitle {{ color: #6b7280; font-size: 14px; }}
            .summary {{ background: #f3f4f6; padding: 20px; border-radius: 8px; margin: 25px 0; border-left: 4px solid #2563eb; }}
            .summary h3 {{ margin-top: 0; color: #1f2937; font-size: 18px; }}
            .summary p {{ margin: 8px 0; font-size: 14px; }}
            .defect-table {{ width: 100%; border-collapse: collapse; margin: 25px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .defect-table thead {{ background: linear-gradient(135deg, #2563eb, #1e40af); }}
            .defect-table th {{ color: white; padding: 12px; text-align: left; font-weight: 600; font-size: 14px; }}
            .defect-table td {{ border: 1px solid #e5e7eb; padding: 10px 12px; font-size: 13px; }}
            .defect-table tbody tr:nth-child(even) {{ background: #f9fafb; }}
            .footer {{ text-align: center; margin-top: 40px; padding-top: 20px; border-top: 2px solid #e5e7eb; font-size: 12px; color: #6b7280; }}
        </style>
    </head>
    <body>
        <div class="header">
            <div class="company-logo">RICI</div>
            <div class="company-name">RICI Industries</div>
            <div class="report-title">AI Defect Detection Report</div>
            <div class="report-subtitle">Generated: {timestamp}</div>
        </div>
        
        <div class="summary">
            <h3>Analysis Summary</h3>
            <p><strong>Detection Type:</strong> {detection_type.capitalize()}</p>
            <p><strong>Total Defects Found:</strong> {len(filtered_results)}</p>
            <p><strong>Unique Defect Types:</strong> {len(defect_counts)}</p>
            <p><strong>Session ID:</strong> {session_id or "N/A"}</p>
        </div>
        
        <h3 style="color: #374151; margin-top: 30px;">Defect Details</h3>
        <table class="defect-table">
            <thead>
                <tr>
                    <th>Defect Type</th>
                    <th style="text-align: center;">Count</th>
                    <th style="text-align: center;">Severity</th>
                </tr>
            </thead>
            <tbody>
                {defect_rows}
            </tbody>
        </table>
        
        <div class="footer">
            <p><strong>RICI Industries AI Defect Detection System</strong></p>
            <p>This report contains automated analysis results and should be reviewed by qualified personnel.</p>
            <p>&copy; {datetime.now().year} RICI Industries. All rights reserved.</p>
        </div>
    </body>
    </html>
    '''
    
    return html_template

###### above is new function i have added ###################



def verify_session(session_id):
    """Verify session is valid and not expired"""
    if not session_id:
        return None

    conn = sqlite3.connect('rici_users.db')
    cursor = conn.cursor()

    try:
        # Use string comparison for datetime, ensure consistent format
        cursor.execute('''
            SELECT u.user_id, u.email, u.full_name, u.job_id, u.department
            FROM user_sessions s
            JOIN users u ON s.user_id = u.user_id
            WHERE s.session_id = ? 
              AND DATETIME(s.expires_at) > DATETIME('now') 
              AND u.is_active = 1
        ''', (session_id,))
        
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return {
                'user_id': user[0],
                'email': user[1],
                'full_name': user[2],
                'job_id': user[3],
                'department': user[4]
            }
        return None
    except Exception as e:
        print(f"❌ verify_session error: {e}")
        conn.close()
        return None







@app.route('/api/clear_session', methods=['POST'])
def clear_session():
    """Clear current session data"""
    global detection_results, session_id
    
    detection_results = []
    session_id = None
    
    return jsonify({'success': True, 'message': 'Session cleared'})

@app.route('/api/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    try:
        info = {
            'success': True,
            'weld_model_path': WELD_MODEL_PATH,
            'coating_model_path': COATING_MODEL_PATH,
            'flange_model_path': FLANGE_MODEL_PATH,
            'weld_model_loaded': weld_model is not None,
            'coating_model_loaded': coating_model is not None,
            'flange_model_loaded': flange_model is not None,
            'flange_specifications_loaded': flange_specifications is not None
        }
        
        if detector:
            info.update({
                'weld_classes': detector.weld_classes,
                'coating_classes': detector.coating_classes
            })
        
        if flange_specifications is not None:
            info['flange_specifications_count'] = len(flange_specifications)
            info['flange_specifications_columns'] = list(flange_specifications.columns)
        
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def aggregate_defects(defects):
    """Aggregate defects by type and return counts"""
    defect_counts = {}
    for defect in defects:
        defect_type = defect['type']
        if defect_type in defect_counts:
            defect_counts[defect_type] += 1
        else:
            defect_counts[defect_type] = 1
    return defect_counts

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'weld_model_loaded': weld_model is not None,
        'coating_model_loaded': coating_model is not None,
        'flange_model_loaded': flange_model is not None,
        'flange_specifications_loaded': flange_specifications is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/')
def index():
    return app.send_static_file('dash_3.html')

if __name__ == '__main__':
    print("Starting Enhanced Defect Detection Backend...")
    print(f"Weld model path: {WELD_MODEL_PATH}")
    print(f"Coating model path: {COATING_MODEL_PATH}")
    print(f"Flange model path: {FLANGE_MODEL_PATH}")
    print(f"Flange specifications path: {FLANGE_CSV_PATH}")
    print(f"Weld model loaded: {weld_model is not None}")
    print(f"Coating model loaded: {coating_model is not None}")
    print(f"Flange model loaded: {flange_model is not None}")
    print(f"Flange specifications loaded: {flange_specifications is not None}")
    print(f"Video output folder: {VIDEO_OUTPUT_FOLDER}")
    
    if detector:
        print(f"Available weld classes: {detector.weld_classes}")
        print(f"Available coating classes: {detector.coating_classes}")
    
    if flange_analyzer:
        print("Flange analyzer initialized successfully")
    
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)


