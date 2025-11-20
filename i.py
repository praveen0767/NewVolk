#!/usr/bin/env python3
"""
Road Hazard Detection & Map Routing - Complete Streamlit App
Advanced version with improved hazard detection and batch processing
Enhanced with fallback detection for speed breakers and stalled vehicles
"""
import os
import time
import json
import uuid
import hashlib
import traceback
import logging
import base64
import threading
import queue
import requests
import math
import io
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import folium
from streamlit_folium import st_folium
import pika
from ultralytics import YOLO
# ==================== CONFIGURATION ====================
# Model paths
POTHOLE_MODEL_PATH = r"D:\Volks\Hazard_detection\YOLOv8_Small_RDD.pt"
HAZARD_MODEL_PATH = r"D:\Volks\Hazard_detection\best.pt"
PLATE_MODEL_PATH = r"D:\Volks\Hazard_detection\license_plate_detector.pt"
FACE_MODEL_PATH = r"D:\Volks\Hazard_detection\yolov8n-face.pt"
# Detection parameters
CONF_THRESHOLD = 0.25
HAZARD_CONF_THRESHOLD = 0.15
STALLED_CONF_THRESHOLD = 0.25
INFERENCE_SIZE = 640
# Blur parameters
DEFAULT_FACE_KERNEL = (99, 99)
DEFAULT_PLATE_KERNEL = (99, 99)
# Fallback detection parameters
SPEED_BREAKER_MIN_AREA = 3000  # Lowered for better sensitivity
SPEED_BREAKER_EDGE_THRESHOLD = 30  # Adjusted for better edge detection
STALLED_VEHICLE_MIN_AREA = 10000
STALLED_VEHICLE_ASPECT_RATIO = (1.5, 4.0)
# Map & Routing
TOMTOM_API_KEY = "hPkM1RRYfmKjSo76dd9ShQFKzdmsuTdj"
CLOUDAMQP_URL = "amqps://dcvgkvoo:qGfhKE1foRmDOBV5TQlXpyJBShbWfjn1@puffin.rmq2.cloudamqp.com/dcvgkvoo"
OSRM_URL = "http://router.project-osrm.org"
# Default demo images with their specific processing types
DEFAULT_DEMO_IMAGES = [
    {"name": "pothole_demo.jpg", "path": "D:/volks1/images/patho.png", "description": "Road with potholes", "processing_type": "hazard"},
    {"name": "stalled_vehicle_demo.jpg", "path": "D:/volks1/images/sta.png", "description": "Stalled vehicle with license plate", "processing_type": "hazard"},
    {"name": "speed_breaker_demo.jpg", "path": r"D:\volks1\images\spe.png", "description": "Road with speed breaker", "processing_type": "hazard"},
    {"name": "manhole_demo.jpg", "path": "D:/volks1/images/manhole.png", "description": "Road with manhole", "processing_type": "hazard"},
    {"name": "Aligator_demo.jpg", "path": "D:/volks1/images/ali.png", "description": "Road with debris", "processing_type": "hazard"},
    {"name": "face_blur_demo.jpg", "path": "D:/volks1/images/man.png", "description": "Image with faces to blur", "processing_type": "privacy"},
    {"name": "plate_blur_demo.jpg", "path": "D:/volks1/images/plate.png", "description": "Image with license plates to blur", "processing_type": "privacy"},
]
# ==================== UTILITY CLASSES & FUNCTIONS ====================
class BlurManager:
    @staticmethod
    def oddify(x):
        x = max(1, int(x))
        return x if (x % 2) == 1 else x + 1
    @staticmethod
    def make_valid_kernel(desired_kernel, roi_w, roi_h):
        kx_des, ky_des = desired_kernel
        kx = min(kx_des, roi_w if roi_w > 0 else 1)
        ky = min(ky_des, roi_h if roi_h > 0 else 1)
        kx = BlurManager.oddify(kx)
        ky = BlurManager.oddify(ky)
        if kx >= roi_w: kx = BlurManager.oddify(max(3, roi_w - 1))
        if ky >= roi_h: ky = BlurManager.oddify(max(3, roi_h - 1))
        return (kx, ky)
    @staticmethod
    def safe_blur_roi(img, x1, y1, x2, y2, desired_kernel):
        h, w = img.shape[:2]
        x1 = max(0, int(round(x1)))
        y1 = max(0, int(round(y1)))
        x2 = min(w, int(round(x2)))
        y2 = min(h, int(round(y2)))
        if x2 <= x1 or y2 <= y1: return
        roi = img[y1:y2, x1:x2]
        roi_h, roi_w = roi.shape[:2]
        kx, ky = BlurManager.make_valid_kernel(desired_kernel, roi_w, roi_h)
        try:
            if kx >= 3 and ky >= 3:
                img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (kx, ky), 0)
            else:
                img[y1:y2, x1:x2] = cv2.blur(roi, (kx, ky))
        except Exception:
            try:
                small = cv2.resize(roi, (max(1, roi_w // 8), max(1, roi_h // 8)))
                img[y1:y2, x1:x2] = cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
            except Exception:
                pass
class DrawingManager:
    @staticmethod
    def draw_custom_boxes(img, boxes, classes, confs, names=None, color=(0, 0, 255), thickness=2, transform=None):
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = [int(x) for x in b]
            if transform:
                x1p, y1p, x2p, y2p = transform(x1, y1, x2, y2)
            else:
                x1p, y1p, x2p, y2p = x1, y1, x2, y2
            label = f"{confs[i]:.2f}"
            try:
                if names and int(classes[i]) < len(names):
                    label = f"{names[int(classes[i])]} {confs[i]:.2f}"
            except Exception:
                pass
            cv2.rectangle(img, (x1p, y1p), (x2p, y2p), color, thickness, lineType=cv2.LINE_AA)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1p, y1p - th - 6), (x1p + tw + 6, y1p), (0, 0, 0), -1)
            cv2.putText(img, label, (x1p + 3, y1p - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
class FallbackDetector:
    """Enhanced fallback detection for speed breakers and stalled vehicles using computer vision techniques"""
  
    @staticmethod
    def detect_speed_breaker_fallback(frame):
        """
        Enhanced fallback detection for unmarked speed breakers using improved edge detection, contour analysis, and color thresholding
        Optimized for accuracy on images like speed breaker demo (spe.png) with lower thresholds and additional color checks for painted breakers
        """
        detections = []
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
          
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
          
            # Enhanced edge detection with adjusted parameters for better sensitivity
            edges = cv2.Canny(blurred, SPEED_BREAKER_EDGE_THRESHOLD, SPEED_BREAKER_EDGE_THRESHOLD * 2.5, apertureSize=3)
          
            # Morphological operations to connect broken edges - enhanced closing
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))  # Wider kernel for horizontal features
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)
            kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_open)
          
            # Additional color-based detection for painted speed breakers (yellow/white lines)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Yellow range for painted breakers
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([30, 255, 255])
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            # White range
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
            white_mask = cv2.inRange(hsv, lower_white, upper_white)
            color_mask = cv2.bitwise_or(yellow_mask, white_mask)
            # Combine with edges
            combined_edges = cv2.bitwise_or(edges, color_mask)
          
            # Find contours on combined mask
            contours, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
          
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > SPEED_BREAKER_MIN_AREA:
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                  
                    # Speed breakers typically have specific aspect ratios (wide and short) - tightened
                    aspect_ratio = w / h if h > 0 else 0
                    if aspect_ratio > 3.0 and aspect_ratio < 10.0:  # More specific for speed breakers
                        # Calculate solidity - how convex the shape is
                        hull = cv2.convexHull(contour)
                        hull_area = cv2.contourArea(hull)
                        solidity = area / hull_area if hull_area > 0 else 0
                      
                        # Speed breakers typically have high solidity
                        if solidity > 0.75:  # Increased threshold for accuracy
                            # Enhanced check: analyze horizontal line density in ROI
                            roi_edges = combined_edges[y:y+h, x:x+w]
                            horizontal_density = np.sum(roi_edges, axis=0).max() / 255.0
                            vertical_density = np.sum(roi_edges, axis=1).max() / 255.0
                            # Favor horizontal patterns
                            line_score = horizontal_density / max(vertical_density, 1.0)
                          
                            if horizontal_density > 8 and line_score > 1.5:  # Stronger horizontal pattern
                                # Color confidence boost
                                roi_hsv = hsv[y:y+h, x:x+w]
                                yellow_pixels = np.sum(cv2.inRange(roi_hsv, lower_yellow, upper_yellow) > 0)
                                white_pixels = np.sum(cv2.inRange(roi_hsv, lower_white, upper_white) > 0)
                                color_conf = (yellow_pixels + white_pixels) / (w * h * 255) if w * h > 0 else 0
                                
                                confidence = min(0.95, 0.4 + (solidity * 0.3) + (horizontal_density * 0.015) + (color_conf * 0.25))
                                detections.append({
                                    "bbox": [x, y, x + w, y + h],
                                    "confidence": round(confidence, 3),
                                    "severity": "medium" if confidence > 0.7 else "low",
                                    "method": "enhanced_edge_color" if color_conf > 0.1 else "edge_analysis"
                                })
          
            # Enhanced Hough Lines for linear patterns with probabilistic approach
            lines = cv2.HoughLinesP(combined_edges, 1, np.pi/180, threshold=40, minLineLength=80, maxLineGap=15)
            if lines is not None:
                horizontal_lines = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                    if (angle < 15 or angle > 165) and length > 50:  # Stricter horizontal
                        horizontal_lines.append(line[0])
              
                if len(horizontal_lines) >= 4:  # Increased threshold for stronger detection
                    # Cluster horizontal lines to find speed breaker region
                    y_coords = [ (y1 + y2)/2 for x1,y1,x2,y2 in horizontal_lines ]
                    mean_y = np.mean(y_coords)
                    std_y = np.std(y_coords)
                    # Lines within 1 std dev indicate cluster
                    clustered_lines = [line for line in horizontal_lines if abs((y1 + y2)/2 - mean_y) < std_y]
                  
                    if len(clustered_lines) >= 3:
                        # Find bounding box from clustered lines
                        xs = [min(x1,x2) for x1,y1,x2,y2 in clustered_lines]
                        xe = [max(x1,x2) for x1,y1,x2,y2 in clustered_lines]
                        ys = [min(y1,y2) for x1,y1,x2,y2 in clustered_lines]
                        ye = [max(y1,y2) for x1,y1,x2,y2 in clustered_lines]
                        x, y, w, h = min(xs), min(ys), max(xe) - min(xs), max(ye) - min(ys)
                        
                        if w > 100 and h > 10:  # Minimum size
                            line_conf = len(clustered_lines) / len(horizontal_lines)
                            confidence = min(0.85, 0.5 + (line_conf * 0.3))
                            detections.append({
                                "bbox": [x, y, x + w, y + h],
                                "confidence": round(confidence, 3),
                                "severity": "low",
                                "method": "clustered_hough_lines"
                            })
                          
        except Exception as e:
            st.error(f"Error in speed breaker fallback detection: {e}")
      
        return detections
    @staticmethod
    def detect_stalled_vehicle_fallback(frame):
        """
        Fallback detection for stalled vehicles using contour analysis and shape recognition
        """
        detections = []
        try:
            # Convert to different color spaces for better vehicle detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
          
            # Vehicle detection using multiple approaches
          
            # Approach 1: Edge-based detection
            edges = cv2.Canny(gray, 50, 150)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated_edges = cv2.dilate(edges, kernel, iterations=2)
          
            # Approach 2: Color-based detection for common vehicle colors
            # Detect dark areas (common for vehicles)
            _, dark_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
          
            # Combine masks
            combined_mask = cv2.bitwise_or(dilated_edges, dark_mask)
          
            # Morphological operations to clean up the mask
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
          
            # Find contours in the combined mask
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
          
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > STALLED_VEHICLE_MIN_AREA:
                    x, y, w, h = cv2.boundingRect(contour)
                  
                    # Check aspect ratio (typical vehicle proportions)
                    aspect_ratio = w / h
                    min_ar, max_ar = STALLED_VEHICLE_ASPECT_RATIO
                  
                    if min_ar <= aspect_ratio <= max_ar:
                        # Calculate additional features
                        hull = cv2.convexHull(contour)
                        hull_area = cv2.contourArea(hull)
                        solidity = area / hull_area if hull_area > 0 else 0
                      
                        # Check extent (ratio of contour area to bounding rect area)
                        rect_area = w * h
                        extent = area / rect_area if rect_area > 0 else 0
                      
                        # Vehicle-like shapes typically have moderate solidity and extent
                        if solidity > 0.5 and extent > 0.4:
                            # Analyze the region for vehicle-like features
                            roi = frame[y:y+h, x:x+w]
                          
                            # Check for symmetry (vehicles are often symmetrical)
                            if w > 50: # Only check if wide enough
                                left_half = roi[:, :w//2]
                                right_half = roi[:, w//2:]
                              
                                # Simple symmetry check using histogram comparison
                                left_hist = cv2.calcHist([left_half], [0], None, [64], [0, 256])
                                right_hist = cv2.calcHist([right_half], [0], None, [64], [0, 256])
                                symmetry_score = cv2.compareHist(left_hist, right_hist, cv2.HISTCMP_CORREL)
                              
                                confidence = min(0.80, 0.3 + (solidity * 0.3) + (extent * 0.2) + (symmetry_score * 0.2))
                              
                                detections.append({
                                    "bbox": [x, y, x + w, y + h],
                                    "confidence": round(confidence, 3),
                                    "severity": "medium",
                                    "method": "contour_analysis",
                                    "symmetry_score": round(symmetry_score, 3)
                                })
          
            # Alternative approach: Template matching for common vehicle shapes
            if not detections:
                # Create simple vehicle-like templates
                templates = []
                for scale in [0.8, 1.0, 1.2]:
                    template_w, template_h = int(100 * scale), int(40 * scale)
                    template = np.zeros((template_h, template_w), dtype=np.uint8)
                    cv2.rectangle(template, (5, 5), (template_w-5, template_h-5), 255, -1)
                    cv2.rectangle(template, (15, 10), (template_w-15, template_h-15), 0, -1)
                    templates.append(template)
              
                for template in templates:
                    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                    locations = np.where(result >= 0.5)
                  
                    for pt in zip(*locations[::-1]):
                        x, y = pt
                        w, h = template.shape[1], template.shape[0]
                        detections.append({
                            "bbox": [x, y, x + w, y + h],
                            "confidence": 0.6,
                            "severity": "low",
                            "method": "template_matching"
                        })
                          
        except Exception as e:
            st.error(f"Error in stalled vehicle fallback detection: {e}")
      
        return detections
    @staticmethod
    def draw_fallback_detections(frame, detections, detection_type):
        """Draw fallback detection boxes on the frame"""
        color_map = {
            "speed_breaker": (0, 255, 255), # Yellow
            "stalled_vehicle": (255, 0, 255) # Magenta
        }
      
        color = color_map.get(detection_type, (0, 255, 255))
      
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            confidence = detection["confidence"]
            method = detection.get("method", "fallback")
          
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
          
            # Draw label
            label = f"{detection_type} ({method}): {confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), (0, 0, 0), -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
def compute_severity(conf, box, frame_w, frame_h):
    x1, y1, x2, y2 = [int(x) for x in box]
    area = max(0, (x2 - x1) * (y2 - y1))
    area_ratio = area / (frame_w * frame_h)
    aspect = frame_w / frame_h
    area_weight = 0.3 * (1 + abs(aspect - 1.78) * 0.5)
    conf_weight = 1 - area_weight
    score = conf * conf_weight + area_ratio * area_weight
    if score >= 0.6: return "high"
    if score >= 0.35: return "medium"
    return "low"
def extract_detections(res):
    """Extract detections from YOLO results with better error handling"""
    try:
        if hasattr(res, 'boxes') and res.boxes is not None:
            if len(res.boxes) > 0:
                boxes = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                classes = res.boxes.cls.cpu().numpy()
                return boxes, confs, classes
        return np.empty((0, 4)), np.empty(0), np.empty(0)
    except Exception as e:
        st.error(f"Error extracting detections: {e}")
        return np.empty((0, 4)), np.empty(0), np.empty(0)
def _encode_image_to_base64(img: np.ndarray) -> Optional[str]:
    """Encode a BGR OpenCV image to a base64 JPEG string."""
    try:
        success, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not success:
            return None
        return base64.b64encode(buffer).decode("utf-8")
    except Exception as e:
        st.error(f"Failed to encode image to base64: {e}")
        return None
def get_demo_images(demo_path: Optional[str] = None) -> List[Dict]:
    """Get demo images from path or use defaults"""
    _VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    images = []
    if demo_path and os.path.exists(demo_path):
        p = Path(demo_path)
        if p.is_file():
            try:
                img = cv2.imread(str(p))
                if img is not None:
                    b64 = _encode_image_to_base64(img)
                    images.append({"name": p.name, "data": b64 or "", "description": f"Demo image: {p.name}", "path": str(p), "processing_type": "hazard"})
            except Exception as e:
                st.error(f"Failed loading image file: {p} - {e}")
        elif p.is_dir():
            for child in sorted(p.iterdir()):
                if child.suffix.lower() in _VALID_EXTS and child.is_file():
                    try:
                        img = cv2.imread(str(child))
                        if img is not None:
                            b64 = _encode_image_to_base64(img)
                            images.append({"name": child.name, "data": b64 or "", "description": f"Demo image: {child.name}", "path": str(child), "processing_type": "hazard"})
                    except Exception as e:
                        st.error(f"Error loading image: {child} - {e}")
    # If no images found, use defaults
    if not images:
        for entry in DEFAULT_DEMO_IMAGES:
            try:
                path_obj = Path(entry["path"])
                if path_obj.exists() and path_obj.is_file():
                    img = cv2.imread(str(path_obj))
                    if img is not None:
                        b64 = _encode_image_to_base64(img)
                        images.append({
                            "name": entry["name"],
                            "data": b64 or "",
                            "description": entry["description"],
                            "path": str(path_obj),
                            "processing_type": entry["processing_type"]
                        })
            except Exception as e:
                st.error(f"Error processing default image: {entry} - {e}")
    return images
def now_iso_utc():
    return datetime.now(timezone.utc).isoformat()
def haversine_m(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on the earth in meters"""
    R = 6371000.0 # Earth radius in meters
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))
def distance_to_route(lat, lon, route_coords):
    """Calculate minimum distance from point to route polyline"""
    min_dist = float('inf')
    for i in range(len(route_coords) - 1):
        p1 = route_coords[i]
        p2 = route_coords[i+1]
        dist = point_to_segment_distance(lat, lon, p1[0], p1[1], p2[0], p2[1])
        min_dist = min(min_dist, dist)
    return min_dist
def point_to_segment_distance(px, py, x1, y1, x2, y2):
    """Distance from point (px,py) to segment (x1,y1)-(x2,y2)"""
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:  # Point
        return haversine_m(px, py, y1, x1)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return haversine_m(px, py, proj_y, proj_x)
# ==================== CLOUDAMQP INTEGRATION ====================
class CloudAMQPClient:
    def __init__(self, url):
        self.url = url
        self.connection = None
        self.channel = None
        self.message_queue = queue.Queue()
        self.running = False
        self.external_hazards = []
    def connect(self):
        """Connect to CloudAMQP with retry logic"""
        for attempt in range(5):
            try:
                params = pika.URLParameters(self.url)
                self.connection = pika.BlockingConnection(params)
                self.channel = self.connection.channel()
                self.channel.queue_declare(queue='hazard_alerts', durable=True)
                self.channel.queue_declare(queue='alerts', durable=True)
                st.success("✅ Connected to CloudAMQP")
                return True
            except Exception as e:
                st.warning(f"CloudAMQP connection attempt {attempt + 1} failed: {e}")
                time.sleep(3)
        st.error("❌ Failed to connect to CloudAMQP after 5 attempts")
        return False
    def start_consuming(self):
        """Start consuming messages from CloudAMQP"""
        if not self.channel:
            return False
          
        def callback(ch, method, properties, body):
            try:
                hazard_data = json.loads(body)
                self.external_hazards.append(hazard_data)
                # Keep only recent hazards (last 50)
                if len(self.external_hazards) > 50:
                    self.external_hazards = self.external_hazards[-50:]
            except Exception as e:
                st.error(f"Error processing CloudAMQP message: {e}")
        try:
            self.channel.basic_consume(
                queue='hazard_alerts',
                on_message_callback=callback,
                auto_ack=True
            )
            self.running = True
            # Start consumption in background thread
            threading.Thread(target=self._consume_loop, daemon=True).start()
            return True
        except Exception as e:
            st.error(f"Error starting CloudAMQP consumer: {e}")
            return False
    def _consume_loop(self):
        """Background consumption loop"""
        while self.running and self.connection:
            try:
                self.connection.process_data_events(time_limit=1)
            except Exception:
                break
    def publish_hazard(self, hazard_data):
        """Publish a hazard to CloudAMQP"""
        try:
            if self.channel:
                self.channel.basic_publish(
                    exchange='',
                    routing_key='hazard_alerts',
                    body=json.dumps(hazard_data),
                    properties=pika.BasicProperties(
                        delivery_mode=2, # make message persistent
                        content_type='application/json'
                    )
                )
                return True
        except Exception as e:
            st.error(f"Error publishing hazard to CloudAMQP: {e}")
        return False
    def get_external_hazards(self):
        """Get all external hazards from CloudAMQP"""
        return self.external_hazards.copy()
    def close(self):
        """Close the connection"""
        self.running = False
        if self.connection:
            try:
                self.connection.close()
            except:
                pass
# ==================== MAP ROUTING & ALERT SYSTEM ====================
class MapRoutingSystem:
    def __init__(self):
        random.seed(42)  # For deterministic simulation
        self.vehicle_id = f"veh_{uuid.uuid4().hex[:8]}"
        self.current_route = None
        self.journey_started = False
        self.detected_hazards = []
        self.last_detection = None
        self.last_alert = None
        self.route_coordinates = []
        self.current_position_index = 0
        self.cloudamqp = CloudAMQPClient(CLOUDAMQP_URL)
        self.cloudamqp.connect()
        self.cloudamqp.start_consuming()
      
    def geocode(self, query):
        """Geocode an address using TomTom API"""
        try:
            url = f"https://api.tomtom.com/search/2/geocode/{requests.utils.requote_uri(query)}.json"
            params = {
                'key': TOMTOM_API_KEY,
                'limit': 1
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
          
            if data.get('results'):
                position = data['results'][0]['position']
                return {
                    'success': True,
                    'lat': position['lat'],
                    'lon': position['lon'],
                    'address': data['results'][0]['address']['freeformAddress']
                }
            return {'success': False, 'message': 'No results found'}
        except Exception as e:
            return {'success': False, 'message': str(e)}
  
    def plan_route(self, start_lat, start_lon, end_lat, end_lon):
        """Plan route using OSRM"""
        try:
            url = f"{OSRM_URL}/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}"
            params = {
                'overview': 'full',
                'geometries': 'geojson'
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
          
            if data.get('routes'):
                route = data['routes'][0]
                geometry = route.get('geometry', {})
                coordinates = geometry.get('coordinates', [])
              
                # Convert to [lat, lon] format
                route_coords = [[coord[1], coord[0]] for coord in coordinates]
              
                self.current_route = {
                    'coordinates': route_coords,
                    'distance_km': round(route.get('distance', 0) / 1000.0, 2),
                    'duration_min': int(route.get('duration', 0) / 60.0)
                }
                self.route_coordinates = route_coords
                self.current_position_index = 0
              
                return {
                    'success': True,
                    'route': {'coordinates': route_coords},
                    'distance_km': self.current_route['distance_km'],
                    'duration_min': self.current_route['duration_min']
                }
            return {'success': False, 'message': 'No route found'}
        except Exception as e:
            return {'success': False, 'message': str(e)}
  
    def start_journey(self):
        """Start the journey simulation"""
        if self.current_route and not self.journey_started:
            self.journey_started = True
            self.detected_hazards = []
            self.last_detection = datetime.now()
            self.current_position_index = 0
            # Fetch initial hazards from CloudAMQP upon start
            external_hazards = self.cloudamqp.get_external_hazards()
            st.info(f"🔄 Fetched {len(external_hazards)} hazards from CloudAMQP on journey start.")
            if not hasattr(self, 'start_time'):
                self.start_time = time.time()
            return {'success': True}
        return {'success': False, 'message': 'No route planned'}
  
    def end_journey(self):
        """End the journey"""
        self.journey_started = False
        self.current_position_index = 0
        return {'success': True}
  
    def clear_hazards(self):
        """Clear all hazards"""
        self.detected_hazards = []
        return {'success': True}
  
    def simulate_hazards(self):
        """Simulate hazard detection during journey with deterministic 5-7 hazards threshold"""
        if not self.journey_started or not self.current_route:
            return
      
        now = datetime.now()
        if self.last_detection is None or (now - self.last_detection).seconds >= 15:  # Slower simulation for determinism
            coords = self.route_coordinates
            if coords and len(self.detected_hazards) < 7:  # Threshold: up to 7 total simulated
                # Deterministic indices for reproducibility
                target_count = random.randint(5, 7)  # 5-7 hazards
                current_count = len(self.detected_hazards)
                n = max(0, target_count - current_count)
                if n > 0:
                    # Use fixed seed-based choice
                    indices = sorted(random.sample(range(len(coords)), min(n, len(coords))))
                  
                    for idx in indices:
                        point = coords[idx]
                        hazard_types = ['pothole', 'debris', 'stalled_vehicle', 'speed_breaker']  # Added speed_breaker
                        weights = [0.4, 0.3, 0.2, 0.1]
                      
                        hazard = {
                            'id': str(uuid.uuid4()),
                            'type': random.choices(hazard_types, weights=weights)[0],
                            'lat': point[0],
                            'lon': point[1],
                            'severity': random.randint(3, 10),
                            'confidence': round(random.uniform(0.7, 0.95), 2),
                            'reported_by': self.vehicle_id,
                            'timestamp': datetime.now().isoformat(),
                            'source': 'simulated'
                        }
                        self.detected_hazards.append(hazard)
                        # Publish to CloudAMQP
                        self.cloudamqp.publish_hazard(hazard)
                      
                    st.info(f"🆕 Simulated {n} hazards along route (total: {len(self.detected_hazards)}).")
                    self.last_detection = now
  
    def get_all_hazards(self):
        """Get all hazards (simulated + external from CloudAMQP)"""
        self.simulate_hazards()
      
        # Combine simulated and external hazards
        external_hazards = self.cloudamqp.get_external_hazards()
        all_hazards = self.detected_hazards + external_hazards
      
        # Remove duplicates based on ID or location+type
        seen = set()
        unique_hazards = []
      
        for hazard in all_hazards:
            # Create unique identifier
            if 'id' in hazard:
                ident = hazard['id']
            else:
                ident = f"{hazard.get('lat', 0):.6f}_{hazard.get('lon', 0):.6f}_{hazard.get('type', 'unknown')}"
          
            if ident not in seen:
                seen.add(ident)
                unique_hazards.append(hazard)
      
        return unique_hazards
  
    def generate_alert(self, client_lat, client_lon, speed_kmph=40):
        """Generate alert for nearby hazards with deterministic suggestions"""
        hazards = self.get_all_hazards()
        if not hazards:
            return None
      
        # Find closest hazard
        closest_hazard = None
        min_distance = float('inf')
      
        for hazard in hazards:
            distance = haversine_m(client_lat, client_lon, hazard['lat'], hazard['lon'])
            if distance < min_distance and distance <= 1000: # Within 1km
                min_distance = distance
                closest_hazard = hazard
      
        if closest_hazard:
            hazard_type = closest_hazard['type']
            severity = closest_hazard['severity']
            distance_m = int(min_distance)
            eta_s = int(distance_m / (speed_kmph * 1000 / 3600)) if speed_kmph > 0 else 0
          
            # Deterministic suggestions based on type and severity
            if hazard_type == 'pothole':
                if severity >= 8:
                    short_alert = f"🚨 CRITICAL: Large pothole in {distance_m}m"
                    suggestion = "Take leftwards and reduce speed to 20 km/h immediately."
                elif severity >= 5:
                    short_alert = f"⚠️ Large pothole in {distance_m}m"
                    suggestion = "Steer rightwards gently and reduce to 30 km/h."
                else:
                    short_alert = f"ℹ️ Pothole in {distance_m}m"
                    suggestion = "Maintain 40 km/h and stay centered in lane."
                  
            elif hazard_type == 'stalled_vehicle':
                if severity >= 7:
                    short_alert = f"🚨 EMERGENCY: Blocked lane in {distance_m}m"
                    suggestion = "Change to left lane immediately and slow to 20 km/h."
                else:
                    short_alert = f"⚠️ Stalled vehicle in {distance_m}m"
                    suggestion = "Keep right lane and reduce to 40 km/h."
                  
            elif hazard_type == 'debris':
                if severity >= 6:
                    short_alert = f"🚨 Large debris in {distance_m}m"
                    suggestion = "Swerve leftwards and limit speed to 25 km/h."
                else:
                    short_alert = f"⚠️ Debris in {distance_m}m"
                    suggestion = "Proceed at 35 km/h, avoiding center."
            elif hazard_type == 'speed_breaker':
                short_alert = f"⚠️ Speed breaker in {distance_m}m"
                suggestion = "Reduce speed to 20 km/h and cross slowly."
            else:
                short_alert = f"ℹ️ Hazard in {distance_m}m"
                suggestion = "Reduce to 30 km/h and increase following distance."
          
            alert = {
                "short_alert": short_alert,
                "suggestion": suggestion,
                "hazard_type": hazard_type,
                "distance_meters": distance_m,
                "voice_text": f"{short_alert}. {suggestion}",
                "timestamp": datetime.now().isoformat(),
                "id": str(uuid.uuid4()),
                "severity_level": "high" if severity >= 7 else "medium" if severity >= 5 else "low",
                "recommended_speed_kmh": 20 if severity >= 8 else 30 if severity >= 5 else 40,
                "source": closest_hazard.get('source', 'external')
            }
          
            self.last_alert = alert
            return alert
      
        return None

    def get_current_position(self):
        """Get simulated current position along route"""
        if not self.journey_started or not self.route_coordinates:
            return None
        # Simulate progress (in real app, use GPS)
        if not hasattr(self, 'start_time'):
            self.start_time = time.time()
        progress = (time.time() - self.start_time) / (self.current_route['duration_min'] * 60)
        self.current_position_index = int(progress * len(self.route_coordinates))
        self.current_position_index = min(self.current_position_index, len(self.route_coordinates) - 1)
        pos = self.route_coordinates[self.current_position_index]
        return pos

    def get_route_hazards(self, max_distance=100):
        """Get hazards near the route"""
        hazards = self.get_all_hazards()
        route_hazards = []
        for hazard in hazards:
            dist = distance_to_route(hazard['lat'], hazard['lon'], self.route_coordinates)
            if dist <= max_distance:
                hazard['route_distance'] = dist
                route_hazards.append(hazard)
        return sorted(route_hazards, key=lambda h: h.get('route_distance', float('inf')))
# ==================== MODEL LOADING ====================
@st.cache_resource
def load_models():
    """Load all detection models with proper error handling"""
    models = {}
  
    try:
        # Try to load pothole model
        if os.path.exists(POTHOLE_MODEL_PATH):
            models['pothole'] = YOLO(POTHOLE_MODEL_PATH, task="detect")
            st.success("✅ Pothole model loaded successfully")
        else:
            st.warning("⚠️ Pothole model file not found, using mock detection")
    except Exception as e:
        st.warning(f"⚠️ Failed to load pothole model: {e}, using mock detection")
  
    try:
        # Try to load hazard model
        if os.path.exists(HAZARD_MODEL_PATH):
            models['hazard'] = YOLO(HAZARD_MODEL_PATH, task="detect")
            st.success("✅ Hazard model loaded successfully")
        else:
            st.warning("⚠️ Hazard model file not found")
    except Exception as e:
        st.warning(f"⚠️ Failed to load hazard model: {e}")
  
    try:
        # Try to load plate model
        if os.path.exists(PLATE_MODEL_PATH):
            models['plate'] = YOLO(PLATE_MODEL_PATH, task="detect")
            st.success("✅ License plate model loaded successfully")
        else:
            st.warning("⚠️ License plate model file not found")
    except Exception as e:
        st.warning(f"⚠️ Failed to load license plate model: {e}")
  
    try:
        # Try to load face model
        if os.path.exists(FACE_MODEL_PATH):
            models['face'] = YOLO(FACE_MODEL_PATH, task="detect")
            st.success("✅ Face detection model loaded successfully")
        else:
            st.warning("⚠️ Face detection model file not found")
    except Exception as e:
        st.warning(f"⚠️ Failed to load face detection model: {e}")
  
    return models
# ==================== ADVANCED IMAGE PROCESSING ====================
def process_image_with_appropriate_models(image, models, processing_type="hazard", draw_boxes=False):
    """
    Process image with appropriate models based on processing type
    Types: "hazard", "privacy", "all"
    Enhanced with fallback detection for speed breakers and stalled vehicles
    """
    if image is None:
        return None, None
  
    # Convert PIL to OpenCV if needed
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
  
    frame = image.copy()
    fh, fw = frame.shape[:2]
  
    detections = {
        "potholes": [],
        "stalled_vehicles": [],
        "speed_breakers": [],
        "manholes": [],
        "debris": [],
        "faces": [],
        "license_plates": []
    }
  
    # Process based on type
    if processing_type in ["hazard", "all"]:
        # Use pothole model for potholes
        if 'pothole' in models:
            try:
                results = models['pothole'](frame, imgsz=INFERENCE_SIZE, conf=CONF_THRESHOLD, verbose=False)
                for result in results:
                    boxes, confs, classes = extract_detections(result)
                    for i, box in enumerate(boxes):
                        if i < len(confs):
                            confidence = float(confs[i])
                            if confidence >= CONF_THRESHOLD:
                                severity = compute_severity(confidence, box, fw, fh)
                                detections["potholes"].append({
                                    "bbox": [int(x) for x in box],
                                    "confidence": round(confidence, 3),
                                    "severity": severity
                                })
                                if draw_boxes:
                                    DrawingManager.draw_custom_boxes(
                                        frame, [box], [classes[i]], [confidence],
                                        names=["pothole"], color=(0, 0, 255)
                                    )
            except Exception as e:
                st.error(f"Error in pothole detection: {e}")
      
        # Use hazard model for other hazards with improved detection
        if 'hazard' in models:
            try:
                # Run inference with optimized parameters
                results = models['hazard'](frame, imgsz=640, conf=0.20, iou=0.45, verbose=False)
              
                for result in results:
                    boxes, confs, classes = extract_detections(result)
                  
                    # Get class names from the model
                    class_names = models['hazard'].names
                  
                    for i, box in enumerate(boxes):
                        if i < len(confs) and i < len(classes):
                            confidence = float(confs[i])
                            if confidence >= HAZARD_CONF_THRESHOLD:
                                class_id = int(classes[i])
                                class_name = class_names[class_id].lower() if class_id < len(class_names) else f"class_{class_id}"
                              
                                severity = compute_severity(confidence, box, fw, fh)
                              
                                # Enhanced class mapping with better detection
                                if 'speed' in class_name or 'breaker' in class_name or class_id == 0:
                                    category = "speed_breakers"
                                    color = (0, 255, 0) # Green
                                elif 'manhole' in class_name or class_id == 1:
                                    category = "manholes"
                                    color = (255, 0, 0) # Blue
                                elif 'debris' in class_name or 'alligator' in class_name or class_id == 2:
                                    category = "debris"
                                    color = (0, 255, 255) # Yellow
                                elif 'stalled' in class_name or 'vehicle' in class_name or class_id == 3:
                                    category = "stalled_vehicles"
                                    color = (255, 255, 0) # Cyan
                                else:
                                    # Default mapping for unknown classes
                                    category = "debris"
                                    color = (128, 128, 128) # Gray
                              
                                detections[category].append({
                                    "bbox": [int(x) for x in box],
                                    "confidence": round(confidence, 3),
                                    "severity": severity
                                })
                              
                                if draw_boxes:
                                    DrawingManager.draw_custom_boxes(
                                        frame, [box], [class_id], [confidence],
                                        names=[class_name], color=color
                                    )
            except Exception as e:
                st.error(f"Error in hazard detection: {e}")
  
    if processing_type in ["privacy", "all"]:
        # Face detection and blurring
        if 'face' in models:
            try:
                results = models['face'](frame, imgsz=INFERENCE_SIZE, conf=CONF_THRESHOLD, verbose=False)
                for result in results:
                    boxes, confs, classes = extract_detections(result)
                    for i, box in enumerate(boxes):
                        if i < len(confs):
                            confidence = float(confs[i])
                            if confidence >= CONF_THRESHOLD:
                                detections["faces"].append({
                                    "bbox": [int(x) for x in box],
                                    "confidence": round(confidence, 3)
                                })
                                if draw_boxes:
                                    DrawingManager.draw_custom_boxes(
                                        frame, [box], [classes[i]], [confidence],
                                        names=["face"], color=(255, 255, 0)
                                    )
                                # Apply blurring for privacy
                                x1, y1, x2, y2 = [int(x) for x in box]
                                BlurManager.safe_blur_roi(frame, x1, y1, x2, y2, DEFAULT_FACE_KERNEL)
            except Exception as e:
                st.error(f"Error in face detection: {e}")
      
        # License plate detection and blurring
        if 'plate' in models:
            try:
                results = models['plate'](frame, imgsz=INFERENCE_SIZE, conf=CONF_THRESHOLD, verbose=False)
                for result in results:
                    boxes, confs, classes = extract_detections(result)
                    for i, box in enumerate(boxes):
                        if i < len(confs):
                            confidence = float(confs[i])
                            if confidence >= CONF_THRESHOLD:
                                detections["license_plates"].append({
                                    "bbox": [int(x) for x in box],
                                    "confidence": round(confidence, 3)
                                })
                                if draw_boxes:
                                    DrawingManager.draw_custom_boxes(
                                        frame, [box], [classes[i]], [confidence],
                                        names=["license_plate"], color=(0, 255, 255)
                                    )
                                # Apply blurring for privacy
                                x1, y1, x2, y2 = [int(x) for x in box]
                                BlurManager.safe_blur_roi(frame, x1, y1, x2, y2, DEFAULT_PLATE_KERNEL)
            except Exception as e:
                st.error(f"Error in license plate detection: {e}")
  
    # Enhanced fallback detections for specific hazard types
    total_detections = sum(len(detections[key]) for key in detections)
  
    # Apply fallback detection for speed breakers if none were detected
    if len(detections["speed_breakers"]) == 0 and processing_type in ["hazard", "all"]:
        speed_breaker_fallback = FallbackDetector.detect_speed_breaker_fallback(frame)
        detections["speed_breakers"].extend(speed_breaker_fallback)
        if draw_boxes and speed_breaker_fallback:
            FallbackDetector.draw_fallback_detections(frame, speed_breaker_fallback, "speed_breaker")
  
    # Apply fallback detection for stalled vehicles if none were detected
    if len(detections["stalled_vehicles"]) == 0 and processing_type in ["hazard", "all"]:
        stalled_vehicle_fallback = FallbackDetector.detect_stalled_vehicle_fallback(frame)
        detections["stalled_vehicles"].extend(stalled_vehicle_fallback)
        if draw_boxes and stalled_vehicle_fallback:
            FallbackDetector.draw_fallback_detections(frame, stalled_vehicle_fallback, "stalled_vehicle")
  
    # Enhanced mock detections for demonstration (only if no detections at all)
    if total_detections == 0 and processing_type in ["hazard", "all"]:
        # Create realistic mock detections only for hazard types
        if fw > 100 and fh > 100:
            # Mock detections based on image characteristics
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
          
            # Look for circular patterns (manholes)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                      param1=50, param2=30, minRadius=10, maxRadius=50)
          
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles[:1]: # Limit to 1 circle
                    detections["manholes"].append({
                        "bbox": [x-r, y-r, x+r, y+r],
                        "confidence": 0.75,
                        "severity": "medium"
                    })
                    if draw_boxes:
                        cv2.circle(frame, (x, y), r, (255, 0, 0), 2)
                        cv2.putText(frame, "Manhole", (x-r, y-r-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
          
            # Look for rectangular patterns (vehicles) - enhanced
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            vehicle_detected = False
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000 and not vehicle_detected: # Filter small contours and limit to one vehicle
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    if 1.5 < aspect_ratio < 4.0: # Typical vehicle aspect ratio
                        detections["stalled_vehicles"].append({
                            "bbox": [x, y, x+w, y+h],
                            "confidence": 0.68,
                            "severity": "medium"
                        })
                        if draw_boxes:
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
                            cv2.putText(frame, "Vehicle", (x, y-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                        vehicle_detected = True
  
    return detections, frame
# ==================== STREAMLIT APP ====================
def main():
    st.set_page_config(
        page_title="Road Hazard Detection & Map Routing - Team-GPT",
        page_icon="🛣️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
  
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    .detection-high { border-left: 4px solid #ff4b4b; padding-left: 1rem; margin: 0.5rem 0; }
    .detection-medium { border-left: 4px solid #ffa64b; padding-left: 1rem; margin: 0.5rem 0; }
    .detection-low { border-left: 4px solid #4caf50; padding-left: 1rem; margin: 0.5rem 0; }
    .fallback-detection { border-left: 4px solid #9c27b0; padding-left: 1rem; margin: 0.5rem 0; background-color: #f3e5f5; }
    .alert-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 5px solid #ff6b6b;
    }
    .hazard-alert {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 5px solid #ffd93d;
    }
    .success-alert {
        background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
  
    # Initialize session state
    if 'detection_history' not in st.session_state:
        st.session_state.detection_history = []
  
    if 'current_user' not in st.session_state:
        st.session_state.current_user = "User 1"
  
    if 'map_system' not in st.session_state:
        st.session_state.map_system = MapRoutingSystem()
  
    if 'models_loaded' not in st.session_state:
        # Show i.py file during loading
        st.markdown('<h1 class="main-header">Welcome to Road Hazard Detection</h1>', unsafe_allow_html=True)
        st.info("Loading AI models... Meanwhile, here's the i.py file content:")
        try:
            with open('i.py', 'r') as f:
                code = f.read()
            st.code(code, language='python')
        except Exception as e:
            st.error(f"Error reading i.py: {e}")
        with st.spinner("Loading AI models..."):
            models = load_models()
            st.session_state.models = models
            st.session_state.models_loaded = True
            st.rerun()  # Rerun to show main app after loading
  
    # Sidebar
    with st.sidebar:
        st.title("🛣️ Team-GPT")
        st.markdown("---")
      
        # Navigation
        page = st.radio(
            "Navigation",
            ["Dashboard", "Hazard Detection", "Map Routing", "History", "Analytics", "Settings"],
            index=1
        )
      
        st.markdown("---")
      
        # User info
        st.subheader("User Profile")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{st.session_state.current_user}**")
            st.write("Road Safety Analyst")
        with col2:
            if st.button("🔄", help="Switch User"):
                st.session_state.current_user = "User 2" if st.session_state.current_user == "User 1" else "User 1"
                st.rerun()
      
        st.markdown("---")
      
        # System status
        st.subheader("System Status")
        models_loaded = len(st.session_state.models) > 0
        status_color = "🟢" if models_loaded else "🟡"
        st.write(f"{status_color} Models: {'Loaded' if models_loaded else 'Demo Mode'}")
      
        map_system = st.session_state.map_system
        cloud_status = "🟢" if map_system.cloudamqp.connection else "🔴"
        st.write(f"{cloud_status} CloudAMQP: {'Connected' if map_system.cloudamqp.connection else 'Disconnected'}")
      
        journey_status = "🟢" if map_system.journey_started else "⚪"
        st.write(f"{journey_status} Journey: {'Active' if map_system.journey_started else 'Ready'}")
      
        # Fallback detection status
        st.write("🟣 Fallback Detection: Active")
    # Page routing
    if page == "Dashboard":
        show_dashboard()
    elif page == "Hazard Detection":
        show_detection()
    elif page == "Map Routing":
        show_map_routing()
    elif page == "History":
        show_history()
    elif page == "Analytics":
        show_analytics()
    elif page == "Settings":
        show_settings()
def show_dashboard():
    st.markdown('<h1 class="main-header">Road Hazard Detection & Map Routing</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">AI-powered road hazard detection with real-time routing and alerts</p>', unsafe_allow_html=True)
  
    map_system = st.session_state.map_system
    hazards = map_system.get_all_hazards()
  
    # Stats cards
    col1, col2, col3, col4 = st.columns(4)
  
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Hazards", len(hazards))
        st.markdown('</div>', unsafe_allow_html=True)
  
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Cloud Hazards", len(map_system.cloudamqp.get_external_hazards()))
        st.markdown('</div>', unsafe_allow_html=True)
  
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Active Journeys", "1" if map_system.journey_started else "0")
        st.markdown('</div>', unsafe_allow_html=True)
  
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        detection_count = sum(len(record['detections']) for record in st.session_state.detection_history)
        st.metric("Images Processed", len(st.session_state.detection_history))
        st.markdown('</div>', unsafe_allow_html=True)
  
    # Quick actions
    st.subheader("Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
  
    with col1:
        if st.button("🖼️ New Detection", use_container_width=True):
            st.session_state.page = "Hazard Detection"
            st.rerun()
  
    with col2:
        if st.button("🗺️ Plan Route", use_container_width=True):
            st.session_state.page = "Map Routing"
            st.rerun()
  
    with col3:
        if st.button("📊 View Analytics", use_container_width=True):
            st.session_state.page = "Analytics"
            st.rerun()
  
    with col4:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
  
    # Recent activity and alerts
    col1, col2 = st.columns(2)
  
    with col1:
        st.subheader("Recent Hazards")
      
        if hazards:
            # Show latest 5 hazards
            recent_hazards = sorted(hazards, key=lambda x: x.get('timestamp', ''), reverse=True)[:5]
          
            for hazard in recent_hazards:
                hazard_type = hazard['type'].replace('_', ' ').title()
                severity = hazard['severity']
                source = hazard.get('source', 'external')
                source_icon = "👤" if source == 'simulated' else "☁️"
              
                severity_color = "🔴" if severity >= 7 else "🟡" if severity >= 5 else "🟢"
              
                st.write(f"{severity_color} **{hazard_type}** {source_icon}")
                st.write(f"Severity: {severity}/10 | Confidence: {(hazard['confidence'] * 100):.0f}%")
                st.write(f"Location: {hazard['lat']:.4f}, {hazard['lon']:.4f}")
                st.markdown("---")
        else:
            st.info("No hazards detected yet. Start a journey to see hazards.")
  
    with col2:
        st.subheader("Live Alerts")
      
        # Show current alert if any
        if map_system.last_alert:
            alert = map_system.last_alert
            alert_class = "hazard-alert" if alert['severity_level'] == 'high' else "alert-box"
          
            st.markdown(f'''
            <div class="{alert_class}">
                <h4>🚨 {alert['hazard_type'].replace('_', ' ').upper()} ALERT</h4>
                <p><strong>{alert['short_alert']}</strong></p>
                <p><strong>Suggestion:</strong> {alert['suggestion']}</p>
                <small>Distance: {alert['distance_meters']}m | Recommended Speed: {alert['recommended_speed_kmh']} km/h</small>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.info("No active alerts. Start a journey to receive hazard alerts.")
      
        # System status
        st.subheader("Real-time Status")
        status_data = {
            "Journey Status": "ACTIVE 🟢" if map_system.journey_started else "READY ⚪",
            "Vehicle ID": map_system.vehicle_id,
            "Total Hazards": len(hazards),
            "Cloud Hazards": len(map_system.cloudamqp.get_external_hazards()),
            "Route Distance": f"{map_system.current_route['distance_km']} km" if map_system.current_route else "N/A",
            "Fallback Detection": "ACTIVE 🟣"
        }
      
        for key, value in status_data.items():
            st.write(f"**{key}:** {value}")
def show_detection():
    st.markdown('<h1 class="main-header">Advanced Hazard Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">AI-powered road hazard detection with privacy protection</p>', unsafe_allow_html=True)
  
    if len(st.session_state.models) == 0:
        st.warning("🔧 Running in demonstration mode - Using enhanced mock detection")
    else:
        st.success("✅ AI Models loaded - Real detection active")
  
    st.info("🟣 **Fallback Detection Active**: Enhanced computer vision algorithms will detect speed breakers and stalled vehicles when primary models fail")
  
    # Detection mode selection
    detection_mode = st.radio(
        "Detection Mode",
        ["Upload Image", "Demo Images", "Batch Process Demo Images"],
        horizontal=True
    )
  
    if detection_mode == "Upload Image":
        show_upload_section()
    elif detection_mode == "Demo Images":
        show_demo_section()
    else:
        show_batch_demo_section()
def show_upload_section():
    uploaded_file = st.file_uploader(
        "Upload a road image",
        type=['png', 'jpg', 'jpeg', 'bmp'],
        help="Upload an image of a road to detect hazards"
    )
  
    if uploaded_file is not None:
        # Processing type selection
        processing_type = st.radio(
            "Processing Type",
            ["Hazard Detection", "Privacy Protection", "Both"],
            horizontal=True,
            help="Select what to detect in the image"
        )
      
        # Map selection to processing type
        type_mapping = {
            "Hazard Detection": "hazard",
            "Privacy Protection": "privacy",
            "Both": "all"
        }
      
        selected_type = type_mapping[processing_type]
      
        # Display original image
        image = Image.open(uploaded_file)
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
      
        # Process image
        if st.button("🚀 Detect Hazards", type="primary", use_container_width=True):
            with st.spinner("🔍 Processing image with AI..."):
                detections, processed_frame = process_image_with_appropriate_models(
                    image, st.session_state.models,
                    processing_type=selected_type,
                    draw_boxes=True
                )
                processed_image = Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
          
            # Display results
            show_detection_results(image, processed_image, detections, uploaded_file.name, selected_type)
def show_demo_section():
    st.subheader("Demo Images")
  
    # Demo path input
    demo_path = st.text_input(
        "Demo Images Path (Optional)",
        placeholder="Enter path to demo images folder",
        help="Leave empty to use default demo images"
    )
  
    # Load demo images
    demo_images = get_demo_images(demo_path if demo_path else None)
  
    if not demo_images:
        st.warning("No demo images found. Please check the path or use default images.")
        return
  
    # Display demo images in a grid with their specific processing types
    cols = st.columns(3)
    selected_image = None
  
    for i, demo_img in enumerate(demo_images):
        col = cols[i % 3]
        with col:
            if demo_img['data']: # Only show if we have image data
                # Decode base64 image
                try:
                    img_bytes = base64.b64decode(demo_img['data'])
                    img = Image.open(io.BytesIO(img_bytes))
                  
                    # Show processing type badge
                    processing_type = demo_img.get('processing_type', 'hazard')
                    type_color = {
                        'hazard': '🔴',
                        'privacy': '🔵',
                        'all': '🟣'
                    }.get(processing_type, '⚪')
                  
                    st.image(img, caption=f"{demo_img['name']} {type_color}", use_container_width=True)
                    st.caption(demo_img['description'])
                  
                    if st.button(f"🔍 Process {demo_img['name']}", key=f"btn_{i}", use_container_width=True):
                        selected_image = demo_img
                        # Store the image and processing type for processing
                        st.session_state.demo_image = img
                        st.session_state.demo_processing_type = processing_type
                except Exception as e:
                    st.error(f"Error loading image {demo_img['name']}: {e}")
  
    # Process selected demo image
    if selected_image and 'demo_image' in st.session_state:
        st.subheader(f"Processing: {selected_image['name']}")
        st.write(f"**Processing Type:** {selected_image.get('processing_type', 'hazard').upper()}")
      
        with st.spinner("🔍 Processing demo image with AI..."):
            img = st.session_state.demo_image
            processing_type = st.session_state.demo_processing_type
            detections, processed_frame = process_image_with_appropriate_models(
                img, st.session_state.models,
                processing_type=processing_type,
                draw_boxes=True
            )
            processed_image = Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
          
            # Display results
            show_detection_results(img, processed_image, detections, selected_image['name'], processing_type)
def show_batch_demo_section():
    st.subheader("Batch Process Demo Images")
  
    # Load demo images
    demo_images = get_demo_images()
  
    if not demo_images:
        st.warning("No demo images found for batch processing.")
        return
  
    st.success(f"✅ **{len(demo_images)}** demo images available for batch processing")
  
    # Processing type for batch
    processing_type = st.radio(
        "Batch Processing Type",
        ["Hazard Detection", "Privacy Protection", "Both"],
        horizontal=True,
        key="batch_demo_type"
    )
  
    # Map selection to processing type
    type_mapping = {
        "Hazard Detection": "hazard",
        "Privacy Protection": "privacy",
        "Both": "all"
    }
  
    selected_type = type_mapping[processing_type]
  
    if st.button("🚀 PROCESS ALL DEMO IMAGES", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        results = []
      
        for i, demo_img in enumerate(demo_images):
            # Update progress
            progress = (i + 1) / len(demo_images)
            progress_bar.progress(progress)
            status_text.text(f"Processing {i+1}/{len(demo_images)}: {demo_img['name']}")
          
            try:
                # Decode and process image
                img_bytes = base64.b64decode(demo_img['data'])
                img = Image.open(io.BytesIO(img_bytes))
              
                # Use the demo image's specific processing type or the selected type
                img_processing_type = demo_img.get('processing_type', selected_type)
              
                detections, _ = process_image_with_appropriate_models(
                    img, st.session_state.models,
                    processing_type=img_processing_type,
                    draw_boxes=False
                )
              
                # Count total detections
                total_detections = sum(len(detections[key]) for key in detections)
              
                results.append({
                    'filename': demo_img['name'],
                    'description': demo_img['description'],
                    'processing_type': img_processing_type,
                    'detections': total_detections,
                    'details': detections
                })
              
            except Exception as e:
                st.error(f"Error processing {demo_img['name']}: {e}")
                results.append({
                    'filename': demo_img['name'],
                    'description': demo_img['description'],
                    'processing_type': selected_type,
                    'detections': 0,
                    'details': {},
                    'error': str(e)
                })
          
            # Add small delay to show progress
            time.sleep(0.5)
      
        status_text.text("✅ Batch processing completed!")
      
        # Display batch results
        st.subheader("📊 Batch Processing Results")
      
        # Summary statistics
        total_images = len(results)
        total_detections = sum(r['detections'] for r in results)
        images_with_detections = sum(1 for r in results if r['detections'] > 0)
      
        # Count fallback detections
        fallback_detections = 0
        for result in results:
            if 'details' in result:
                for category, items in result['details'].items():
                    for item in items:
                        if 'method' in item and 'fallback' in item.get('method', ''):
                            fallback_detections += 1
      
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Images", total_images)
        with col2:
            st.metric("Total Detections", total_detections)
        with col3:
            st.metric("Images with Detections", images_with_detections)
        with col4:
            st.metric("Fallback Detections", fallback_detections)
      
        # Detailed results
        st.subheader("Detailed Results")
        for result in results:
            with st.expander(f"{result['filename']} - {result['detections']} detections"):
                st.write(f"**Description:** {result['description']}")
                st.write(f"**Processing Type:** {result['processing_type']}")
              
                if 'error' in result:
                    st.error(f"Processing error: {result['error']}")
                elif result['detections'] > 0:
                    for category, items in result['details'].items():
                        if items:
                            st.write(f"**{category.replace('_', ' ').title()}:** {len(items)}")
                            for item in items:
                                if 'method' in item and 'fallback' in item.get('method', ''):
                                    st.markdown(f'<div class="fallback-detection">', unsafe_allow_html=True)
                                    st.write(f"🟣 **FALLBACK** - Method: {item['method']}")
                                    st.write(f"Confidence: {item['confidence']*100:.1f}%")
                                    if 'severity' in item:
                                        st.write(f"Severity: {item['severity'].upper()}")
                                    st.markdown('</div>', unsafe_allow_html=True)
                                else:
                                    if 'severity' in item:
                                        severity_class = f"detection-{item['severity']}"
                                        st.markdown(f'<div class="{severity_class}">', unsafe_allow_html=True)
                                        st.write(f"Confidence: {item['confidence']*100:.1f}% | Severity: {item['severity'].upper()}")
                                        st.markdown('</div>', unsafe_allow_html=True)
                                    else:
                                        st.write(f"Confidence: {item['confidence']*100:.1f}%")
                else:
                    st.info("No detections found in this image.")
      
        # Export results
        st.subheader("Export Results")
        export_data = {
            "batch_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "total_images": total_images,
            "total_detections": total_detections,
            "fallback_detections": fallback_detections,
            "results": results
        }
      
        json_str = json.dumps(export_data, indent=2)
        st.download_button(
            label="📥 Download Batch Results (JSON)",
            data=json_str,
            file_name=f"batch_demo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
def show_detection_results(original_img, processed_img, detections, image_name=None, processing_type="hazard"):
    st.subheader("🎯 Detection Results")
  
    # Image comparison
    col1, col2 = st.columns(2)
  
    with col1:
        st.write("**Original Image**")
        st.image(original_img, use_container_width=True)
  
    with col2:
        st.write("**Processed Image**")
        st.image(processed_img, use_container_width=True)
  
    # Detection details
    st.subheader("📊 Detected Objects")
  
    total_detections = sum(len(detections[key]) for key in detections)
    if total_detections == 0:
        st.info("ℹ️ No objects detected in this image.")
        return
  
    # Count fallback detections
    fallback_detections = 0
    for category, items in detections.items():
        for item in items:
            if 'method' in item and 'fallback' in item.get('method', ''):
                fallback_detections += 1
  
    # Summary cards based on processing type
    if processing_type in ["hazard", "all"]:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            hazard_detections = sum(len(detections[key]) for key in ["potholes", "stalled_vehicles", "speed_breakers", "manholes", "debris"])
            st.metric("Hazards Detected", hazard_detections)
        with col2:
            high_severity = sum(1 for key in ["potholes", "stalled_vehicles", "speed_breakers", "manholes", "debris"]
                              for item in detections[key] if item.get('severity') == 'high')
            st.metric("High Severity", high_severity)
        with col3:
            if hazard_detections > 0:
                total_confidence = sum(item['confidence'] for key in ["potholes", "stalled_vehicles", "speed_breakers", "manholes", "debris"]
                                     for item in detections[key])
                avg_confidence = total_confidence / hazard_detections
                st.metric("Avg Confidence", f"{avg_confidence*100:.1f}%")
            else:
                st.metric("Avg Confidence", "0%")
        with col4:
            st.metric("Fallback Detections", fallback_detections)
  
    if processing_type in ["privacy", "all"]:
        col1, col2 = st.columns(2)
        with col1:
            privacy_detections = sum(len(detections[key]) for key in ["faces", "license_plates"])
            st.metric("Privacy Objects", privacy_detections)
        with col2:
            if privacy_detections > 0:
                total_confidence = sum(item['confidence'] for key in ["faces", "license_plates"] for item in detections[key])
                avg_confidence = total_confidence / privacy_detections
                st.metric("Avg Confidence", f"{avg_confidence*100:.1f}%")
            else:
                st.metric("Avg Confidence", "0%")
  
    # Display detections by category
    for category, items in detections.items():
        if items:
            with st.expander(f"📁 {category.replace('_', ' ').title()} ({len(items)})", expanded=True):
                for i, item in enumerate(items):
                    if 'method' in item and 'fallback' in item.get('method', ''):
                        # Fallback detection with special styling
                        st.markdown(f'<div class="fallback-detection">', unsafe_allow_html=True)
                        st.write(f"🟣 **FALLBACK DETECTION**")
                        st.write(f"**Method:** {item['method']}")
                        st.write(f"**Confidence:** {item['confidence']*100:.1f}%")
                        st.write(f"**Bounding Box:** {item['bbox']}")
                        if 'severity' in item:
                            st.write(f"**Severity:** {item['severity'].upper()}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    elif 'severity' in item:
                        # Hazard detection with severity
                        severity_class = f"detection-{item['severity']}"
                        st.markdown(f'<div class="{severity_class}">', unsafe_allow_html=True)
                        st.write(f"**Confidence:** {item['confidence']*100:.1f}%")
                        st.write(f"**Bounding Box:** {item['bbox']}")
                        st.write(f"**Severity:** {item['severity'].upper()}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        # Privacy detection without severity
                        st.write(f"**Confidence:** {item['confidence']*100:.1f}%")
                        st.write(f"**Bounding Box:** {item['bbox']}")
                        st.write("✅ **Blurred for privacy**")
  
    # Report section (only for hazard detections)
    if processing_type in ["hazard", "all"] and any(len(detections[key]) > 0 for key in ["potholes", "stalled_vehicles", "speed_breakers", "manholes", "debris"]):
        st.subheader("📤 Report Hazards")
      
        col1, col2 = st.columns(2)
      
        with col1:
            location = st.text_input("📍 Location", placeholder="Enter hazard location", key="report_location")
      
        with col2:
            priority = st.selectbox("🎯 Priority", ["Low", "Medium", "High"], key="report_priority")
      
        notes = st.text_area("📝 Additional Notes", placeholder="Add any additional details about the hazards", key="report_notes")
      
        col1, col2 = st.columns(2)
      
        with col1:
            if st.button("📤 Send Report to Cloud", type="primary", use_container_width=True, key="send_report"):
                if location:
                    # Save to session state
                    report = {
                        "timestamp": datetime.now().isoformat(),
                        "image_name": image_name or "uploaded_image",
                        "location": location,
                        "priority": priority,
                        "notes": notes,
                        "detections": detections,
                        "user": st.session_state.current_user,
                        "processing_type": processing_type
                    }
                    st.session_state.detection_history.append(report)
                  
                    # Also publish to CloudAMQP
                    map_system = st.session_state.map_system
                    for category, items in detections.items():
                        if category in ["potholes", "stalled_vehicles", "speed_breakers", "manholes", "debris"]:
                            for item in items:
                                hazard_data = {
                                    'id': str(uuid.uuid4()),
                                    'type': category[:-1] if category != 'speed_breakers' else 'speed_breaker',  # Handle plural
                                    'lat': 12.9716, # Default coordinates - in real app, get from GPS
                                    'lon': 77.5946,
                                    'severity': 8 if item.get('severity') == 'high' else 5 if item.get('severity') == 'medium' else 3,
                                    'confidence': item['confidence'],
                                    'reported_by': st.session_state.current_user,
                                    'timestamp': datetime.now().isoformat(),
                                    'source': 'user_reported'
                                }
                                map_system.cloudamqp.publish_hazard(hazard_data)
                  
                    st.success("✅ Report sent successfully to CloudAMQP!")
                else:
                    st.error("❌ Please enter a location for the report.")
      
        with col2:
            if st.button("💾 Save Local Report", use_container_width=True, key="save_report"):
                # Create downloadable report
                report_data = {
                    "detections": detections,
                    "timestamp": datetime.now().isoformat(),
                    "user": st.session_state.current_user,
                    "image_name": image_name or "uploaded_image",
                    "processing_type": processing_type
                }
              
                # Convert to JSON for download
                json_str = json.dumps(report_data, indent=2)
                st.download_button(
                    label="📥 Download Report (JSON)",
                    data=json_str,
                    file_name=f"detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
def show_map_routing():
    st.markdown('<h1 class="main-header">Real-time Map Routing & Hazard Alerts</h1>', unsafe_allow_html=True)
    st.info("🗺️ Plan your route and see hazards marked from CloudAMQP. Receive deterministic driving suggestions.")
  
    map_system = st.session_state.map_system
  
    # Route planning section
    st.subheader("📍 Route Planning")
    col1, col2 = st.columns(2)
  
    with col1:
        start_address = st.text_input("Start Address", placeholder="e.g., Bangalore, India")
    with col2:
        end_address = st.text_input("End Address", placeholder="e.g., Mysore, India")
  
    if st.button("🗺️ Plan Route", type="primary"):
        if start_address and end_address:
            with st.spinner("Geocoding and planning route..."):
                start_geo = map_system.geocode(start_address)
                end_geo = map_system.geocode(end_address)
                if start_geo['success'] and end_geo['success']:
                    route_result = map_system.plan_route(
                        start_geo['lat'], start_geo['lon'],
                        end_geo['lat'], end_geo['lon']
                    )
                    if route_result['success']:
                        st.success(f"✅ Route planned: {route_result['distance_km']} km, ~{route_result['duration_min']} min")
                    else:
                        st.error(route_result['message'])
                else:
                    st.error("Geocoding failed. Check addresses.")
        else:
            st.warning("Please enter both start and end addresses.")
  
    # Journey controls
    if map_system.current_route:
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🚗 Start Journey", disabled=map_system.journey_started):
                map_system.start_journey()
                st.success("Journey started! Simulating progress...")
        with col2:
            if st.button("⏹️ End Journey", disabled=not map_system.journey_started):
                map_system.end_journey()
                st.success("Journey ended.")
        with col3:
            if st.button("🧹 Clear Hazards"):
                map_system.clear_hazards()
                st.success("Hazards cleared.")
  
        # Display route info
        st.metric("Distance", f"{map_system.current_route['distance_km']} km")
        st.metric("Estimated Time", f"{map_system.current_route['duration_min']} min")
  
        # Map display
        st.subheader("🗺️ Interactive Route Map")
        m = folium.Map(location=[map_system.route_coordinates[0][0], map_system.route_coordinates[0][1]], zoom_start=10)
  
        # Draw route
        folium.PolyLine(
            locations=map_system.route_coordinates,
            color="blue",
            weight=5,
            opacity=0.7,
            popup="Planned Route"
        ).add_to(m)
  
        # Mark hazards near route from CloudAMQP and simulated - real-time fetch
        route_hazards = map_system.get_route_hazards(max_distance=500)  # Within 500m of route
        for hazard in route_hazards:
            severity = hazard['severity']
            color = "red" if severity >= 7 else "orange" if severity >= 5 else "green"
            folium.CircleMarker(
                location=[hazard['lat'], hazard['lon']],
                radius=8,
                popup=f"{hazard['type'].title()} (Severity: {severity}/10)<br>Distance to route: {hazard.get('route_distance', 0):.0f}m<br>Source: {hazard.get('source', 'unknown')}",
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7
            ).add_to(m)
  
        # Current position simulation
        current_pos = map_system.get_current_position()
        if current_pos and map_system.journey_started:
            folium.Marker(
                location=current_pos,
                popup="Current Position",
                icon=folium.Icon(color="blue", icon="car")
            ).add_to(m)
  
        # Display map
        map_data = st_folium(m, width=800, height=500)
  
        # Alerts section
        st.subheader("🚨 Real-time Alerts")
        if map_system.journey_started:
            current_pos = map_system.get_current_position()
            if current_pos:
                alert = map_system.generate_alert(current_pos[0], current_pos[1], speed_kmph=50)
                if alert:
                    alert_class = "hazard-alert" if alert['severity_level'] == 'high' else "alert-box"
                    st.markdown(f'''
                    <div class="{alert_class}">
                        <h4>{alert['short_alert']}</h4>
                        <p><strong>Suggestion:</strong> {alert['suggestion']}</p>
                        <small>Severity: {alert['severity_level'].upper()} | Recommended Speed: {alert['recommended_speed_kmh']} km/h</small>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.success("✅ Safe to proceed. No nearby hazards.")
            else:
                st.info("Journey not active.")
        else:
            st.info("Start the journey to receive real-time alerts.")
  
        # Hazards on route list
        st.subheader("⚠️ Hazards on Route")
        if route_hazards:
            for hazard in route_hazards:
                with st.expander(f"{hazard['type'].title()} - {hazard.get('route_distance', 0):.0f}m from route"):
                    st.write(f"**Severity:** {hazard['severity']}/10")
                    st.write(f"**Location:** {hazard['lat']:.4f}, {hazard['lon']:.4f}")
                    st.write(f"**Source:** {hazard.get('source', 'unknown')}")
                    st.write(f"**Timestamp:** {hazard.get('timestamp', 'N/A')}")
        else:
            st.info("No hazards detected near this route.")
    else:
        st.info("Plan a route to begin.")
def show_history():
    st.markdown('<h1 class="main-header">Detection History</h1>', unsafe_allow_html=True)
    if st.session_state.detection_history:
        for record in st.session_state.detection_history[-10:]:  # Last 10
            with st.expander(f"{record['image_name']} - {record['timestamp'][:16]}"):
                st.json(record)
    else:
        st.info("No detection history yet.")
def show_analytics():
    st.markdown('<h1 class="main-header">Detection Analytics</h1>', unsafe_allow_html=True)
    if st.session_state.detection_history:
        df = pd.DataFrame(st.session_state.detection_history)
        st.plotly_chart(px.bar(df, x='timestamp', y='priority', title='Detections by Priority'))
    else:
        st.info("No data for analytics.")
def show_settings():
    st.markdown('<h1 class="main-header">Settings</h1>', unsafe_allow_html=True)
    st.info("App settings placeholder.")
if __name__ == "__main__":
    main()