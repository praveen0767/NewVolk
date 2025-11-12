#!/usr/bin/env python3
"""
Road Hazard Detection - Streamlit App
Combined frontend and backend in a single Streamlit application
"""

import os
import time
import json
import uuid
import hashlib
import traceback
import logging
import base64
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional

import streamlit as st
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# ------------- CONFIGURATION -------------
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

# Default demo images
DEFAULT_DEMO_IMAGES = [
    {"name": "pothole_demo.jpg", "path": "D:/volks1/images/patho.png", "description": "Road with potholes"},
    {"name": "stalled_vehicle_demo.jpg", "path": "D:/volks1/images/sta.png", "description": "Stalled vehicle with license plate"},
    {"name": "speed_breaker_demo.jpg", "path": "D:/volks1/images/ali.png", "description": "Road with speed breaker"},
    {"name": "manhole_demo.jpg", "path": "D:/volks1/images/manhole.png", "description": "Road with manhole"},
    {"name": "Aligator_demo.jpg", "path": "D:/volks1/images/ali.png", "description": "Road with debris"},
    {"name": "face_blur_demo.jpg", "path": "D:/volks1/images/man.png", "description": "Image with faces to blur"},
    {"name": "plate_blur_demo.jpg", "path": "D:/volks1/images/plate.png", "description": "Image with license plates to blur"},
]

# ------------- UTILITY CLASSES & FUNCTIONS -------------
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
                    images.append({"name": p.name, "data": b64 or "", "description": f"Demo image: {p.name}", "path": str(p)})
            except Exception as e:
                st.error(f"Failed loading image file: {p} - {e}")
        elif p.is_dir():
            for child in sorted(p.iterdir()):
                if child.suffix.lower() in _VALID_EXTS and child.is_file():
                    try:
                        img = cv2.imread(str(child))
                        if img is not None:
                            b64 = _encode_image_to_base64(img)
                            images.append({"name": child.name, "data": b64 or "", "description": f"Demo image: {child.name}", "path": str(child)})
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
                            "path": str(path_obj)
                        })
            except Exception as e:
                st.error(f"Error processing default image: {entry} - {e}")

    return images

def now_iso_utc():
    return datetime.now(timezone.utc).isoformat()

# ------------- MODEL LOADING -------------
@st.cache_resource
def load_models():
    """Load all detection models"""
    # Device detection
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    models = {}
    
    try:
        if os.path.exists(POTHOLE_MODEL_PATH):
            models['pothole'] = YOLO(POTHOLE_MODEL_PATH, task="detect")
            models['pothole'].to(device)
        else:
            st.error(f"Pothole model not found: {POTHOLE_MODEL_PATH}")
    except Exception as e:
        st.error(f"Failed to load pothole model: {e}")
    
    try:
        if HAZARD_MODEL_PATH and os.path.exists(HAZARD_MODEL_PATH):
            models['hazard'] = YOLO(HAZARD_MODEL_PATH, task="detect")
            models['hazard'].to(device)
    except Exception as e:
        st.error(f"Failed to load hazard model: {e}")
    
    try:
        if PLATE_MODEL_PATH and os.path.exists(PLATE_MODEL_PATH):
            models['plate'] = YOLO(PLATE_MODEL_PATH, task="detect")
            models['plate'].to(device)
    except Exception as e:
        st.error(f"Failed to load plate model: {e}")
    
    try:
        if FACE_MODEL_PATH and os.path.exists(FACE_MODEL_PATH):
            models['face'] = YOLO(FACE_MODEL_PATH, task="detect")
            models['face'].to(device)
    except Exception as e:
        st.error(f"Failed to load face model: {e}")
    
    return models, device

# ------------- DETECTION FUNCTIONS -------------
def process_image(image, models, draw_boxes=False):
    """Process image and return detections"""
    if image is None:
        return None
    
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
        "debris": []
    }
    
    # Pothole detection
    if 'pothole' in models:
        try:
            res = models['pothole'](frame, imgsz=INFERENCE_SIZE, conf=CONF_THRESHOLD, verbose=False)
            if len(res) > 0:
                boxes, confs, classes = extract_detections(res[0])
                for bi, box in enumerate(boxes):
                    if bi < len(confs):
                        conf = float(confs[bi])
                        if conf >= CONF_THRESHOLD:
                            severity = compute_severity(conf, box, fw, fh)
                            detections["potholes"].append({
                                "bbox": [int(x) for x in box],
                                "confidence": round(conf, 3),
                                "severity": severity
                            })
                            if draw_boxes:
                                DrawingManager.draw_custom_boxes(frame, [box], [0], [conf], names=["pothole"], color=(0, 0, 255))
        except Exception as e:
            st.error(f"Pothole detection failed: {e}")
    
    # Hazard detection
    if 'hazard' in models:
        try:
            hres = models['hazard'](frame, imgsz=INFERENCE_SIZE, conf=HAZARD_CONF_THRESHOLD, verbose=False)
            if len(hres) > 0:
                hboxes, hconfs, hcls = extract_detections(hres[0])
                for bi, box in enumerate(hboxes):
                    if bi < len(hconfs) and bi < len(hcls):
                        conf = float(hconfs[bi])
                        if conf >= HAZARD_CONF_THRESHOLD:
                            cls = int(hcls[bi])
                            severity = compute_severity(conf, box, fw, fh)
                            det = {
                                "bbox": [int(x) for x in box],
                                "confidence": round(conf, 3),
                                "severity": severity
                            }
                            colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255)]
                            color = colors[cls] if cls < len(colors) else (0, 255, 0)
                            
                            if cls == 0:
                                detections["speed_breakers"].append(det)
                                if draw_boxes:
                                    DrawingManager.draw_custom_boxes(frame, [box], [cls], [conf], names=["speed_breaker", "manhole", "debris"], color=color)
                            elif cls == 1:
                                detections["manholes"].append(det)
                                if draw_boxes:
                                    DrawingManager.draw_custom_boxes(frame, [box], [cls], [conf], names=["speed_breaker", "manhole", "debris"], color=color)
                            elif cls == 2:
                                detections["debris"].append(det)
                                if draw_boxes:
                                    DrawingManager.draw_custom_boxes(frame, [box], [cls], [conf], names=["speed_breaker", "manhole", "debris"], color=color)
        except Exception as e:
            st.error(f"Hazard detection failed: {e}")
    
    # Plate detection for stalled vehicles
    if 'plate' in models:
        try:
            pres = models['plate'](frame, imgsz=416, conf=STALLED_CONF_THRESHOLD, verbose=False)
            if len(pres) > 0:
                pboxes, pconfs, pcls = extract_detections(pres[0])
                for bi, box in enumerate(pboxes):
                    if bi < len(pconfs):
                        conf = float(pconfs[bi])
                        if conf >= STALLED_CONF_THRESHOLD:
                            x1, y1, x2, y2 = [int(x) for x in box]
                            plate_w = x2 - x1
                            plate_h = y2 - y1
                            veh_x1 = max(0, x1 - plate_w * 1.5)
                            veh_y1 = max(0, y1 - plate_h * 0.5)
                            veh_x2 = min(fw, x2 + plate_w * 1.5)
                            veh_y2 = min(fh, y2 + plate_h * 1.5)
                            vehicle_box = [veh_x1, veh_y1, veh_x2, veh_y2]
                            detections["stalled_vehicles"].append({
                                "bbox": vehicle_box,
                                "confidence": round(conf, 3),
                                "severity": "medium"
                            })
                            if draw_boxes:
                                DrawingManager.draw_custom_boxes(frame, [vehicle_box], [0], [conf], names=["stalled_vehicle"], color=(255, 0, 255))
        except Exception as e:
            st.error(f"Plate detection failed: {e}")
    
    # Face blurring
    if 'face' in models:
        try:
            fres = models['face'](frame, imgsz=416, conf=0.01, verbose=False)
            if len(fres) > 0:
                face_boxes, face_confs, _ = extract_detections(fres[0])
                for bi, box in enumerate(face_boxes):
                    if bi < len(face_confs) and float(face_confs[bi]) >= 0.01:
                        x1, y1, x2, y2 = [int(x) for x in box]
                        BlurManager.safe_blur_roi(frame, x1, y1, x2, y2, DEFAULT_FACE_KERNEL)
        except Exception as e:
            st.error(f"Face blurring failed: {e}")
    
    # Plate blurring
    if 'plate' in models:
        try:
            pres = models['plate'](frame, imgsz=416, conf=0.20, verbose=False)
            if len(pres) > 0:
                pboxes, pconfs, _ = extract_detections(pres[0])
                for bi, box in enumerate(pboxes):
                    if bi < len(pconfs) and float(pconfs[bi]) >= 0.20:
                        x1, y1, x2, y2 = [int(x) for x in box]
                        BlurManager.safe_blur_roi(frame, x1, y1, x2, y2, DEFAULT_PLATE_KERNEL)
        except Exception as e:
            st.error(f"Plate blurring failed: {e}")
    
    return detections, frame

# ------------- STREAMLIT APP -------------
def main():
    st.set_page_config(
        page_title="Road Hazard Detector - Team-GPT",
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
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    .detection-high {
        border-left: 4px solid #ff4b4b;
        padding-left: 1rem;
        margin: 0.5rem 0;
    }
    .detection-medium {
        border-left: 4px solid #ffa64b;
        padding-left: 1rem;
        margin: 0.5rem 0;
    }
    .detection-low {
        border-left: 4px solid #4caf50;
        padding-left: 1rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'models_loaded' not in st.session_state:
        with st.spinner("Loading AI models..."):
            models, device = load_models()
            st.session_state.models = models
            st.session_state.device = device
            st.session_state.models_loaded = True
    
    if 'detection_history' not in st.session_state:
        st.session_state.detection_history = []
    
    if 'current_user' not in st.session_state:
        st.session_state.current_user = "User 1"
    
    # Sidebar
    with st.sidebar:
        st.title("🛣️ Team-GPT")
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["Dashboard", "Detection", "History", "Analytics", "Settings"],
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
        status_color = "🟢" if models_loaded else "🔴"
        st.write(f"{status_color} Models: {'Loaded' if models_loaded else 'Failed'}")
        st.write(f"🔧 Device: {st.session_state.device}")
    
    # Main content based on page selection
    if page == "Dashboard":
        show_dashboard()
    elif page == "Detection":
        show_detection()
    elif page == "History":
        show_history()
    elif page == "Analytics":
        show_analytics()
    elif page == "Settings":
        show_settings()

def show_dashboard():
    st.markdown('<h1 class="main-header">Road Hazard Detection Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered detection of road hazards with privacy protection</p>', unsafe_allow_html=True)
    
    # Stats cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("High Severity Hazards", "24", "+12%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Potholes Detected", "42", "-5%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Stalled Vehicles", "18", "+8%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Detection Accuracy", "92%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Hazard map and recent detections
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Hazard Map Overview")
        # Create a simple map visualization
        map_data = pd.DataFrame({
            'lat': [12.9716, 12.9718, 12.9714, 12.9719],
            'lon': [77.5946, 77.5948, 77.5944, 77.5949],
            'size': [10, 15, 8, 12],
            'severity': ['High', 'Medium', 'Low', 'High']
        })
        
        fig = px.scatter_mapbox(
            map_data,
            lat="lat",
            lon="lon",
            size="size",
            color="severity",
            color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'},
            zoom=12,
            height=400
        )
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Recent Detections")
        
        # Sample recent detections
        recent_detections = [
            {"type": "Pothole", "severity": "high", "location": "Main St & 5th Ave", "time": "2 hours ago", "confidence": 92},
            {"type": "Stalled Vehicle", "severity": "medium", "location": "Highway 101, Exit 42", "time": "5 hours ago", "confidence": 87},
            {"type": "Debris", "severity": "low", "location": "Oak Street", "time": "1 day ago", "confidence": 78},
            {"type": "Speed Breaker", "severity": "medium", "location": "School Zone, Elm St", "time": "2 days ago", "confidence": 82}
        ]
        
        for detection in recent_detections:
            severity_class = f"detection-{detection['severity'].lower()}"
            st.markdown(f'<div class="{severity_class}">', unsafe_allow_html=True)
            st.write(f"**{detection['type']}** - {detection['confidence']}%")
            st.write(f"📍 {detection['location']}")
            st.write(f"⏰ {detection['time']}")
            st.markdown('</div>', unsafe_allow_html=True)

def show_detection():
    st.markdown('<h1 class="main-header">Road Hazard Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload road images for AI-powered hazard detection</p>', unsafe_allow_html=True)
    
    # Detection mode selection
    detection_mode = st.radio(
        "Detection Mode",
        ["Upload Image", "Demo Images"],
        horizontal=True
    )
    
    if detection_mode == "Upload Image":
        show_upload_section()
    else:
        show_demo_section()

def show_upload_section():
    uploaded_file = st.file_uploader(
        "Upload a road image",
        type=['png', 'jpg', 'jpeg', 'bmp'],
        help="Upload an image of a road to detect hazards"
    )
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
        
        # Process image
        if st.button("Detect Hazards", type="primary"):
            with st.spinner("Processing image..."):
                detections, processed_frame = process_image(image, st.session_state.models, draw_boxes=True)
            
            # Display results
            show_detection_results(image, processed_frame, detections)

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
    
    # Display demo images in a grid
    cols = st.columns(3)
    selected_image = None
    
    for i, demo_img in enumerate(demo_images):
        col = cols[i % 3]
        with col:
            if demo_img['data']:  # Only show if we have image data
                # Decode base64 image
                try:
                    img_bytes = base64.b64decode(demo_img['data'])
                    img = Image.open(io.BytesIO(img_bytes))
                    
                    st.image(img, caption=demo_img['name'], use_column_width=True)
                    if st.button(f"Process {demo_img['name']}", key=f"btn_{i}"):
                        selected_image = demo_img
                except Exception as e:
                    st.error(f"Error loading image {demo_img['name']}: {e}")
    
    # Process selected demo image
    if selected_image:
        st.subheader(f"Processing: {selected_image['name']}")
        
        with st.spinner("Processing demo image..."):
            # Decode and process
            img_bytes = base64.b64decode(selected_image['data'])
            img = Image.open(io.BytesIO(img_bytes))
            
            detections, processed_frame = process_image(img, st.session_state.models, draw_boxes=True)
            
            # Display results
            show_detection_results(img, processed_frame, detections, selected_image['name'])

def show_detection_results(original_img, processed_frame, detections, image_name=None):
    st.subheader("Detection Results")
    
    # Convert processed frame back to PIL for display
    processed_img = Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
    
    # Image comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Original Image**")
        st.image(original_img, use_column_width=True)
    
    with col2:
        st.write("**Processed Image**")
        st.image(processed_img, use_column_width=True)
    
    # Detection details
    st.subheader("Detected Hazards")
    
    total_detections = sum(len(detections[key]) for key in detections)
    if total_detections == 0:
        st.info("No hazards detected in this image.")
        return
    
    # Display detections by category
    for category, items in detections.items():
        if items:
            with st.expander(f"{category.replace('_', ' ').title()} ({len(items)})", expanded=True):
                for i, item in enumerate(items):
                    severity_class = f"detection-{item['severity']}"
                    st.markdown(f'<div class="{severity_class}">', unsafe_allow_html=True)
                    st.write(f"**Confidence:** {item['confidence']*100:.1f}%")
                    st.write(f"**Bounding Box:** {item['bbox']}")
                    st.write(f"**Severity:** {item['severity'].upper()}")
                    st.markdown('</div>', unsafe_allow_html=True)
    
    # Report section
    st.subheader("Report Hazards")
    
    col1, col2 = st.columns(2)
    
    with col1:
        location = st.text_input("Location", placeholder="Enter hazard location")
    
    with col2:
        priority = st.selectbox("Priority", ["Low", "Medium", "High"])
    
    notes = st.text_area("Additional Notes", placeholder="Add any additional details about the hazards")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📤 Send Report", type="primary", use_container_width=True):
            if location:
                # Save to session state
                report = {
                    "timestamp": now_iso_utc(),
                    "image_name": image_name or "uploaded_image",
                    "location": location,
                    "priority": priority,
                    "notes": notes,
                    "detections": detections,
                    "user": st.session_state.current_user
                }
                st.session_state.detection_history.append(report)
                st.success("Report sent successfully!")
            else:
                st.error("Please enter a location for the report.")
    
    with col2:
        if st.button("💾 Save Report", use_container_width=True):
            # Create downloadable report
            report_data = {
                "detections": detections,
                "timestamp": now_iso_utc(),
                "user": st.session_state.current_user,
                "image_name": image_name or "uploaded_image"
            }
            
            # Convert to JSON for download
            json_str = json.dumps(report_data, indent=2)
            st.download_button(
                label="Download Report JSON",
                data=json_str,
                file_name=f"hazard_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

def show_history():
    st.markdown('<h1 class="main-header">Detection History</h1>', unsafe_allow_html=True)
    
    if not st.session_state.detection_history:
        st.info("No detection history available. Process some images to see history here.")
        return
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        date_filter = st.selectbox("Date Range", ["Last 7 days", "Last 30 days", "Last 3 months", "All time"])
    
    with col2:
        type_filter = st.selectbox("Hazard Type", ["All types", "Potholes", "Stalled Vehicles", "Speed Breakers", "Manholes", "Debris"])
    
    with col3:
        severity_filter = st.selectbox("Severity", ["All severities", "High", "Medium", "Low"])
    
    # History table
    st.subheader("Detection Records")
    
    # Convert history to DataFrame for display
    history_data = []
    for record in st.session_state.detection_history:
        # Count detections by type
        detection_counts = {k: len(v) for k, v in record['detections'].items() if v}
        
        history_data.append({
            "Date": record['timestamp'][:10],
            "Time": record['timestamp'][11:19],
            "Image": record['image_name'],
            "Location": record['location'],
            "Priority": record['priority'],
            "Potholes": detection_counts.get('potholes', 0),
            "Stalled Vehicles": detection_counts.get('stalled_vehicles', 0),
            "Speed Breakers": detection_counts.get('speed_breakers', 0),
            "Manholes": detection_counts.get('manholes', 0),
            "Debris": detection_counts.get('debris', 0),
            "User": record['user']
        })
    
    if history_data:
        df = pd.DataFrame(history_data)
        st.dataframe(df, use_container_width=True)
        
        # Export option
        if st.button("Export History to CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"detection_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    else:
        st.info("No records match the current filters.")

def show_analytics():
    st.markdown('<h1 class="main-header">Detection Analytics</h1>', unsafe_allow_html=True)
    
    # Analytics cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Hazards Detected", "1,247", "+18%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Average Detection Accuracy", "84%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("System Uptime", "92%", "+3%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Hazard Distribution")
        
        # Sample data for pie chart
        hazard_data = pd.DataFrame({
            'Type': ['Potholes', 'Stalled Vehicles', 'Speed Breakers', 'Manholes', 'Debris'],
            'Count': [42, 18, 15, 8, 12]
        })
        
        fig = px.pie(hazard_data, values='Count', names='Type', hole=0.3)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Detection Trends")
        
        # Sample data for line chart
        trend_data = pd.DataFrame({
            'Date': pd.date_range('2023-09-01', periods=30, freq='D'),
            'Detections': np.random.randint(10, 50, 30)
        })
        
        fig = px.line(trend_data, x='Date', y='Detections', title="Daily Detections")
        st.plotly_chart(fig, use_container_width=True)
    
    # Severity analysis
    st.subheader("Severity Analysis")
    
    severity_data = pd.DataFrame({
        'Severity': ['High', 'Medium', 'Low'],
        'Count': [24, 35, 28],
        'Color': ['#ff4b4b', '#ffa64b', '#4caf50']
    })
    
    fig = px.bar(severity_data, x='Severity', y='Count', color='Severity',
                 color_discrete_map={'High': '#ff4b4b', 'Medium': '#ffa64b', 'Low': '#4caf50'})
    st.plotly_chart(fig, use_container_width=True)

def show_settings():
    st.markdown('<h1 class="main-header">Settings</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("User Preferences")
        
        dark_mode = st.toggle("Dark Mode", value=True, help="Use dark theme for the application")
        notifications = st.toggle("Notifications", value=True, help="Receive alerts for new detections")
        auto_save = st.toggle("Auto-save Reports", value=False, help="Automatically save detection reports")
        
        st.subheader("Detection Settings")
        
        confidence_threshold = st.selectbox(
            "Confidence Threshold",
            ["Low (50%)", "Medium (70%)", "High (85%)"],
            index=1,
            help="Minimum confidence for hazard detection"
        )
        
        privacy_mode = st.toggle("Privacy Mode", value=True, help="Blur sensitive information in images")
        auto_process = st.toggle("Auto-process Images", value=True, help="Process images immediately after upload")
    
    with col2:
        st.subheader("Security & Privacy")
        
        data_encryption = st.toggle("Data Encryption", value=True, help="Encrypt all stored data")
        auto_logout = st.toggle("Auto-logout", value=False, help="Automatically logout after 30 minutes")
        audit_logging = st.toggle("Audit Logging", value=True, help="Keep logs of all user activities")
        
        st.subheader("Data Management")
        
        retention_period = st.selectbox(
            "Retention Period",
            ["30 days", "90 days", "1 year", "Indefinitely"],
            index=1,
            help="How long to keep detection data"
        )
        
        backup_frequency = st.selectbox(
            "Backup Frequency",
            ["Daily", "Weekly", "Monthly"],
            index=1,
            help="How often to backup data"
        )
    
    if st.button("Save All Settings", type="primary"):
        st.success("Settings saved successfully!")

if __name__ == "__main__":
    import io  # Add this import for BytesIO
    main()