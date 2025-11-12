#!/usr/bin/env python3
"""
app.py — Flask Backend for Road Hazard Detection.

Fixed version with proper demo image integration.
"""
import os, time, json, uuid, hashlib, socket, sys, traceback, logging, base64
from datetime import datetime, timezone
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import cv2, numpy as np, torch
from ultralytics import YOLO
import io
from demo_images import get_demo_images

# ------------- LOGGING SETUP -------------
LOG_LEVEL = logging.INFO
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('detector.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Fix for Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

# Metrics logging
METRICS_PATH = "metrics.jsonl"

def log_metric(event, **kwargs):
    record = {"event": event, "timestamp": now_iso_utc(), **kwargs}
    try:
        with open(METRICS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Failed to log metric: {e}")

# ------------- USER CONFIG -------------
POTHOLE_MODEL_PATH = r"D:\Volks\Hazard_detection\YOLOv8_Small_RDD.pt"
HAZARD_MODEL_PATH = r"D:\Volks\Hazard_detection\best.pt"
PLATE_MODEL_PATH = r"D:\Volks\Hazard_detection\license_plate_detector.pt"
FACE_MODEL_PATH = r"D:\Volks\Hazard_detection\yolov8n-face.pt"

CLOUDAMQP_URL = "amqps://dcvgkvoo:qGfhKE1foRmDOBV5TQlXpyJBShbWfjn1@puffin.rmq2.cloudamqp.com/dcvgkvoo"
CLOUDAMQP_QUEUE = "pothole_events"

WEBHOOK_URL = None

PHONE_GPS_IP = "192.168.0.112"
PHONE_GPS_PORT = 11123
PHONE_GPS_TIMEOUT = 1.0

CAMERA_INDEX = 0
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
INFERENCE_SIZE = 640  # Increased for better detection

TARGET_PROCESS_FPS = 20.0
TARGET_DISPLAY_FPS = 30.0

USE_GPU = False
USE_HALF = False

# Adjusted confidence thresholds
CONF_THRESHOLD = 0.25
CONF_REPORT_THRESHOLD = 0.45
HAZARD_CONF_THRESHOLD = 0.15
HAZARD_REPORT_THRESHOLD = 0.25
STABLE_SECONDS = 5.0
MISS_TIMEOUT = 2.0
IOU_MATCH_THRESHOLD = 0.3
REPORT_COOLDOWN_SECONDS = 60.0
MIN_RUNTIME_SECONDS = 5.0

# Stalled vehicle detection
STALLED_CONF_THRESHOLD = 0.25
STALLED_REPORT_THRESHOLD = 0.45
STALLED_SECONDS = 10.0
MOVEMENT_THRESHOLD_PX = 50

SNAPSHOT_DIR = "snapshots"
REPORTS_PATH = "reports.jsonl"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

DEFAULT_FACE_KERNEL = (99, 99)
DEFAULT_PLATE_KERNEL = (99, 99)

SEND_RETRIES = 3
RETRY_BACKOFF = 1.5

# ------------- UTILITY FUNCTIONS -------------
def now_iso_utc():
    return datetime.now(timezone.utc).isoformat()

def stable_vehicle_hash():
    try:
        node = uuid.getnode()
        mac = f"{node:012x}"
    except Exception:
        mac = uuid.uuid4().hex
    return hashlib.sha256(mac.encode()).hexdigest()[:32]

VEHICLE_ID_HASH = stable_vehicle_hash()

def make_hazard_id(prefix="ph"):
    t = int(time.time())
    rnd = hashlib.sha256(f"{t}_{uuid.uuid4().hex}".encode()).hexdigest()[:8]
    return f"{prefix}_{t}_{rnd}"

def save_report_local(record, path=REPORTS_PATH):
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"Report saved locally: {record.get('hazard_id', 'unknown')}")
    except Exception as e:
        logger.warning(f"Failed to save report locally: {e}")

# ASCII Boxed JSON printer
def print_json_boxed(json_str, title="Payload", width=100):
    try:
        data = json.loads(json_str)
        pretty_json = json.dumps(data, indent=2, ensure_ascii=False)
    except:
        pretty_json = json_str
    lines = pretty_json.split('\n')
    truncated_lines = []
    for line in lines:
        if len(line) > width - 4:
            truncated_lines.append(line[:width-7] + "...")
        else:
            truncated_lines.append(line)
    max_len = max(len(line) for line in truncated_lines)
    max_len = min(max_len, width - 4)
    
    top = "+" + "-" * (max_len + 2) + "+"
    mid_title = "| " + title.center(max_len) + " |"
    mid_sep = "+" + "-" * (max_len + 2) + "+"
    bottom = "+" + "-" * (max_len + 2) + "+"
    sides = "|"

    print(top)
    print(mid_title)
    print(mid_sep)
    for line in truncated_lines:
        print(f"{sides} {line:<{max_len}} {sides}")
    print(bottom)
    print()

def send_webhook_notification(message):
    if not WEBHOOK_URL:
        return
    try:
        import requests
        requests.post(WEBHOOK_URL, json={"content": message}, timeout=5)
        logger.info("Webhook notification sent")
    except Exception as e:
        logger.warning(f"Webhook send failed: {e}")

def send_to_cloudamqp(payload, url=CLOUDAMQP_URL, queue=CLOUDAMQP_QUEUE, retries=SEND_RETRIES):
    try:
        import pika
    except Exception as e:
        logger.error(f"pika missing — install `pip install pika` to enable CloudAMQP. Error: {e}")
        return False, "pika-missing"
    
    attempt = 0
    backoff = 1.0
    while attempt < retries:
        attempt += 1
        try:
            params = pika.URLParameters(url)
            conn = pika.BlockingConnection(params)
            ch = conn.channel()
            ch.queue_declare(queue=queue, durable=True)
            body = json.dumps(payload, ensure_ascii=False, indent=2)
            logger.info(f"AMQP: CloudAMQP connected (attempt {attempt}). Publishing payload:")
            print_json_boxed(body, title=f"CloudAMQP {payload.get('type', 'event')} Payload (Attempt {attempt})")
            ch.basic_publish(exchange='', routing_key=queue, body=json.dumps(payload, ensure_ascii=False),
                             properties=pika.BasicProperties(delivery_mode=2, content_type='application/json'))
            conn.close()
            logger.info("SUCCESS: Published to CloudAMQP")
            send_webhook_notification(f"ALERT: {payload.get('type', 'event')} reported: {payload.get('hazard_id', 'unknown')}")
            log_metric("cloud_send_success", hazard_id=payload.get('hazard_id'), type=payload.get('type'), attempt=attempt)
            return True, None
        except Exception as e:
            logger.error(f"WARNING: CloudAMQP attempt {attempt} failed: {e}")
            time.sleep(backoff)
            backoff *= RETRY_BACKOFF
            log_metric("cloud_send_fail", hazard_id=payload.get('hazard_id', 'unknown'), attempt=attempt, error=str(e))
    return False, "max_retries"

# Blur Manager Class
class BlurManager:
    @staticmethod
    def oddify(x):
        x = max(1, int(x))
        return x if (x%2)==1 else x+1

    @staticmethod
    def make_valid_kernel(desired_kernel, roi_w, roi_h):
        kx_des, ky_des = desired_kernel
        kx = min(kx_des, roi_w if roi_w>0 else 1)
        ky = min(ky_des, roi_h if roi_h>0 else 1)
        kx = BlurManager.oddify(kx)
        ky = BlurManager.oddify(ky)
        if kx >= roi_w: kx = BlurManager.oddify(max(3, roi_w-1))
        if ky >= roi_h: ky = BlurManager.oddify(max(3, roi_h-1))
        return (kx, ky)

    @staticmethod
    def safe_blur_roi(img, x1,y1,x2,y2, desired_kernel):
        h,w = img.shape[:2]
        x1 = max(0,int(round(x1)))
        y1 = max(0,int(round(y1)))
        x2 = min(w,int(round(x2)))
        y2 = min(h,int(round(y2)))
        if x2<=x1 or y2<=y1: return
        roi = img[y1:y2, x1:x2]
        roi_h, roi_w = roi.shape[:2]
        kx, ky = BlurManager.make_valid_kernel(desired_kernel, roi_w, roi_h)
        try:
            if kx>=3 and ky>=3: 
                img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (kx,ky), 0)
            else: 
                img[y1:y2, x1:x2] = cv2.blur(roi, (kx,ky))
        except Exception:
            try:
                small = cv2.resize(roi, (max(1, roi_w//8), max(1, roi_h//8)))
                img[y1:y2, x1:x2] = cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
            except Exception:
                pass

# Drawing Manager Class
class DrawingManager:
    @staticmethod
    def draw_custom_boxes(img, boxes, classes, confs, names=None, color=(0,0,255), thickness=2, transform=None):
        for i, b in enumerate(boxes):
            x1,y1,x2,y2 = [int(x) for x in b]
            if transform:
                x1p,y1p,x2p,y2p = transform(x1,y1,x2,y2)
            else:
                x1p,y1p,x2p,y2p = x1,y1,x2,y2
            label = f"{confs[i]:.2f}"
            try:
                if names and int(classes[i]) < len(names):
                    label = f"{names[int(classes[i])]} {confs[i]:.2f}"
            except Exception:
                pass
            cv2.rectangle(img, (x1p,y1p), (x2p,y2p), color, thickness, lineType=cv2.LINE_AA)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1p, y1p-th-6), (x1p+tw+6, y1p), (0,0,0), -1)
            cv2.putText(img, label, (x1p+3, y1p-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

# Enhanced Severity with aspect ratio
def compute_severity(conf, box, frame_w, frame_h):
    x1,y1,x2,y2 = [int(x) for x in box]
    area = max(0, (x2-x1)*(y2-y1))
    area_ratio = area / (frame_w*frame_h)
    aspect = frame_w / frame_h
    area_weight = 0.3 * (1 + abs(aspect - 1.78) * 0.5)
    conf_weight = 1 - area_weight
    score = conf * conf_weight + area_ratio * area_weight
    if score >= 0.6: return "high"
    if score >= 0.35: return "medium"
    return "low"

# Detection extraction - FIXED VERSION
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
        logger.error(f"Error extracting detections: {e}")
        return np.empty((0, 4)), np.empty(0), np.empty(0)

def iou(a,b):
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    ix1 = max(ax1,bx1)
    iy1 = max(ay1,by1)
    ix2 = min(ax2,bx2)
    iy2 = min(ay2,by2)
    iw = max(0, ix2-ix1)
    ih = max(0, iy2-iy1)
    inter = iw*ih
    area_a = max(0,(ax2-ax1)*(ay2-ay1))
    area_b = max(0,(bx2-bx1)*(by2-by1))
    union = area_a + area_b - inter
    return inter/union if union>0 else 0.0

# ------------- MODELS & CAMERA SETUP -------------
# Device detection
if torch.backends.mps.is_available() and sys.platform == "darwin":
    device = "mps"
elif USE_GPU and torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
half = USE_HALF and device in ["cuda", "mps"]
logger.info(f"Device: {device} | half: {half}")

# Load models with better error handling
pothole_model = None
hazard_model = None
plate_model = None
face_model = None

try:
    logger.info(f"Loading pothole model: {POTHOLE_MODEL_PATH}")
    if os.path.exists(POTHOLE_MODEL_PATH):
        pothole_model = YOLO(POTHOLE_MODEL_PATH, task="detect")
        pothole_model.to(device)
        logger.info("Pothole model loaded successfully")
    else:
        logger.error(f"Pothole model file not found: {POTHOLE_MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load pothole model: {e}")
    pothole_model = None

try:
    if HAZARD_MODEL_PATH and os.path.exists(HAZARD_MODEL_PATH):
        hazard_model = YOLO(HAZARD_MODEL_PATH, task="detect")
        hazard_model.to(device)
        logger.info("Hazard model loaded successfully")
    else:
        logger.warning("Hazard model path not set or file not found")
except Exception as e:
    logger.error(f"Failed to load hazard model: {e}")
    hazard_model = None

try:
    if PLATE_MODEL_PATH and os.path.exists(PLATE_MODEL_PATH):
        plate_model = YOLO(PLATE_MODEL_PATH, task="detect")
        plate_model.to(device)
        logger.info("Plate model loaded successfully")
    else:
        logger.warning("Plate model path not set or file not found")
except Exception as e:
    logger.error(f"Failed to load plate model: {e}")
    plate_model = None

try:
    if FACE_MODEL_PATH and os.path.exists(FACE_MODEL_PATH):
        face_model = YOLO(FACE_MODEL_PATH, task="detect")
        face_model.to(device)
        logger.info("Face model loaded successfully")
    else:
        logger.warning("Face model path not set or file not found")
except Exception as e:
    logger.error(f"Failed to load face model: {e}")
    face_model = None

# Check if at least one model is loaded
if not any([pothole_model, hazard_model, plate_model]):
    logger.error("No models loaded successfully! Check model paths and files.")

def open_camera(index):
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF] if hasattr(cv2, 'CAP_DSHOW') else [0]
    for b in backends:
        try:
            cap = cv2.VideoCapture(index, b)
            if cap.isOpened(): return cap
            else: cap.release()
        except Exception:
            pass
    return cv2.VideoCapture(index)

# Deterministic GPS/IMU mocks for backend
def get_mock_gps():
    return {"lat": 12.9716, "lon": 77.5946, "src": "mock"}

def get_mock_imu():
    t = int(time.time())
    ax = 0.01 * ((t % 7) + 1)
    ay = -0.02 * ((t % 5) + 1)
    az = 9.81 + 0.01 * ((t % 3) + 1)
    return {"acc": (round(ax,4), round(ay,4), round(az,4)), "gyro": (0.0,0.0,0.0), "src":"mock"}

# Flask App
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'bmp'}

@app.route('/')
def index():
    return jsonify({
        "status": "Road Hazard Detection API is running",
        "models_loaded": {
            "pothole": pothole_model is not None,
            "hazard": hazard_model is not None,
            "plate": plate_model is not None,
            "face": face_model is not None
        },
        "device": device
    })

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    try:
        # Read and validate image
        img_bytes = file.read()
        if len(img_bytes) == 0:
            return jsonify({"error": "Empty image file"}), 400
            
        nparr = np.frombuffer(img_bytes, np.uint8)
        orig_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if orig_frame is None:
            return jsonify({"error": "Could not decode image"}), 400

        fh, fw = orig_frame.shape[:2]
        logger.info(f"Processing image: {fw}x{fh}")
        
        # Create working copy
        frame = orig_frame.copy()

        detections = {
            "potholes": [],
            "stalled_vehicles": [],
            "speed_breakers": [],
            "manholes": [],
            "debris": []
        }

        # Pothole detection
        if pothole_model is not None:
            try:
                res = pothole_model(frame, imgsz=INFERENCE_SIZE, conf=CONF_THRESHOLD, half=half, verbose=False)
                if len(res) > 0:
                    boxes, confs, classes = extract_detections(res[0])
                    logger.info(f"Pothole model found {len(boxes)} raw detections")
                    
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
                    logger.info(f"Pothole detections after filtering: {len(detections['potholes'])}")
            except Exception as e:
                logger.error(f"Pothole detection failed: {e}")

        # Hazard detection
        if hazard_model is not None:
            try:
                hres = hazard_model(frame, imgsz=INFERENCE_SIZE, conf=HAZARD_CONF_THRESHOLD, half=half, verbose=False)
                if len(hres) > 0:
                    hboxes, hconfs, hcls = extract_detections(hres[0])
                    logger.info(f"Hazard model found {len(hboxes)} raw detections")
                    
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
                                if cls == 0:
                                    detections["speed_breakers"].append(det)
                                elif cls == 1:
                                    detections["manholes"].append(det)
                                elif cls == 2:
                                    detections["debris"].append(det)
            except Exception as e:
                logger.error(f"Hazard detection failed: {e}")

        # Plate detection for stalled vehicles
        if plate_model is not None:
            try:
                pres = plate_model(frame, imgsz=416, conf=STALLED_CONF_THRESHOLD, half=half, verbose=False)
                if len(pres) > 0:
                    pboxes, pconfs, pcls = extract_detections(pres[0])
                    logger.info(f"Plate model found {len(pboxes)} raw detections")
                    
                    for bi, box in enumerate(pboxes):
                        if bi < len(pconfs):
                            conf = float(pconfs[bi])
                            if conf >= STALLED_CONF_THRESHOLD:
                                x1,y1,x2,y2 = [int(x) for x in box]
                                plate_w = x2 - x1
                                plate_h = y2 - y1
                                veh_x1 = max(0, x1 - plate_w * 1.5)
                                veh_y1 = max(0, y1 - plate_h * 0.5)
                                veh_x2 = min(fw, x2 + plate_w * 1.5)
                                veh_y2 = min(fh, y2 + plate_h * 1.5)
                                detections["stalled_vehicles"].append({
                                    "bbox": [veh_x1, veh_y1, veh_x2, veh_y2],
                                    "confidence": round(conf, 3),
                                    "severity": "medium"
                                })
            except Exception as e:
                logger.error(f"Plate detection failed: {e}")

        # Face detection for blurring
        face_boxes = np.empty((0,4))
        face_confs = np.empty(0)
        if face_model is not None:
            try:
                fres = face_model(frame, imgsz=416, conf=0.01, half=half, verbose=False)
                if len(fres) > 0:
                    face_boxes, face_confs, _ = extract_detections(fres[0])
                    logger.info(f"Face model found {len(face_boxes)} faces")
            except Exception as e:
                logger.error(f"Face detection failed: {e}")

        # Blur faces
        for bi, box in enumerate(face_boxes):
            if bi < len(face_confs) and float(face_confs[bi]) >= 0.01:
                x1,y1,x2,y2 = [int(x) for x in box]
                BlurManager.safe_blur_roi(frame, x1,y1,x2,y2, DEFAULT_FACE_KERNEL)

        # Blur plates
        if plate_model is not None:
            try:
                pres = plate_model(frame, imgsz=416, conf=0.20, half=half, verbose=False)
                if len(pres) > 0:
                    pboxes, pconfs, _ = extract_detections(pres[0])
                    for bi, box in enumerate(pboxes):
                        if bi < len(pconfs) and float(pconfs[bi]) >= 0.20:
                            x1,y1,x2,y2 = [int(x) for x in box]
                            BlurManager.safe_blur_roi(frame, x1,y1,x2,y2, DEFAULT_PLATE_KERNEL)
            except Exception as e:
                logger.error(f"Plate blurring failed: {e}")

        # Encode blurred image to base64
        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            return jsonify({"error": "Failed to encode image"}), 500
            
        blurred_image_b64 = base64.b64encode(buffer).decode('utf-8')

        response_data = {
            "detections": detections,
            "blurred_image": blurred_image_b64,
            "image_info": {
                "width": fw,
                "height": fh
            }
        }
        
        logger.info(f"Detection completed: {sum(len(v) for v in detections.values())} total detections")
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Detection error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.route('/send_report', methods=['POST'])
def send_report():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data"}), 400
    try:
        ok, err = send_to_cloudamqp(data, CLOUDAMQP_URL, CLOUDAMQP_QUEUE, retries=SEND_RETRIES)
        if ok:
            return jsonify({"status": "Report sent successfully to CloudAMQP", "hazard_id": data.get('hazard_id')})
        else:
            return jsonify({"error": err}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_demo_images', methods=['GET'])
def get_demo_images_endpoint():
    """Return list of demo images with base64 encoded data."""
    try:
        demo_path = request.args.get('path')
        logger.info(f"Loading demo images from path: {demo_path}")
        images = get_demo_images(demo_path)
        
        # Filter out images with empty data
        valid_images = [img for img in images if img.get('data')]
        logger.info(f"Loaded {len(valid_images)} valid demo images")
        
        return jsonify({
            "demo_images": valid_images,
            "total_count": len(valid_images),
            "loaded_from": demo_path or "default"
        })
    except Exception as e:
        logger.error(f"Error loading demo images: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/process_demo', methods=['POST'])
def process_demo():
    """Process a demo image by name and return detections with annotated image."""
    data = request.get_json()
    if not data or 'image_name' not in data:
        return jsonify({"error": "image_name required"}), 400

    image_name = data['image_name']
    demo_path = data.get('demo_path')  # Optional custom demo path
    
    logger.info(f"Processing demo image: {image_name} from path: {demo_path}")

    try:
        # Get demo images from the specified path or default
        demo_images = get_demo_images(demo_path)
        
        # Find the requested image
        image_data = None
        image_description = ""
        for img in demo_images:
            if img['name'] == image_name:
                image_data = img.get('data')
                image_description = img.get('description', '')
                break

        if not image_data:
            logger.error(f"Demo image not found: {image_name}")
            return jsonify({"error": f"Demo image '{image_name}' not found"}), 404

        # Decode base64 image
        try:
            img_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            orig_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if orig_frame is None:
                return jsonify({"error": "Invalid demo image data"}), 400
        except Exception as e:
            logger.error(f"Failed to decode demo image: {e}")
            return jsonify({"error": "Invalid image data"}), 400

        fh, fw = orig_frame.shape[:2]
        logger.info(f"Processing demo image: {image_name} ({fw}x{fh})")
        frame = orig_frame.copy()

        detections = {
            "potholes": [],
            "stalled_vehicles": [],
            "speed_breakers": [],
            "manholes": [],
            "debris": []
        }

        # Pothole detection with bounding boxes
        if pothole_model is not None:
            try:
                res = pothole_model(frame, imgsz=INFERENCE_SIZE, conf=CONF_THRESHOLD, half=half, verbose=False)
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
                                # Draw bounding box
                                DrawingManager.draw_custom_boxes(frame, [box], [0], [conf], names=["pothole"], color=(0, 0, 255))
            except Exception as e:
                logger.error(f"Pothole detection failed in demo: {e}")

        # Hazard detection with bounding boxes
        if hazard_model is not None:
            try:
                hres = hazard_model(frame, imgsz=INFERENCE_SIZE, conf=HAZARD_CONF_THRESHOLD, half=half, verbose=False)
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
                                # Different colors for different hazard types
                                colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255)]  # green, blue, yellow
                                color = colors[cls] if cls < len(colors) else (0, 255, 0)
                                
                                if cls == 0:
                                    detections["speed_breakers"].append(det)
                                    DrawingManager.draw_custom_boxes(frame, [box], [cls], [conf], names=["speed_breaker", "manhole", "debris"], color=color)
                                elif cls == 1:
                                    detections["manholes"].append(det)
                                    DrawingManager.draw_custom_boxes(frame, [box], [cls], [conf], names=["speed_breaker", "manhole", "debris"], color=color)
                                elif cls == 2:
                                    detections["debris"].append(det)
                                    DrawingManager.draw_custom_boxes(frame, [box], [cls], [conf], names=["speed_breaker", "manhole", "debris"], color=color)
            except Exception as e:
                logger.error(f"Hazard detection failed in demo: {e}")

        # Plate detection for stalled vehicles with bounding boxes
        if plate_model is not None:
            try:
                pres = plate_model(frame, imgsz=416, conf=STALLED_CONF_THRESHOLD, half=half, verbose=False)
                if len(pres) > 0:
                    pboxes, pconfs, pcls = extract_detections(pres[0])
                    for bi, box in enumerate(pboxes):
                        if bi < len(pconfs):
                            conf = float(pconfs[bi])
                            if conf >= STALLED_CONF_THRESHOLD:
                                x1,y1,x2,y2 = [int(x) for x in box]
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
                                # Draw bounding box for vehicle (purple)
                                DrawingManager.draw_custom_boxes(frame, [vehicle_box], [0], [conf], names=["stalled_vehicle"], color=(255, 0, 255))
            except Exception as e:
                logger.error(f"Plate detection failed in demo: {e}")

        # Face detection for blurring
        if face_model is not None:
            try:
                fres = face_model(frame, imgsz=416, conf=0.01, half=half, verbose=False)
                if len(fres) > 0:
                    face_boxes, face_confs, _ = extract_detections(fres[0])
                    for bi, box in enumerate(face_boxes):
                        if bi < len(face_confs) and float(face_confs[bi]) >= 0.01:
                            x1,y1,x2,y2 = [int(x) for x in box]
                            BlurManager.safe_blur_roi(frame, x1,y1,x2,y2, DEFAULT_FACE_KERNEL)
            except Exception as e:
                logger.error(f"Face blurring failed in demo: {e}")

        # Plate blurring
        if plate_model is not None:
            try:
                pres = plate_model(frame, imgsz=416, conf=0.20, half=half, verbose=False)
                if len(pres) > 0:
                    pboxes, pconfs, _ = extract_detections(pres[0])
                    for bi, box in enumerate(pboxes):
                        if bi < len(pconfs) and float(pconfs[bi]) >= 0.20:
                            x1,y1,x2,y2 = [int(x) for x in box]
                            BlurManager.safe_blur_roi(frame, x1,y1,x2,y2, DEFAULT_PLATE_KERNEL)
            except Exception as e:
                logger.error(f"Plate blurring failed in demo: {e}")

        # Encode processed image
        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            return jsonify({"error": "Failed to encode processed image"}), 500
            
        processed_image_b64 = base64.b64encode(buffer).decode('utf-8')

        logger.info(f"Demo processing completed: {sum(len(v) for v in detections.values())} detections found")

        return jsonify({
            "detections": detections,
            "processed_image": processed_image_b64,
            "image_name": image_name,
            "image_description": image_description,
            "image_info": {
                "width": fw,
                "height": fh
            }
        })

    except Exception as e:
        logger.error(f"Demo processing error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": now_iso_utc(),
        "models_loaded": {
            "pothole": pothole_model is not None,
            "hazard": hazard_model is not None,
            "plate": plate_model is not None,
            "face": face_model is not None
        }
    })

if __name__ == "__main__":
    logger.info("Starting Flask backend for Road Hazard Detection")
    logger.info(f"Models loaded - Pothole: {pothole_model is not None}, Hazard: {hazard_model is not None}, Plate: {plate_model is not None}, Face: {face_model is not None}")
    app.run(host='0.0.0.0', port=5000, debug=True)