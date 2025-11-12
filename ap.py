from flask import Flask, request, jsonify, send_from_directory
import threading
import queue
import pika
import json
import time
import requests
import numpy as np
import uuid
import math
from datetime import datetime, timedelta
import toml

# --- LLM IMPORTS (from your notebook) ---
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__, static_folder='../frontend', static_url_path='')

# ==================== CONFIG ====================
try:
    secrets = toml.load("secrets.toml")
    TOMTOM_API_KEY = secrets["TOMTOM_API_KEY"]
except:
    TOMTOM_API_KEY = "hPkM1RRYfmKjSo76dd9ShQFKzdmsuTdj"

CLOUDAMQP_URL = "amqps://dcvgkvoo:qGfhKE1foRmDOBV5TQlXpyJBShbWfjn1@puffin.rmq2.cloudamqp.com/dcvgkvoo"
OSRM_URL = "http://router.project-osrm.org"

# ==================== LLM SETUP ====================
print("Loading LLM models...")
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embed_dim = embed_model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(embed_dim)
metadatas = []

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)

def build_metadata_text(h):
    return f"{h['type']}; lat:{h['lat']}; lon:{h['lon']}; severity:{h.get('severity',0)}; conf:{h.get('confidence',0):.2f}; id:{h.get('id','')}"

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

# ==================== CloudAMQP ====================
class CloudAMQPClient:
    def __init__(self, url):
        self.url = url
        self.connection = None
        self.channel = None
        self.message_queue = queue.Queue()
        self.running = False

    def connect(self):
        for _ in range(5):
            try:
                params = pika.URLParameters(self.url)
                self.connection = pika.BlockingConnection(params)
                self.channel = self.connection.channel()
                self.channel.queue_declare(queue='hazard_alerts', durable=True)
                self.channel.queue_declare(queue='alerts', durable=True)
                print("CloudAMQP Connected")
                return True
            except: time.sleep(3)
        return False

    def publish_hazard(self, data):
        try:
            if self.channel:
                self.channel.basic_publish(exchange='', routing_key='hazard_alerts',
                    body=json.dumps(data, default=str),
                    properties=pika.BasicProperties(delivery_mode=2))
        except: pass

    def publish_alert(self, data):
        try:
            if self.channel:
                self.channel.basic_publish(exchange='', routing_key='alerts',
                    body=json.dumps(data, default=str),
                    properties=pika.BasicProperties(delivery_mode=2))
        except: pass

    def start_consuming(self):
        if not self.channel: return
        def cb(ch, method, props, body):
            try: self.message_queue.put(json.loads(body))
            except: pass
        self.channel.basic_consume(queue='hazard_alerts', on_message_callback=cb, auto_ack=True)
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self.running and self.connection:
            try: self.connection.process_data_events(time_limit=1)
            except: break

    def get_messages(self):
        msgs = []
        while not self.message_queue.empty():
            try: msgs.append(self.message_queue.get_nowait())
            except: break
        return msgs

    def close(self):
        self.running = False
        try: self.connection.close()
        except: pass

# ==================== Global State ====================
class State:
    def __init__(self):
        self.amqp = CloudAMQPClient(CLOUDAMQP_URL)
        self.amqp.connect()
        self.amqp.start_consuming()

        self.vehicle_id = f"veh_{uuid.uuid4().hex[:8]}"
        self.current_route = None
        self.journey_started = False
        self.detected_hazards = []
        self.last_detection = None
        self.simulation_thread = None
        self.last_alert = None

state = State()

# ==================== LLM Alert Generator ====================
def generate_alert(hazards, client_lat, client_lon, speed_kmph=40):
    if not hazards: return None

    # Sort by distance
    hazards = sorted(hazards, key=lambda h: haversine_m(client_lat, client_lon, h['lat'], h['lon']))
    first = hazards[0]
    dist_m = haversine_m(client_lat, client_lon, first['lat'], first['lon'])
    eta_s = int(dist_m / (speed_kmph * 1000 / 3600)) if speed_kmph > 0 else 0

    # Generate detailed instructions based on hazard type and severity
    hazard_type = first['type']
    severity = first['severity']
    
    if hazard_type == 'pothole':
        if severity >= 8:
            instructions = f"CRITICAL: Large pothole {int(dist_m)}m ahead. Immediately reduce speed to 20 km/h, check mirrors, and change lanes if safe. Avoid sudden braking."
            short_alert = f"Critical pothole in {int(dist_m)}m - Reduce to 20 km/h"
        elif severity >= 5:
            instructions = f"Large pothole detected {int(dist_m)}m ahead. Reduce speed to 30 km/h and steer gently around it. Maintain firm grip on steering wheel."
            short_alert = f"Large pothole in {int(dist_m)}m - Reduce to 30 km/h"
        else:
            instructions = f"Pothole {int(dist_m)}m ahead. Reduce speed to 40 km/h and proceed with caution. Minor suspension impact expected."
            short_alert = f"Pothole in {int(dist_m)}m - Proceed with caution"
            
    elif hazard_type == 'stalled_vehicle':
        if severity >= 7:
            instructions = f"EMERGENCY: Stalled vehicle blocking lane {int(dist_m)}m ahead. Immediately reduce speed to 20 km/h, activate hazard lights, and change lanes safely."
            short_alert = f"Blocked lane in {int(dist_m)}m - Change lanes"
        else:
            instructions = f"Stalled vehicle on roadside {int(dist_m)}m ahead. Reduce speed to 40 km/h, maintain safe distance, and be prepared for pedestrians."
            short_alert = f"Stalled vehicle in {int(dist_m)}m - Reduce speed"
            
    elif hazard_type == 'debris':
        if severity >= 6:
            instructions = f"Large debris obstruction {int(dist_m)}m ahead. Reduce speed to 25 km/h, check surroundings, and navigate carefully around obstacle."
            short_alert = f"Large debris in {int(dist_m)}m - Navigate carefully"
        else:
            instructions = f"Road debris {int(dist_m)}m ahead. Reduce speed to 35 km/h and avoid if possible. Watch for scattering objects."
            short_alert = f"Debris in {int(dist_m)}m - Reduce speed"
    else:
        instructions = f"Road hazard {int(dist_m)}m ahead. Reduce speed to 30 km/h and proceed with increased awareness."
        short_alert = f"Hazard in {int(dist_m)}m - Proceed with caution"

    # Build enhanced prompt for LLM
    prompt = "You are a professional driving safety assistant. Provide clear, calm, and instructional guidance.\n\n"
    prompt += "HAZARD DETAILS:\n"
    prompt += f"- Type: {hazard_type.replace('_', ' ')}\n"
    prompt += f"- Distance: {int(dist_m)} meters ahead\n"
    prompt += f"- Severity: {severity}/10\n"
    prompt += f"- Current speed: {speed_kmph} km/h\n"
    prompt += f"- ETA to hazard: {eta_s} seconds\n\n"
    prompt += "Provide specific driving instructions including:\n"
    prompt += "1. Recommended speed\n2. Lane change if needed\n3. Special precautions\n4. Any additional warnings\n\n"
    prompt += "Reply in JSON format: {short_alert, detailed_instructions, severity_level, recommended_speed_kmh}"

    try:
        out = generator(prompt, max_length=len(prompt.split())+100, do_sample=False, temperature=0.7)[0]['generated_text']
        text = out[len(prompt):].strip()
        # Simple JSON extract
        import re
        json_match = re.search(r'\{.*\}', text)
        if json_match:
            alert = json.loads(json_match.group())
            # Use LLM-generated content if available
            short_alert = alert.get('short_alert', short_alert)
            instructions = alert.get('detailed_instructions', instructions)
        else:
            # Fallback to our generated instructions
            alert = {
                "short_alert": short_alert,
                "detailed_instructions": instructions,
                "severity_level": "high" if severity >= 7 else "medium" if severity >= 5 else "low",
                "recommended_speed_kmh": 20 if severity >= 8 else 30 if severity >= 5 else 40
            }
    except:
        # Fallback if LLM fails
        alert = {
            "short_alert": short_alert,
            "detailed_instructions": instructions,
            "severity_level": "high" if severity >= 7 else "medium" if severity >= 5 else "low", 
            "recommended_speed_kmh": 20 if severity >= 8 else 30 if severity >= 5 else 40
        }

    alert.update({
        "hazard_type": hazard_type,
        "distance_meters": int(dist_m),
        "voice_text": instructions,  # Use the detailed instructions for voice
        "timestamp": datetime.now().isoformat(),
        "id": str(uuid.uuid4())
    })
    return alert

# ==================== Simulation Loop ====================
def simulation_loop():
    while state.journey_started:
        now = datetime.now()
        # Limit total hazards to prevent overflow
        if len(state.detected_hazards) >= 50:  # Overall limit
            state.detected_hazards = state.detected_hazards[-30:]  # Keep recent ones
        
        if state.last_detection is None or (now - state.last_detection).seconds >= 15:
            if state.current_route:
                coords = state.current_route['coordinates']
                # Generate fewer hazards to stay within limits
                n = min(np.random.randint(1, 3), 10 - len(state.detected_hazards))
                if n > 0:
                    idxs = np.random.choice(len(coords), n, replace=False)
                    for i in idxs:
                        pt = coords[i]
                        h = {
                            'id': str(uuid.uuid4()),
                            'type': np.random.choice(['pothole', 'debris', 'stalled_vehicle'], p=[0.5,0.3,0.2]),
                            'lat': pt['lat'],
                            'lon': pt['lon'],
                            'severity': np.random.randint(3,10),
                            'confidence': round(np.random.uniform(0.7,0.95),2),
                            'reported_by': state.vehicle_id,
                            'timestamp': datetime.now().isoformat()
                        }
                        state.detected_hazards.append(h)
                        state.amqp.publish_hazard(h)
                        # Index for LLM
                        emb = embed_model.encode(build_metadata_text(h)).astype('float32')
                        index.add(np.array([emb]))
                        metadatas.append(h)
                    state.last_detection = now
        time.sleep(1)

# ==================== Routes ====================
@app.route('/geocode', methods=['POST'])
def geocode():
    try:
        data = request.get_json(force=True)
        q = (data.get('query') if isinstance(data, dict) else None) or ''
        q = q.strip()
        if not q:
            return jsonify({'success': False, 'message': 'Empty query'}), 400

        # Use TomTom Search API to geocode
        try:
            url = f"https://api.tomtom.com/search/2/geocode/{requests.utils.requote_uri(q)}.json?key={TOMTOM_API_KEY}&limit=1"
            r = requests.get(url, timeout=5)
            r.raise_for_status()
            j = r.json()
            if j.get('results'):
                pos = j['results'][0]['position']
                return jsonify({'success': True, 'lat': pos['lat'], 'lon': pos['lon']})
        except Exception:
            # fall through to failure below
            pass

        return jsonify({'success': False, 'message': 'Geocoding failed'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/plan_route', methods=['POST'])
def plan_route():
    try:
        data = request.get_json(force=True)
        s_lat = float(data.get('start_lat'))
        s_lon = float(data.get('start_lon'))
        e_lat = float(data.get('end_lat'))
        e_lon = float(data.get('end_lon'))

        # Query OSRM for route
        coords_url = f"{OSRM_URL}/route/v1/driving/{s_lon},{s_lat};{e_lon},{e_lat}?overview=full&geometries=geojson"
        r = requests.get(coords_url, timeout=10)
        r.raise_for_status()
        j = r.json()
        if not j.get('routes'):
            return jsonify({'success': False, 'message': 'No route found'})

        route = j['routes'][0]
        geom = route.get('geometry', {})
        poly = geom.get('coordinates', [])
        # OSRM returns [lon, lat]
        coordinates = [{'lat': float(p[1]), 'lon': float(p[0])} for p in poly]

        # save to state for simulation
        state.current_route = {
            'coordinates': coordinates,
            'distance_km': round(route.get('distance', 0) / 1000.0, 2),
            'duration_min': int(route.get('duration', 0) / 60.0)
        }

        return jsonify({'success': True, 'route': {'coordinates': coordinates}, 'distance_km': state.current_route['distance_km'], 'duration_min': state.current_route['duration_min']})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/start_journey', methods=['POST'])
def start_journey():
    if state.current_route and not state.journey_started:
        state.journey_started = True
        state.detected_hazards = []
        state.last_detection = datetime.now()
        if not state.simulation_thread:
            state.simulation_thread = threading.Thread(target=simulation_loop, daemon=True)
            state.simulation_thread.start()
        return jsonify({'success': True})
    return jsonify({'success': False})

@app.route('/end_journey', methods=['POST'])
def end_journey():
    state.journey_started = False
    return jsonify({'success': True})

@app.route('/clear_hazards', methods=['POST'])
def clear_hazards():
    state.detected_hazards = []
    return jsonify({'success': True})

@app.route('/get_hazards')
def get_hazards():
    if state.current_route:
        route_coords = state.current_route['coordinates']
        nearby = []
        
        # Combine detected hazards and external messages
        all_hazards = state.detected_hazards + state.amqp.get_messages()
        
        # Remove duplicates based on ID or location
        seen = set()
        unique_hazards = []
        for h in all_hazards:
            # Create a unique identifier for the hazard
            if 'id' in h:
                ident = h['id']
            else:
                ident = f"{h.get('lat', 0):.4f}_{h.get('lon', 0):.4f}_{h.get('type', 'unknown')}"
            
            if ident not in seen:
                seen.add(ident)
                unique_hazards.append(h)
        
        # Filter hazards near route and limit to 10
        for h in unique_hazards:
            if len(nearby) >= 10:  # Limit to 10 hazards
                break
                
            for pt in route_coords:
                if haversine_m(h.get('lat', 0), h.get('lon', 0), pt['lat'], pt['lon']) <= 500:
                    nearby.append(h)
                    break
        
        return jsonify(nearby)
    return jsonify([])

@app.route('/get_alert')
def get_alert():
    if not state.current_route or not state.journey_started:
        return jsonify(None)
    # Assume vehicle is at start
    client_lat = state.current_route['coordinates'][0]['lat']
    client_lon = state.current_route['coordinates'][0]['lon']
    hazards = json.loads(requests.get('http://127.0.0.1:5000/get_hazards').text)
    alert = generate_alert(hazards, client_lat, client_lon)
    if alert:
        state.amqp.publish_alert(alert)
        state.last_alert = alert
    return jsonify(alert or None)

@app.route('/get_status')
def get_status():
    return jsonify({
        'journey_started': state.journey_started,
        'detected_hazards_count': len(state.detected_hazards),
        'external_hazards_count': 0,
        'vehicle_id': state.vehicle_id,
        'amqp_connected': True,
        'distance_km': state.current_route['distance_km'] if state.current_route else 0,
        'duration_min': state.current_route['duration_min'] if state.current_route else 0
    })

@app.route('/')
def index():
    return send_from_directory('../frontend',  'user2.html')

if __name__ == '__main__':
    print("SmartRoadGuard + LLM Alerts Running → http://127.0.0.1:5000")
    app.run(debug=True, port=5000)