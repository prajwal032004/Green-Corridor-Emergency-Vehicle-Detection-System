from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from ultralytics import YOLO
import cv2
import numpy as np
import os
import json
from datetime import datetime, timezone
import threading
import time
import base64
from werkzeug.utils import secure_filename
import uuid
import math
import requests

app = Flask(__name__)
app.config['SECRET_KEY'] = 'green_corridor_2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs('static/detected_images', exist_ok=True)

# Initialize YOLO model
try:
    model = YOLO('best.pt')
    print("✓ YOLOv8 model loaded successfully!")
except Exception as e:
    print(f"✗ Error loading YOLOv8 model: {e}")
    model = None

# Global variables for live detection
camera_active = False
camera_lock = threading.Lock()
detection_results = []
current_detections = []
current_frame = None
frame_lock = threading.Lock()
ambulance_detected_flag = False
detection_count = 0
ambulance_count = 0

# Traffic simulation
JUNCTIONS = [
    {"id": 1, "name": "Kengeri Junction", "lat": 12.899811261500984, "lng": 77.48159002679168, "status": "red"},
    {"id": 2, "name": "Naagarabhaavi Junction", "lat": 12.971770656692353, "lng": 77.51334819815241, "status": "red"},
    {"id": 3, "name": "Yeswanthpur Junction", "lat": 13.025172009759848, "lng": 77.53282768617794, "status": "red"},
    {"id": 4, "name": "Hebbal Junction", "lat": 13.035207229228284, "lng":  77.5987195473465, "status": "red"},
    {"id": 5, "name": "Uttarahalli Junction", "lat": 12.907694222351116, "lng": 77.54739811674267, "status": "red"},
    {"id": 6, "name": "Electronic City Junction", "lat": 12.83963148948628, "lng": 77.677003456789, "status": "red"},
    {"id": 7, "name": "Marathahalli Junction", "lat": 12.956345678901234, "lng": 77.70123456789012, "status": "red"},
    {"id": 8, "name": "Whitefield Junction", "lat": 12.969567890123456, "lng": 77.75012345678901, "status": "red"},
    {"id": 9, "name": "Koramangala Junction", "lat": 12.935678901234567, "lng": 77.62456789012345, "status": "red"},
    {"id": 10, "name": "Jayanagar Junction", "lat": 12.925123456789012, "lng": 77.59345678901234, "status": "red"},
    {"id": 11, "name": "Indiranagar Junction", "lat": 12.971234567890123, "lng": 77.6390123456789, "status": "red"},
    {"id": 12, "name": "MG Road Junction", "lat": 12.976345678901234, "lng": 77.59912345678901, "status": "red"}
]

HOSPITALS = [
    {"id": 1, "name": "Bangalore Baptist Hospital", "lat": 13.037313957001425, "lng": 77.58987556883473, "address": "Sample Road 1"},
    {"id": 2, "name": "Manipal Hospital Hebbal", "lat": 13.054313718818214, "lng": 77.59258940763026, "address": "Sample Road 2"},
    {"id": 3, "name": "Ramaiah Memorial Hospital", "lat": 13.029020176347379, "lng": 77.56991568104787, "address": "Sample Road 3"},
    {"id": 4, "name": "Abhay Hospital", "lat": 12.951710337604903, "lng": 77.49403748319067, "address": "Sample Road 4"},
    {"id": 5, "name": "Banu Multispeciality Hospital", "lat": 12.951096143399294, "lng": 77.49190285234876, "address": "Sample Road 5"},
    {"id": 6, "name": "Aryan Multispeciality Hospital", "lat": 12.947002872774917, "lng": 77.49624795506206, "address": "Sample Road 6"},
    {"id": 7, "name": "Fortis Hospital, Nagarbhavi", "lat": 12.96022208061866, "lng": 77.51102500712904, "address": "Sample Road 7"},
    {"id": 8, "name": "SPARSH Hospital, Yeshwanthpur", "lat": 13.027838919462717, "lng": 77.54288857834604, "address": "Sample Road 8"},
    {"id": 9, "name": "People Tree Hospitals", "lat": 13.03582379748628, "lng": 77.53934931209913, "address": "Sample Road 9"},
    {"id": 10, "name": "Narayana Multispeciality Hospital", "lat": 13.029839954683057, "lng": 77.53934931209913, "address": "Sample Road 10"},
    {"id": 11, "name": "Columbia Asia Hospital, Hebbal", "lat": 13.041567890123456, "lng": 77.59876543210987, "address": "Sample Road 11"},
    {"id": 12, "name": "Vasantha Hospital", "lat": 12.908123456789012, "lng": 77.53678901234567, "address": "Sample Road 12"},
    {"id": 13, "name": "Apollo Hospital, Bannerghatta Road", "lat": 12.908765432109876, "lng": 77.58432109876543, "address": "Sample Road 13"},
    {"id": 14, "name": "Fortis Hospital, Bannerghatta Road", "lat": 12.914567890123456, "lng": 77.58456789012345, "address": "Sample Road 14"},
    {"id": 15, "name": "Narayana Multispeciality Hospital, Bannerghatta Road", "lat": 12.910123456789012, "lng": 77.58567890123456, "address": "Sample Road 15"}
]

_sim_state = {
    "ambulance": None,
    "active_route": None,
    "green_junctions": [],
    "last_route_time": None
}

OSRM_BASE_URL = "https://router.project-osrm.org/route/v1/driving"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov', 'mkv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2*R*math.asin(math.sqrt(a))

def reset_junctions():
    for j in JUNCTIONS:
        j['status'] = 'red'
    _sim_state['green_junctions'] = []

def activate_green_corridor():
    global _sim_state
    for i, signal in enumerate(JUNCTIONS):
        if i < 2:
            signal['status'] = 'green'
            signal['activated_at'] = datetime.now().isoformat()
            signal['reason'] = 'Emergency Vehicle Detected'
        else:
            signal['status'] = 'red'
    _sim_state['green_junctions'] = [1, 2]
    _sim_state['last_route_time'] = time.time()

def reset_signals():
    if _sim_state['green_junctions']:
        reset_junctions()

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload')
def upload_page():
    return render_template('upload.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/map')
def map_page():
    return render_template('map.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/team')
def team():
    return render_template('team.html')

# Camera control endpoints
@app.route('/api/start_camera')
def start_camera():
    global camera_active
    with camera_lock:
        if not camera_active:
            camera_active = True
            threading.Thread(target=camera_detection_thread, daemon=True).start()
            return jsonify({'success': True, 'message': 'Camera started'})
        else:
            return jsonify({'success': False, 'message': 'Camera already running'})

@app.route('/api/stop_camera')
def stop_camera():
    global camera_active, ambulance_detected_flag, detection_count, ambulance_count
    with camera_lock:
        camera_active = False
        ambulance_detected_flag = False
        detection_count = 0
        ambulance_count = 0
        reset_signals()
    return jsonify({'success': True, 'message': 'Camera stopped'})

# Add these endpoints to your Flask app (app.py)
# Place them after the existing API endpoints

@app.route('/api/set_ambulance', methods=['POST'])
def set_ambulance():
    data = request.json
    _sim_state['ambulance'] = {
        'lat': data['lat'],
        'lng': data['lng'],
        'timestamp': datetime.now().isoformat()
    }
    return jsonify({'success': True})

@app.route('/api/nearest_hospital', methods=['GET'])
def nearest_hospital():
    lat = float(request.args.get('lat'))
    lng = float(request.args.get('lng'))
    
    nearest = None
    min_dist = float('inf')
    
    for hospital in HOSPITALS:
        dist = haversine_m(lat, lng, hospital['lat'], hospital['lng'])
        if dist < min_dist:
            min_dist = dist
            nearest = hospital
    
    return jsonify({'hospital': nearest, 'distance': min_dist})

@app.route('/api/directions', methods=['POST'])
def get_directions():
    data = request.json
    origin = data['origin']
    destination = data['destination']
    
    # Build OSRM request URL
    coords = f"{origin['lng']},{origin['lat']};{destination['lng']},{destination['lat']}"
    url = f"{OSRM_BASE_URL}/{coords}?overview=full&geometries=geojson&steps=true"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        osrm_data = response.json()
        
        if osrm_data.get('code') != 'Ok':
            return jsonify({'ok': False, 'error': 'No route found'}), 400
        
        route = osrm_data['routes'][0]
        geometry = route['geometry']
        
        # Convert GeoJSON coordinates to lat/lng points
        points = [{'lat': coord[1], 'lng': coord[0]} for coord in geometry['coordinates']]
        
        # Get route details
        leg = route['legs'][0]
        distance_m = leg['distance']
        duration_s = leg['duration']
        
        # Format distance and duration
        distance_text = f"{distance_m / 1000:.1f} km" if distance_m >= 1000 else f"{int(distance_m)} m"
        duration_text = f"{int(duration_s / 60)} min" if duration_s >= 60 else f"{int(duration_s)} sec"
        
        # Reset all junctions first
        reset_junctions()
        
        # Find junctions along the route and activate green corridor
        route_junctions = []
        for junction in JUNCTIONS:
            # Check if junction is near the route (within 500m of any point)
            for point in points[::5]:  # Check every 5th point for better accuracy
                dist = haversine_m(junction['lat'], junction['lng'], point['lat'], point['lng'])
                if dist < 500:  # 500 meters threshold for better coverage
                    route_junctions.append(junction['id'])
                    junction['status'] = 'green'
                    junction['activated_at'] = datetime.now().isoformat()
                    junction['reason'] = 'Emergency Route Active'
                    break
        
        # Update global state
        _sim_state['active_route'] = {
            'origin': origin,
            'destination': destination,
            'junctions': route_junctions,
            'timestamp': datetime.now().isoformat()
        }
        _sim_state['green_junctions'] = route_junctions
        _sim_state['last_route_time'] = time.time()
        
        return jsonify({
            'ok': True,
            'route': {
                'points': points,
                'legs': [{
                    'distance': {
                        'value': distance_m,
                        'text': distance_text
                    },
                    'duration': {
                        'value': duration_s,
                        'text': duration_text
                    }
                }],
                'green_junctions': route_junctions
            }
        })
        
    except requests.exceptions.RequestException as e:
        print(f"OSRM API Error: {e}")
        return jsonify({'ok': False, 'error': 'Routing service unavailable'}), 503
    except Exception as e:
        print(f"Direction error: {e}")
        return jsonify({'ok': False, 'error': str(e)}), 500
    
def camera_detection_thread():
    global camera_active, current_detections, current_frame, ambulance_detected_flag
    global detection_count, ambulance_count
    
    print("Starting camera detection thread...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        camera_active = False
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Camera opened successfully, starting detection loop...")
    frame_count = 0
    
    while camera_active:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {frame_count}")
            time.sleep(0.1)
            continue
        
        frame_count += 1
        
        if model and frame_count % 2 == 0:  # Process every 2nd frame for performance
            try:
                # Run detection
                results = model(frame, conf=0.4, verbose=False)
                detections = []
                ambulance_detected = False
                annotated_frame = frame.copy()
                
                for result in results:
                    for box in result.boxes:
                        class_name = model.names[int(box.cls)]
                        confidence = float(box.conf)
                        bbox = box.xyxy[0].cpu().numpy()
                        
                        detections.append({
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': bbox.tolist(),
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        # Check for ambulance
                        if 'ambulance' in class_name.lower():
                            ambulance_detected = True
                            ambulance_count += 1
                            color = (0, 0, 255)  # Red
                            thickness = 3
                        else:
                            color = (0, 255, 0)  # Green
                            thickness = 2
                        
                        # Draw bounding box
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                        
                        # Draw label
                        label = f"{class_name}: {confidence:.2f}"
                        (label_width, label_height), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                        )
                        cv2.rectangle(
                            annotated_frame,
                            (x1, y1 - label_height - 10),
                            (x1 + label_width, y1),
                            color,
                            -1
                        )
                        cv2.putText(
                            annotated_frame,
                            label,
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2
                        )
                
                # Update global state
                current_detections = detections
                detection_count = len(detections)
                ambulance_detected_flag = ambulance_detected
                
                with frame_lock:
                    current_frame = annotated_frame.copy()
                
                # Handle green corridor
                if ambulance_detected:
                    activate_green_corridor()
                    print(f"AMBULANCE DETECTED! Frame: {frame_count}")
                else:
                    reset_signals()
                
                if detections:
                    print(f"Frame {frame_count}: Detected {len(detections)} objects (Ambulance: {ambulance_detected})")
                
            except Exception as e:
                print(f"Detection error on frame {frame_count}: {e}")
                with frame_lock:
                    current_frame = frame.copy()
        else:
            with frame_lock:
                current_frame = frame.copy()
        
        time.sleep(0.033)  # ~30 FPS
    
    cap.release()
    print("Camera detection thread stopped")

@app.route('/api/camera_feed')
def camera_feed():
    def generate():
        print("Starting camera feed stream...")
        while camera_active:
            with frame_lock:
                if current_frame is not None:
                    frame = current_frame.copy()
                else:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "Initializing...", (200, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.033)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/detections')
def get_detections():
    return jsonify({
        'detections': current_detections,
        'timestamp': datetime.now().isoformat(),
        'ambulance_detected': ambulance_detected_flag,
        'total_count': detection_count,
        'ambulance_count': ambulance_count
    })

@app.route('/api/traffic_signals')
def get_traffic_signals():
    return jsonify({
        'signals': JUNCTIONS,
        'green_junctions': _sim_state['green_junctions'],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/emergency_status')
def get_emergency_status():
    return jsonify({
        'emergency_active': ambulance_detected_flag,
        'green_corridor_status': 'active' if ambulance_detected_flag else 'inactive',
        'active_signals': len(_sim_state['green_junctions']),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/hospitals')
def api_hospitals():
    return jsonify({"hospitals": HOSPITALS})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        try:
            detection_result = process_file_detection(filepath, unique_filename)
            return jsonify({
                'success': True,
                'filename': unique_filename,
                'detection_result': detection_result
            })
        except Exception as e:
            return jsonify({'error': f'Detection failed: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

def process_file_detection(filepath, filename):
    if not model:
        return {'error': 'Model not loaded'}
    
    file_ext = filename.split('.')[-1].lower()
    
    if file_ext in ['jpg', 'jpeg', 'png', 'gif']:
        results = model(filepath, conf=0.4)
        detected_objects = []
        
        for result in results:
            for box in result.boxes:
                if box.conf > 0.4:
                    detected_objects.append({
                        'class': model.names[int(box.cls)],
                        'confidence': float(box.conf),
                        'bbox': box.xyxy[0].tolist()
                    })
        
        annotated_image = results[0].plot()
        output_path = os.path.join('static/detected_images', f"detected_{filename}")
        cv2.imwrite(output_path, annotated_image)
        
        ambulance_detected = any('ambulance' in obj['class'].lower() for obj in detected_objects)
        
        return {
            'type': 'image',
            'detections': detected_objects,
            'ambulance_detected': ambulance_detected,
            'output_image': f"/static/detected_images/detected_{filename}",
            'timestamp': datetime.now().isoformat()
        }

@app.route('/static/detected_images/<filename>')
def detected_images(filename):
    return send_from_directory('static/detected_images', filename)

if __name__ == '__main__':
    print("=" * 60)
    print("Green Corridor Emergency Vehicle Detection System")
    print("=" * 60)
    print("✓ Server starting...")
    print("✓ Make sure 'best.pt' YOLOv8 model is in the directory")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)