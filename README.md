# 🚑 Green Corridor Emergency Vehicle Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-green)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

An intelligent real-time emergency vehicle detection system that automatically creates green corridors for ambulances using computer vision and traffic signal control simulation.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🎯 Overview

The Green Corridor Emergency Vehicle Detection System uses state-of-the-art computer vision (YOLOv8) to detect ambulances in real-time through camera feeds or uploaded media. When an ambulance is detected, the system automatically activates a "green corridor" by simulating traffic signal changes along the route to the nearest hospital, potentially saving critical minutes in emergency situations.

### Key Highlights

- **Real-time Detection**: Live camera feed processing with instant ambulance detection
- **Intelligent Routing**: Automatic calculation of optimal routes to nearest hospitals using OSRM
- **Traffic Signal Control**: Automated green corridor activation along emergency routes
- **Multi-modal Input**: Supports live camera, image uploads, and video processing
- **Interactive Dashboard**: Real-time monitoring of detections, traffic signals, and emergency status
- **Geospatial Visualization**: Interactive map showing ambulance location, hospitals, and active routes

## ✨ Features

### Core Functionality

- 🎥 **Live Camera Detection**: Real-time ambulance detection from webcam feed
- 📸 **Image/Video Upload**: Process pre-recorded media for ambulance detection
- 🗺️ **Smart Routing**: Automatic route calculation to nearest hospital
- 🚦 **Green Corridor Activation**: Automated traffic signal control simulation
- 📊 **Analytics Dashboard**: Real-time statistics and detection history
- 🏥 **Hospital Database**: Pre-configured hospital locations across Bangalore

### Technical Features

- High-accuracy YOLOv8 object detection
- Multi-threaded video processing
- RESTful API architecture
- Responsive web interface
- OSRM-based routing engine
- Haversine distance calculations
- Real-time WebSocket communication

## 🎬 Demo

### Live Detection Interface

```
[Camera Feed] → [YOLOv8 Detection] → [Green Corridor Activation] → [Traffic Signal Control]
```

### Workflow

1. System detects ambulance via camera or uploaded media
2. Calculates nearest hospital using geospatial algorithms
3. Generates optimal route using OSRM routing service
4. Activates green corridor by changing traffic signals along route
5. Provides real-time status updates and analytics

## 🛠️ Technology Stack

### Backend

- **Flask** - Web framework
- **YOLOv8 (Ultralytics)** - Object detection model
- **OpenCV** - Computer vision processing
- **NumPy** - Numerical computing

### Frontend

- **HTML5/CSS3** - Structure and styling
- **JavaScript** - Interactive functionality
- **Leaflet.js** - Map visualization
- **Bootstrap** - Responsive design

### External Services

- **OSRM** - Open Source Routing Machine for route optimization
- **OpenStreetMap** - Map data

### Core Libraries

```python
Flask==2.0+
ultralytics==8.0+
opencv-python==4.8+
numpy==1.24+
requests==2.31+
```

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- Webcam (for live detection)
- 4GB+ RAM recommended
- CUDA-compatible GPU (optional, for faster processing)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/green-corridor-system.git
cd green-corridor-system
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download YOLOv8 Model

Place your trained YOLOv8 model file (`best.pt`) in the project root directory.

**Note**: The model should be trained to detect ambulances. You can train your own model using the [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) framework or contact the repository maintainers for the pre-trained model.

### Step 5: Create Required Directories

```bash
mkdir -p uploads results static/detected_images
```

## 🚀 Usage

### Starting the Server

```bash
python app.py
```

The application will start on `http://localhost:5000`

### Accessing the Application

1. **Home Page**: `http://localhost:5000/`
2. **Upload Detection**: `http://localhost:5000/upload`
3. **Live Camera**: `http://localhost:5000/dashboard`
4. **Map View**: `http://localhost:5000/map`
5. **About**: `http://localhost:5000/about`

### Using Live Camera Detection

1. Navigate to the Dashboard
2. Click "Start Camera"
3. Grant camera permissions when prompted
4. System will automatically detect ambulances and activate green corridor
5. View real-time statistics and active signals

### Uploading Media

1. Go to Upload page
2. Select an image or video file
3. Click "Upload and Detect"
4. View detection results and analysis

### Map Simulation

1. Open Map View
2. Click on the map to set ambulance location
3. System automatically calculates route to nearest hospital
4. Green corridor activates along the route
5. View affected junctions and route details

## 📚 API Documentation

### Traffic Signal Endpoints

#### Get All Traffic Signals

```http
GET /api/traffic_signals
```

**Response:**

```json
{
  "signals": [
    {
      "id": 1,
      "name": "Kengeri Junction",
      "lat": 12.8998,
      "lng": 77.4816,
      "status": "red"
    }
  ],
  "green_junctions": [1, 2],
  "timestamp": "2024-01-01T12:00:00"
}
```

### Camera Endpoints

#### Start Camera

```http
GET /api/start_camera
```

#### Stop Camera

```http
GET /api/stop_camera
```

#### Get Detections

```http
GET /api/detections
```

**Response:**

```json
{
  "detections": [
    {
      "class": "ambulance",
      "confidence": 0.95,
      "bbox": [100, 100, 300, 300],
      "timestamp": "2024-01-01T12:00:00"
    }
  ],
  "ambulance_detected": true,
  "total_count": 1,
  "ambulance_count": 1
}
```

### Routing Endpoints

#### Get Nearest Hospital

```http
GET /api/nearest_hospital?lat=12.9716&lng=77.5946
```

#### Get Directions

```http
POST /api/directions
Content-Type: application/json

{
  "origin": {"lat": 12.9716, "lng": 77.5946},
  "destination": {"lat": 13.0373, "lng": 77.5899}
}
```

### Upload Endpoints

#### Upload File for Detection

```http
POST /api/upload
Content-Type: multipart/form-data

file: <image/video file>
```

## 📁 Project Structure

```
green-corridor-system/
│
├── app.py                      # Main Flask application
├── best.pt                     # YOLOv8 trained model
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
│
├── templates/                  # HTML templates
│   ├── index.html             # Home page
│   ├── upload.html            # Upload interface
│   ├── dashboard.html         # Live detection dashboard
│   ├── map.html               # Interactive map
│   ├── about.html             # About page
│   └── team.html              # Team information
│
├── uploads/                    # Uploaded files
├── results/                    # Processing results
└── models/                     # ML models directory
```

## ⚙️ Configuration

### Application Settings

Edit `app.py` to modify these configurations:

```python
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
```

### Model Configuration

```python
# Detection confidence threshold
model = YOLO('best.pt')
results = model(frame, conf=0.4)  # Adjust confidence threshold
```

### Junction & Hospital Data

Update the `JUNCTIONS` and `HOSPITALS` lists in `app.py` to add or modify locations:

```python
JUNCTIONS = [
    {
        "id": 1,
        "name": "Junction Name",
        "lat": 12.9716,
        "lng": 77.5946,
        "status": "red"
    }
]
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guide for Python code
- Write clear commit messages
- Add comments for complex logic
- Update documentation for new features
- Test thoroughly before submitting PR

## 🐛 Known Issues

- Camera access may require HTTPS in some browsers
- GPU acceleration requires CUDA setup
- Large video files may cause memory issues
- OSRM routing requires internet connection

## 🔮 Future Enhancements

- [ ] Multi-camera support
- [ ] Real-time vehicle tracking
- [ ] SMS/Email alerts to hospitals
- [ ] Integration with actual traffic control systems
- [ ] Mobile application
- [ ] AI-based ETA predictions
- [ ] Cloud deployment with scalability
- [ ] Support for multiple cities

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**⭐ Star this repository if you find it helpful!**

**Made with ❤️ for saving lives**
