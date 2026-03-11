# Helmet Violation Detection System

An AI-based motorcycle helmet detection system using YOLOv8 object detection. Detects riders, helmets, and helmet violations from images, videos, and live webcam feeds.

## Project Structure

```
HelmetDetection/
├── detector/
│   └── helmet_violation_detector.py
├── utils/
│   └── geometry.py
├── data/
├── train/
├── violations/
├── config.py
├── main.py
├── run_video.py
├── run_webcam.py
├── train.py
├── download_dataset.py
└── .env
```

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
ROBOFLOW_API_KEY=your_key
WORKSPACE=your_workspace
PROJECT=your_project
VERSION=1
```

## Running the API

```bash
cd "HelmetDetection/New Workspace"
uvicorn main:app --reload
```

API will be available at `http://localhost:8000`

Interactive docs at `http://localhost:8000/docs`

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/stream/start` | POST | Start webcam detection |
| `/stream/stop` | POST | Stop webcam detection |
| `/stream/webcam` | GET | Live MJPEG stream |
| `/upload/image` | POST | Upload image, returns annotated image |
| `/upload/video?output=video` | POST | Upload video, returns annotated video |
| `/upload/video?output=snapshots` | POST | Upload video, returns violation stats + snapshots |
| `/violations` | GET | List all saved violation snapshots |
| `/stats` | GET | Current active violation count |

## Frontend Integration

### Live webcam feed

```html
<img src="http://localhost:8000/stream/webcam">
```

Start the stream before displaying:

```javascript
await fetch('http://localhost:8000/stream/start', { method: 'POST' })
```

### Image upload

```javascript
const formData = new FormData()
formData.append('file', imageFile)
const response = await fetch('http://localhost:8000/upload/image', {
    method: 'POST',
    body: formData
})
const blob = await response.blob()
const url = URL.createObjectURL(blob)
// display annotated image using url
// violation count is in response header: X-Violation-Count
```

### Video upload

```javascript
const formData = new FormData()
formData.append('file', videoFile)
const response = await fetch('http://localhost:8000/upload/video?output=snapshots', {
    method: 'POST',
    body: formData
})
const data = await response.json()
// data.total_violations
// data.snapshots → array of snapshot URLs
```

## Local Scripts

Run detection locally without the API:

```bash
# Video file
python run_video.py

# Webcam
python run_webcam.py
```

## Model

- Architecture: YOLO26m
- Classes: helmet, no-helmet, rider
- mAP50: 0.956
- Trained on real Indian traffic dashcam footage

## Training

```bash
python train.py
```