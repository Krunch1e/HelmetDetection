import cv2
import os
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import VIOLATION_DIR
from detector.helmet_violation_detector import HelmetViolationDetector

app = FastAPI(title="Helmet Violation Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(VIOLATION_DIR, exist_ok=True)
app.mount("/violations", StaticFiles(directory=str(VIOLATION_DIR)), name="violations")

webcam_running = False
webcam_thread = None
webcam_frame = None
webcam_lock = threading.Lock()
violation_count = 0
violation_counter_lock = threading.Lock()


def draw_detections(frame, detections, helmet_boxes, nohelmet_boxes):
    for det in detections:
        track_id = det["track_id"]
        x1, y1, x2, y2 = det["box"]
        hx1, hy1, hx2, hy2 = det["head_region"]
        violation = det["violation"]

        color = (0, 0, 255) if violation else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 0, 0), 1)

        text = f"ID {track_id} - {'NO HELMET' if violation else 'HELMET'}"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if not detections:
        for x1, y1, x2, y2 in helmet_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                frame,
                "helmet",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        for x1, y1, x2, y2 in nohelmet_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(
                frame,
                "no helmet",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

    return frame


def webcam_worker():
    global webcam_running, webcam_frame, violation_count

    detector = HelmetViolationDetector()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        webcam_running = False
        return

    frame_id = 0
    last_detections = []
    last_helmet_boxes = []
    last_nohelmet_boxes = []

    while webcam_running:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        if frame_id % 2 == 0:
            last_detections, last_helmet_boxes, last_nohelmet_boxes = (
                detector.process_frame(frame)
            )

            with violation_counter_lock:
                violation_count = sum(1 for d in last_detections if d["violation"])

        frame = draw_detections(
            frame, last_detections, last_helmet_boxes, last_nohelmet_boxes
        )

        with webcam_lock:
            webcam_frame = frame.copy()

    cap.release()


@app.post("/stream/start")
def start_webcam():
    global webcam_running, webcam_thread

    if webcam_running:
        return JSONResponse({"status": "already running"})

    webcam_running = True
    webcam_thread = threading.Thread(target=webcam_worker, daemon=True)
    webcam_thread.start()

    return JSONResponse({"status": "started"})


@app.post("/stream/stop")
def stop_webcam():
    global webcam_running
    webcam_running = False
    return JSONResponse({"status": "stopped"})


@app.get("/stream/webcam")
def stream_webcam():
    def generate():
        while webcam_running:
            with webcam_lock:
                frame = webcam_frame

            if frame is None:
                time.sleep(0.03)
                continue

            _, buffer = cv2.imencode(".jpg", frame)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
            time.sleep(0.03)

    return StreamingResponse(
        generate(), media_type="multipart/x-mixed-replace;boundary=frame"
    )


@app.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    detector = HelmetViolationDetector()
    detections, helmet_boxes, nohelmet_boxes = detector.process_frame(frame)
    frame = draw_detections(frame, detections, helmet_boxes, nohelmet_boxes)

    violations = sum(1 for d in detections if d["violation"])

    _, buffer = cv2.imencode(".jpg", frame)
    return StreamingResponse(
        iter([buffer.tobytes()]),
        media_type="image/jpeg",
        headers={"X-Violation-Count": str(violations)},
    )


@app.post("/upload/video")
async def upload_video(
    file: UploadFile = File(...),
    output: Optional[str] = Query("snapshots", enum=["video", "snapshots"]),
):
    temp_input = f"temp_{file.filename}"
    temp_output = f"output_{file.filename}"

    with open(temp_input, "wb") as f:
        f.write(await file.read())

    detector = HelmetViolationDetector()
    cap = cv2.VideoCapture(temp_input)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output == "video":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    frame_id = 0
    total_violations = 0
    snapshot_paths = []
    last_detections = []
    last_helmet_boxes = []
    last_nohelmet_boxes = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        if frame_id % 2 == 0:
            last_detections, last_helmet_boxes, last_nohelmet_boxes = (
                detector.process_frame(frame)
            )

        detections = last_detections
        helmet_boxes = last_helmet_boxes
        nohelmet_boxes = last_nohelmet_boxes

        annotated = draw_detections(
            frame.copy(), detections, helmet_boxes, nohelmet_boxes
        )

        violations_in_frame = [d for d in detections if d["violation"]]
        total_violations += len(violations_in_frame)

        if violations_in_frame and frame_id % 20 == 0:
            snapshot_path = str(VIOLATION_DIR / f"violation_{frame_id}.jpg")
            cv2.imwrite(snapshot_path, annotated)
            snapshot_paths.append(f"/violations/violation_{frame_id}.jpg")

        if writer:
            writer.write(annotated)

    cap.release()
    if writer:
        writer.release()

    os.remove(temp_input)

    if output == "video":

        def cleanup():
            if os.path.exists(temp_output):
                os.remove(temp_output)

        from starlette.background import BackgroundTask

        return FileResponse(
            temp_output,
            media_type="video/mp4",
            filename="annotated_output.mp4",
            background=BackgroundTask(cleanup),
        )

    return JSONResponse(
        {
            "total_violations": total_violations,
            "frames_processed": frame_id,
            "snapshots": snapshot_paths,
        }
    )


@app.get("/violations")
def get_violations():
    files = list(Path(VIOLATION_DIR).glob("*.jpg"))
    return JSONResponse(
        {"violations": [f"/violations/{f.name}" for f in sorted(files)]}
    )


@app.get("/stats")
def get_stats():
    with violation_counter_lock:
        count = violation_count
    return JSONResponse({"active_violations": count})
