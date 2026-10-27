"""
Real-time multi-object tracking pipeline using YOLO11m and ByteTrack.

The pipeline performs object detection on video or webcam frames, passes
detections to ByteTrack for multi-object tracking, and displays tracking
results with class labels, track IDs, FPS, and GPU memory usage.

Optional recording and screenshot functionality is provided for qualitative
evaluation of the tracking pipeline.
"""



import argparse
import os
import threading
import time
import cv2
import torch
import numpy as np
from pathlib import Path
from collections import deque
from ultralytics import YOLO
from boxmot.trackers.bytetrack.bytetrack import ByteTrack as BYTETracker
# best detector
DetectorWeights = r"C:\Users\USER\PycharmProjects\MOT-RAG\model_fp16.torchscript"
# the output path for videos and screenshots
OutDir = Path(r"C:\Users\USER\PycharmProjects\Multi-Object-Tracking-in-Surveillance-Videos\realtime_outputs")
TrackerParams = { # best tracker paramters for ByteTrack
    "track_high_thresh": 0.4,
    "track_buffer": 20,
    "match_thresh": 0.8,
}
Conf = 0.01
IOU = 0.5
Imgsz = 640
Device = 0
# Live webcam settings prioritize responsiveness over maximum resolution.
LiveCaptureWidth = 960
LiveCaptureHeight = 540
LiveImgsz = 256
_model_lock = threading.Lock()
_model = None
_model_warmed_up = False

ClassNames = {
    0: "airplane",
    1: "fish",
    2: "ball",
    3: "bird",
    4: "boat",
    5: "balloon",
    6: "person",
    7: "insect",
    8: "stock",
    9: "car"
}
ClassColors = {
    0: (0, 165, 255),  # orange
    1: (255, 0, 0),  #blue
    2: (0, 255, 0),  #green
    3: (255, 255, 0),  # cyan
    4: (0, 0, 255),  # red
    5: (255, 0, 255),  # magenta
    6: (0, 255, 255),  # yellow
    7: (128, 0, 255),  #purple
    8: (255, 128, 0),  # light blue
    9: (0, 128, 255)  # orange-red
}


def draw_box(frame, x1, y1, x2, y2, tid, cls_id, conf):
    color = ClassColors.get(int(cls_id), (200, 200, 200))  # get the class colour
    label = f"ID: {tid} {ClassNames.get(int(cls_id), str(int(cls_id)))} {conf:.2f}"  # arrange the label that will be displayed
    # draw a rectangle
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    # draw label background
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(
        frame, (int(x1), int(y1) - th - 6),
        (int(x1) + tw + 4, int(y1)),
        color, -1
    )
    # draw label text in black
    cv2.putText(frame, label, (int(x1) + 2, int(y1) - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def draw_overlay(frame, fps, n_tracks, paused=False):
    # display fps and track count in the top left corner
    h, w = frame.shape[:2]
    overlay = frame.copy()
    # create a semi-transparent background panel
    cv2.rectangle(overlay, (0, 0), (220, 68), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    fpsColor = (0, 255, 0) if fps >= 25 else (0, 165, 255) if fps >= 15 else (0, 0, 255)
    cv2.putText(frame, f"FPS: {fps:.2f}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fpsColor, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Tracks: {n_tracks}", (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    if paused:
        cv2.putText(frame, "Paused", (8, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)
    return frame


def run(
    source,
    save_video=False,
    frame_callback=None,
    stop_event=None,
    command_callback=None,
    capture=None,
):
    OutDir.mkdir(parents=True, exist_ok=True)
    global _model, _model_warmed_up
    with _model_lock:
        if _model is None:
            print(f"Loading detector from {DetectorWeights}")
            _model = YOLO(DetectorWeights)
        model = _model
    # tracker initialisation
    tracker = BYTETracker(
        track_thresh=TrackerParams["track_high_thresh"],
        match_thresh=TrackerParams["match_thresh"],
        track_buffer=TrackerParams["track_buffer"],
        frame_rate=30,
    )
    if capture is not None:
        cap = capture
    elif frame_callback is not None and isinstance(source, int) and os.name == "nt":
        # DirectShow avoids the extra buffering commonly seen with MSMF on Windows.
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)
    if frame_callback is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, LiveCaptureWidth)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, LiveCaptureHeight)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source {source}")
    web_mode = frame_callback is not None
    inference_device = Device if torch.cuda.is_available() else "cpu"
    if not web_mode:
        # Create a resizable OpenCV window
        cv2.namedWindow("Real-Time Tracking", cv2.WINDOW_NORMAL)
    # get frame dimensions
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Source opened: {frame_w}x{frame_h}")
    # optional video writer
    videoWriter = None
    recording = save_video
    if recording:
        ts = time.strftime("%Y%m%d_%H%M%S")
        videoPath = OutDir / f"tracking_{ts}.mp4"
        videoWriter = cv2.VideoWriter(str(videoPath), cv2.VideoWriter.fourcc(*"mp4v"), 30, (frame_w, frame_h))
        print(f"Recording video: {videoPath}")
    # fps smoothing for last 30 frames
    fps_buffer = deque(maxlen=30)
    paused = False
    frame_idx = 0
    last_tracks = np.empty((0, 7))
    display_fps_buffer = deque(maxlen=30)
    last_frame_time = None
    detector_stop = threading.Event()
    detector_lock = threading.Lock()
    latest_frame = None
    latest_frame_number = -1
    detected_frame_number = -1
    inference_imgsz = LiveImgsz if web_mode else Imgsz
    displayed_tracks = np.empty((0, 7))
    applied_detection_number = -1
    previous_gray = None

    def follow_tracks(frame, tracks):
        if not len(tracks):
            return tracks
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if previous_gray is None:
            return tracks
        followed = tracks.copy()
        for index, track in enumerate(followed):
            x1, y1, x2, y2 = (int(value) for value in track[:4])
            x1 = max(0, min(x1, gray.shape[1] - 1))
            y1 = max(0, min(y1, gray.shape[0] - 1))
            x2 = max(x1 + 1, min(x2, gray.shape[1]))
            y2 = max(y1 + 1, min(y2, gray.shape[0]))
            points = cv2.goodFeaturesToTrack(
                previous_gray[y1:y2, x1:x2],
                maxCorners=12,
                qualityLevel=0.01,
                minDistance=5,
            )
            if points is None:
                continue
            points[:, 0, 0] += x1
            points[:, 0, 1] += y1
            next_points, valid, _ = cv2.calcOpticalFlowPyrLK(
                previous_gray, gray, points, None
            )
            if next_points is None or valid is None:
                continue
            displacement = next_points[valid.reshape(-1) == 1] - points[valid.reshape(-1) == 1]
            if len(displacement):
                dx, dy = np.median(displacement, axis=0).ravel()
                followed[index, 0] += dx
                followed[index, 2] += dx
                followed[index, 1] += dy
                followed[index, 3] += dy
        return followed

    def detect_latest() -> None:
        nonlocal last_tracks, latest_frame, latest_frame_number, detected_frame_number
        while not detector_stop.is_set():
            with detector_lock:
                if latest_frame is None or latest_frame_number == detected_frame_number:
                    frame_to_detect = None
                else:
                    frame_to_detect = latest_frame.copy()
                    frame_number = latest_frame_number
            if frame_to_detect is None:
                detector_stop.wait(0.005)
                continue
            results = model.predict(
                frame_to_detect,
                conf=Conf,
                iou=IOU,
                imgsz=inference_imgsz,
                verbose=False,
                augment=False,
                device=inference_device,
            )
            dets = []
            if results[0].boxes is not None and len(results[0].boxes):
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                clss = results[0].boxes.cls.cpu().numpy()
                for box, c, cls in zip(boxes, confs, clss):
                    dets.append([*box, c, cls])
            dets_np = np.array(dets) if dets else np.empty((0, 6))
            tracks = tracker.update(dets_np, frame_to_detect) if len(dets_np) else np.empty((0, 7))
            with detector_lock:
                last_tracks = tracks
                detected_frame_number = frame_number

    # warm up running a dummy inference to avoid first frame being slow
    if not _model_warmed_up:
        dummy = np.zeros((frame_h, frame_w, 3), np.uint8)
        model.predict(
            dummy,
            conf=Conf,
            iou=IOU,
            imgsz=inference_imgsz,
            verbose=False,
            device=inference_device,
        )
        _model_warmed_up = True
    detector_thread = None
    if web_mode:
        detector_thread = threading.Thread(target=detect_latest, daemon=True, name="mot-detector")
        detector_thread.start()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    print("Press Q to quit S to screenshot and V to toggle recording.")
    while True:
        if stop_event is not None and stop_event.is_set():
            break
        command = command_callback() if command_callback is not None else None
        if command == "stop":
            break
        if command == "screenshot":
            ts = time.strftime("%Y%m%d_%H%M%S")
            ss_path = OutDir / f"ss_{ts}.jpg"
            if "frame" in locals():
                cv2.imwrite(str(ss_path), frame)
                print(f"Saved SS image: {ss_path}")
        elif command == "toggle_recording":
            if not recording:
                ts = time.strftime("%Y%m%d_%H%M%S")
                videoPath = OutDir / f"tracking_{ts}.mp4"
                videoWriter = cv2.VideoWriter(
                    str(videoPath),
                    cv2.VideoWriter.fourcc(*"mp4v"),
                    30,
                    (frame_w, frame_h),
                )
                recording = True
                print(f"Recording started: {videoPath}")
            else:
                if videoWriter:
                    videoWriter.release()
                videoWriter = None
                recording = False
                print("Recording stopped")
        if paused:
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = False
            continue
        if web_mode:
            # Discard buffered frames so the browser receives the newest image
            # instead of displaying delayed frames after a slow inference pass.
            for _ in range(2):
                if not cap.grab():
                    break
            ret, frame = cap.retrieve()
        else:
            ret, frame = cap.read()
        if not ret:  # end of video file loop back to start or break for webcam
            if isinstance(source, str) and source != "0":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        frame_time = time.perf_counter()
        if last_frame_time is not None:
            frame_interval = frame_time - last_frame_time
            if frame_interval > 0:
                display_fps_buffer.append(1.0 / frame_interval)
        last_frame_time = frame_time
        if web_mode:
            with detector_lock:
                latest_frame = frame.copy()
                latest_frame_number = frame_idx
                if detected_frame_number != applied_detection_number:
                    displayed_tracks = last_tracks.copy()
                    applied_detection_number = detected_frame_number
                displayed_tracks = follow_tracks(frame, displayed_tracks)
                tracks = displayed_tracks
            previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            results = model.predict(
                frame,
                conf=Conf,
                iou=IOU,
                imgsz=inference_imgsz,
                verbose=False,
                augment=False,
                device=inference_device,
            )
            dets = []
            if results[0].boxes is not None and len(results[0].boxes):
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                clss = results[0].boxes.cls.cpu().numpy()
                for box, c, cls in zip(boxes, confs, clss):
                    dets.append([*box, c, cls])
            dets_np = np.array(dets) if dets else np.empty((0, 6))
            tracks = tracker.update(dets_np, frame) if len(dets_np) else np.empty((0, 7))
        # drawing the tracks
        for t in tracks:
            x1, y1, x2, y2 = t[0], t[1], t[2], t[3]
            tid = int(t[4])
            conf_t = float(t[5])
            cls_id = int(t[6]) if len(t) > 6 else 6 # if class missing default to person class
            frame = draw_box(frame, x1, y1, x2, y2, tid, cls_id, conf_t)

        # compute soothed fps
        fps = float(np.mean(display_fps_buffer))
        frame = draw_overlay(frame, fps, len(tracks))
        # if it is recording create a video
        if recording and videoWriter is not None:
            videoWriter.write(frame)
        if frame_callback is not None:
            frame_callback(frame)
        else:
            # Get current window size dynamically
            _, _, win_w, win_h = cv2.getWindowImageRect("Real-Time Tracking")

            # Resize only for display
            if win_w > 0 and win_h > 0:
                display_frame = cv2.resize(
                    frame,
                    (win_w, win_h),
                    interpolation=cv2.INTER_LINEAR
                )
            else:
                display_frame = frame

            cv2.imshow("Real-Time Tracking", display_frame)
            # key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = True
            elif key == ord("s"):
                ts = time.strftime("%Y%m%d_%H%M%S")
                ss_path = OutDir / f"ss_{ts}.jpg"
                cv2.imwrite(str(ss_path), frame)
                print("Saved SS image")
            elif key == ord("v"):
                if not recording:
                    ts=time.strftime("%Y%m%d_%H%M%S")
                    videoPath = OutDir / f"tracking_{ts}.mp4"
                    videoWriter = cv2.VideoWriter(
                        str(videoPath),
                        cv2.VideoWriter.fourcc(*"mp4v"),
                        30,(frame_w, frame_h),
                    )
                    recording = True
                    print(f"Recording started:{videoPath}")
                else:
                    if videoWriter:
                        videoWriter.release()
                    recording = False
                    print(f"Recording stopped:{videoPath}")
        frame_idx += 1

    # cleanup
    cap.release()
    detector_stop.set()
    if detector_thread is not None:
        detector_thread.join(timeout=2)
    if videoWriter:
        videoWriter.release()
    if not web_mode:
        cv2.destroyAllWindows()
    print(f"Session ended. Processed {frame_idx} frames.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Tracking Pipeline")
    parser.add_argument(
        "--source", default=0,
        help="Webcam index (0, 1, ...) or path to video file (default: 0)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Start recording immediately on launch"
    )
    args = parser.parse_args()

    # convert source to int if it's a digit string (webcam index)
    source = int(args.source) if str(args.source).isdigit() else args.source

    run(source=source, save_video=args.save)
