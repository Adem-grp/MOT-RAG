"""Web controls and MJPEG streaming for the real-time tracking pipeline."""

from __future__ import annotations

import threading
import traceback
import io
import zipfile
from collections import deque
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

ROOT_DIR = Path(__file__).resolve().parents[2]
router = APIRouter(prefix="/api/realtime", tags=["realtime"])


class TrackingService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame_condition = threading.Condition(self._lock)
        self._frame: bytes | None = None
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._camera = None
        self._error: str | None = None
        self._commands = deque()
        self._recording = False

    def status(self) -> dict[str, str | bool | None]:
        with self._lock:
            running = self._thread is not None and self._thread.is_alive()
            return {
                "running": running,
                "frame_available": self._frame is not None,
                "error": self._error,
                "recording": self._recording,
            }

    def start(self) -> None:
        # Initialize native vision modules on the server thread before starting
        # the worker. This avoids concurrent NumPy/OpenCV initialization.
        import cv2
        from Real_Time_Pipeline import run

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("Tracking is already running.")
            self._frame = None
            self._error = None
            self._commands.clear()
            self._recording = False
            stop_event = threading.Event()
            self._stop_event = stop_event
            camera = self._camera
            self._camera = None
            self._thread = threading.Thread(
                target=self._run,
                args=(stop_event, cv2, run, camera),
                daemon=True,
                name="mot-realtime-tracker",
            )
            self._thread.start()

    def stop(self) -> None:
        self.command("stop")
        with self._lock:
            if self._stop_event is not None:
                self._stop_event.set()
            self._frame_condition.notify_all()

    def command(self, command: str) -> None:
        with self._lock:
            self._commands.append(command)
            if command == "toggle_recording":
                self._recording = not self._recording
            elif command == "stop":
                self._recording = False

    def next_command(self) -> str | None:
        with self._lock:
            return self._commands.popleft() if self._commands else None

    def frame_stream(self):
        while True:
            with self._frame_condition:
                if self._frame is None:
                    self._frame_condition.wait(timeout=1)
                frame = self._frame
                running = self._thread is not None and self._thread.is_alive()
                error = self._error
            if frame is not None:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            elif error:
                return
            elif not running:
                return

    def prepare_camera(self) -> None:
        import cv2

        with self._lock:
            if self._camera is not None and self._camera.isOpened():
                return
            camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not camera.isOpened():
                camera.release()
                raise RuntimeError("Cannot open webcam 0.")
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._camera = camera

    def _run(self, stop_event: threading.Event, cv2, run, camera) -> None:
        try:
            def publish(frame) -> None:
                success, encoded = cv2.imencode(".jpg", frame)
                if not success:
                    return
                with self._frame_condition:
                    self._frame = encoded.tobytes()
                    self._frame_condition.notify_all()

            run(
                source=0,
                frame_callback=publish,
                stop_event=stop_event,
                command_callback=self.next_command,
                capture=camera,
            )
        except Exception as error:
            traceback.print_exc()
            with self._frame_condition:
                self._recording = False
                self._error = str(error)
                self._frame_condition.notify_all()


tracking_service = TrackingService()


@router.get("/status")
def status():
    return tracking_service.status()


@router.post("/start")
def start():
    try:
        tracking_service.start()
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except Exception as error:
        traceback.print_exc()
        raise HTTPException(
            status_code=503,
            detail=f"Tracking dependencies could not be loaded: {error}",
        ) from error
    return tracking_service.status()


@router.post("/prepare")
def prepare():
    try:
        tracking_service.prepare_camera()
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"ready": True}


@router.post("/stop")
def stop():
    tracking_service.stop()
    return tracking_service.status()


@router.post("/command/{command}")
def command(command: str):
    if command not in {"screenshot", "toggle_recording", "stop"}:
        raise HTTPException(status_code=400, detail="Unsupported tracking command.")
    tracking_service.command(command)
    return tracking_service.status()


@router.get("/download")
def download_outputs():
    from Real_Time_Pipeline import OutDir

    files = [path for path in OutDir.glob("*") if path.is_file()]
    if not files:
        raise HTTPException(status_code=404, detail="No screenshots or videos have been created.")
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for path in files:
            zip_file.write(path, arcname=path.name)
    archive.seek(0)
    return StreamingResponse(
        archive,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="realtime_outputs.zip"'},
    )


@router.get("/stream")
def stream():
    if not tracking_service.status()["running"]:
        raise HTTPException(status_code=409, detail="Start tracking before opening the stream.")
    return StreamingResponse(
        tracking_service.frame_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
