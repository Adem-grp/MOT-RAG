"""Single-command launcher for the FastAPI backend and browser frontend.

Run with:
    python main.py

The preserved desktop comparison version is:
    python tkinter_version.py
"""

import threading
import time
from urllib.error import URLError
from urllib.request import urlopen
import webbrowser

import uvicorn


# Keep the server local for development; change these when deploying remotely.
HOST = "127.0.0.1"
PORT = 8000


def _open_frontend() -> None:
    # Wait until Uvicorn has completed startup instead of guessing how long imports take.
    health_url = f"http://{HOST}:{PORT}/api/health"
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            with urlopen(health_url, timeout=1) as response:
                if response.status == 200:
                    webbrowser.open(f"http://{HOST}:{PORT}")
                    return
        except (OSError, URLError):
            time.sleep(0.25)
    print(f"Could not open the browser: {health_url} did not become ready in 120 seconds.")


def main() -> None:
    # Opening the browser in a daemon thread keeps server startup non-blocking.
    threading.Thread(target=_open_frontend, daemon=True).start()
    # Import the ASGI app by path so Uvicorn owns the server lifecycle.
    uvicorn.run("backend.main:app", host=HOST, port=PORT)


if __name__ == "__main__":
    main()
