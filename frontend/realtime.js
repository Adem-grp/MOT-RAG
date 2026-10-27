const controls = document.querySelector("#tracking-controls");
const stopButton = document.querySelector("#stop-tracking");
const screenshotButton = document.querySelector("#screenshot");
const recordingButton = document.querySelector("#recording");
const stream = document.querySelector("#tracking-stream");
const placeholder = document.querySelector("#stream-placeholder");
const status = document.querySelector("#status");
const errorMessage = document.querySelector("#tracking-error");

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = !message;
}

stream.src = "/tracking-preview.svg";
stream.alt = "Camera ready preview";
stream.hidden = false;
placeholder.hidden = true;

async function readResponse(response) {
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) return response.json();
  const text = await response.text();
  return {detail: text || `Request failed with status ${response.status}.`};
}

controls.addEventListener("submit", async event => {
  event.preventDefault();
  showError("");
  status.textContent = "Starting...";
  try {
    const response = await fetch("/api/realtime/start", {method: "POST"});
    const data = await readResponse(response);
    if (!response.ok) throw new Error(data.detail || "Could not start tracking.");
    status.textContent = "Starting camera...";
    await waitForStream();
  } catch (error) {
    status.textContent = "Stopped";
    showError(error.message);
  }
});

async function waitForStream() {
  for (let attempt = 0; attempt < 120; attempt += 1) {
    const response = await fetch("/api/realtime/status");
    const data = await readResponse(response);
    if (!response.ok) throw new Error(data.detail || "Could not read tracking status.");
    if (data.error) throw new Error(data.error);
    if (data.running && data.frame_available) {
      stream.src = `/api/realtime/stream?started=${Date.now()}`;
      stream.alt = "Live annotated tracking stream";
      stream.hidden = false;
      placeholder.hidden = true;
      status.textContent = "Running";
      return;
    }
    if (!data.running) {
      throw new Error(data.error || "The tracking worker stopped before producing a camera frame.");
    }
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  throw new Error(
    "The camera is taking too long to produce a frame. Check the terminal for the tracking error."
  );
}

stopButton.addEventListener("click", async () => {
  await fetch("/api/realtime/stop", {method: "POST"});
  stream.removeAttribute("src");
  stream.src = "/tracking-preview.svg";
  stream.alt = "Camera ready preview";
  stream.hidden = false;
  placeholder.hidden = true;
  status.textContent = "Stopped";
});

async function sendCommand(command) {
  const response = await fetch(`/api/realtime/command/${command}`, {method: "POST"});
  const data = await readResponse(response);
  if (!response.ok) throw new Error(data.detail || "Tracking command failed.");
  return data;
}

screenshotButton.addEventListener("click", async () => {
  try {
    await sendCommand("screenshot");
    showError("");
  } catch (error) {
    showError(error.message);
  }
});

recordingButton.addEventListener("click", async () => {
  try {
    const data = await sendCommand("toggle_recording");
    recordingButton.textContent = data.recording
      ? "Stop Recording (V)"
      : "Start Recording (V)";
  } catch (error) {
    showError(error.message);
  }
});

document.addEventListener("keydown", async event => {
  if (event.key.toLowerCase() === "s") await sendCommand("screenshot");
  if (event.key.toLowerCase() === "v") {
    const data = await sendCommand("toggle_recording");
    recordingButton.textContent = data.recording
      ? "Stop Recording (V)"
      : "Start Recording (V)";
  }
  if (event.key.toLowerCase() === "q") await sendCommand("stop");
});

stream.addEventListener("error", () => {
  stream.hidden = false;
  placeholder.hidden = true;
  status.textContent = "Stopped";
  showError("The camera stream could not be displayed. Check the terminal for the tracking error.");
});
stream.hidden = false;
