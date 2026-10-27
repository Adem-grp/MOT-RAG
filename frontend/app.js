const chat = document.querySelector("#chat");
const composer = document.querySelector("#composer");
const question = document.querySelector("#question");
const send = document.querySelector("#send");
const status = document.querySelector("#status");

function addMessage(speaker, text, sources = []) {
  // Build DOM nodes instead of using innerHTML so model output stays plain text.
  const message = document.createElement("article");
  message.className = `message ${speaker}`;
  const label = document.createElement("strong");
  label.textContent = speaker === "user" ? "You" : "MOT Research Assistant";
  message.append(label, document.createTextNode(text));
  if (sources.length) {
    // The API returns full source objects; the first UI only shows paper and page.
    const sourceList = document.createElement("div");
    sourceList.className = "sources";
    sourceList.textContent = "Sources: " + sources.map(
      source => `${source.source} p.${source.page}`
    ).join(", ");
    message.append(sourceList);
  }
  chat.append(message);
  chat.scrollTop = chat.scrollHeight;
}

function showWelcome() {
  // A new chat resets only browser state; it does not affect the RAG database.
  chat.replaceChildren();
  addMessage("assistant",
    "Hello! You can ask any questions about MOT trackers, tracking methods, and research papers.");
}

composer.addEventListener("submit", async event => {
  // Prevent a full-page form reload and keep the conversation in the current view.
  event.preventDefault();
  const value = question.value.trim();
  // Disable duplicate submissions while retrieval and generation are running.
  if (!value || send.disabled) return;
  addMessage("user", value);
  question.value = "";
  send.disabled = true;
  status.textContent = "Thinking...";
  try {
    // This request shape matches backend.schemas.ChatRequest.
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({question: value})
    });
    const data = await response.json();
    // FastAPI places server-side errors in the detail field.
    if (!response.ok) throw new Error(data.detail || "Request failed.");
    addMessage("assistant", data.answer, data.sources);
  } catch (error) {
    addMessage("assistant", `I could not answer that request.\n\nDetails: ${error.message}`);
  } finally {
    // Restore the controls after success or failure and return focus to the input.
    send.disabled = false;
    status.textContent = "Ready";
    question.focus();
  }
});

document.querySelector("#new-chat").addEventListener("click", showWelcome);

const liveTrackingLink = document.querySelector(".live-tracking");
liveTrackingLink.addEventListener("click", async event => {
  event.preventDefault();
  window.location.href = liveTrackingLink.href;
});
showWelcome();
