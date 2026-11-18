const conversation = document.querySelector("#conversation");
const welcome = document.querySelector("#welcome");
const form = document.querySelector("#chat-form");
const questionInput = document.querySelector("#question-input");
const sendButton = document.querySelector("#send-button");
const tokenInput = document.querySelector("#api-token");
const history = [];

tokenInput.value = sessionStorage.getItem("financial-rag-token") || "";
tokenInput.addEventListener("input", () => {
  sessionStorage.setItem("financial-rag-token", tokenInput.value.trim());
});

function authHeaders() {
  const token = tokenInput.value.trim();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { ...authHeaders(), ...options.headers },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${response.status}).`);
  }
  return response.status === 204 ? null : response.json();
}

function setUploadStatus(text, type = "") {
  const status = document.querySelector("#upload-status");
  status.textContent = text;
  status.className = `upload-status ${type}`;
}

function renderDocuments(documents) {
  const list = document.querySelector("#documents-list");
  document.querySelector("#document-count").textContent = documents.length;
  list.replaceChildren();
  if (!documents.length) {
    const note = document.createElement("p");
    note.className = "empty-note";
    note.textContent = "Your uploaded filings will appear here.";
    list.append(note);
    return;
  }
  for (const document of documents) {
    const card = document.createElement("article");
    card.className = "document-card";
    const icon = document.createElement("span");
    icon.className = "file-icon";
    icon.textContent = document.name.split(".").pop().slice(0, 4);
    const info = document.createElement("div");
    info.className = "document-info";
    const name = document.createElement("div");
    name.className = "document-name";
    name.title = document.name;
    name.textContent = document.name;
    const meta = document.createElement("div");
    meta.className = "document-meta";
    meta.textContent = `${document.chunk_count} indexed passages`;
    info.append(name, meta);
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "delete-document";
    remove.title = "Remove document";
    remove.setAttribute("aria-label", `Remove ${document.name}`);
    remove.textContent = "×";
    remove.addEventListener("click", async () => {
      if (!confirm(`Remove "${document.name}" and its indexed passages?`)) return;
      try {
        await api(`/api/documents/${encodeURIComponent(document.id)}`, { method: "DELETE" });
        await refreshDocuments();
      } catch (error) {
        setUploadStatus(error.message, "error");
      }
    });
    card.append(icon, info, remove);
    list.append(card);
  }
}

async function refreshDocuments() {
  const result = await api("/api/documents");
  renderDocuments(result.documents);
}

function addMessage(role, text, citations = []) {
  if (welcome.isConnected) welcome.remove();
  const message = document.createElement("article");
  message.className = `message ${role === "user" ? "user-message" : "assistant-message"}`;
  if (role === "assistant") {
    const label = document.createElement("div");
    label.className = "assistant-label";
    label.textContent = "LedgerLens";
    message.append(label);
  }
  const body = document.createElement("div");
  body.className = "message-text";
  body.textContent = text;
  message.append(body);

  if (citations.length) {
    const citationList = document.createElement("div");
    citationList.className = "citations";
    for (const citation of citations) {
      const card = document.createElement("div");
      card.className = "citation-card";
      const top = document.createElement("div");
      top.className = "citation-top";
      const id = document.createElement("span");
      id.className = "citation-id";
      id.textContent = citation.id;
      const source = document.createElement("span");
      source.textContent = citation.source;
      const location = document.createElement("span");
      location.className = "citation-location";
      const parts = [];
      if (citation.page !== null && citation.page !== undefined) parts.push(`p. ${citation.page}`);
      if (citation.section) parts.push(citation.section);
      location.textContent = parts.join(" · ");
      top.append(id, source);
      if (parts.length) top.append(location);
      const excerpt = document.createElement("div");
      excerpt.className = "citation-excerpt";
      excerpt.textContent = citation.excerpt;
      card.append(top, excerpt);
      citationList.append(card);
    }
    message.append(citationList);
  }
  conversation.append(message);
  conversation.scrollTop = conversation.scrollHeight;
}

document.querySelector("#file-input").addEventListener("change", async (event) => {
  const [file] = event.target.files;
  if (!file) return;
  setUploadStatus(`Indexing ${file.name}...`);
  try {
    const body = new FormData();
    body.append("file", file);
    const result = await api("/api/documents", { method: "POST", body });
    setUploadStatus(
      result.duplicate ? `${result.name} is already indexed.` : `Indexed ${result.chunk_count} passages from ${result.name}.`,
      "success",
    );
    await refreshDocuments();
  } catch (error) {
    setUploadStatus(error.message, "error");
  } finally {
    event.target.value = "";
  }
});

const dropzone = document.querySelector("#dropzone");
for (const name of ["dragenter", "dragover"]) {
  dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.add("drag-over");
  });
}
for (const name of ["dragleave", "drop"]) {
  dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.remove("drag-over");
  });
}
dropzone.addEventListener("drop", (event) => {
  const input = document.querySelector("#file-input");
  if (!event.dataTransfer.files.length) return;
  input.files = event.dataTransfer.files;
  input.dispatchEvent(new Event("change", { bubbles: true }));
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question || sendButton.disabled) return;
  addMessage("user", question);
  questionInput.value = "";
  sendButton.disabled = true;
  const thinking = document.createElement("div");
  thinking.className = "thinking";
  thinking.setAttribute("aria-label", "Thinking");
  for (let i = 0; i < 3; i++) thinking.append(document.createElement("span"));
  conversation.append(thinking);
  conversation.scrollTop = conversation.scrollHeight;
  try {
    const result = await api("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history: history.slice(-8) }),
    });
    thinking.remove();
    addMessage("assistant", result.answer, result.citations);
    history.push({ role: "user", content: question }, { role: "assistant", content: result.answer });
  } catch (error) {
    thinking.remove();
    addMessage("assistant", `I couldn't complete that request: ${error.message}`);
  } finally {
    sendButton.disabled = false;
    questionInput.focus();
  }
});

document.querySelectorAll(".suggestion").forEach((button) => {
  button.addEventListener("click", () => {
    questionInput.value = button.textContent;
    form.requestSubmit();
  });
});
document.querySelector("#clear-chat").addEventListener("click", () => {
  history.length = 0;
  conversation.replaceChildren(welcome);
  questionInput.focus();
});
document.querySelector("#refresh-documents").addEventListener("click", () => {
  refreshDocuments().catch((error) => setUploadStatus(error.message, "error"));
});
tokenInput.addEventListener("change", () => {
  refreshDocuments().catch(() => {});
});

async function checkHealth() {
  const indicator = document.querySelector("#health-indicator");
  try {
    const health = await api("/api/health");
    indicator.classList.add("ready");
    indicator.innerHTML = "";
    const dot = document.createElement("i");
    indicator.append(dot, document.createTextNode(
      health.openai_configured ? " Ready" : " Add API key",
    ));
    if (!health.openai_configured) indicator.classList.remove("ready");
    document.querySelector("#upload-limit").textContent =
      `Maximum size: ${health.max_upload_mb} MB`;
  } catch {
    indicator.classList.add("error");
    indicator.textContent = " Offline";
  }
}

refreshDocuments().catch((error) => setUploadStatus(error.message, "error"));
checkHealth();
