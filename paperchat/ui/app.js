import * as pdfjsLib from "./vendor/pdf.min.mjs";

pdfjsLib.GlobalWorkerOptions.workerSrc = "./vendor/pdf.worker.min.mjs";

const statusEl = document.getElementById("status");
const statusSpinner = document.getElementById("status-spinner");
const progressWrap = document.getElementById("progress-wrap");
const progressFill = document.getElementById("progress-fill");
const paperListEl = document.getElementById("paper-list");
const chatLog = document.getElementById("chat-log");
const emptyState = document.getElementById("empty-state");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const chatSend = document.getElementById("chat-send");

const viewerTitle = document.getElementById("viewer-title");
const pageIndicator = document.getElementById("page-indicator");
const prevPageBtn = document.getElementById("prev-page");
const nextPageBtn = document.getElementById("next-page");
const zoomInBtn = document.getElementById("zoom-in");
const zoomOutBtn = document.getElementById("zoom-out");
const zoomIndicator = document.getElementById("zoom-indicator");
const canvas = document.getElementById("pdf-canvas");
const ctx = canvas.getContext("2d");
const highlightLayer = document.getElementById("highlight-layer");
const pageContainer = document.getElementById("page-container");
const viewerEmpty = document.getElementById("viewer-empty");

let currentScale = 1.5;
const MIN_SCALE = 0.6;
const MAX_SCALE = 3.0;

let currentAssistantBody = null;
let currentAssistantText = "";
let currentTypingRow = null;
let pdfCache = new Map(); // doc_path -> pdfjs document proxy
let viewerState = { docPath: null, pageNum: null, bboxes: null, pageWidth: null, pageHeight: null };

// ---------- tiny markdown + inline-citation renderer ----------

function escapeHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function renderAnswerHtml(text) {
  let html = escapeHtml(text);

  // inline citation markers like [1] or [1, 4] -> clickable badges
  html = html.replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, (_, nums) => {
    return nums
      .split(",")
      .map((n) => `<span class="cite-badge" data-cite="${n.trim()}">${n.trim()}</span>`)
      .join("");
  });

  // bold / inline code
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

  // paragraphs (blank-line separated); single newlines become <br>
  const paragraphs = html.split(/\n{2,}/).map((p) => p.replace(/\n/g, "<br>"));
  return paragraphs.map((p) => `<p>${p}</p>`).join("");
}

// ---------- chat rendering ----------

function hideEmptyState() {
  if (emptyState) emptyState.remove();
}

function addUserMessage(text) {
  hideEmptyState();
  const row = document.createElement("div");
  row.className = "msg-row user";
  row.innerHTML = `
    <div class="avatar user">You</div>
    <div class="msg-body">${escapeHtml(text)}</div>
  `;
  chatLog.appendChild(row);
  chatLog.scrollTop = chatLog.scrollHeight;
  return row;
}

function addTypingIndicator() {
  const row = document.createElement("div");
  row.className = "msg-row assistant";
  row.innerHTML = `
    <div class="avatar assistant">P</div>
    <div class="msg-body">
      <div class="typing-dots"><span></span><span></span><span></span></div>
    </div>
  `;
  chatLog.appendChild(row);
  chatLog.scrollTop = chatLog.scrollHeight;
  return row;
}

function beginAssistantMessage() {
  currentTypingRow = addTypingIndicator();
  currentAssistantBody = null;
  currentAssistantText = "";
}

function ensureAssistantBodyVisible() {
  if (currentAssistantBody) return;
  // swap the typing indicator's body content the first time real text arrives
  currentAssistantBody = currentTypingRow.querySelector(".msg-body");
  currentAssistantBody.innerHTML = "";
}

function appendAssistantToken(token) {
  ensureAssistantBodyVisible();
  currentAssistantText += token;
  currentAssistantBody.innerHTML = renderAnswerHtml(currentAssistantText);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function finalizeAssistantMessage(result, citations) {
  ensureAssistantBodyVisible();
  if (result.error) {
    currentAssistantBody.innerHTML = `<p>Error: ${escapeHtml(result.error)}</p>`;
    currentAssistantBody.classList.add("error");
    return;
  }
  currentAssistantText = result.answer;
  currentAssistantBody.innerHTML = renderAnswerHtml(currentAssistantText);
  currentTypingRow._citations = citations;
  chatLog.scrollTop = chatLog.scrollHeight;
}

// clicking an inline citation badge -> open + highlight that passage
chatLog.addEventListener("click", (e) => {
  const badge = e.target.closest(".cite-badge");
  if (!badge) return;
  const row = badge.closest(".msg-row");
  const citations = row && row._citations;
  if (!citations) return;
  const idx = Number(badge.dataset.cite);
  const citation = citations.find((c) => c.index === idx);
  if (citation) openCitation(citation);
});

// ---------- PDF viewer ----------

async function getPdfDoc(docPath) {
  if (pdfCache.has(docPath)) return pdfCache.get(docPath);
  const base64 = await window.pywebview.api.get_pdf_data(docPath);
  const bytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
  const doc = await pdfjsLib.getDocument({ data: bytes }).promise;
  pdfCache.set(docPath, doc);
  return doc;
}

async function renderPage(docPath, pageNum, bboxes, pageWidth, pageHeight) {
  viewerEmpty.classList.add("hidden");
  pageContainer.classList.remove("hidden");

  const doc = await getPdfDoc(docPath);
  const page = await doc.getPage(pageNum + 1); // pdf.js pages are 1-indexed
  const viewport = page.getViewport({ scale: currentScale });

  canvas.width = viewport.width;
  canvas.height = viewport.height;
  highlightLayer.style.width = `${viewport.width}px`;
  highlightLayer.style.height = `${viewport.height}px`;

  await page.render({ canvasContext: ctx, viewport }).promise;

  highlightLayer.innerHTML = "";
  if (bboxes && bboxes.length) {
    const scaleX = viewport.width / pageWidth;
    const scaleY = viewport.height / pageHeight;
    for (const [x0, y0, x1, y1] of bboxes) {
      const box = document.createElement("div");
      box.className = "highlight-box";
      box.style.left = `${x0 * scaleX}px`;
      box.style.top = `${y0 * scaleY}px`;
      box.style.width = `${(x1 - x0) * scaleX}px`;
      box.style.height = `${(y1 - y0) * scaleY}px`;
      highlightLayer.appendChild(box);
    }
    highlightLayer.firstChild.scrollIntoView({ block: "center", behavior: "smooth" });
  }

  viewerState = { docPath, pageNum, bboxes, pageWidth, pageHeight, numPages: doc.numPages };
  viewerTitle.textContent = docPath.split("/").pop();
  viewerTitle.title = docPath.split("/").pop();
  pageIndicator.textContent = `${pageNum + 1} / ${doc.numPages}`;
  zoomIndicator.textContent = `${Math.round((currentScale / 1.5) * 100)}%`;

  zoomInBtn.disabled = currentScale >= MAX_SCALE;
  zoomOutBtn.disabled = currentScale <= MIN_SCALE;
  prevPageBtn.disabled = pageNum <= 0;
  nextPageBtn.disabled = pageNum + 1 >= doc.numPages;
}

async function openCitation(c) {
  await renderPage(c.doc_path, c.page, c.bboxes, c.page_width, c.page_height);
}

async function openPaperByPath(path) {
  await renderPage(path, 0, null, null, null);
}

prevPageBtn.addEventListener("click", async () => {
  if (!viewerState.docPath || viewerState.pageNum <= 0) return;
  await renderPage(viewerState.docPath, viewerState.pageNum - 1, null, null, null);
});
nextPageBtn.addEventListener("click", async () => {
  if (!viewerState.docPath) return;
  const doc = await getPdfDoc(viewerState.docPath);
  if (viewerState.pageNum + 1 >= doc.numPages) return;
  await renderPage(viewerState.docPath, viewerState.pageNum + 1, null, null, null);
});
zoomInBtn.addEventListener("click", async () => {
  if (!viewerState.docPath || currentScale >= MAX_SCALE) return;
  currentScale = Math.min(MAX_SCALE, currentScale + 0.25);
  await renderPage(viewerState.docPath, viewerState.pageNum, viewerState.bboxes, viewerState.pageWidth, viewerState.pageHeight);
});
zoomOutBtn.addEventListener("click", async () => {
  if (!viewerState.docPath || currentScale <= MIN_SCALE) return;
  currentScale = Math.max(MIN_SCALE, currentScale - 0.25);
  await renderPage(viewerState.docPath, viewerState.pageNum, viewerState.bboxes, viewerState.pageWidth, viewerState.pageHeight);
});

// ---------- chat input (auto-resize + enter-to-send) ----------

function autoResizeInput() {
  chatInput.style.height = "auto";
  chatInput.style.height = `${Math.min(chatInput.scrollHeight, 120)}px`;
}
chatInput.addEventListener("input", autoResizeInput);
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    chatForm.requestSubmit();
  }
});

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = chatInput.value.trim();
  if (!question) return;
  chatInput.value = "";
  autoResizeInput();
  chatInput.disabled = true;
  chatSend.disabled = true;

  addUserMessage(question);
  beginAssistantMessage();

  try {
    const result = await window.pywebview.api.ask_question(question);
    finalizeAssistantMessage(result, result.citations || []);
  } catch (err) {
    ensureAssistantBodyVisible();
    currentAssistantBody.innerHTML = `<p>Error: ${escapeHtml(String(err))}</p>`;
    currentAssistantBody.classList.add("error");
  } finally {
    chatInput.disabled = false;
    chatSend.disabled = false;
    chatInput.focus();
  }
});

// ---------- Callbacks invoked by the Python backend via window.evaluate_js ----------

window.onStatus = (msg) => {
  statusEl.textContent = msg;
  statusSpinner.classList.remove("hidden");
};

window.onDownloadProgress = (pct) => {
  progressWrap.classList.remove("hidden");
  progressFill.style.width = `${pct}%`;
};

window.onReady = (papers) => {
  statusEl.textContent = "Ready";
  statusSpinner.classList.add("hidden");
  progressWrap.classList.add("hidden");
  chatInput.disabled = false;
  chatSend.disabled = false;

  paperListEl.innerHTML = "";
  papers.forEach(({ name, path }) => {
    const li = document.createElement("li");
    li.innerHTML = `
      <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.8">
        <path d="M6 2h9l5 5v13a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2z"></path>
        <path d="M15 2v5h5"></path>
      </svg>
      <span class="paper-name">${escapeHtml(name)}</span>
    `;
    li.addEventListener("click", () => openPaperByPath(path));
    paperListEl.appendChild(li);
  });
};

window.onError = (msg) => {
  statusEl.textContent = `Error: ${msg}`;
  statusSpinner.classList.add("hidden");
};

window.onToken = (token) => {
  appendAssistantToken(token);
};

window.onDone = () => {};
