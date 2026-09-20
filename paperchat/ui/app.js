import * as pdfjsLib from "./vendor/pdf.min.mjs";

pdfjsLib.GlobalWorkerOptions.workerSrc = "./vendor/pdf.worker.min.mjs";

const statusEl = document.getElementById("status");
const progressWrap = document.getElementById("progress-wrap");
const progressFill = document.getElementById("progress-fill");
const paperListEl = document.getElementById("paper-list");
const chatLog = document.getElementById("chat-log");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const chatSend = document.getElementById("chat-send");

const viewerTitle = document.getElementById("viewer-title");
const pageIndicator = document.getElementById("page-indicator");
const prevPageBtn = document.getElementById("prev-page");
const nextPageBtn = document.getElementById("next-page");
const canvas = document.getElementById("pdf-canvas");
const ctx = canvas.getContext("2d");
const highlightLayer = document.getElementById("highlight-layer");

const RENDER_SCALE = 1.5;

let currentAssistantBubble = null;
let pdfCache = new Map(); // doc_path -> pdfjs document proxy
let viewerState = { docPath: null, pageNum: null };

function addMessage(role, text) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
  return div;
}

function renderCitations(container, citations) {
  if (!citations || !citations.length) return;
  const wrap = document.createElement("div");
  wrap.className = "citations";
  citations.forEach((c) => {
    const chip = document.createElement("span");
    chip.className = "citation-chip";
    chip.textContent = `[${c.index}] ${c.doc_name} p.${c.page + 1}`;
    chip.addEventListener("click", () => openCitation(c));
    wrap.appendChild(chip);
  });
  container.appendChild(wrap);
}

async function getPdfDoc(docPath) {
  if (pdfCache.has(docPath)) return pdfCache.get(docPath);
  const base64 = await window.pywebview.api.get_pdf_data(docPath);
  const bytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
  const doc = await pdfjsLib.getDocument({ data: bytes }).promise;
  pdfCache.set(docPath, doc);
  return doc;
}

async function renderPage(docPath, pageNum, bboxes, pageWidth, pageHeight) {
  const doc = await getPdfDoc(docPath);
  const page = await doc.getPage(pageNum + 1); // pdf.js pages are 1-indexed
  const viewport = page.getViewport({ scale: RENDER_SCALE });

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

  viewerState = { docPath, pageNum, numPages: doc.numPages };
  viewerTitle.textContent = docPath.split("/").pop();
  pageIndicator.textContent = `page ${pageNum + 1} / ${doc.numPages}`;
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

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = chatInput.value.trim();
  if (!question) return;
  chatInput.value = "";
  chatInput.disabled = true;
  chatSend.disabled = true;

  addMessage("user", question);
  currentAssistantBubble = addMessage("assistant", "");

  try {
    const result = await window.pywebview.api.ask_question(question);
    if (result.error) {
      currentAssistantBubble.textContent = `Error: ${result.error}`;
    } else {
      currentAssistantBubble.textContent = result.answer;
      renderCitations(currentAssistantBubble.parentElement, result.citations);
    }
  } catch (err) {
    currentAssistantBubble.textContent = `Error: ${err}`;
  } finally {
    chatInput.disabled = false;
    chatSend.disabled = false;
    chatInput.focus();
  }
});

// --- Callbacks invoked by the Python backend via window.evaluate_js ---

window.onStatus = (msg) => {
  statusEl.textContent = msg;
};

window.onDownloadProgress = (pct) => {
  progressWrap.classList.remove("hidden");
  progressFill.style.width = `${pct}%`;
};

window.onReady = (papers) => {
  statusEl.textContent = "Ready";
  progressWrap.classList.add("hidden");
  chatInput.disabled = false;
  chatSend.disabled = false;

  paperListEl.innerHTML = "";
  papers.forEach(({ name, path }) => {
    const li = document.createElement("li");
    li.textContent = name;
    li.addEventListener("click", () => openPaperByPath(path));
    paperListEl.appendChild(li);
  });
};

window.onError = (msg) => {
  statusEl.textContent = `Error: ${msg}`;
};

window.onToken = (token) => {
  if (currentAssistantBubble) {
    currentAssistantBubble.textContent += token;
    chatLog.scrollTop = chatLog.scrollHeight;
  }
};

window.onDone = () => {};
