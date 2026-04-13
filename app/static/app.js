const initialState = window.__INITIAL_STATE__ || {};

const uploadForm = document.getElementById("upload-form");
const pdfInput = document.getElementById("pdf-file");
const uploadMessage = document.getElementById("upload-message");
const pdfScroll = document.getElementById("pdf-scroll");
const pdfEmptyState = document.getElementById("pdf-empty-state");
const pdfPages = document.getElementById("pdf-pages");
const chatLog = document.getElementById("chat-log");
const questionForm = document.getElementById("question-form");
const questionInput = document.getElementById("question-input");
const sendButton = document.getElementById("send-button");
const chunksSection = document.getElementById("chunks-section");
const chunksList = document.getElementById("chunks-list");
const turnTemplate = document.getElementById("turn-template");
const chunkTemplate = document.getElementById("chunk-template");

const MAX_VISIBLE_CHUNKS = 3;

let activeTurn = null;
let currentPdfUrl = null;
let currentPdfDoc = null;
let renderGeneration = 0;

function renderEmptyConversation() {
  chatLog.innerHTML = "";
  const empty = document.createElement("div");
  empty.className = "empty-state conversation-empty";
  empty.innerHTML = `
    <strong>Aún no hay preguntas</strong>
    <p>Sube un PDF y escribe una pregunta para ver la respuesta y los fragmentos recuperados.</p>
  `;
  chatLog.appendChild(empty);
}

function showPdfEmptyState(show) {
  pdfEmptyState.style.display = show ? "grid" : "none";
  pdfPages.style.display = show ? "none" : "flex";
}

async function renderPdf(url) {
  if (!window.pdfjsLib) {
    pdfEmptyState.textContent = "No se pudo cargar el visor de PDF.";
    showPdfEmptyState(true);
    return;
  }

  currentPdfUrl = url;
  const generation = ++renderGeneration;

  pdfPages.innerHTML = "";
  pdfEmptyState.textContent = "Cargando PDF...";
  showPdfEmptyState(true);

  try {
    const loadingTask = pdfjsLib.getDocument({ url });
    currentPdfDoc = await loadingTask.promise;

    if (generation !== renderGeneration) {
      return;
    }

    showPdfEmptyState(false);

    const scaleBase = Math.max(1, pdfScroll.clientWidth - 32);
    const fragment = document.createDocumentFragment();

    for (let pageNum = 1; pageNum <= currentPdfDoc.numPages; pageNum += 1) {
      const page = await currentPdfDoc.getPage(pageNum);
      const baseViewport = page.getViewport({ scale: 1 });
      const scale = Math.min(1.6, Math.max(0.8, scaleBase / baseViewport.width));
      const viewport = page.getViewport({ scale });
      const pageWrap = document.createElement("article");
      pageWrap.className = "pdf-page";

      const label = document.createElement("div");
      label.className = "pdf-page-label";
      label.textContent = `Página ${pageNum} de ${currentPdfDoc.numPages}`;

      const canvas = document.createElement("canvas");
      const context = canvas.getContext("2d");
      const outputScale = window.devicePixelRatio || 1;

      canvas.width = Math.floor(viewport.width * outputScale);
      canvas.height = Math.floor(viewport.height * outputScale);
      canvas.style.width = `${Math.floor(viewport.width)}px`;
      canvas.style.height = `${Math.floor(viewport.height)}px`;

      const renderContext = {
        canvasContext: context,
        viewport,
        transform: outputScale !== 1 ? [outputScale, 0, 0, outputScale, 0, 0] : null,
      };

      await page.render(renderContext).promise;

      pageWrap.appendChild(label);
      pageWrap.appendChild(canvas);
      fragment.appendChild(pageWrap);
    }

    pdfPages.innerHTML = "";
    pdfPages.appendChild(fragment);
  } catch (error) {
    console.error(error);
    pdfEmptyState.textContent = "No se pudo mostrar el PDF cargado.";
    showPdfEmptyState(true);
  }
}

function setViewerState(state) {
  if (state && state.ready) {
    questionInput.disabled = false;
    sendButton.disabled = false;
    renderPdf(`/pdf?ts=${Date.now()}`);
  } else {
    currentPdfDoc = null;
    currentPdfUrl = null;
    pdfPages.innerHTML = "";
    pdfEmptyState.textContent = "Aquí se mostrará el PDF cargado";
    showPdfEmptyState(true);
    questionInput.disabled = true;
    sendButton.disabled = true;
  }
}

function syncState(state, message = "") {
  setViewerState(state);
  if (message) {
    uploadMessage.textContent = message;
    uploadMessage.className = state?.ready ? "helper success-message" : "helper";
  }
}

function clearConversation() {
  renderEmptyConversation();
  chunksList.innerHTML = "";
  chunksSection.style.display = "";
  activeTurn = null;
}

function createTurn(question) {
  if (chatLog.querySelector(".conversation-empty")) {
    chatLog.innerHTML = "";
  }

  const node = turnTemplate.content.cloneNode(true);
  const questionEl = node.querySelector(".turn-question-text");
  const answerEl = node.querySelector(".turn-answer-text");

  questionEl.textContent = question;
  answerEl.textContent = "Consultando el documento...";
  answerEl.classList.add("is-pending");

  chatLog.appendChild(node);
  chatLog.scrollTop = chatLog.scrollHeight;

  return { answerEl };
}

function normalizeChunkText(text) {
  return String(text || "").replace(/\s+/g, " ").trim();
}

function renderChunkList(container, chunks) {
  container.innerHTML = "";
  const visibleChunks = (chunks || []).slice(0, MAX_VISIBLE_CHUNKS);

  if (visibleChunks.length === 0) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "No hay chunks recuperados para mostrar.";
    container.appendChild(empty);
    return;
  }

  visibleChunks.forEach((chunk, index) => {
    const node = chunkTemplate.content.cloneNode(true);
    const title = node.querySelector(".chunk-title");
    const score = node.querySelector(".chunk-score");
    const text = node.querySelector(".chunk-text");
    const scoreText =
      typeof chunk.score === "number" ? `${Math.round(chunk.score * 100)}%` : "";

    title.textContent = `Chunk ${index + 1}`;
    score.textContent = scoreText;
    text.textContent = normalizeChunkText(chunk.content);
    container.appendChild(node);
  });
}

function updateTurn(turn, answer, chunks, isError = false) {
  turn.answerEl.textContent = answer;
  turn.answerEl.classList.remove("is-pending", "is-error");
  if (isError) {
    turn.answerEl.classList.add("is-error");
  }
  renderChunkList(chunksList, chunks);
  chatLog.scrollTop = chatLog.scrollHeight;
}

renderEmptyConversation();
renderChunkList(chunksList, []);
syncState(initialState, initialState.message || "");

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = pdfInput.files[0];

  if (!file) {
    uploadMessage.textContent = "Selecciona un archivo PDF antes de cargarlo.";
    uploadMessage.className = "helper error-message";
    return;
  }

  uploadMessage.textContent = "Procesando PDF...";
  uploadMessage.className = "helper";

  try {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("/upload", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();

    if (!response.ok || !data.ok) {
      throw new Error(data.error || "No se pudo procesar el PDF.");
    }

    window.__INITIAL_STATE__ = data.state;
    clearConversation();
    syncState(data.state, data.message);
    pdfInput.value = "";
  } catch (error) {
    uploadMessage.textContent = error.message;
    uploadMessage.className = "helper error-message";
    window.__INITIAL_STATE__ = { ready: false, chunks: [] };
    clearConversation();
    syncState(window.__INITIAL_STATE__, "");
  }
});

questionForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const question = questionInput.value.trim();
  if (!question) {
    return;
  }

  activeTurn = createTurn(question);
  questionInput.value = "";
  questionInput.disabled = true;
  sendButton.disabled = true;

  try {
    const formData = new FormData();
    formData.append("question", question);

    const response = await fetch("/ask", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();

    if (!response.ok || !data.ok) {
      throw new Error(data.error || "No se pudo responder la pregunta.");
    }

    const outOfContext = Boolean(data.out_of_context);

    if (outOfContext) {
      chunksSection.style.display = "none";
      chunksList.innerHTML = "";
      updateTurn(activeTurn, data.answer, [], false);
    } else {
      chunksSection.style.display = "";
      updateTurn(activeTurn, data.answer, data.chunks || [], false);
    }
  } catch (error) {
    chunksSection.style.display = "";
    updateTurn(activeTurn, error.message, [], true);
  } finally {
    questionInput.disabled = !window.__INITIAL_STATE__?.ready;
    sendButton.disabled = !window.__INITIAL_STATE__?.ready;
    questionInput.focus();
  }
});

window.addEventListener("resize", () => {
  if (currentPdfUrl && window.__INITIAL_STATE__?.ready) {
    renderPdf(currentPdfUrl);
  }
});
