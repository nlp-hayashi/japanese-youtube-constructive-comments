/**
 * YouTube comments: scrape → API scores → filter / sort by constructiveness.
 * Update SELECTORS when YouTube changes the DOM.
 */
const MIN_LEN = 1;

// Typical DOM (circa 2024–2025); update in DevTools if selectors break
const SELECTORS = {
  thread: "ytd-comment-thread-renderer",
  text: "#content-text, yt-formatted-string#content-text",
};

function getCommentThreads() {
  return Array.from(document.querySelectorAll(SELECTORS.thread));
}

function getTextFromThread(thread) {
  const el = thread.querySelector(SELECTORS.text);
  if (!el) return "";
  return el.innerText.replace(/\s+/g, " ").trim();
}

function scoreBatch(texts) {
  return new Promise((resolve, reject) => {
    chrome.runtime.sendMessage({ type: "SCORE", texts }, (resp) => {
      if (chrome.runtime.lastError) {
        reject(new Error(chrome.runtime.lastError.message));
        return;
      }
      if (!resp || !resp.ok) {
        reject(new Error((resp && resp.error) || "Failed to fetch scores"));
        return;
      }
      resolve(resp.data);
    });
  });
}

const PANEL_POS_KEY = "constructive-al-panel-pos";

function applySavedPanelPosition(panel) {
  try {
    const raw = localStorage.getItem(PANEL_POS_KEY);
    if (!raw) return;
    const { left, top } = JSON.parse(raw);
    panel.style.right = "auto";
    panel.style.left = `${left}px`;
    panel.style.top = `${top}px`;
  } catch (_) {
    /* ignore invalid saved position */
  }
}

function savePanelPosition(panel) {
  const rect = panel.getBoundingClientRect();
  localStorage.setItem(
    PANEL_POS_KEY,
    JSON.stringify({ left: rect.left, top: rect.top })
  );
}

function clampPanelPosition(panel, left, top) {
  const rect = panel.getBoundingClientRect();
  const margin = 8;
  return {
    left: Math.max(margin, Math.min(window.innerWidth - rect.width - margin, left)),
    top: Math.max(margin, Math.min(window.innerHeight - rect.height - margin, top)),
  };
}

function enablePanelDrag(panel) {
  const handle = panel.querySelector(".constructive-panel-header");
  if (!handle || handle.dataset.dragBound === "1") return;
  handle.dataset.dragBound = "1";

  let dragging = false;
  let startX = 0;
  let startY = 0;
  let startLeft = 0;
  let startTop = 0;

  handle.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    dragging = true;
    handle.setPointerCapture(e.pointerId);
    handle.classList.add("dragging");

    const rect = panel.getBoundingClientRect();
    panel.style.right = "auto";
    panel.style.left = `${rect.left}px`;
    panel.style.top = `${rect.top}px`;

    startX = e.clientX;
    startY = e.clientY;
    startLeft = rect.left;
    startTop = rect.top;
    e.preventDefault();
  });

  handle.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const next = clampPanelPosition(
      panel,
      startLeft + (e.clientX - startX),
      startTop + (e.clientY - startY)
    );
    panel.style.left = `${next.left}px`;
    panel.style.top = `${next.top}px`;
  });

  const endDrag = (e) => {
    if (!dragging) return;
    dragging = false;
    handle.classList.remove("dragging");
    try {
      handle.releasePointerCapture(e.pointerId);
    } catch (_) {
      /* pointer may already be released */
    }
    savePanelPosition(panel);
  };

  handle.addEventListener("pointerup", endDrag);
  handle.addEventListener("pointercancel", endDrag);
}

function ensurePanel() {
  let p = document.getElementById("constructive-al-panel");
  if (p) return p;
  p = document.createElement("div");
  p.id = "constructive-al-panel";
  p.innerHTML = `
    <div class="constructive-panel-header" title="Drag to move">
      <strong>Constructive comments (demo)</strong>
      <span class="constructive-drag-hint" aria-hidden="true">⋮⋮</span>
    </div>
    <button type="button" id="constructive-btn-score">Rescore &amp; sort</button>
    <button type="button" id="constructive-btn-filter" class="secondary">Show p ≥ 0.5 only</button>
    <button type="button" id="constructive-btn-reset" class="secondary">Reset view</button>
    <div class="status" id="constructive-status">API: 127.0.0.1:8765 (via background)</div>
  `;
  document.body.appendChild(p);
  applySavedPanelPosition(p);
  enablePanelDrag(p);
  return p;
}

function setStatus(msg) {
  const s = document.getElementById("constructive-status");
  if (s) s.textContent = msg;
}

/** Last comment-list parent we changed flex/order for sorting (for reset) */
let lastReorderedParent = null;

/**
 * Reorder display by descending score.
 * Do not appendChild ytd-comment-thread-renderer (breaks custom element lifecycle / infinite scroll).
 * Use column flex on the parent and style.order only (lower order = higher score = earlier).
 */
function reorderThreads(threads, scores) {
  if (threads.length === 0) return;
  const parent = threads[0].parentElement;
  if (!parent) return;
  lastReorderedParent = parent;

  const paired = threads.map((t, i) => ({ t, s: scores[i] }));
  paired.sort((a, b) => b.s - a.s);

  const cs = getComputedStyle(parent);
  const alreadyFlex = cs.display === "flex" || cs.display === "inline-flex";
  if (!alreadyFlex) {
    if (parent.dataset.constructiveSavedDisplay === undefined) {
      parent.dataset.constructiveSavedDisplay = parent.style.display || "";
      parent.dataset.constructiveSavedFlexDir = parent.style.flexDirection || "";
    }
    parent.style.display = "flex";
    parent.style.flexDirection = "column";
    parent.dataset.constructiveAppliedFlex = "1";
  }

  const ORDER_BASE = -1_000_000;
  paired.forEach(({ t }, rank) => {
    t.style.order = String(ORDER_BASE + rank);
  });
}

function restoreNativeCommentOrder() {
  document.querySelectorAll(SELECTORS.thread).forEach((t) => {
    t.style.order = "";
  });
  if (
    lastReorderedParent &&
    lastReorderedParent.dataset.constructiveAppliedFlex === "1"
  ) {
    lastReorderedParent.style.display =
      lastReorderedParent.dataset.constructiveSavedDisplay ?? "";
    lastReorderedParent.style.flexDirection =
      lastReorderedParent.dataset.constructiveSavedFlexDir ?? "";
    delete lastReorderedParent.dataset.constructiveAppliedFlex;
    delete lastReorderedParent.dataset.constructiveSavedDisplay;
    delete lastReorderedParent.dataset.constructiveSavedFlexDir;
  }
  lastReorderedParent = null;
}

function filterThreads(threads, scores, threshold) {
  threads.forEach((t, i) => {
    t.style.display = scores[i] >= threshold ? "" : "none";
  });
}

function showAllThreads(threads) {
  threads.forEach((t) => {
    t.style.display = "";
  });
}

let lastThreads = [];
let lastScores = [];

async function runScoreAndSort() {
  const threads = getCommentThreads().filter((t) => getTextFromThread(t).length >= MIN_LEN);
  if (threads.length === 0) {
    setStatus("No comments found. Expand the comment section or update SELECTORS.");
    return;
  }
  const texts = threads.map(getTextFromThread);
  setStatus(`Sending ${texts.length} comment(s) to API…`);
  try {
    const data = await scoreBatch(texts);
    const scores = data.scores;
    if (!scores || scores.length !== threads.length) throw new Error("Score count mismatch");
    lastThreads = threads;
    lastScores = scores;
    reorderThreads(threads, scores);
    setStatus(`Done. Sorted ${threads.length} comment(s) by score (high first).`);
  } catch (e) {
    setStatus(`Error: ${e.message} (Is the API running?)`);
  }
}

function runFilter() {
  if (lastThreads.length === 0) {
    setStatus('Run "Rescore & sort" first.');
    return;
  }
  filterThreads(lastThreads, lastScores, 0.5);
  setStatus("Hidden comments with p < 0.5 (rescore to refresh).");
}

function runReset() {
  showAllThreads(getCommentThreads());
  restoreNativeCommentOrder();
  setStatus("Restored original display and layout.");
}

function init() {
  const panel = ensurePanel();
  panel.querySelector("#constructive-btn-score").addEventListener("click", runScoreAndSort);
  panel.querySelector("#constructive-btn-filter").addEventListener("click", runFilter);
  panel.querySelector("#constructive-btn-reset").addEventListener("click", runReset);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
