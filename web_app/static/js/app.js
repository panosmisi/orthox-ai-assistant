function setupTabs() {
  document.querySelectorAll("[data-tabs]").forEach((tabs) => {
    const buttons = tabs.querySelectorAll("[data-tab]");
    const panels = tabs.querySelectorAll("[data-panel]");
    buttons.forEach((button) => {
      button.addEventListener("click", () => {
        const target = button.dataset.tab;
        buttons.forEach((b) => b.classList.toggle("active", b === button));
        panels.forEach((panel) => panel.classList.toggle("active", panel.dataset.panel === target));
      });
    });
  });
}

function setupThemeToggle() {
  const root = document.documentElement;
  const button = document.querySelector("[data-theme-toggle]");
  const label = document.querySelector("[data-theme-toggle-label]");
  if (!button) return;

  const applyTheme = (theme) => {
    const isDark = theme === "dark";
    if (isDark) {
      root.dataset.theme = "dark";
    } else {
      delete root.dataset.theme;
    }
    button.setAttribute("aria-pressed", isDark ? "true" : "false");
    button.setAttribute("aria-label", isDark ? "Switch to light mode" : "Switch to dark mode");
    if (label) label.textContent = isDark ? "Light" : "Dark";
  };

  let savedTheme = "light";
  try {
    savedTheme = localStorage.getItem("orthox-theme") || "light";
  } catch (error) {}
  applyTheme(savedTheme === "dark" ? "dark" : "light");

  button.addEventListener("click", () => {
    const nextTheme = root.dataset.theme === "dark" ? "light" : "dark";
    try {
      localStorage.setItem("orthox-theme", nextTheme);
    } catch (error) {}
    applyTheme(nextTheme);
  });
}

function setupBboxToggle() {
  const button = document.querySelector("[data-toggle-bbox]");
  const boxes = Array.from(document.querySelectorAll("[data-bbox]"));
  if (!button || boxes.length === 0) {
    if (button) button.disabled = true;
    return;
  }
  button.addEventListener("click", () => {
    const shouldHide = !boxes.every((box) => box.classList.contains("hidden"));
    boxes.forEach((box) => box.classList.toggle("hidden", shouldHide));
  });
}

function setupCueSelection() {
  const frame = document.querySelector("#imageFrame");
  const boxes = Array.from(document.querySelectorAll("[data-bbox]"));
  const cards = Array.from(document.querySelectorAll("[data-cue-target]"));
  if (!boxes.length || !cards.length) return;

  const selectCue = (id) => {
    boxes.forEach((box) => box.classList.toggle("active", box.dataset.bboxId === id));
    cards.forEach((card) => card.classList.toggle("active", card.dataset.cueTarget === id));
    const target = boxes.find((box) => box.dataset.bboxId === id);
    if (frame && target) {
      const boxCenterX = target.offsetLeft + target.offsetWidth / 2;
      const boxCenterY = target.offsetTop + target.offsetHeight / 2;
      frame.scrollTo({
        left: Math.max(0, boxCenterX - frame.clientWidth / 2),
        top: Math.max(0, boxCenterY - frame.clientHeight / 2),
        behavior: "smooth",
      });
    }
  };

  cards.forEach((card) => {
    card.addEventListener("click", () => selectCue(card.dataset.cueTarget));
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectCue(card.dataset.cueTarget);
      }
    });
  });

  if (!boxes.some((box) => box.classList.contains("active"))) {
    boxes[0].classList.add("active");
  }
}

function setupAnalysisProgress() {
  const shell = document.querySelector(".analysis-shell[data-target]");
  if (!shell) return;
  const stages = Array.from(document.querySelectorAll("#stageList li"));
  const fill = document.querySelector("[data-progress-fill]");
  const readout = document.querySelector("[data-progress-readout]");
  const current = document.querySelector("[data-progress-current]");
  const label = document.querySelector("[data-progress-label]");
  const target = shell.dataset.target;
  let index = 0;

  const updateProgress = (activeIndex) => {
    const total = Math.max(1, stages.length);
    const visibleStage = Math.min(activeIndex + 1, total);
    const completed = Math.min(activeIndex + 1, total);
    const percent = Math.round((completed / total) * 100);
    if (fill) fill.style.width = `${percent}%`;
    if (readout) readout.textContent = `${percent}%`;
    if (label) label.textContent = percent >= 100 ? "Finalizing" : `Stage ${visibleStage} of ${total}`;
    if (current) {
      const stageText = stages[Math.min(activeIndex, total - 1)]?.textContent?.trim();
      current.textContent = percent >= 100 ? "Opening the review result." : stageText || "Running review.";
    }
  };

  const tick = () => {
    stages.forEach((stage, i) => {
      stage.classList.toggle("done", i < index);
      stage.classList.toggle("active", i === index);
    });
    updateProgress(index);
    index += 1;
    if (index <= stages.length) {
      setTimeout(tick, 520);
    } else {
      setTimeout(() => { window.location.href = target; }, 420);
    }
  };
  tick();
}

function setupImageZoom() {
  const frame = document.querySelector("#imageFrame");
  const canvas = document.querySelector("[data-image-canvas]");
  const image = document.querySelector("[data-review-image]");
  const zoomIn = document.querySelector("[data-zoom-in]");
  const zoomOut = document.querySelector("[data-zoom-out]");
  const zoomReset = document.querySelector("[data-zoom-reset]");
  const readout = document.querySelector("[data-zoom-readout]");

  if (!frame || !canvas || !image || !zoomIn || !zoomOut || !zoomReset) return;

  const levels = [1, 1.25, 1.5, 2, 3, 4, 6, 8];
  let baseWidth = 0;
  let baseHeight = 0;
  let zoomIndex = 0;

  const updateReadout = () => {
    if (readout) readout.textContent = `${Math.round(levels[zoomIndex] * 100)}%`;
  };

  const applyZoom = (preserveCenter = true) => {
    if (!baseWidth || !baseHeight) return;

    const oldCenterX = frame.scrollLeft + frame.clientWidth / 2;
    const oldCenterY = frame.scrollTop + frame.clientHeight / 2;
    const oldWidth = canvas.offsetWidth || baseWidth;
    const oldHeight = canvas.offsetHeight || baseHeight;
    const scale = levels[zoomIndex];
    const newWidth = Math.round(baseWidth * scale);
    const newHeight = Math.round(baseHeight * scale);

    canvas.style.width = `${newWidth}px`;
    canvas.style.height = `${newHeight}px`;
    image.style.width = `${newWidth}px`;
    image.style.height = `${newHeight}px`;

    updateReadout();

    if (preserveCenter && oldWidth && oldHeight) {
      const ratioX = oldCenterX / oldWidth;
      const ratioY = oldCenterY / oldHeight;
      frame.scrollLeft = Math.max(0, ratioX * newWidth - frame.clientWidth / 2);
      frame.scrollTop = Math.max(0, ratioY * newHeight - frame.clientHeight / 2);
    }
  };

  const fitImage = () => {
    if (!image.naturalWidth || !image.naturalHeight) return;
    const availableWidth = Math.max(220, frame.clientWidth - 28);
    const availableHeight = 680;
    const fitScale = Math.min(availableWidth / image.naturalWidth, availableHeight / image.naturalHeight, 1);
    baseWidth = Math.max(1, Math.round(image.naturalWidth * fitScale));
    baseHeight = Math.max(1, Math.round(image.naturalHeight * fitScale));
    zoomIndex = 0;
    applyZoom(false);
    frame.scrollTo({ top: 0, left: 0 });
  };

  zoomIn.addEventListener("click", () => {
    zoomIndex = Math.min(levels.length - 1, zoomIndex + 1);
    applyZoom(true);
  });

  zoomOut.addEventListener("click", () => {
    zoomIndex = Math.max(0, zoomIndex - 1);
    applyZoom(true);
  });

  zoomReset.addEventListener("click", () => {
    zoomIndex = 0;
    applyZoom(false);
    frame.scrollTo({ top: 0, left: 0, behavior: "smooth" });
  });

  if (image.complete) {
    fitImage();
  } else {
    image.addEventListener("load", fitImage, { once: true });
  }

  window.addEventListener("resize", fitImage);
}

function setupUploadPicker() {
  const input = document.querySelector("[data-file-input]");
  const fileName = document.querySelector("[data-file-name]");
  if (!input || !fileName) return;

  input.addEventListener("change", () => {
    const file = input.files && input.files[0];
    fileName.textContent = file ? file.name : "PNG or JPEG, up to 15 MB";
  });
}

function setupProblemDescriptionLoading() {
  const form = document.querySelector("[data-problem-form]");
  if (!form) return;
  const button = document.querySelector("[data-problem-submit]");
  const loading = document.querySelector("[data-problem-loading]");
  const fill = document.querySelector("[data-problem-loading-fill]");
  const percent = document.querySelector("[data-problem-loading-percent]");
  const message = document.querySelector("[data-problem-loading-message]");
  const live = document.querySelector("[data-problem-loading-live]");
  const steps = Array.from(document.querySelectorAll("[data-problem-loading-steps] li"));

  const milestones = [
    { pct: 10, step: 0, text: "Reading the description and checking that it fits this mode.", live: "Reading submitted details" },
    { pct: 24, step: 0, text: "Finding the body area, mechanism, symptoms, and warning signs.", live: "Organizing the clinical details" },
    { pct: 42, step: 1, text: "Preparing a focused evidence query from the case details.", live: "Preparing evidence context" },
    { pct: 58, step: 1, text: "Collecting evidence context when the description is specific enough.", live: "Checking source context" },
    { pct: 73, step: 2, text: "Drafting a cautious response from the clinical details.", live: "Writing the review text" },
    { pct: 86, step: 2, text: "The local language model may take extra time on this step.", live: "Still writing - this can take a little while" },
    { pct: 93, step: 3, text: "Checking the response before showing it.", live: "Safety-checking the response" },
  ];

  const waitingMessages = [
    "Still working - the local model can be slow on longer cases.",
    "Review is still running. The page has not frozen.",
    "Final checks are continuing before the answer is shown.",
    "Almost there - keeping the response cautious and structured.",
  ];

  form.addEventListener("submit", () => {
    form.classList.add("is-submitting");
    if (loading) loading.setAttribute("aria-hidden", "false");
    if (button) {
      button.disabled = true;
      button.textContent = "Analyzing...";
    }

    let idx = 0;
    let currentPct = 0;
    let waitIdx = 0;
    const startTime = Date.now();

    const applyMilestone = () => {
      const item = milestones[Math.min(idx, milestones.length - 1)];
      currentPct = Math.max(currentPct, item.pct);
      if (fill) fill.style.width = `${item.pct}%`;
      if (percent) percent.textContent = `${item.pct}%`;
      if (message) message.textContent = item.text;
      if (live) live.textContent = item.live;
      steps.forEach((step, i) => {
        step.classList.toggle("done", i < item.step);
        step.classList.toggle("active", i === item.step);
      });
      idx += 1;
      if (idx < milestones.length) {
        window.setTimeout(applyMilestone, idx < 3 ? 750 : 1300);
      } else {
        window.setTimeout(keepAlive, 2200);
      }
    };

    const keepAlive = () => {
      const elapsed = Math.max(1, Math.round((Date.now() - startTime) / 1000));
      currentPct = Math.min(99, currentPct + (currentPct < 97 ? 1 : 0));
      if (fill) fill.style.width = `${currentPct}%`;
      if (percent) percent.textContent = `${currentPct}%`;
      if (message) message.textContent = waitingMessages[waitIdx % waitingMessages.length];
      if (live) live.textContent = `Working for ${elapsed}s`;
      steps.forEach((step, i) => {
        step.classList.toggle("done", i < 3);
        step.classList.toggle("active", i === 3);
      });
      waitIdx += 1;
      window.setTimeout(keepAlive, 3600);
    };

    applyMilestone();
  });
}

function setupReportExport() {
  const printButton = document.querySelector("[data-print-report]");
  const copyButton = document.querySelector("[data-copy-report]");
  const copyTemplate = document.querySelector("#reportCopyTemplate");
  const copyStatus = document.querySelector("[data-copy-status]");

  if (printButton) {
    printButton.addEventListener("click", () => window.print());
  }

  if (!copyButton || !copyTemplate) return;

  const setStatus = (message) => {
    if (!copyStatus) return;
    copyStatus.textContent = message;
    window.setTimeout(() => {
      copyStatus.textContent = "";
    }, 2800);
  };

  copyButton.addEventListener("click", async () => {
    const reportText = copyTemplate.innerText.replace(/\n{3,}/g, "\n\n").trim();
    try {
      await navigator.clipboard.writeText(reportText);
      setStatus("Report text copied.");
    } catch (error) {
      const fallback = document.createElement("textarea");
      fallback.value = reportText;
      fallback.setAttribute("readonly", "");
      fallback.style.position = "fixed";
      fallback.style.left = "-9999px";
      document.body.appendChild(fallback);
      fallback.select();
      document.execCommand("copy");
      fallback.remove();
      setStatus("Report text copied.");
    }
  });
}

setupThemeToggle();
setupTabs();
setupBboxToggle();
setupCueSelection();
setupAnalysisProgress();
setupImageZoom();
setupUploadPicker();
setupProblemDescriptionLoading();
setupReportExport();
