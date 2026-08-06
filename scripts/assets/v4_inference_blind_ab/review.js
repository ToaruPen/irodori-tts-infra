"use strict";

const wrapper = window.IRODORI_BLIND_AB_MANIFEST;
const manifest = wrapper.manifest;
const storageKey = `irodori-blind-ab:${manifest.packet_id}`;
const choices = [
  ["a", "Aが良い"],
  ["b", "Bが良い"],
  ["same", "同等"],
  ["unsure", "判断できない"],
];
const reasonLabels = {
  reading: "読み",
  voice: "声",
  noise: "ノイズ",
  prosody: "自然さ・韻律",
  emotion: "感情",
};
const validPairIds = new Set(manifest.pairs.map((pair) => pair.pair_id));
const validChoices = new Set(choices.map(([value]) => value));
const validReasons = new Set(
  manifest.reasons.filter((reason) => Object.hasOwn(reasonLabels, reason)),
);

function emptyState() {
  return { index: 0, answers: {} };
}

function normalizeReasons(value) {
  if (!Array.isArray(value)) {
    return [];
  }
  const selected = new Set(value.filter((reason) => validReasons.has(reason)));
  return manifest.reasons.filter((reason) => selected.has(reason));
}

function normalizeAnswers(value) {
  const answers = {};
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return answers;
  }
  for (const [pairId, restored] of Object.entries(value)) {
    if (
      !validPairIds.has(pairId) ||
      !restored ||
      typeof restored !== "object" ||
      Array.isArray(restored)
    ) {
      continue;
    }
    const choice = validChoices.has(restored.choice) ? restored.choice : null;
    const reasons = normalizeReasons(restored.reasons);
    if (choice !== null || reasons.length > 0) {
      answers[pairId] = { choice, reasons };
    }
  }
  return answers;
}

function loadState() {
  try {
    const restored = JSON.parse(localStorage.getItem(storageKey) || "null");
    if (!restored || typeof restored !== "object" || Array.isArray(restored)) {
      return emptyState();
    }
    const lastIndex = Math.max(0, manifest.pairs.length - 1);
    const index = Number.isInteger(restored.index)
      ? Math.max(0, Math.min(lastIndex, restored.index))
      : 0;
    return { index, answers: normalizeAnswers(restored.answers) };
  } catch (error) {
    try {
      localStorage.removeItem(storageKey);
    } catch (removeError) {
      // Storage access can be denied for local files; in-memory review still works.
    }
    return emptyState();
  }
}

const state = loadState();

function save() {
  try {
    localStorage.setItem(storageKey, JSON.stringify(state));
  } catch (error) {
    // Keep the current in-memory answers when local file storage is unavailable.
  }
}

function currentAnswer(pairId) {
  return state.answers[pairId] || { choice: null, reasons: [] };
}

function answeredCount() {
  return manifest.pairs.filter((pair) => {
    const answer = state.answers[pair.pair_id];
    return answer && validChoices.has(answer.choice);
  }).length;
}

function updateStatus() {
  const answered = answeredCount();
  document.getElementById("progress").textContent =
    `${state.index + 1} / ${manifest.pairs.length}`;
  document.getElementById("remaining").textContent =
    `未回答 ${manifest.pairs.length - answered} 件`;
  document.getElementById("previous").disabled = state.index === 0;
  document.getElementById("next").disabled = state.index === manifest.pairs.length - 1;
  document.getElementById("download").disabled = answered !== manifest.pairs.length;
}

function rebuildOptions(rootId, options, inputType, selected, onChange) {
  const root = document.getElementById(rootId).querySelector(".option-grid");
  while (root.firstChild) {
    root.removeChild(root.firstChild);
  }
  for (const [value, labelText] of options) {
    const label = document.createElement("label");
    const input = document.createElement("input");
    label.className = "option-label";
    input.type = inputType;
    input.name = inputType === "radio" ? "choice" : `reason-${value}`;
    input.value = value;
    input.checked = selected.has(value);
    input.addEventListener("change", () => onChange(value, input.checked));
    label.append(input, document.createTextNode(labelText));
    root.append(label);
  }
}

function renderSelection(pairId) {
  const answer = currentAnswer(pairId);
  rebuildOptions(
    "choices",
    choices,
    "radio",
    new Set(answer.choice === null ? [] : [answer.choice]),
    (choice) => {
      state.answers[pairId] = {
        choice,
        reasons: [...currentAnswer(pairId).reasons],
      };
      save();
      updateStatus();
    },
  );
  rebuildOptions(
    "reasons",
    manifest.reasons
      .filter((reason) => validReasons.has(reason))
      .map((reason) => [reason, reasonLabels[reason]]),
    "checkbox",
    new Set(answer.reasons),
    (reason, checked) => {
      const latest = currentAnswer(pairId);
      const selected = new Set(latest.reasons);
      if (checked) {
        selected.add(reason);
      } else {
        selected.delete(reason);
      }
      state.answers[pairId] = {
        choice: latest.choice,
        reasons: manifest.reasons.filter((value) => selected.has(value)),
      };
      save();
      updateStatus();
    },
  );
}

function render() {
  const pair = manifest.pairs[state.index];
  document.getElementById("sample-text").textContent = pair.text;
  document.getElementById("audio-a").src = pair.a_audio;
  document.getElementById("audio-b").src = pair.b_audio;
  renderSelection(pair.pair_id);
  updateStatus();
}

function resetAudio() {
  for (const audio of [
    document.getElementById("audio-a"),
    document.getElementById("audio-b"),
  ]) {
    audio.pause();
    audio.currentTime = 0;
  }
}

function move(delta) {
  const nextIndex = Math.max(
    0,
    Math.min(manifest.pairs.length - 1, state.index + delta),
  );
  if (nextIndex === state.index) {
    return;
  }
  resetAudio();
  state.index = nextIndex;
  save();
  render();
}

function downloadResults() {
  if (answeredCount() !== manifest.pairs.length) {
    return;
  }
  const answers = manifest.pairs.map((pair) => {
    const answer = currentAnswer(pair.pair_id);
    return {
      pair_id: pair.pair_id,
      choice: answer.choice,
      reasons: [...answer.reasons],
    };
  });
  const result = {
    schema_version: "irodori-v4-inference-blind-ab-results/v1",
    packet_id: manifest.packet_id,
    manifest_sha256: wrapper.manifest_sha256,
    answers,
  };
  const blob = new Blob([JSON.stringify(result, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "irodori-blind-ab-results.json";
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

document.getElementById("previous").addEventListener("click", () => move(-1));
document.getElementById("next").addEventListener("click", () => move(1));
document.getElementById("download").addEventListener("click", downloadResults);
render();
