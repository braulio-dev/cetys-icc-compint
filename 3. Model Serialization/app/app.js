const LABEL_MAP = { 0: "Low", 1: "Moderate", 2: "High", 3: "Severe" };

const RISK_COLORS = {
  Low: "#22c55e",
  Moderate: "#f59e0b",
  High: "#f97316",
  Severe: "#ef4444",
};

const NUMERICAL_COLS = [
  "age", "daily_gaming_hours", "sleep_hours", "weight_change_kg",
  "exercise_hours_weekly", "social_isolation_score",
  "face_to_face_social_hours_weekly", "monthly_game_spending_usd", "years_gaming",
];

const CATEGORICAL_COLS = [
  "gender", "game_genre", "gaming_platform", "sleep_quality",
  "sleep_disruption_frequency", "academic_work_performance",
  "mood_state", "mood_swing_frequency",
];

const BOOLEAN_FEATURES = [
  "withdrawal_symptoms", "loss_of_other_interests",
  "continued_despite_problems", "eye_strain", "back_neck_pain",
];

// validation rules for each numeric field
const VALIDATION_RULES = {
  age: { min: 5,  max: 60,   label: "Age" },
  daily_gaming_hours: { min: 0.1, max: 24.0, label: "Daily gaming hours" },
  sleep_hours: { min: 3,   max: 9,    label: "Sleep hours" },
  weight_change_kg: { min: 0,   max: 9,    label: "Weight change" },
  exercise_hours_weekly: { min: 0,   max: 12,   label: "Exercise hours" },
  social_isolation_score: { min: 1,   max: 10,   label: "Social isolation score" },
  face_to_face_social_hours_weekly: { min: 0,   max: 17,   label: "Social hours" },
  monthly_game_spending_usd: { min: 0,   max: 500,  label: "Monthly spending" },
  years_gaming: { min: 1,   max: 20,   label: "Years gaming" },
};

let session = null;

async function loadModel() {
  const statusEl = document.getElementById("model-status");
  const MODEL_PATH = "../best_model.onnx";

  const probe = await fetch(MODEL_PATH);
  if (!probe.ok) {
    statusEl.textContent =
      `Model file not found (HTTP ${probe.status}). ` +
      `Make sure best_model.onnx exists in "3. Model Serialization/" and Live Server is open on that folder.`;
    statusEl.className = "status error";
    return;
  }

  try {
    session = await ort.InferenceSession.create(MODEL_PATH);
    statusEl.textContent = "Model loaded.";
    statusEl.className = "status ready";
    document.getElementById("submit-btn").disabled = false;
  } catch (err) {
    const detail = err?.message ?? err?.toString?.() ?? String(err);
    console.log("Model initialization error:", err.stack);
    statusEl.textContent = `Model file found but failed to initialise: ${detail}`;
    statusEl.className = "status error";
  }
}

function validateField(input) {
  const errorEl = input.closest(".field")?.querySelector(".error");
  if (!errorEl) return true;

  const name = input.name;

  if (input.type === "number") {
    const value = parseFloat(input.value);
    const rule = VALIDATION_RULES[name];

    if (input.value === "" || isNaN(value)) {
      showError(input, errorEl, "This field is required.");
      return false;
    }
    if (rule && (value < rule.min || value > rule.max)) {
      showError(input, errorEl, `Must be between ${rule.min} and ${rule.max}.`);
      return false;
    }
  } else if (input.tagName === "SELECT") {
    if (!input.value) {
      showError(input, errorEl, "Please select an option.");
      return false;
    }
  }

  clearError(input, errorEl);
  return true;
}

function showError(input, errorEl, message) {
  errorEl.textContent = message;
  input.classList.add("invalid");
}

function clearError(input, errorEl) {
  errorEl.textContent = "";
  input.classList.remove("invalid");
}

function buildFeeds(form) {
  const feeds = {};
  const data = new FormData(form);

  for (const col of NUMERICAL_COLS) {
    const val = parseFloat(data.get(col));
    feeds[col] = new ort.Tensor("float32", Float32Array.from([val]), [1, 1]);
  }

  for (const col of CATEGORICAL_COLS) {
    feeds[col] = new ort.Tensor("string", [data.get(col)], [1, 1]);
  }

  for (const col of BOOLEAN_FEATURES) {
    const checked = form.querySelector(`#${col}`).checked;
    feeds[col] = new ort.Tensor("int64", BigInt64Array.from([checked ? 1n : 0n]), [1, 1]);
  }

  return feeds;
}

async function handleSubmit(e) {
  e.preventDefault();
  const form = e.target;

  // validate all required fields before running inference
  const requiredInputs = form.querySelectorAll("input[required], select[required]");
  let valid = true;
  requiredInputs.forEach(input => { if (!validateField(input)) valid = false; });
  if (!valid) return;

  const btn = document.getElementById("submit-btn");
  btn.textContent = "Running...";
  btn.disabled = true;

  try {
    const feeds = buildFeeds(form);
    const results = await session.run(feeds);
    const label = Number(results[session.outputNames[0]].data[0]);
    const riskLevel = LABEL_MAP[label];

    const badgeEl = document.getElementById("risk-badge");
    badgeEl.textContent = riskLevel;
    badgeEl.style.backgroundColor = RISK_COLORS[riskLevel];

    const resultEl = document.getElementById("result");
    resultEl.classList.remove("hidden");
    resultEl.scrollIntoView({ behavior: "smooth" });
  } catch (err) {
    alert("Inference failed: " + err.message);
  } finally {
    btn.textContent = "Run Prediction";
    btn.disabled = false;
  }
}

// real-time validation on blur / after first error
document.querySelectorAll("input[required], select[required]").forEach(input => {
  input.addEventListener("blur", () => validateField(input));
  input.addEventListener("input", () => {
    if (input.classList.contains("invalid")) validateField(input);
  });
});

document.getElementById("prediction-form").addEventListener("submit", handleSubmit);

loadModel();
