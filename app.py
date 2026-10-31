import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="CarbonBudget AI", page_icon="🌱", layout="wide")

@st.cache_data
def generate_hard_dataset(seed=42):
    rng = np.random.default_rng(seed)
    n = 500
    time_step = np.arange(n)
    temperature = rng.normal(50, 4, n)
    vibration = rng.normal(1.0, 0.3, n)
    fault = np.zeros(n, dtype=int)
    for start, end in [(100, 125), (270, 300), (410, 435)]:
        fault[start:end] = 1
        temperature[start:end] += rng.normal(5, 3, end-start)
        vibration[start:end] += rng.normal(0.5, 0.3, end-start)
    vibration = np.clip(vibration, 0, None)
    return pd.DataFrame({"time": time_step, "temperature": temperature,
                         "vibration": vibration, "fault": fault})

def deep_review(data, index):
    start = max(0, index - 4)
    window = data.iloc[start:index + 1]
    suspicious = (window["temperature"].mean() >= 55 or
                  window["vibration"].mean() >= 1.5 or
                  window["temperature"].max() >= 58 or
                  window["vibration"].max() >= 1.8)
    return "FAULT" if suspicious else "NORMAL"

def run_agent_v4(data, model, budget=250, protected_reserve=30):
    remaining = budget
    records = []
    total = len(data)
    for i, (_, row) in enumerate(data.iterrows()):
        temp, vib = row["temperature"], row["vibration"]
        actual = int(row["fault"])
        steps_left = total - i
        if remaining < 3:
            prediction, action, cost = "REVIEW", "Energy exhausted", 0
        elif temp < 55 and vib < 1.5:
            remaining -= 1
            prediction, action, cost = "NORMAL", "Quick screen", 1
        else:
            remaining -= 3
            cost = 3
            features = pd.DataFrame([[temp, vib]], columns=["temperature", "vibration"])
            probability = model.predict_proba(features)[0][1]
            uncertainty = 1 - abs(2 * probability - 1)
            prediction, action = ("FAULT" if probability >= 0.5 else "NORMAL"), "AI check"
            wants_deep = probability >= 0.65 or uncertainty >= 0.55
            future_reserve = min(protected_reserve, max(0, steps_left - 1) * 0.4)
            can_deep = remaining - future_reserve >= 6
            if wants_deep and can_deep:
                remaining -= 6
                cost += 6
                if deep_review(data, i) == "FAULT":
                    prediction = "FAULT"
                action = "AI + deep review"
        records.append({"actual": actual, "prediction": prediction,
                        "action": action, "energy_used": cost,
                        "reading": i + 1, "time": int(row["time"]),
                        "temperature": float(temp), "vibration": float(vib)})
    results = pd.DataFrame(records)
    reviewed = results[results["prediction"] != "REVIEW"]
    actual_faults = results["actual"] == 1
    detected = (actual_faults & (results["prediction"] == "FAULT"))
    predicted_faults = results["prediction"] == "FAULT"
    precision = 100 * detected.sum() / predicted_faults.sum() if predicted_faults.sum() else 0
    recall = 100 * detected.sum() / actual_faults.sum() if actual_faults.sum() else 0
    accuracy = 100 * (reviewed["actual"] == reviewed["prediction"].map({"NORMAL": 0, "FAULT": 1})).mean() if len(reviewed) else 0
    summary = {"Energy used": budget - remaining, "Energy remaining": remaining,
               "Coverage (%)": round(100 * len(reviewed) / total, 1),
               "Accuracy (%)": round(accuracy, 1),
               "Fault precision (%)": round(precision, 1),
               "Fault recall (%)": round(recall, 1)}
    return summary, results

st.title("🌱 CarbonBudget AI")
st.subheader("Energy-aware AI for simulated industrial motor monitoring")
st.info("This demo uses synthetic sensor readings and simulated energy units. It is a prototype, not a live industrial monitoring system. REVIEW means the reading was not classified because the energy budget was exhausted.")

with st.sidebar:
    st.header("Agent settings")
    budget = st.slider("Simulated energy budget", 100, 600, 250, 10)
    reserve = st.slider("Protected reserve", 0, 80, 30, 5)
    run = st.button("▶ Run CarbonBudget AI", type="primary")

# Rebuild the same synthetic dataset and train/test split used in the notebook.
dataset = generate_hard_dataset(seed=42)
train_data = dataset.iloc[:350]
test_data = dataset.iloc[350:].reset_index(drop=True)
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
model.fit(train_data[["temperature", "vibration"]], train_data["fault"])

if "run_results" not in st.session_state or run:
    st.session_state.run_summary, st.session_state.run_results = run_agent_v4(
        test_data, model, budget=budget, protected_reserve=reserve)
summary = st.session_state.run_summary
results = st.session_state.run_results

st.caption(f"Showing {len(results)} simulated test readings | Training readings: {len(train_data)}")
cols = st.columns(4)
metrics = [("Energy used", summary["Energy used"]), ("Energy remaining", summary["Energy remaining"]),
          ("Coverage", f'{summary["Coverage (%)"]}%'), ("Classified accuracy", f'{summary["Accuracy (%)"]}%')]
for col, (label, value) in zip(cols, metrics):
    col.metric(label, value)

cols = st.columns(2)
cols[0].metric("Fault precision", f'{summary["Fault precision (%)"]}%')
cols[1].metric("Fault recall", f'{summary["Fault recall (%)"]}%')

st.subheader("Sensor readings")
fig, ax = plt.subplots(figsize=(11, 3.5))
ax.plot(results["time"], results["temperature"], label="Temperature (°C)")
ax.set_xlabel("Time step")
ax.set_ylabel("Temperature (°C)")
ax.legend(loc="upper left")
ax.grid(alpha=0.25)
st.pyplot(fig)
plt.close(fig)

counts = results["prediction"].value_counts().reindex(["NORMAL", "FAULT", "REVIEW"], fill_value=0)
left, right = st.columns(2)
with left:
    st.subheader("Prediction breakdown")
    st.bar_chart(counts)
with right:
    st.subheader("Agent actions")
    st.bar_chart(results["action"].value_counts())

st.subheader("Human review queue")
review = results[results["prediction"] == "REVIEW"].copy()
if len(review):
    review["priority"] = np.where((review["temperature"] >= 55) | (review["vibration"] >= 1.5), "HIGH", "NORMAL")
    review["review_reason"] = "Energy budget exhausted - AI processing not completed"
    priority_filter = st.selectbox("Filter by priority", ["All", "HIGH", "NORMAL"])
    shown = review if priority_filter == "All" else review[review["priority"] == priority_filter]
    st.dataframe(shown[["reading", "time", "temperature", "vibration", "priority", "review_reason"]], use_container_width=True, hide_index=True)
else:
    st.success("No readings are waiting for human review in this run.")

with st.expander("View all simulated readings"):
    st.dataframe(results, use_container_width=True, hide_index=True)

st.caption("Prototype note: priority labels are threshold-based attention flags, not confirmed fault diagnoses. Energy units are simulated; no real electricity or CO₂ savings have been measured.")
