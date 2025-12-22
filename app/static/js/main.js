async function loadInsights() {
  const panel = document.getElementById("insights-panel");
  if (!panel) return;
  panel.innerHTML = "Loading AI insights...";
  try {
    const res = await fetch("/api/insights");
    const data = await res.json();
    if (!data.length) {
      panel.innerHTML = "No insights yet. Run the pipeline to populate outputs.";
      return;
    }
    panel.innerHTML = "";
    const list = document.createElement("ul");
    data.forEach((item) => {
      const li = document.createElement("li");
      li.innerHTML = `<strong>Site ${item.site_id}</strong> · ${item.country} · Risk ${item.risk_probability.toFixed(2)}<br>${item.ai_insight}`;
      list.appendChild(li);
    });
    panel.appendChild(list);
  } catch (err) {
    panel.innerHTML = "Failed to load insights.";
    console.error(err);
  }
}

document.addEventListener("DOMContentLoaded", loadInsights);

