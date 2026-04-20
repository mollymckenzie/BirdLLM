const API_URL = "/api/query";

const WEEK_LABELS = [];
const MONTH_STARTS = {
  1: "Jan", 5: "Feb", 9: "Mar", 13: "Apr", 17: "May", 21: "Jun",
  25: "Jul", 29: "Aug", 33: "Sep", 37: "Oct", 41: "Nov", 45: "Dec",
};
for (let w = 1; w <= 52; w++) {
  WEEK_LABELS.push(MONTH_STARTS[w] || "");
}

let chartInstances = [];

function fillExample(el) {
  const input = document.getElementById("userInput");
  input.value = el.textContent;
  input.focus();
}

function scrollToBottom() {
  const container = document.getElementById("chatContainer");
  container.scrollTop = container.scrollHeight;
}

function appendMessage(role, content) {
  const container = document.getElementById("chatContainer");
  const wrap = document.createElement("div");
  wrap.className = `message ${role === "user" ? "user-message" : "bot-message"}`;

  const icon = document.createElement("div");
  icon.className = "message-icon";
  const iconImg = document.createElement("img");
  iconImg.src = role === "user" ? "/assets/binoculars.png" : "/assets/yellowwarbler.png";
  iconImg.alt = role === "user" ? "binoculars" : "warbler";
  icon.appendChild(iconImg);

  const body = document.createElement("div");
  body.className = "message-body";
  if (typeof content === "string") {
    body.textContent = content;
  } else {
    body.appendChild(content);
  }

  wrap.appendChild(icon);
  wrap.appendChild(body);
  container.appendChild(wrap);
  scrollToBottom();
  return body;
}

function appendLoading() {
  const container = document.getElementById("chatContainer");
  const wrap = document.createElement("div");
  wrap.className = "message bot-message loading-message";
  wrap.id = "loadingMsg";

  const icon = document.createElement("div");
  icon.className = "message-icon";
  const loadingImg = document.createElement("img");
  loadingImg.src = "/assets/yellowwarbler.png";
  loadingImg.alt = "warbler";
  icon.appendChild(loadingImg);

  const body = document.createElement("div");
  body.className = "message-body";
  const spinner = document.createElement("div");
  spinner.className = "spinner";
  const txt = document.createElement("span");
  txt.textContent = "Searching the dataset...";
  body.appendChild(spinner);
  body.appendChild(txt);

  wrap.appendChild(icon);
  wrap.appendChild(body);
  container.appendChild(wrap);
  scrollToBottom();
}

function removeLoading() {
  const el = document.getElementById("loadingMsg");
  if (el) el.remove();
}

function buildBotResponseNode(data) {
  const frag = document.createDocumentFragment();

  // Main text response
  const text = document.createElement("p");
  text.textContent = data.response;
  frag.appendChild(text);

  // Location note warning
  if (data.location_note) {
    const note = document.createElement("div");
    note.className = "location-note";
    note.textContent = "⚠ " + data.location_note;
    frag.appendChild(note);
  }

  // Peak week badges
  if (data.peak_weeks && data.peak_weeks.length) {
    const badges = document.createElement("div");
    badges.className = "peak-badges";
    data.peak_weeks.forEach((p) => {
      const badge = document.createElement("span");
      badge.className = "peak-badge";
      const month = weekToMonth(p.week);
      badge.textContent = `Peak: ${month} (W${p.week}) — ${(p.normalized_frequency * 100).toFixed(1)}%`;
      badges.appendChild(badge);
    });
    frag.appendChild(badges);
  }

  // Meta line
  if (data.total_records) {
    const meta = document.createElement("div");
    meta.className = "meta-line";
    const sp = data.parsed_query?.species || "";
    const loc = data.parsed_query?.location || "all locations";
    meta.textContent = `${data.total_records.toLocaleString()} records for "${sp}" in ${loc}`;
    frag.appendChild(meta);
  }

  // Frequency chart
  if (data.chart_data) {
    const chartWrap = buildChart(data.chart_data);
    frag.appendChild(chartWrap);
  }

  return frag;
}

function weekToMonth(week) {
  const months = [
    "", "Jan","Jan","Jan","Jan",
    "Feb","Feb","Feb","Feb",
    "Mar","Mar","Mar","Mar",
    "Apr","Apr","Apr","Apr",
    "May","May","May","May",
    "Jun","Jun","Jun","Jun",
    "Jul","Jul","Jul","Jul",
    "Aug","Aug","Aug","Aug",
    "Sep","Sep","Sep","Sep",
    "Oct","Oct","Oct","Oct",
    "Nov","Nov","Nov","Nov",
    "Dec","Dec","Dec","Dec","Dec","Dec","Dec","Dec",
  ];
  return months[week] || `W${week}`;
}

function buildChart(chartData) {
  const wrapper = document.createElement("div");
  wrapper.className = "chart-wrapper";

  const title = document.createElement("div");
  title.className = "chart-title";
  title.textContent = `Weekly sighting probability — ${chartData.species} (${chartData.location})`;
  wrapper.appendChild(title);

  const legend = document.createElement("div");
  legend.className = "chart-legend";
  legend.innerHTML = `
    <span><span class="legend-dot normal"></span>Normal</span>
    <span><span class="legend-dot peak"></span>Peak week</span>
  `;
  wrapper.appendChild(legend);

  const canvas = document.createElement("canvas");
  canvas.className = "freq-chart";
  wrapper.appendChild(canvas);

  const peakSet = new Set(chartData.peak_weeks);
  const bgColors = chartData.frequencies.map((_, i) => {
    const w = i + 1;
    return peakSet.has(w) ? "rgba(244, 162, 97, 0.85)" : "rgba(82, 183, 136, 0.65)";
  });

  // Defer chart creation until after element is in the DOM
  requestAnimationFrame(() => {
    const chart = new Chart(canvas, {
      type: "bar",
      data: {
        labels: WEEK_LABELS,
        datasets: [
          {
            label: "Sighting probability (%)",
            data: chartData.frequencies,
            backgroundColor: bgColors,
            borderRadius: 3,
            borderSkipped: false,
          },
        ],
      },
      options: {
        responsive: true,
        animation: { duration: 600 },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              title: (items) => {
                const w = items[0].dataIndex + 1;
                return `Week ${w} (${weekToMonth(w)})`;
              },
              label: (item) => ` ${item.raw.toFixed(2)}% of checklists`,
            },
          },
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: {
              maxRotation: 0,
              font: { size: 10 },
              callback: (val, idx) => WEEK_LABELS[idx] || "",
              maxTicksLimit: 12,
            },
          },
          y: {
            beginAtZero: true,
            ticks: {
              font: { size: 10 },
              callback: (v) => v + "%",
            },
            grid: { color: "rgba(0,0,0,0.05)" },
          },
        },
      },
    });
    chartInstances.push(chart);
  });

  return wrapper;
}

async function handleSubmit(e) {
  e.preventDefault();
  const input = document.getElementById("userInput");
  const btn = document.getElementById("sendBtn");
  const message = input.value.trim();
  if (!message) return;

  input.value = "";
  btn.disabled = true;

  appendMessage("user", message);
  appendLoading();

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });

    const data = await res.json();
    removeLoading();

    if (data.error) {
      appendMessage("bot", "Sorry, something went wrong: " + data.error);
    } else {
      const node = buildBotResponseNode(data);
      appendMessage("bot", node);
    }
  } catch (err) {
    removeLoading();
    appendMessage("bot", "Network error — make sure the server is running.");
    console.error(err);
  } finally {
    btn.disabled = false;
    input.focus();
  }
}
