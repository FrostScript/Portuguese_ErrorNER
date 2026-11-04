// 实验看板前端：每 2 秒拉一次 /api/state，日志每 1.5 秒 tail 一次
const POLL_STATE = 2000;
const POLL_LOG = 1500;

let currentLog = null;
let logTimer = null;

const $ = (id) => document.getElementById(id);

function fmt(v, digits = 4) {
  if (v === null || v === undefined || v === "") return "—";
  const n = Number(v);
  if (Number.isNaN(n)) return String(v);
  return n.toFixed(digits);
}

function fmtTime(iso) {
  if (!iso) return "—";
  return String(iso).replace("T", " ");
}

function duration(start, end) {
  if (!start) return "—";
  const s = new Date(start).getTime();
  const e = end ? new Date(end).getTime() : Date.now();
  if (Number.isNaN(s)) return "—";
  const sec = Math.max(0, Math.round((e - s) / 1000));
  const m = Math.floor(sec / 60);
  return m > 0 ? `${m}m ${sec % 60}s` : `${sec}s`;
}

async function getJSON(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

function renderRunning(state) {
  const box = $("running-box");
  const badge = $("status");
  if (!state.running || state.running.length === 0) {
    badge.textContent = "空闲";
    badge.className = "badge idle";
    box.className = "running-box";
    const last = (state.runs || []).find((r) => r.status === "done" || r.status === "failed");
    box.innerHTML = last
      ? `最近一次：<b>${last.name}</b> · ${last.status === "failed" ? '<span class="err">失败</span>' : "完成"} · 耗时 ${duration(last.start, last.end)} · exit ${last.exit ?? "—"}`
      : "当前没有运行中的任务";
    return;
  }
  badge.textContent = `运行中 (${state.running.length})`;
  badge.className = "badge run";
  box.className = "running-box active";
  box.innerHTML = state.running
    .map(
      (r) => `<div><b>${r.name}</b> <span class="err"></span>
        <div class="row">
          <span><span class="k">开始</span> ${fmtTime(r.start)}</span>
          <span><span class="k">已运行</span> ${duration(r.start)}</span>
          <span><span class="k">日志</span> ${r.file || "—"}</span>
        </div>
        <div class="row"><span class="k">命令</span> <code>${(r.cmd || "").slice(0, 160)}</code></div>
      </div>`
    )
    .join("");
}

function renderExperiments(rows) {
  const body = $("exp-table").querySelector("tbody");
  if (!rows.length) {
    body.innerHTML = `<tr><td colspan="12" class="err">还没有实验记录（results/experiments.csv 为空）</td></tr>`;
    return;
  }
  body.innerHTML = rows
    .map(
      (r) => `<tr>
        <td>${r.arm || "—"}</td>
        <td>${r.split || "—"}</td>
        <td>${r.seed || "—"}</td>
        <td>${fmt(r.threshold, 2)}</td>
        <td>${fmt(r.precision)}</td>
        <td>${fmt(r.recall)}</td>
        <td class="num-ok"><b>${fmt(r.micro_f1)}</b></td>
        <td>${fmt(r.macro_f1)}</td>
        <td>${fmt(r.per_f1)}</td>
        <td>${fmt(r.org_f1)}</td>
        <td>${fmt(r.loc_f1)}</td>
        <td>${fmt(r.ci_low)}–${fmt(r.ci_high)}</td>
      </tr>`
    )
    .join("");
}

function renderDocs(docs) {
  const ul = $("doc-list");
  if (!docs.length) {
    ul.innerHTML = `<li>docs/runs/ 里还没有运行记录</li>`;
    return;
  }
  ul.innerHTML = docs
    .map(
      (d) => `<li><b>${d.name}</b><span class="right">${d.mtime} · ${(d.size / 1024).toFixed(1)} KB</span></li>`
    )
    .join("");
}

function renderCheckpoints(items) {
  const body = $("ckpt-table").querySelector("tbody");
  if (!items.length) {
    body.innerHTML = `<tr><td colspan="9">D:\\pt_ner_work\\checkpoints 下还没有训练产物</td></tr>`;
    return;
  }
  body.innerHTML = items
    .map((c) => {
      const last = (c.log_history || []).filter((h) => h.loss !== undefined).pop();
      const loss = last ? Number(last.loss).toFixed(4) : "—";
      return `<tr>
        <td>${c.path}</td>
        <td>${c.arm || "—"}</td>
        <td>${c.seed ?? "—"}</td>
        <td>${c.max_steps ?? "—"}</td>
        <td>${c.batch ?? "—"}</td>
        <td>${c.lr ?? "—"}</td>
        <td>${c.train_seconds ?? "—"}</td>
        <td>${c.size_mb ?? "—"}</td>
        <td>${loss}</td>
      </tr>`;
    })
    .join("");
}

function renderLogSelect(logs, runs) {
  const sel = $("log-select");
  const prev = sel.value;
  const names = new Set((logs || []).map((l) => l.name));
  (runs || []).forEach((r) => r.file && names.add(r.file));
  const sorted = [...names].sort().reverse();
  sel.innerHTML = sorted.map((n) => `<option value="${n}">${n}</option>`).join("") || `<option value="">无日志</option>`;
  if (prev && names.has(prev)) sel.value = prev;
  else if (sorted.length && !currentLog) {
    sel.value = sorted[0];
    currentLog = sorted[0];
    loadLog();
  }
}

async function loadLog() {
  if (!currentLog) return;
  try {
    const data = await getJSON(`/api/log?name=${encodeURIComponent(currentLog)}`);
    const pre = $("log-view");
    const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 40;
    pre.textContent = data.text || "（空文件）";
    if ($("follow").checked && (atBottom || true)) pre.scrollTop = pre.scrollHeight;
  } catch (e) {
    $("log-view").textContent = `[读取失败] ${e.message}`;
  }
}

async function refresh() {
  try {
    const st = await getJSON("/api/state");
    $("clock").textContent = st.now;
    $("foot").textContent = `工作区 ${st.workspace}`;
    renderRunning(st);
    renderExperiments(st.experiments || []);
    renderDocs(st.run_docs || []);
    renderCheckpoints(st.checkpoints || []);
    renderLogSelect(st.logs, st.runs);
  } catch (e) {
    $("status").textContent = "未连接";
    $("status").className = "badge idle";
    $("foot").textContent = `无法连接服务：${e.message}`;
  }
}

$("log-select").addEventListener("change", (e) => {
  currentLog = e.target.value || null;
  $("log-view").textContent = "加载中…";
  loadLog();
});

$("refresh").addEventListener("click", () => {
  refresh();
  loadLog();
});

refresh();
logTimer = setInterval(loadLog, POLL_LOG);
setInterval(refresh, POLL_STATE);
