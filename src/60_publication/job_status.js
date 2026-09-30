/* Public workflow metadata only: never embed a token in this file. */
(async () => {
  const notice = document.getElementById("live-status");
  if (!notice) return;
  const history = document.getElementById("run-history");
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 10000);
  try {
    const query = new URLSearchParams({per_page: "100", branch: notice.dataset.branch});
    const response = await fetch(
      `https://api.github.com/repos/${notice.dataset.repository}/actions/workflows/daily-racing.yml/runs?${query}`,
      {headers: {Accept: "application/vnd.github+json"}, credentials: "omit",
       cache: "no-store", signal: controller.signal}
    );
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    if (!Array.isArray(payload.workflow_runs)) throw new Error("Invalid response");
    const runs = payload.workflow_runs.filter(run =>
      ["schedule", "workflow_dispatch"].includes(run.event) && run.head_branch === notice.dataset.branch
    ).slice(0, 7);
    const saved = new Map(Array.from(history.querySelectorAll("tr[data-run-id]"), row =>
      [`${row.dataset.runId}:${row.dataset.attempt}`, Array.from(row.cells, cell => cell.textContent)]));
    const labels = {success: "成功", failure: "失敗", timed_out: "失敗", cancelled: "中止",
      skipped: "スキップ", neutral: "中立", action_required: "要対応", stale: "期限切れ"};
    const body = document.createElement("tbody");
    for (const run of runs) {
      if (!Number.isSafeInteger(run.id) || !Number.isSafeInteger(run.run_attempt) ||
          !run.status || !Number.isFinite(Date.parse(run.run_started_at))) throw new Error("Invalid run");
      const previous = saved.get(`${run.id}:${run.run_attempt}`);
      const result = run.status === "completed" ? (labels[run.conclusion] || "結果不明") :
        (run.status === "in_progress" ? "実行中" : "待機中");
      const started = new Intl.DateTimeFormat("ja-JP", {timeZone: "Asia/Tokyo",
        year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit",
        minute: "2-digit", second: "2-digit", hourCycle: "h23"}).format(new Date(run.run_started_at));
      const row = document.createElement("tr");
      for (const value of [started, previous?.[1] || "—", result, previous?.[3] || "—", previous?.[4] || "—"]) {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      }
      body.append(row);
    }
    let table = history.querySelector("table");
    if (!table) {
      table = document.createElement("table");
      const head = table.createTHead().insertRow();
      for (const title of ["実行日時（日本時間）", "予想・結果照合", "全体結果（配信含む）", "予想レース数", "レース結果照合数"]) {
        const cell = document.createElement("th"); cell.textContent = title; head.append(cell);
      }
      history.replaceChildren(table);
    }
    const oldBody = table.querySelector("tbody");
    if (oldBody) oldBody.replaceWith(body); else table.append(body);
    notice.textContent = runs.length ? "全体結果を最新情報に更新しました。" : "実行記録はありません。";
  } catch (error) {
    notice.textContent = "最新結果を確認できません。保存時点の情報を表示しています。GitHub Actionsでも確認できます。";
  } finally {
    clearTimeout(timer);
  }
})();
