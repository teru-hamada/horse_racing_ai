/* Public workflow metadata only: never embed a token in this file. */
// 保存時点の履歴を、公開GitHub APIの情報で更新する。認証トークンは利用・埋め込みしない。
// 通信できない場合も既存の履歴表を残し、最新情報を取得できなかった旨だけを通知する。
(async () => {
  // 状況表示を置いたページだけで実行する。要素のないページではAPI要求も送らない。
  const notice = document.getElementById("live-status");
  if (!notice) return;
  // サーバー側で作った履歴表の格納先。件数などAPIにない情報の引継ぎ元にもなる。
  const history = document.getElementById("run-history");
  // API応答が止まっても待ち続けないよう、10秒で取得を中断する。
  const controller = new AbortController();
  // このタイマーは成功・失敗のどちらの場合もfinallyで解除する。
  const timer = setTimeout(() => controller.abort(), 10000);
  try {
    // APIでは最大100件取得し、その後に必要なイベント・ブランチへ絞って7件表示する。
    const query = new URLSearchParams({per_page: "100", branch: notice.dataset.branch});
    const response = await fetch(
      `https://api.github.com/repos/${notice.dataset.repository}/actions/workflows/daily-racing.yml/runs?${query}`,
      {headers: {Accept: "application/vnd.github+json"}, credentials: "omit",
       cache: "no-store", signal: controller.signal}
    );
    // HTTP失敗や想定外のJSONを、実行履歴ゼロ件として表示しない。
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    if (!Array.isArray(payload.workflow_runs)) throw new Error("Invalid response");
    // 別ブランチや別イベントを混ぜず、定期・手動の日次実行の最新7件だけを表示する。
    const runs = payload.workflow_runs.filter(run =>
      ["schedule", "workflow_dispatch"].includes(run.event) && run.head_branch === notice.dataset.branch
    ).slice(0, 7);
    // 公開APIには予想・照合件数がないため、同じ実行IDと試行回数の保存済みセルから引き継ぐ。
    // 別の再実行へ件数を誤って付けないよう、両方をキーにする。
    const saved = new Map(Array.from(history.querySelectorAll("tr[data-run-id]"), row =>
      [`${row.dataset.runId}:${row.dataset.attempt}`, Array.from(row.cells, cell => cell.textContent)]));
    // completedのときだけconclusionを結果へ変換する。未登録の結果値には別途フォールバックする。
    const labels = {success: "成功", failure: "失敗", timed_out: "失敗", cancelled: "中止",
      skipped: "スキップ", neutral: "中立", action_required: "要対応", stale: "期限切れ"};
    // 既存表を直接更新せず、全行の構築に成功するまで差し替え用tbodyにためる。
    const body = document.createElement("tbody");
    for (const run of runs) {
      // DOMへ出す前にID・試行回数・状態・開始日時の型を確認する。
      if (!Number.isSafeInteger(run.id) || !Number.isSafeInteger(run.run_attempt) ||
          !run.status || !Number.isFinite(Date.parse(run.run_started_at))) throw new Error("Invalid run");
      // 同じ実行でも再試行ごとに結果が異なるため、試行回数も含めて保存行を検索する。
      const previous = saved.get(`${run.id}:${run.run_attempt}`);
      // 未完了は実行中／待機中とし、まだ存在しない最終結果を成功と推測しない。
      const result = run.status === "completed" ? (labels[run.conclusion] || "結果不明") :
        (run.status === "in_progress" ? "実行中" : "待機中");
      // ブラウザの地域設定にかかわらず日本時間、24時間表記で公開時刻とそろえる。
      const started = new Intl.DateTimeFormat("ja-JP", {timeZone: "Asia/Tokyo",
        year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit",
        minute: "2-digit", second: "2-digit", hourCycle: "h23"}).format(new Date(run.run_started_at));
      const row = document.createElement("tr");
      // 予想ジョブの結果と件数は保存行から、配信を含む全体結果は最新APIから組み合わせる。
      // 保存行がなければ件数をゼロと断定せず「—」で示す。
      for (const value of [started, previous?.[1] || "—", result, previous?.[3] || "—", previous?.[4] || "—"]) {
        const cell = document.createElement("td");
        // API由来の文字列をHTMLとして解釈させず、テキストとして表示する。
        cell.textContent = value;
        row.append(cell);
      }
      body.append(row);
    }
    let table = history.querySelector("table");
    // 保存時点で履歴が空だったページには表がないため、初回の更新で見出しごと作る。
    if (!table) {
      table = document.createElement("table");
      const head = table.createTHead().insertRow();
      for (const title of ["実行日時（日本時間）", "予想・結果照合", "全体結果（配信含む）", "予想レース数", "レース結果照合数"]) {
        const cell = document.createElement("th"); cell.textContent = title; head.append(cell);
      }
      history.replaceChildren(table);
    }
    // 全行の検査とDOM構築が済んでから差し替え、途中の例外で既存履歴を消さない。
    const oldBody = table.querySelector("tbody");
    if (oldBody) oldBody.replaceWith(body); else table.append(body);
    notice.textContent = runs.length ? "全体結果を最新情報に更新しました。" : "実行記録はありません。";
  } catch (error) {
    // 通信制限、タイムアウト、データ形式不正は表示上同じ取得失敗として知らせる。
    // 既存の保存版履歴は残し、例外の内部情報を公開ページへ出さない。
    notice.textContent = "最新結果を確認できません。保存時点の情報を表示しています。";
  } finally {
    // 成功時にもタイマーを解除し、終了後に不要なabortが発生しないようにする。
    clearTimeout(timer);
  }
})();
