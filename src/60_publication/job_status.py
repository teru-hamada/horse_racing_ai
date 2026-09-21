"""Build a public status snapshot using only workflow metadata (stdlib only)."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from html import escape
import json
import os
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

JST = timezone(timedelta(hours=9))


def result_label(run: dict) -> str:
    if run.get("status") != "completed":
        return "実行中" if run.get("status") == "in_progress" else "待機中"
    return {"success": "成功", "cancelled": "中止", "skipped": "スキップ"}.get(
        run.get("conclusion"), "失敗"
    )


def render_status(runs: list[dict], now: datetime) -> str:
    rows = []
    for run in runs:
        started = datetime.fromisoformat(run["run_started_at"].replace("Z", "+00:00"))
        method = "定期実行" if run.get("event") == "schedule" else "手動実行"
        rows.append("<tr>" + "".join(f"<td>{escape(value)}</td>" for value in (
            started.astimezone(JST).strftime("%Y-%m-%d %H:%M:%S"),
            method, str(run.get("run_attempt", 1)), result_label(run),
        )) + "</tr>")
    content = (
        "<table><thead><tr><th>実行日時（日本時間）</th><th>実行方法</th>"
        "<th>試行回数</th><th>結果</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        if rows else "<p>実行記録はありません。</p>"
    )
    updated = now.astimezone(JST).strftime("%Y-%m-%d %H:%M:%S")
    return f'''<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ジョブ実行状況</title><style>
body{{font-family:system-ui,sans-serif;background:#07130f;color:#f4f7f5;margin:24px auto;padding:16px;max-width:1000px}}
a{{color:#38d996}}table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;padding:12px;border-bottom:1px solid #25483a}}
.table{{overflow-x:auto}}p{{line-height:1.7}}
</style></head><body><a href="index.html">予想一覧へ戻る</a><h1>ジョブ実行状況</h1>
<p>更新日時：{updated}（日本時間）</p>
<p>当日予想・前日結果照合の実行記録（最新100件まで）。定期実行は毎日日本時間3:00予定です。
成功には非開催日の正常終了も含みます。実行日時は予想対象日とは異なる場合があります。</p>
<p>表示はページ公開時点の記録です。記録がない日は、この一覧では実行を確認できません。
実行の遅延や履歴の保存期間により、すべての実行を表示できない場合があります。</p>
<div class="table">{content}</div></body></html>'''


def fetch_runs(repository: str, branch: str, token: str) -> list[dict]:
    query = urlencode({"branch": branch, "per_page": 100})
    request = Request(
        f"https://api.github.com/repos/{repository}/actions/workflows/publish-predictions.yml/runs?{query}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                 "X-GitHub-Api-Version": "2022-11-28"},
    )
    with urlopen(request, timeout=30) as response:
        payload = json.load(response)
    # Missing or invalid API data must not be presented as an empty run history.
    runs = payload["workflow_runs"]
    if not isinstance(runs, list):
        raise ValueError("Invalid workflow run response")
    return [run for run in runs if run.get("event") in {"schedule", "workflow_dispatch"}]


def build_status(site: Path, runs: list[dict], now: datetime) -> None:
    page = render_status(runs, now)
    site.mkdir(parents=True, exist_ok=True)
    (site / "job-status.html").write_text(page, encoding="utf-8")
    # Include the navigation even for prediction pages generated before this feature.
    for path in [site / "index.html", *(site / "predictions").glob("*.html")]:
        if not path.exists():
            continue
        html = path.read_text(encoding="utf-8")
        if 'href="job-status.html"' in html or 'href="../job-status.html"' in html:
            continue
        href = "job-status.html" if path.parent == site else "../job-status.html"
        link = f'<p><a style="color:#38d996" href="{href}">ジョブ実行状況 →</a></p>'
        path.write_text(html.replace("<main>", "<main>" + link, 1), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=Path, default=Path("docs"))
    args = parser.parse_args()
    runs = fetch_runs(os.environ["GITHUB_REPOSITORY"], os.environ["DEFAULT_BRANCH"], os.environ["GH_TOKEN"])
    build_status(args.site, runs, datetime.now(JST))


if __name__ == "__main__":
    main()
