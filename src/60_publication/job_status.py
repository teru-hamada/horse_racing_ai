"""Build a public status snapshot from workflow metadata and race counts (stdlib only)."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from html import escape
from io import BytesIO
import json
import logging
import os
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zipfile import ZipFile, BadZipFile

JST = timezone(timedelta(hours=9))
MAX_RUNS = 7
COUNT_KEYS = ("predicted_races", "compared_races")


def collect_counts(output: Path) -> dict:
    """Count completed predictions/comparisons, not discovered races or horses."""
    def read(relative):
        path = output / relative
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    daily = read("daily_report.json")
    prediction = read("prediction/date_report.json")
    comparison = read("comparison/report.json")
    tasks = (daily or {}).get("tasks", {})
    predicted = sum(row.get("prediction") == "ok" for row in prediction["races"]) if prediction else None
    compared = comparison["summary"]["compared_races"] if comparison else None
    if tasks.get("prediction", {}).get("status") == "no_races":
        predicted = 0
    if tasks.get("comparison", {}).get("status") == "no_races" or (
        daily and daily.get("status") != "running" and "comparison" not in tasks
    ):
        compared = 0
    return dict(zip(COUNT_KEYS, (predicted, compared)))


def count_label(value) -> str:
    return str(value) if type(value) is int and value >= 0 else "—"


def result_label(run: dict) -> str:
    if run.get("status") != "completed":
        return "実行中" if run.get("status") == "in_progress" else "待機中"
    return {"success": "成功", "cancelled": "中止", "skipped": "スキップ"}.get(
        run.get("conclusion"), "失敗"
    )


def render_status(runs: list[dict], now: datetime) -> str:
    rows = []
    for run in runs[:MAX_RUNS]:
        started = datetime.fromisoformat(run["run_started_at"].replace("Z", "+00:00"))
        rows.append("<tr>" + "".join(f"<td>{escape(value)}</td>" for value in (
            started.astimezone(JST).strftime("%Y-%m-%d %H:%M:%S"),
            result_label({"status": "completed", "conclusion": run["prediction_result"]})
            if "prediction_result" in run else "—",
            "配信中（最終結果未確定）" if "prediction_result" in run else result_label(run),
            count_label(run.get("predicted_races")), count_label(run.get("compared_races")),
        )) + "</tr>")
    content = (
        "<table><thead><tr><th>実行日時（日本時間）</th><th>予想・結果照合</th><th>全体結果（配信含む）</th>"
        "<th>予想レース数</th><th>レース結果照合数</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
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
<p>当日予想・前日結果照合の実行記録（最新7件まで）。毎日日本時間3:00に実行予定です。
成功には非開催日の正常終了も含みます。実行日時は予想対象日とは異なる場合があります。</p>
<p>レース数は各処理が完了した件数です。非開催・照合未指定は0、記録を取得できない場合は「—」で表示します。</p>
<p>今回の予想・結果照合欄は予想ジョブの結果です。配信中のため、全体の最終結果はGitHubのActions画面で確認してください。
過去分の全体結果は次回配信時に更新します。予想ジョブ単独の結果を取得していない過去分は「—」で表示します。</p>
<p>表示はページ公開時点の記録です。記録がない日は、この一覧では実行を確認できません。
実行の遅延や履歴の保存期間により、すべての実行を表示できない場合があります。</p>
<div class="table">{content}</div></body></html>'''


def api_request(url: str, token: str) -> Request:
    request = Request(
        url,
        headers={"Accept": "application/vnd.github+json",
                 "X-GitHub-Api-Version": "2022-11-28"},
    )
    # Artifact downloads redirect to signed storage URLs. Do not forward the token.
    request.add_unredirected_header("Authorization", f"Bearer {token}")
    return request


def fetch_counts(repository: str, run: dict, token: str) -> dict:
    name = f"daily-racing-counts-{run['id']}-{run.get('run_attempt', 1)}"
    query = urlencode({"name": name, "per_page": 1})
    base = f"https://api.github.com/repos/{repository}/actions/artifacts"
    with urlopen(api_request(f"{base}?{query}", token), timeout=30) as response:
        artifacts = json.load(response)["artifacts"]
    if not artifacts or artifacts[0].get("expired"):
        return {}
    artifact = artifacts[0]
    if artifact["name"] != name:
        return {}
    with urlopen(api_request(f"{base}/{int(artifact['id'])}/zip", token), timeout=30) as response:
        archive = response.read(65537)
    if len(archive) > 65536:
        raise ValueError("Race count artifact is too large")
    with ZipFile(BytesIO(archive)) as zipped:
        if zipped.getinfo("race_counts.json").file_size > 65536:
            raise ValueError("Race count report is too large")
        counts = json.loads(zipped.read("race_counts.json"))
    return {key: counts.get(key) for key in COUNT_KEYS
            if type(counts.get(key)) is int and counts[key] >= 0}


def fetch_runs(repository: str, branch: str, token: str) -> list[dict]:
    query = urlencode({"branch": branch, "per_page": 100})
    request = api_request(
        f"https://api.github.com/repos/{repository}/actions/workflows/daily-racing.yml/runs?{query}", token)
    with urlopen(request, timeout=30) as response:
        payload = json.load(response)
    # Missing or invalid API data must not be presented as an empty run history.
    runs = payload["workflow_runs"]
    if not isinstance(runs, list):
        raise ValueError("Invalid workflow run response")
    runs = [run for run in runs if run.get("event") in {"schedule", "workflow_dispatch"}][:MAX_RUNS]
    for run in runs:
        try:
            run.update(fetch_counts(repository, run, token))
        except (OSError, ValueError, KeyError, TypeError, BadZipFile):
            logging.warning("レース数の記録を取得できません: run_id=%s", run.get("id"))
    return runs


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


def annotate_current_run(runs: list[dict], run_id: str, attempt: str, result: str) -> None:
    """Attach the caller's job result only to this run and this retry attempt."""
    if result not in {"success", "failure", "cancelled", "skipped"}:
        raise ValueError("Invalid prediction job result")
    for run in runs:
        if str(run.get("id")) == run_id and str(run.get("run_attempt", 1)) == attempt:
            run["prediction_result"] = result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=Path, default=Path("docs"))
    parser.add_argument("--write-counts", type=Path)
    parser.add_argument("--run-output", type=Path, default=Path("data/daily_racing"))
    args = parser.parse_args()
    if args.write_counts:
        args.write_counts.parent.mkdir(parents=True, exist_ok=True)
        args.write_counts.write_text(json.dumps(collect_counts(args.run_output)), encoding="utf-8")
        return
    runs = fetch_runs(os.environ["GITHUB_REPOSITORY"], os.environ["DEFAULT_BRANCH"], os.environ["GH_TOKEN"])
    if os.environ.get("PREDICTION_RESULT"):
        annotate_current_run(runs, os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_RUN_ATTEMPT"],
                             os.environ["PREDICTION_RESULT"])
    build_status(args.site, runs, datetime.now(JST))


if __name__ == "__main__":
    main()
