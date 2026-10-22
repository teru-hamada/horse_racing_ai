# GitHub Actionsの実行履歴と完了レース件数から、公開用のジョブ状況ページを作る。
# 再実行回数まで一致するartifactを読み、取得できない件数はゼロとせず不明として扱う。
# 認証情報はAPIへのアクセス時だけ使い、公開HTMLやJavaScriptへ埋め込まない。
#
"""Build a public status snapshot from workflow metadata and race counts (stdlib only)."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from html import escape
from io import BytesIO
import json
import logging
import os
import re
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zipfile import ZipFile, BadZipFile

# 日本標準時（UTC+9）。日付判定と公開ページの時刻表示を統一する。
JST = timezone(timedelta(hours=9))
# 公開する最新の実行履歴数の上限。件数artifactの取得対象もこの範囲へ絞る。
MAX_RUNS = 7
# 公開してよい予想・照合の完了レース件数のキー。
COUNT_KEYS = ("predicted_races", "compared_races")


# 発見したレース数ではなく、レポートに記録された完了済みの予想・照合レース数を数える。
def collect_counts(output: Path) -> dict:
    """Count completed predictions/comparisons, not discovered races or horses."""
    # 実行出力配下のレポートJSONを読む。未生成のレポートはNoneとして件数不明と区別する。
    def read(relative):
        path = output / relative
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    daily = read("daily_report.json")
    prediction = read("prediction/date_report.json")
    comparison = read("comparison/report.json")
    tasks = (daily or {}).get("tasks", {})
    # レポートから確認した予想完了レース数。不明はNoneで保持する。
    predicted = sum(row.get("prediction") == "ok" for row in prediction["races"]) if prediction else None
    # 確認できた結果照合レース数。不明と開催なしのゼロを区別する。
    compared = comparison["summary"]["compared_races"] if comparison else None
    if tasks.get("prediction", {}).get("status") == "no_races":
        predicted = 0
    if tasks.get("comparison", {}).get("status") == "no_races" or (
        daily and daily.get("status") != "running" and "comparison" not in tasks
    ):
        compared = 0
    return dict(zip(COUNT_KEYS, (predicted, compared)))


# 確認済みの非負整数だけを件数として表示し、取得不能などはダッシュで示す。
def count_label(value) -> str:
    return str(value) if type(value) is int and value >= 0 else "—"


# GitHubの実行状態と結論を日本語へ変換する。未完了は成功・失敗と断定しない。
def result_label(run: dict) -> str:
    if run.get("status") != "completed":
        return "実行中" if run.get("status") == "in_progress" else "待機中"
    return {"success": "成功", "cancelled": "中止", "skipped": "スキップ"}.get(
        run.get("conclusion"), "失敗"
    )


# API情報をエスケープして日本時間の履歴表へ変換し、保存版ページのHTMLを返す。
def render_status(runs: list[dict], now: datetime, repository: str = "", branch: str = "") -> str:
    rows = []
    for run in runs[:MAX_RUNS]:
        started = datetime.fromisoformat(run["run_started_at"].replace("Z", "+00:00"))
        # ブラウザ更新で同じ実行と再試行を識別するdata属性。埋め込む値はエスケープする。
        attrs = f' data-run-id="{escape(str(run.get("id", "")))}" data-attempt="{escape(str(run.get("run_attempt", 1)))}"' if repository else ""
        rows.append(f"<tr{attrs}>" + "".join(f"<td>{escape(value)}</td>" for value in (
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
    # 公開APIによる最新履歴表示を有効にするHTML断片。リポジトリ指定がない場合は空にする。
    live = ""
    # 公開リポジトリが指定された場合だけ、ブラウザからの最新状況取得を有効にする。
    if repository:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise ValueError("Invalid public repository")
        live = (f'<p id="live-status" role="status" data-repository="{escape(repository)}" '
                f'data-branch="{escape(branch)}">最新結果を確認しています…</p>'
                '<noscript><p>JavaScriptが無効なため、保存時点の情報を表示しています。</p></noscript>'
                '<script src="job-status.js" defer></script>')
    return f'''<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ジョブ実行状況</title><style>
body{{font-family:system-ui,sans-serif;background:#07130f;color:#f4f7f5;margin:24px auto;padding:16px;max-width:1000px}}
a{{color:#38d996}}table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;padding:12px;border-bottom:1px solid #25483a}}
.table{{overflow-x:auto}}p{{line-height:1.7}}
</style></head><body><a href="index.html">予想一覧へ戻る</a><h1>ジョブ実行状況</h1>
<p>当日予想・前日結果照合の実行記録（最新7件まで）。毎日日本時間3:00に実行予定です。
成功には非開催日の正常終了も含みます。実行日時は予想対象日とは異なる場合があります。</p>
<p>レース数は各処理が完了した件数です。非開催・照合未指定は0、記録を取得できない場合は「—」で表示します。</p>
<p>全体結果はページを開いたときに最新情報を取得します。実行中の場合は、しばらくしてページを再読み込みしてください。
予想・結果照合とレース数は保存時点の情報です。再実行などで対応する記録がない場合は「—」で表示します。</p>
<p>最新結果を取得できない場合は、保存時点の記録を表示します。
実行の遅延や履歴の保存期間により、すべての実行を表示できない場合があります。</p>
{live}<div class="table" id="run-history">{content}</div></body></html>'''


# API要求のURLと認証ヘッダーを組み立てる。トークンは公開用の戻り値へ含めない。
def api_request(url: str, token: str) -> Request:
    # token: GitHub API認証用の値。公開HTMLやログへ書き出さない。
    request = Request(
        url,
        headers={"Accept": "application/vnd.github+json",
                 "X-GitHub-Api-Version": "2022-11-28"},
    )
    # Artifact downloads redirect to signed storage URLs. Do not forward the token.
    request.add_unredirected_header("Authorization", f"Bearer {token}")
    return request


# run_idとrun_attemptに一致する件数artifactを読む。取得不可・期限切れなどは不明として扱う。
def fetch_counts(repository: str, run: dict, token: str) -> dict:
    # token: GitHub API認証用の値。公開HTMLやログへ書き出さない。
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
    # 件数しか含めない小さなartifactなので、想定外に大きいZIPを受け入れない。
    if len(archive) > 65536:
        raise ValueError("Race count artifact is too large")
    with ZipFile(BytesIO(archive)) as zipped:
        # 圧縮後のサイズだけでなく展開後のJSONサイズも制限する。
        if zipped.getinfo("race_counts.json").file_size > 65536:
            raise ValueError("Race count report is too large")
        counts = json.loads(zipped.read("race_counts.json"))
    return {key: counts.get(key) for key in COUNT_KEYS
            if type(counts.get(key)) is int and counts[key] >= 0}


# 対象イベントとブランチの実行から最新7件を選び、それぞれの件数情報を補う。
def fetch_runs(repository: str, branch: str, token: str) -> list[dict]:
    # token: GitHub API認証用の値。公開HTMLやログへ書き出さない。
    query = urlencode({"branch": branch, "per_page": 100})
    request = api_request(
        f"https://api.github.com/repos/{repository}/actions/workflows/daily-racing.yml/runs?{query}", token)
    with urlopen(request, timeout=30) as response:
        payload = json.load(response)
    # Missing or invalid API data must not be presented as an empty run history.
    runs = payload["workflow_runs"]
    # APIの予期しない応答形式を、実行履歴なしの正常結果として受け入れない。
    if not isinstance(runs, list):
        raise ValueError("Invalid workflow run response")
    runs = [run for run in runs if run.get("event") in {"schedule", "workflow_dispatch"}][:MAX_RUNS]
    for run in runs:
        try:
            run.update(fetch_counts(repository, run, token))
        except (OSError, ValueError, KeyError, TypeError, BadZipFile):
            logging.warning("レース数の記録を取得できません: run_id=%s", run.get("id"))
    return runs


# ジョブ状況ページと公開JSを書き込み、既存ページへ状況ページの導線を補う。
def build_status(site: Path, runs: list[dict], now: datetime) -> None:
    repository = os.environ.get("GITHUB_REPOSITORY", "")
    page = render_status(runs, now, repository, os.environ.get("DEFAULT_BRANCH", ""))
    site.mkdir(parents=True, exist_ok=True)
    (site / "job-status.html").write_text(page, encoding="utf-8")
    # 公開リポジトリが指定された場合だけ、ブラウザからの最新状況取得を有効にする。
    if repository:
        (site / "job-status.js").write_text(Path(__file__).with_name("job_status.js").read_text(encoding="utf-8"), encoding="utf-8")
    # Include the navigation even for prediction pages generated before this feature.
    for path in [site / "index.html", *(site / "predictions").glob("*.html")]:
        # 対象ファイルが未生成のケースを扱い、別のファイルを推測して読み込まない。
        if not path.exists():
            continue
        html = path.read_text(encoding="utf-8")
        if 'href="job-status.html"' not in html and 'href="../job-status.html"' not in html:
            href = "job-status.html" if path.parent == site else "../job-status.html"
            link = f'<p><a style="color:#38d996" href="{href}">ジョブ実行状況 →</a></p>'
            html = html.replace("<main>", "<main>" + link, 1)
        if path == site / "index.html":
            updated = now.astimezone(JST).strftime("%Y-%m-%d %H:%M:%S")
            stamp = f'<p id="job-status-updated" class="meta">ジョブ実行状況の更新日時：{updated}（日本時間）</p>'
            if 'id="job-status-updated"' in html:
                html = re.sub(r'<p\b[^>]*id="job-status-updated"[^>]*>.*?</p>', stamp, html, flags=re.DOTALL)
            elif "</header>" in html:
                html = html.replace("</header>", stamp + "</header>", 1)
            else:
                html = html.replace("<main>", "<main>" + stamp, 1)
        path.write_text(html, encoding="utf-8")


# 今回と同じ実行ID・試行回数の行だけに予想ジョブの結果を付け、過去の試行を上書きしない。
def annotate_current_run(runs: list[dict], run_id: str, attempt: str, result: str) -> None:
    """Attach the caller's job result only to this run and this retry attempt."""
    # 呼出元から付与する予想ジョブの結果は、定義済みの状態値だけを受け入れる。
    if result not in {"success", "failure", "cancelled", "skipped"}:
        raise ValueError("Invalid prediction job result")
    for run in runs:
        if str(run.get("id")) == run_id and str(run.get("run_attempt", 1)) == attempt:
            run["prediction_result"] = result


# コマンドライン引数を解析して処理を起動する入口。省略時の保存先やモードはparserで定義する。
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


# モジュールとしてimportしただけでは実行せず、CLIとして起動された場合だけ入口を呼ぶ。
if __name__ == "__main__":
    main()
