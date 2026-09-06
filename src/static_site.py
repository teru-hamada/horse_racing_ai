from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
import re

import pandas as pd


_STYLE = """
:root{color-scheme:dark;--bg:#07130f;--panel:#10251d;--line:#25483a;--text:#f4f7f5;--muted:#a9bdb5;--accent:#38d996;--gold:#ffd166}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top,#163b2c 0,var(--bg) 42%);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans JP",sans-serif}
main{width:100%;margin:0;padding:42px 28px 80px}header{margin-bottom:30px}.eyebrow{color:var(--accent);font-size:.78rem;font-weight:800;letter-spacing:.16em;text-transform:uppercase}h1{font-size:clamp(2rem,6vw,4.5rem);line-height:1;margin:.25em 0}.lead,.meta{color:var(--muted)}.race{width:100%;background:color-mix(in srgb,var(--panel) 92%,transparent);border:1px solid var(--line);border-radius:18px;box-shadow:0 18px 50px #0004;margin:16px 0;overflow:hidden}.race>summary{display:flex;align-items:center;justify-content:space-between;gap:18px;padding:22px 24px;cursor:pointer;font-size:1.25rem;font-weight:800;list-style:none}.race.graded>summary{color:var(--gold)}.race>summary::-webkit-details-marker{display:none}.race>summary::after{content:"＋";color:var(--accent);font-size:1.5rem}.race[open]>summary::after{content:"−"}.race[open]>summary{border-bottom:1px solid var(--line)}.race-content{width:100%;padding:20px 24px 24px;overflow-x:auto}.race-head{display:flex;align-items:center;justify-content:space-between;gap:14px}.course{color:var(--accent);font-weight:800}table{width:100%;min-width:760px;border-collapse:collapse;margin-top:18px;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:13px 10px;border-bottom:1px solid var(--line)}th{color:var(--muted);font-size:.78rem}.rank{font-size:1.1rem;font-weight:900}.prob{color:var(--gold);font-weight:800}.top3 td{background:#38d9960c}.back{display:inline-block;color:var(--accent);margin-bottom:18px;text-decoration:none}.empty{padding:50px 0;color:var(--muted)}footer{margin-top:36px;color:var(--muted);font-size:.8rem}@media(max-width:720px){main{padding:28px 12px 60px}.race>summary{padding:18px 16px;font-size:1.05rem}.race-content{padding:14px 12px 18px}th,td{white-space:nowrap}.race-head{align-items:start;flex-direction:column}}
"""

_STYLE += """
.comparison-summary{display:flex;flex-wrap:wrap;gap:10px 22px;padding:18px 22px;margin:0 0 24px;background:#10251dcc;border:1px solid var(--line);border-radius:14px}.comparison-summary strong{width:100%;color:var(--gold)}.comparison-summary span{color:var(--muted)}.comparison-summary .failures{width:100%;color:#ffb4a8}.hit{color:var(--accent);font-weight:800}.miss{color:#ffb4a8}
"""


_GRADED_RACE_PATTERN = re.compile(
    r"(?:Jpn\s*[123ⅠⅡⅢ]|G\s*[123ⅠⅡⅢ])",
    re.IGNORECASE,
)


def _text(value: object, fallback: str = "-") -> str:
    return fallback if pd.isna(value) else escape(str(value))


def _number(value: object, digits: int = 2) -> str:
    return "-" if pd.isna(value) else f"{float(value):.{digits}f}"


def _page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="競馬予想AIが生成した3着以内確率"><title>{escape(title)}</title><style>{_STYLE}</style></head>
<body><main>{body}<footer>研究・学習用の予想です。的中や利益を保証するものではありません。</footer></main></body></html>
"""


def prediction_date_status(
    race_date: object,
    predictions_dir: Path,
    site_dir: Path,
) -> tuple[bool, bool]:
    """Return whether a date has a saved prediction and a completed comparison."""

    target_date = pd.to_datetime(race_date, errors="coerce")
    if pd.isna(target_date):
        return False, False
    date_text = target_date.strftime("%Y-%m-%d")
    prediction_page = site_dir / "predictions" / f"{date_text}.html"
    prediction_files = predictions_dir.glob(
        f"*/predictions_{date_text}.parquet"
    )
    prediction_created = prediction_page.is_file() or any(prediction_files)
    comparison_completed = False
    if prediction_page.is_file():
        page_html = prediction_page.read_text(encoding="utf-8", errors="replace")
        comparison_completed = (
            'data-comparison-status="completed"' in page_html
            or 'class="comparison-summary"' in page_html
        )
    return prediction_created, comparison_completed


def latest_prediction_file(
    race_date: object,
    predictions_dir: Path,
) -> Path | None:
    """Return the most recently updated saved date-prediction file."""

    target_date = pd.to_datetime(race_date, errors="coerce")
    if pd.isna(target_date):
        return None
    date_text = target_date.strftime("%Y-%m-%d")
    candidates = list(
        predictions_dir.glob(f"*/predictions_{date_text}.parquet")
    )
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _write_index(site_dir: Path) -> Path:
    pages = sorted((site_dir / "predictions").glob("*.html"), reverse=True)
    if pages:
        links = "".join(
            f'<article class="race"><a class="back" href="predictions/{escape(page.name)}">'
            f'{escape(page.stem)} の予想を見る →</a></article>'
            for page in pages
        )
    else:
        links = '<p class="empty">公開済みの予想はまだありません。</p>'
    body = f"""<header><div class="eyebrow">Horse Racing AI</div><h1>開催日別<br>レース予想</h1>
<p class="lead">各レースの3着以内確率を、開催日ごとに掲載しています。</p></header>{links}"""
    index_path = site_dir / "index.html"
    index_path.write_text(_page("競馬予想AI", body), encoding="utf-8")
    return index_path


def build_prediction_site(
    predictions: pd.DataFrame,
    race_date: object,
    model_id: str,
    site_dir: Path,
    comparison: pd.DataFrame | None = None,
    comparison_summary: dict[str, object] | None = None,
) -> Path:
    """Write a dependency-free GitHub Pages site for one prediction date."""

    required = {
        "race_id", "race_number", "race_name", "course_name", "horse_number",
        "horse_name", "jockey_name", "prediction_rank", "top3_probability",
        "odds", "expected_value_index",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"静的ページ用の予想列が不足しています: {sorted(missing)}")
    target_date = pd.to_datetime(race_date, errors="coerce")
    if pd.isna(target_date):
        raise ValueError("静的ページの開催日が正しくありません。")
    date_text = target_date.strftime("%Y-%m-%d")
    prediction_dir = site_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / ".nojekyll").touch()

    page_predictions = predictions.copy()
    has_comparison = comparison is not None or comparison_summary is not None
    if comparison is not None and not comparison.empty:
        merge_keys = ["race_id", "horse_number"]
        if "horse_id" in page_predictions.columns and "horse_id" in comparison.columns:
            merge_keys = ["race_id", "horse_id"]
        comparison_columns = merge_keys + [
            column
            for column in ("finish_position", "predicted_top3", "actual_top3", "top3_hit")
            if column in comparison.columns
        ]
        page_predictions = page_predictions.merge(
            comparison[comparison_columns].drop_duplicates(merge_keys),
            on=merge_keys,
            how="left",
        )
    for column in ("finish_position", "predicted_top3", "actual_top3", "top3_hit"):
        if column not in page_predictions.columns:
            page_predictions[column] = pd.NA

    race_sections = []
    for _, race in page_predictions.groupby("race_id", sort=False):
        first = race.iloc[0]
        rows = []
        for runner in race.sort_values("prediction_rank").itertuples():
            rank = int(runner.prediction_rank)
            comparison_cells = ""
            if has_comparison:
                finish_position = getattr(runner, "finish_position", pd.NA)
                top3_hit = getattr(runner, "top3_hit", pd.NA)
                judgment = "-"
                judgment_class = ""
                if pd.notna(finish_position) and rank <= 3 and pd.notna(top3_hit):
                    judgment = "的中" if bool(top3_hit) else "不的中"
                    judgment_class = "hit" if bool(top3_hit) else "miss"
                comparison_cells = (
                    f'<td>{_text(finish_position)}</td>'
                    f'<td class="{judgment_class}">{judgment}</td>'
                )
            rows.append(
                f'<tr class="{"top3" if rank <= 3 else ""}"><td class="rank">{rank}</td>'
                f'<td>{_text(runner.horse_number)}</td><td>{_text(runner.horse_name)}</td>'
                f'<td class="prob">{float(runner.top3_probability):.1%}</td>'
                f'{comparison_cells}'
                f'<td>{_number(runner.odds, 1)}</td><td>{_number(runner.expected_value_index)}</td>'
                f'<td>{_text(runner.jockey_name)}</td></tr>'
            )
        race_number = int(first["race_number"]) if pd.notna(first["race_number"]) else "-"
        race_name = str(first["race_name"]) if pd.notna(first["race_name"]) else ""
        race_class = "race graded" if _GRADED_RACE_PATTERN.search(race_name) else "race"
        race_sections.append(
            f'<details class="{race_class}"><summary>{_text(first["course_name"])} {race_number}R {_text(first["race_name"])}</summary>'
            f'<div class="race-content">'
            '<table><thead><tr><th>予測</th><th>馬番</th><th>馬名</th><th>3着以内確率</th>'
            + ('<th>実着順</th><th>判定</th>' if has_comparison else '')
            + '<th>オッズ</th><th>期待値指数</th><th>騎手</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div></details>'
        )
    comparison_block = ""
    if has_comparison:
        summary = comparison_summary or {}
        failures = summary.get("failures", []) or []
        failure_text = ""
        if failures:
            failure_items = " / ".join(
                f'{item.get("race_id", "-")} ({item.get("message", "-")})'
                for item in failures
            )
            failure_text = f'<span class="failures">未比較: {_text(failure_items)}</span>'
        comparison_block = (
            '<section class="comparison-summary" data-comparison-status="completed">'
            '<strong>確定結果との比較</strong>'
            f'<span>比較完了: {_text(summary.get("compared_races", 0))} / '
            f'{_text(summary.get("requested_races", 0))} レース</span>'
            f'<span>予測上位3頭の的中: {_text(summary.get("top3_hit_count", 0))} 頭</span>'
            f'<span>上位3頭完全的中: {_text(summary.get("perfect_top3_races", 0))} レース</span>'
            f'{failure_text}</section>'
        )
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    body = f"""<a class="back" href="../index.html">← 開催日一覧</a><header><div class="eyebrow">Daily Predictions</div>
<h1>{date_text}</h1><p class="lead">開催日単位の全レース予想</p></header>
{comparison_block}{''.join(race_sections)}<p class="meta">生成日時: {escape(generated_at)}</p>"""
    page_path = prediction_dir / f"{date_text}.html"
    page_path.write_text(_page(f"{date_text} 競馬予想", body), encoding="utf-8")
    _write_index(site_dir)
    return page_path


def initialize_prediction_site(site_dir: Path) -> Path:
    """Create the empty landing page committed before the first prediction."""
    (site_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (site_dir / ".nojekyll").touch()
    return _write_index(site_dir)
