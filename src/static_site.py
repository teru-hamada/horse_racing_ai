from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path

import pandas as pd


_STYLE = """
:root{color-scheme:dark;--bg:#07130f;--panel:#10251d;--line:#25483a;--text:#f4f7f5;--muted:#a9bdb5;--accent:#38d996;--gold:#ffd166}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top,#163b2c 0,var(--bg) 42%);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans JP",sans-serif}
main{width:min(1180px,calc(100% - 28px));margin:0 auto;padding:42px 0 80px}header{margin-bottom:30px}.eyebrow{color:var(--accent);font-size:.78rem;font-weight:800;letter-spacing:.16em;text-transform:uppercase}h1{font-size:clamp(2rem,6vw,4.5rem);line-height:1;margin:.25em 0}.lead,.meta{color:var(--muted)}.summary{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:28px 0}.metric,.race{background:color-mix(in srgb,var(--panel) 92%,transparent);border:1px solid var(--line);border-radius:18px;box-shadow:0 18px 50px #0004}.metric{padding:18px}.metric strong{display:block;font-size:1.7rem;color:var(--gold)}.race{padding:22px;margin:18px 0}.race-head{display:flex;align-items:end;justify-content:space-between;gap:14px}.race h2{margin:0}.course{color:var(--accent);font-weight:800}table{width:100%;border-collapse:collapse;margin-top:18px;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:11px 8px;border-bottom:1px solid var(--line)}th{color:var(--muted);font-size:.78rem}.rank{font-size:1.1rem;font-weight:900}.prob{color:var(--gold);font-weight:800}.top3 td{background:#38d9960c}.back{display:inline-block;color:var(--accent);margin-bottom:18px;text-decoration:none}.empty{padding:50px 0;color:var(--muted)}footer{margin-top:36px;color:var(--muted);font-size:.8rem}@media(max-width:720px){.summary{grid-template-columns:1fr}.race{padding:15px;overflow-x:auto}th,td{white-space:nowrap}.race-head{align-items:start;flex-direction:column}}
"""


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

    race_sections = []
    race_count = predictions["race_id"].nunique()
    for _, race in predictions.groupby("race_id", sort=False):
        first = race.iloc[0]
        rows = []
        for runner in race.sort_values("prediction_rank").itertuples():
            rank = int(runner.prediction_rank)
            rows.append(
                f'<tr class="{"top3" if rank <= 3 else ""}"><td class="rank">{rank}</td>'
                f'<td>{_text(runner.horse_number)}</td><td>{_text(runner.horse_name)}</td>'
                f'<td class="prob">{float(runner.top3_probability):.1%}</td>'
                f'<td>{_number(runner.odds, 1)}</td><td>{_number(runner.expected_value_index)}</td>'
                f'<td>{_text(runner.jockey_name)}</td></tr>'
            )
        race_number = int(first["race_number"]) if pd.notna(first["race_number"]) else "-"
        race_sections.append(
            f'<section class="race"><div class="race-head"><div><div class="course">{_text(first["course_name"])}</div>'
            f'<h2>{race_number}R {_text(first["race_name"])}</h2></div><span class="meta">{_text(first["race_id"])}</span></div>'
            '<table><thead><tr><th>予測</th><th>馬番</th><th>馬名</th><th>3着以内確率</th><th>オッズ</th><th>期待値指数</th><th>騎手</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></section>'
        )
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    body = f"""<a class="back" href="../index.html">← 開催日一覧</a><header><div class="eyebrow">Daily Predictions</div>
<h1>{date_text}</h1><p class="lead">開催日単位の全レース予想</p></header>
<div class="summary"><div class="metric"><span>レース数</span><strong>{race_count}</strong></div>
<div class="metric"><span>出走馬数</span><strong>{len(predictions)}</strong></div>
<div class="metric"><span>モデル</span><strong>{escape(str(model_id))}</strong></div></div>
{''.join(race_sections)}<p class="meta">生成日時: {escape(generated_at)}</p>"""
    page_path = prediction_dir / f"{date_text}.html"
    page_path.write_text(_page(f"{date_text} 競馬予想", body), encoding="utf-8")
    _write_index(site_dir)
    return page_path


def initialize_prediction_site(site_dir: Path) -> Path:
    """Create the empty landing page committed before the first prediction."""
    (site_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (site_dir / ".nojekyll").touch()
    return _write_index(site_dir)
