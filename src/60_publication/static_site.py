# 予想、推奨買い目、確定結果の比較を、外部依存のない静的HTMLへ整形する。
# 入力文字列をHTMLエスケープし、日付別ページと一覧を生成する。
# 画面に掲載した買い目の購入額・払戻額を100円単位で集計し、返還は的中件数と区別する。
#
from __future__ import annotations
from importlib import import_module as _import_module

from datetime import datetime
from html import escape
from pathlib import Path
import re
import json

import pandas as pd

compare_recommended_bets = _import_module('src.50_result_comparison.prediction_comparison').compare_recommended_bets
# 完走・取消・除外・中止などの内部状態と日本語表示の対応。
RESULT_STATUS_LABELS = _import_module('src.50_result_comparison.prediction_comparison').RESULT_STATUS_LABELS


# 公開ページへ埋め込む共通CSS。Python文字列内へ説明を入れると生成HTMLも変わるため、説明は文字列外へ置く。
_STYLE = """
:root{color-scheme:dark;--bg:#07130f;--panel:#10251d;--line:#25483a;--text:#f4f7f5;--muted:#a9bdb5;--accent:#38d996;--gold:#ffd166}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top,#163b2c 0,var(--bg) 42%);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans JP",sans-serif}
main{width:100%;margin:0;padding:42px 28px 80px}header{margin-bottom:30px}.eyebrow{color:var(--accent);font-size:.78rem;font-weight:800;letter-spacing:.16em;text-transform:uppercase}h1{font-size:clamp(2rem,6vw,4.5rem);line-height:1;margin:.25em 0}.lead,.meta{color:var(--muted)}.race{width:100%;background:color-mix(in srgb,var(--panel) 92%,transparent);border:1px solid var(--line);border-radius:18px;box-shadow:0 18px 50px #0004;margin:16px 0;overflow:hidden}.race>summary{display:flex;align-items:center;justify-content:space-between;gap:18px;padding:22px 24px;cursor:pointer;font-size:1.25rem;font-weight:800;list-style:none}.race.graded>summary{color:var(--gold)}.race>summary::-webkit-details-marker{display:none}.race>summary::after{content:"＋";color:var(--accent);font-size:1.5rem}.race[open]>summary::after{content:"−"}.race[open]>summary{border-bottom:1px solid var(--line)}.race-content{width:100%;padding:20px 24px 24px;overflow-x:auto}.race-head{display:flex;align-items:center;justify-content:space-between;gap:14px}.course{color:var(--accent);font-weight:800}table{width:100%;min-width:760px;border-collapse:collapse;margin-top:18px;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:13px 10px;border-bottom:1px solid var(--line)}th{color:var(--muted);font-size:.78rem}.rank{font-size:1.1rem;font-weight:900}.prob{color:var(--gold);font-weight:800}.top3 td{background:#38d9960c}.back{display:inline-block;color:var(--accent);margin-bottom:18px;text-decoration:none}.empty{padding:50px 0;color:var(--muted)}footer{margin-top:36px;color:var(--muted);font-size:.8rem}@media(max-width:720px){main{padding:28px 12px 60px}.race>summary{padding:18px 16px;font-size:1.05rem}.race-content{padding:14px 12px 18px}th,td{white-space:nowrap}.race-head{align-items:start;flex-direction:column}}
"""

_STYLE += """
.comparison-summary{display:flex;flex-wrap:wrap;gap:10px 22px;padding:18px 22px;margin:0 0 24px;background:#10251dcc;border:1px solid var(--line);border-radius:14px}.comparison-summary strong{width:100%;color:var(--gold)}.comparison-summary span{color:var(--muted)}.comparison-summary .failures{width:100%;color:#ffb4a8}.hit{color:var(--accent);font-weight:800}.miss{color:#ffb4a8}
.bets{width:max-content;min-width:100%;margin:0 0 24px;padding:16px 18px;border:1px solid #ffd16666;border-radius:12px;background:#ffd1660a}.bets h3{margin:0 0 8px;color:var(--gold)}.bets table{margin-top:8px;min-width:680px}.bets-note{color:var(--muted);font-size:.8rem}
"""


# レース名から重賞表記を識別する正規表現。金色見出しの判定に使う。
_GRADED_RACE_PATTERN = re.compile(
    r"(?:Jpn\s*[123ⅠⅡⅢ]|G\s*[123ⅠⅡⅢ])",
    re.IGNORECASE,
)


# 動的な文字列はHTML本文や属性へ埋め込む前にエスケープする。
def _text(value: object, fallback: str = "-") -> str:
    return fallback if pd.isna(value) else escape(str(value))


# 数値を指定小数桁の表示文字列にする。欠損はゼロと表示せず「-」にする。
def _number(value: object, digits: int = 2) -> str:
    return "-" if pd.isna(value) else f"{float(value):.{digits}f}"


# 一つのレースで画面へ掲載する買い目を選び、表示分の精算と同じ集合を使う。
def _displayed_bets(bets: pd.DataFrame) -> pd.DataFrame:
    return _import_module('src.00_common.purchase_plan').resolve_plan(bets)



# 保存した購入金額で精算し、返還金も払戻合計へ含める。
def _bet_return_summary(
    bets: pd.DataFrame,
    official_payouts: dict[str, dict[str, int]],
    official_refunds: dict[str, dict[str, list[int]]] | None = None,
) -> dict[str, object]:
    """保存済み購入対象の金額に基づく回収率を共通処理で集計する。"""
    # official_payouts: レース・券種・組合せをキーとする公式の100円当たり払戻金。
    # official_refunds: 馬番・枠番・同枠組合せに関する公式返還情報。
    displayed = _displayed_bets(bets)
    judged = compare_recommended_bets(displayed, official_payouts, official_refunds)
    return _import_module('src.00_common.purchase_plan').return_summary(judged)



# 本文を共通のHTML枠とスタイルで包み、日別ページとして表示できる形にする。
def _page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="競馬予想AIが生成した3着以内確率"><title>{escape(title)}</title><style>{_STYLE}</style></head>
<body><main>{body}<footer>研究・学習用の予想です。的中や利益を保証するものではありません。</footer></main></body></html>
"""


# メモリ上の状態ではなく保存済み予想と公開ページから、指定日の処理状況を復元する。
def prediction_date_status(
    race_date: object,
    predictions_dir: Path,
    site_dir: Path,
) -> tuple[bool, str | None]:
    """Return prediction presence and comparison status for a date."""

    # race_date: 対象開催日。日別の抽出と過去履歴の境界に使う。
    target_date = pd.to_datetime(race_date, errors="coerce")
    if pd.isna(target_date):
        return False, None
    date_text = target_date.strftime("%Y-%m-%d")
    prediction_page = site_dir / "predictions" / f"{date_text}.html"
    prediction_files = predictions_dir.glob(
        f"*/predictions_{date_text}.parquet"
    )
    prediction_created = prediction_page.is_file() or any(prediction_files)
    comparison_status = None
    if prediction_page.is_file():
        page_html = prediction_page.read_text(encoding="utf-8", errors="replace")
        status_match = re.search(
            r'data-comparison-status="(completed|partial|none)"',
            page_html,
        )
        if status_match:
            stored_status = status_match.group(1)
            comparison_status = (
                stored_status if stored_status != "none" else None
            )
        elif 'class="comparison-summary"' in page_html:
            comparison_status = "completed"
    return prediction_created, comparison_status


# 指定日に対応する保存済み予想を探し、更新日時が最も新しいファイルを返す。
def latest_prediction_file(
    race_date: object,
    predictions_dir: Path,
) -> Path | None:
    """Return the most recently updated saved date-prediction file."""

    # race_date: 対象開催日。日別の抽出と過去履歴の境界に使う。
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


# 保存済み日付別ページへのリンク一覧を作り、公開サイトの入口を更新する。
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
<p class="lead">各レースの3着以内確率を、開催日ごとに掲載しています。</p>
<a class="back" href="job-status.html">ジョブ実行状況 →</a></header>{links}"""
    index_path = site_dir / "index.html"
    index_path.write_text(_page("競馬予想AI", body), encoding="utf-8")
    return index_path


# 日別の予想・買い目・結果比較ページを生成して一覧を更新する。モデル内部情報の表示は最小限にする。
def _calibration_return_block(returns: dict | None) -> str:
    """公式結果で精算した比較方式の成績。未精算・データなしは回収率と区別する。"""
    items = []
    labels = {'class_weight': '重み付け補正', 'sigmoid': 'sigmoid補正'}
    for method, label in labels.items():
        result = (returns or {}).get(method, {})
        status = result.get('status', 'missing')
        if status == 'completed':
            text = (f'{int(result["total_bets"])}点・購入 {int(result["purchase"]):,}円・'
                    f'的中 {int(result["hit_bets"])}点・払戻 {float(result["payout"]):,.0f}円・'
                    f'回収率 {float(result["recovery_rate"]):.1f}%')
        elif status == 'no_bets':
            text = '購入対象なし（見送り）'
        elif status == 'pending':
            text = f'集計中（{int(result.get("settled_bets", 0))} / {int(result.get("total_bets", 0))}点確定）'
        elif status == 'incomplete':
            text = '検証未完了'
        else:
            text = '比較データなし'
        items.append(f'<span data-calibration-method="{method}" data-status="{_text(status)}">'
                     f'{label}【検証中】: {_text(text)}</span>')
    return ('<section class="comparison-summary calibration-verification">'
            '<strong>補正方式の回収率（検証中）</strong>' + ''.join(items)
            + '<span>予想時に保存した購入対象・購入金額を、JRA公式払戻金・返還で精算。</span></section>')


def build_prediction_site(
    predictions: pd.DataFrame,
    race_date: object,
    model_id: str,
    site_dir: Path,
    comparison: pd.DataFrame | None = None,
    comparison_summary: dict[str, object] | None = None,
    bet_recommendations: pd.DataFrame | None = None,
    calibration_returns: dict | None = None,
) -> Path:
    """Write a dependency-free GitHub Pages site for one prediction date."""

    # race_date: 対象開催日。日別の抽出と過去履歴の境界に使う。
    required = {
        "race_id", "race_number", "race_name", "course_name", "horse_number",
        "horse_name", "jockey_name", "prediction_rank", "top3_probability",
        "odds", "expected_value_index",
    }
    missing = required.difference(predictions.columns)
    # 必要な列・定義が不足する場合は、内容を補完で隠さず不足一覧付きで処理を止める。
    if missing:
        raise ValueError(f"静的ページ用の予想列が不足しています: {sorted(missing)}")
    target_date = pd.to_datetime(race_date, errors="coerce")
    if pd.isna(target_date):
        raise ValueError("静的ページの開催日が正しくありません。")
    date_text = target_date.strftime("%Y-%m-%d")
    prediction_dir = site_dir / "predictions"
    # アプリから既存ページを再生成する場合も、保存済みの検証回収率を失わないようにする。
    returns_path = prediction_dir / 'calibration' / date_text / 'returns.json'
    if calibration_returns is None and returns_path.is_file():
        calibration_returns = json.loads(returns_path.read_text(encoding='utf-8'))
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
            for column in ("finish_position", "result_status", "predicted_top3", "actual_top3", "top3_hit")
            if column in comparison.columns
        ]
        page_predictions = page_predictions.merge(
            comparison[comparison_columns].drop_duplicates(merge_keys),
            on=merge_keys,
            how="left",
        )
    for column in ("finish_position", "result_status", "predicted_top3", "actual_top3", "top3_hit"):
        if column not in page_predictions.columns:
            page_predictions[column] = pd.NA

    race_sections = []
    # レース単位で処理し、別レースの馬番や払戻金を混ぜない。
    for race_id, race in page_predictions.groupby("race_id", sort=False):
        first = race.iloc[0]
        rows = []
        for runner in race.sort_values("prediction_rank").itertuples():
            rank = int(runner.prediction_rank)
            comparison_cells = ""
            if has_comparison:
                finish_position = getattr(runner, "finish_position", pd.NA)
                result_status = getattr(runner, "result_status", pd.NA)
                status = str(result_status) if pd.notna(result_status) else "unknown"
                finish_label = _text(finish_position) if pd.notna(finish_position) else _text(
                    RESULT_STATUS_LABELS.get(status), "-"
                )
                top3_hit = getattr(runner, "top3_hit", pd.NA)
                judgment = "-"
                judgment_class = ""
                if (pd.notna(finish_position) or status in {"did_not_finish", "disqualified"}) and rank <= 3 and pd.notna(top3_hit):
                    judgment = "的中" if bool(top3_hit) else "不的中"
                    judgment_class = "hit" if bool(top3_hit) else "miss"
                comparison_cells = (
                    f'<td>{finish_label}</td>'
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
        bet_block = ""
        if bet_recommendations is not None:
            race_bets = _displayed_bets(bet_recommendations[
                bet_recommendations["race_id"].astype(str).eq(str(race_id))
            ])
            if race_bets.empty:
                bet_block = (
                    '<section class="bets"><h3>AIおすすめ買い目 上位3つ</h3>'
                    '<p class="bets-note">購入対象なし（見送り）。</p></section>'
                )
            else:
                race_bets = compare_recommended_bets(
                    race_bets, (comparison_summary or {}).get("official_payouts", {}),
                    (comparison_summary or {}).get("official_refunds", {}),
                )
                bet_rows = "".join(
                    "<tr>"
                    + (f'<td>{_text(bet.bet_result)}</td>' if has_comparison else "")
                    + f"<td>{_text(bet.bet_type_label)}</td>"
                    f"<td>{_text(bet.selection)}</td>"
                    f"<td>{float(bet.estimated_probability):.1%}</td>"
                    f"<td>{_number(bet.odds_used, 1)}</td>"
                    f"<td>{float(bet.recovery_rate_percent):.1f}%</td>"
                    f"<td>{float(bet.bet_type_reliability):.0%}</td>"
                    f"<td>{float(bet.recommendation_score):.1f}</td>"
                    f"<td>{float(bet.expected_profit_per_100):+.0f}円</td>"
                    f"<td>{int(bet.stake_yen):,}円</td>"
                    + (
                        f'<td>{_number(bet.payout_yen, 0)}円</td>'
                        if has_comparison else ""
                    )
                    + "</tr>"
                    for bet in race_bets.itertuples()
                )
                bet_block = (
                    '<section class="bets"><h3>AIおすすめ買い目 上位3つ</h3>'
                    '<table><thead><tr>'
                    + ('<th>的中結果</th>' if has_comparison else '')
                    + '<th>券種</th><th>買い目</th>'
                    '<th>推定的中確率</th><th>使用オッズ</th>'
                    '<th>推定回収率</th><th>券種適合度</th><th>おすすめスコア</th>'
                    '<th>100円当たり期待損益</th><th>購入額</th>'
                    + ('<th>払戻金</th>' if has_comparison else '')
                    +
                    f'</tr></thead><tbody>{bet_rows}</tbody></table>'
                    '<p class="bets-note">おすすめスコア＝推定回収率×券種適合度。'
                    '複勝以外はPlackett-Luce法による近似値。'
                    '推定回収率110%以上かつ券種別最低的中確率を満たす候補から、'
                    '複勝・ワイドを優先。範囲オッズは下限値を使用しています。</p></section>'
                )
        race_sections.append(
            f'<details class="{race_class}"><summary>{_text(first["course_name"])} {race_number}R {_text(first["race_name"])}</summary>'
            f'<div class="race-content">{bet_block}'
            '<table><thead><tr><th>予測</th><th>馬番</th><th>馬名</th><th>3着以内確率</th>'
            + ('<th>実着順</th><th>判定</th>' if has_comparison else '')
            + '<th>オッズ</th><th>期待値指数</th><th>騎手</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div></details>'
        )
    comparison_block = ""
    if has_comparison:
        summary = comparison_summary or {}
        requested_races = int(summary.get("requested_races", 0) or 0)
        compared_races = int(summary.get("compared_races", 0) or 0)
        if requested_races > 0 and compared_races >= requested_races:
            comparison_status = "completed"
        elif compared_races > 0:
            comparison_status = "partial"
        else:
            comparison_status = "none"
        # 失敗したレースIDと原因を残す一覧。成功分を失わず不足を報告する。
        failures = summary.get("failures", []) or []
        failure_text = ""
        if failures:
            failure_items = " / ".join(
                f'{item.get("race_id", "-")} ({item.get("message", "-")})'
                for item in failures
            )
            failure_text = f'<span class="failures">未比較: {_text(failure_items)}</span>'
        bet_return_text = ""
        if bet_recommendations is not None:
            returns = _bet_return_summary(
                bet_recommendations,
                summary.get("official_payouts", {}) or {},
                summary.get("official_refunds", {}) or {},
            )
            if returns["total_bets"] == 0:
                bet_return_text = '<span>おすすめ買い目の回収率: 対象なし</span>'
            elif returns["settled_bets"] < returns["total_bets"]:
                bet_return_text = (
                    '<span>おすすめ買い目の回収率: 集計中 '
                    f'（{returns["settled_bets"]} / {returns["total_bets"]} 点確定）</span>'
                )
            else:
                bet_return_text = (
                    f'<span>保存済みおすすめ買い目を購入: '
                    f'{returns["total_bets"]} 点・購入 {returns["purchase"]:,}円・'
                    f'的中 {returns["hit_bets"]} 点・払戻 {returns["payout"]:,.0f}円・'
                    f'回収率 {returns["recovery_rate"]:.1f}%（JRA公式払戻金）</span>'
                )
        comparison_block = (
            '<section class="comparison-summary" '
            f'data-comparison-status="{comparison_status}">'
            '<strong>確定結果との比較</strong>'
            f'<span>比較完了: {_text(summary.get("compared_races", 0))} / '
            f'{_text(summary.get("requested_races", 0))} レース</span>'
            f'<span>予測上位3頭の的中: {_text(summary.get("top3_hit_count", 0))} 頭</span>'
            f'<span>上位3頭完全的中: {_text(summary.get("perfect_top3_races", 0))} レース</span>'
            f'{bet_return_text}'
            f'{failure_text}</section>'
        )
        comparison_block += _calibration_return_block(calibration_returns)
    # この生成runで共通に付与する生成日時。行ごとにばらつかせない。
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    body = f"""<a class="back" href="../index.html">← 開催日一覧</a><header><div class="eyebrow">Daily Predictions</div>
<h1>{date_text}</h1><p class="lead">開催日単位の全レース予想</p></header>
{comparison_block}{''.join(race_sections)}<p class="meta">生成日時: {escape(generated_at)}</p>"""
    page_path = prediction_dir / f"{date_text}.html"
    page_path.write_text(_page(f"{date_text} 競馬予想", body), encoding="utf-8")
    _write_index(site_dir)
    return page_path


# 予想がまだない場合にも、公開サイトの入口ページを用意する。
def initialize_prediction_site(site_dir: Path) -> Path:
    """Create the empty landing page committed before the first prediction."""
    (site_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (site_dir / ".nojekyll").touch()
    return _write_index(site_dir)
