"""Validate a complete date result and prepare only its public files."""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np
import pandas as pd

from .static_site import build_prediction_site


def prepare_publication(results: Path, site: Path) -> list[str]:
    report = json.loads((results / "date_report.json").read_text(encoding="utf-8"))
    day = date.fromisoformat(report["race_date"]).isoformat()
    ids = report["race_ids"]
    rows = report["races"]
    if (report["status"] != "ok" or not ids or len(ids) != len(set(ids))
            or len(rows) != len(ids) or {r["race_id"] for r in rows} != set(ids)
            or any(r["status"] != "ok" for r in rows)):
        raise ValueError("全レースの検証が成功していないため公開しません。")
    stages = ("card", "jra_odds", "database", "prediction", "odds_match", "betting")
    if any(r.get(stage) != "ok" for r in rows for stage in stages):
        raise ValueError("未完了の検証工程があるため公開しません。")
    predictions = pd.read_csv(results / "predictions.csv", dtype={"race_id": str, "horse_id": str})
    bets = pd.read_csv(results / "bets.csv", dtype={"race_id": str, "selection": str})
    if set(predictions.race_id) != set(ids) or not set(bets.race_id).issubset(ids):
        raise ValueError("公開CSVと検証対象のレースが一致しません。")
    if predictions.model_run_id.nunique() != 1 or predictions.model_run_id.isna().any():
        raise ValueError("使用モデルが不正です。")
    if not pd.to_datetime(predictions.race_date).dt.date.eq(date.fromisoformat(day)).all():
        raise ValueError("公開CSVの開催日が一致しません。")
    probabilities = predictions.top3_probability.to_numpy(dtype=float)
    numbers = pd.to_numeric(predictions.horse_number, errors="coerce")
    if (not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all()
            or not (numbers.between(1, 18) & numbers.mod(1).eq(0)).all()
            or predictions.duplicated(["race_id", "horse_number"]).any()):
        raise ValueError("予想の確率または対象馬に異常があります。")
    for row in rows:
        runners = predictions[predictions.race_id.eq(row["race_id"])]
        if (len(runners) != row["prediction_rows"] or len(runners) != row["runners"]
                or row["model_id"] != predictions.model_run_id.iloc[0]
                or len(bets[bets.race_id.eq(row["race_id"])]) != row["bet_rows"]):
            raise ValueError("検証結果と公開する頭数・モデルが一致しません。")
    if not bets.empty:
        values = bets[["estimated_probability", "odds_used", "recovery_rate_percent"]].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not bets.estimated_probability.between(0, 1).all() or not bets.odds_used.gt(0).all():
            raise ValueError("公開する買い目の数値が不正です。")

    # Prepare the complete site elsewhere first; failures leave published files alone.
    changed = []
    with tempfile.TemporaryDirectory() as temporary:
        staged = Path(temporary) / "site"
        if site.exists():
            shutil.copytree(site, staged)
        else:
            staged.mkdir()
        build_prediction_site(predictions, day, predictions.model_run_id.iloc[0], staged, bet_recommendations=bets)
        export = staged / "predictions"
        predictions.to_csv(export / f"{day}.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
        bets.to_csv(export / f"{day}_bets.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
        names = [".nojekyll", "index.html", f"predictions/{day}.html",
                 f"predictions/{day}.csv", f"predictions/{day}_bets.csv"]
        for name in names:
            source, target = staged / name, site / name
            content = source.read_bytes()
            if target.exists():
                previous = target.read_bytes()
                if name.endswith(".html"):
                    # Keep the previous generation time if visible content is identical.
                    pattern = '<p class="meta">生成日時: [^<]*</p>'.encode("utf-8")
                    clean = lambda b: re.sub(pattern, b'', b.replace(b'\r\n', b'\n'))
                    if clean(previous) == clean(content):
                        continue
                elif previous == content:
                    continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            changed.append(name)
    return changed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--site", type=Path, default=Path("docs"))
    args = parser.parse_args()
    print(json.dumps({"changed_files": prepare_publication(args.results, args.site)}, ensure_ascii=False))
