# 取得済み出馬表とオッズを検証用DuckDBへ登録し、読み戻した内容を照合する。
# 再登録で行が増えないこと、出走馬と単勝オッズが対応することを検査する。
# 取消・除外馬を予想対象から外し、本番DBを使わずに保存処理の整合性を確かめる。
#
"""Register and validate a single race in a new, explicitly isolated database."""
from __future__ import annotations

from datetime import date
from importlib import import_module
from pathlib import Path

import pandas as pd

storage = import_module("src.00_common.storage")


# 券種をキーに行数を数え、保存前後で券種ごとのデータ欠落がないか比較できるようにする。
def _counts(odds: pd.DataFrame) -> dict:
    return odds.groupby("bet_type").size().to_dict()


# 表示形式や行順の違いを正規化して、保存前後の値が一致するかを比較する。
def _same_values(left: pd.DataFrame, right: pd.DataFrame, columns: list[str]) -> bool:
    # Compare payloads independent of row order, dtype, and registration timestamps.
    def canonical(frame):
        return frame.reindex(columns=columns).sort_values(columns, na_position="last").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(canonical(left), canonical(right), check_dtype=False)
        return True
    # 比較の不一致を検出し、正常一致の戻り値と区別して呼出元へ知らせる。
    except AssertionError:
        return False


# 検証用DBへ登録して件数・キー・値・再登録時の不変性を確認し、検査結果をまとめて返す。
def verify_database(card: pd.DataFrame, odds: pd.DataFrame, race_date: date,
                    race_id: str, output: Path) -> dict:
    # race_date: 対象開催日。日別の抽出と過去履歴の境界に使う。
    # race_id: レースの識別子。日付や馬番とは別にレース単位の照合へ使う。
    from .html_acquisition import inspect_card

    statuses = card.get("entry_status", pd.Series("active", index=card.index))
    withdrawn = card.loc[statuses.isin({"scratched", "excluded"})]
    # 出馬表から確認できた取消・除外馬の馬番。馬番式買い目の除外に使う。
    withdrawn_numbers = withdrawn.horse_number.dropna().astype(int).tolist()
    card = card.loc[~statuses.isin({"scratched", "excluded"})].copy()
    # Horse-number tickets involving withdrawn horses are not prediction inputs.
    # Frame numbers are a separate namespace; do not filter frame tickets by horse number.
    involves_withdrawn = odds[["selection_1", "selection_2", "selection_3"]].isin(withdrawn_numbers).any(axis=1)
    odds = odds.loc[~(odds.bet_type.ne("bracket_quinella") & involves_withdrawn)].copy()

    database = output / "smoke.duckdb"
    # 本番DBや既存ファイルへ検証データを上書きしないよう、新しい隔離DBだけを許可する。
    if database.resolve() == storage.PATHS.database.resolve() or database.exists():
        raise ValueError("検証DBには新しい専用ファイルが必要です。")
    checks = {}
    # 保存前の券種別オッズ件数。登録後の欠落や重複の検査に使う。
    expected_counts = _counts(odds)
    # 初回登録と再登録後の読み戻し結果。再登録で値や件数が変わらないか比較する。
    snapshots = []
    # 同じ入力を二回登録し、重複せず置換されることと最新runへの更新を実際に検証する。
    for attempt in (1, 2):
        # 今回の生成・登録を識別する実行ID。保存行と履歴の対応付けに用いる。
        run_id = f"database-smoke-{attempt}"
        if not card.empty:
            storage.save_race_records(card, run_id, "upcoming", database=database)
        storage.save_race_odds(odds, run_id, database=database)
        # 検証用DBから読み戻した出馬表。入力と対応するかを検査する。
        loaded_card = storage.load_records("upcoming", database=database)
        # Exercise the same race-ID based lookup used by predictions.
        loaded_odds = storage.load_race_odds([race_id], database=database)
        with storage.connect(database) as con:
            total_odds = con.execute("SELECT count(*) FROM race_odds").fetchone()[0]
        snapshots.append((loaded_card, loaded_odds))
        prefix = f"registration_{attempt}"
        checks[f"{prefix}_runner_count"] = len(loaded_card) == len(card) > 0
        checks[f"{prefix}_odds_counts"] = (
            len(loaded_odds) == len(odds) == total_odds and len(odds) > 0
            and _counts(loaded_odds) == expected_counts
        )
        checks[f"{prefix}_horse_and_frame_numbers"] = inspect_card(loaded_card)["status"] == "ok"
        checks[f"{prefix}_race_and_date"] = all(
            not frame.empty
            and frame["race_id"].astype(str).eq(race_id).all()
            and pd.to_datetime(frame["race_date"]).dt.date.eq(race_date).all()
            for frame in (loaded_card, loaded_odds)
        )

    first_card, first_odds = snapshots[0]
    # 収集日時など毎回変わる管理列を除いた、出馬表の比較対象列。
    card_columns = [c for c in storage.RACE_RECORD_COLUMNS if c not in {"collection_run_id", "collected_at"}]
    # 登録日時などを除いた、オッズ保存前後の比較対象列。
    odds_columns = [c for c in storage.ODDS_COLUMNS if c not in {"collection_run_id", "registered_at"}]
    checks["reregistration_unchanged"] = (
        _same_values(first_card, loaded_card, card_columns)
        and _same_values(first_odds, loaded_odds, odds_columns)
    )
    checks["reregistration_replaced"] = all(
        not frame.empty and frame["collection_run_id"].eq("database-smoke-2").all()
        for frame in (loaded_card, loaded_odds)
    )
    checks["unique_odds_selections"] = not loaded_odds.duplicated(
        ["race_id", "bet_type", "selection_1", "selection_2", "selection_3"]
    ).any()
    win = loaded_odds[loaded_odds["bet_type"].eq("win")]
    joined = loaded_card[["race_id", "horse_number", "horse_name"]].merge(
        win[["race_id", "selection_1", "odds_min"]],
        left_on=["race_id", "horse_number"], right_on=["race_id", "selection_1"],
        how="outer", indicator=True,
    )
    matched = joined["_merge"].eq("both") & joined["odds_min"].gt(0)
    checks["win_odds_match_all_runners"] = (
        len(joined) == len(loaded_card) > 0 and bool(matched.all())
        and not win.duplicated(["race_id", "selection_1"]).any()
    )
    loaded_card.to_csv(output / "db_card.csv", index=False, encoding="utf-8-sig")
    loaded_odds.to_csv(output / "db_odds.csv", index=False, encoding="utf-8-sig")
    joined.to_csv(output / "db_win_join.csv", index=False, encoding="utf-8-sig")
    checks = {key: bool(value) for key, value in checks.items()}
    return {
        "status": "ok" if all(checks.values()) else "incomplete",
        "database": database.name, "checks": checks,
        "withdrawn_horse_numbers": withdrawn_numbers,
        "failed_checks": [key for key, value in checks.items() if not value],
        "expected_runners": len(card), "registered_runners": len(loaded_card),
        "expected_odds_by_bet_type": expected_counts,
        "registered_odds_by_bet_type": _counts(loaded_odds),
        "matched_win_runners": int(matched.sum()),
        "unmatched_card_numbers": joined.loc[joined["_merge"].eq("left_only"), "horse_number"].dropna().astype(int).tolist(),
        "unmatched_odds_numbers": joined.loc[joined["_merge"].eq("right_only"), "selection_1"].dropna().astype(int).tolist(),
    }
