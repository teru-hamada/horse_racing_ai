# アプリで共有するディレクトリとDuckDBファイルのパスを定義する。
# 基準はこのソースの配置場所であり、起動時のカレントディレクトリに依存させない。
# PATHSの初期化時には必要な保存用ディレクトリも作成される。
#
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# 各保存先をひとまとめにした変更不可の設定。DBパスとディレクトリパスを区別する。
@dataclass(frozen=True)
class AppPaths:
    # rootはプロジェクト基準、dataは収集物・DB・予想の保存先。
    root: Path
    data: Path
    raw_html: Path
    # 過去結果と未開催の出馬表を分け、取得・再解析で対象種別を選べるようにする。
    historical_html: Path
    upcoming_html: Path
    predictions: Path
    models: Path
    # 画面から参照するログ行の一覧。インスタンスごとに独立したリストを持つ。
    logs: Path
    # レース・実行履歴と特徴量は別DB。特徴量の再生成が元レース表を置換しない構成にする。
    database: Path
    # 特徴量の読み書き先。Noneを許容するAPIではプロジェクトの既定DBを使用する。
    feature_database: Path


# rootを省略するとリポジトリ直下を採用する。戻す前に保存用フォルダーを作成する。
def get_paths(root: Path | None = None) -> AppPaths:
    root_path = (root or Path(__file__).resolve().parents[2]).resolve()
    data = root_path / "data"
    raw_html = data / "raw_html"
    paths = AppPaths(
        root=root_path,
        data=data,
        raw_html=raw_html,
        historical_html=raw_html / "historical",
        upcoming_html=raw_html / "upcoming",
        predictions=data / "predictions",
        models=root_path / "models",
        logs=root_path / "logs",
        database=data / "racing.duckdb",
        feature_database=data / "features.duckdb",
    )
    for directory in (
        paths.data,
        paths.raw_html,
        paths.historical_html,
        paths.upcoming_html,
        paths.predictions,
        paths.models,
        paths.logs,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return paths


# プロジェクト既定の保存先。DBを差し替えるテストではこの参照も隔離先へ変更する。
PATHS = get_paths()
