# 日別のUTF-8ログファイルと画面向けコールバックへログを送る。
# 通常メッセージと例外のトレースをファイルへ残し、画面には時刻付きの行を通知する。
# インスタンス専用のloggerを使い、親loggerへの伝播による二重出力を避ける。
#
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Callable


LogCallback = Callable[[str], None]


# ファイル記録とUI通知を同じ呼出しで行うロガー。画面向け通知先はcallbackで注入する。
class AppLogger:
    """File logger that can also stream formatted lines into the Streamlit UI."""

    # 同じ日に追記するFileHandlerをUTF-8で作り、既存ハンドラーを整理して重複出力を防ぐ。
    def __init__(self, log_dir: Path, callback: LogCallback | None = None) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"app_{datetime.now():%Y%m%d}.log"
        # UIなどへのログ通知先。Noneならファイルへの記録だけを行う。
        self.callback = callback

        # 取得や解析の経過・警告・例外を通知するロガー。
        self.logger = logging.getLogger(f"horse_racing_ai.{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        handler = logging.FileHandler(self.log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        self.logger.handlers.clear()
        self.logger.addHandler(handler)

    # ファイルへ記録した後、コールバックがあれば画面表示用に整形した文字列を通知する。
    def _emit(self, level: str, message: str) -> None:
        getattr(self.logger, level.lower())(message)
        line = f"[{datetime.now():%H:%M:%S}] {level.upper():7s} {message}"
        # 画面への通知先がある場合だけ整形済みの行を送る。ファイル記録は別に実施済み。
        if self.callback:
            self.callback(line)

    # 通常の処理経過をINFOレベルでファイルと任意の通知先へ送る。
    def info(self, message: str) -> None:
        self._emit("info", message)

    # 処理継続は可能だが確認が必要な状態をWARNINGとして通知する。
    def warning(self, message: str) -> None:
        self._emit("warning", message)

    # 失敗をERRORレベルで通知する。例外トレースを記録する場合はexceptionを使う。
    def error(self, message: str) -> None:
        self._emit("error", message)

    # ファイルには現在の例外トレースを残し、画面には簡潔なエラー行を通知する。
    def exception(self, message: str) -> None:
        self.logger.exception(message)
        line = f"[{datetime.now():%H:%M:%S}] ERROR   {message}"
        # 画面への通知先がある場合だけ整形済みの行を送る。ファイル記録は別に実施済み。
        if self.callback:
            self.callback(line)
