# 日次ワークフロー用に、netkeiba共通処理のHTML保存先を差し替える。
# 通常のキャッシュと診断用の取得ファイルを分け、実行ごとの出力先にHTMLを集める。
#
"""Collect race HTML into an isolated run directory."""
from .scrapers_html_collection_netkeiba import NetkeibaHtmlCollector


# 日付別ワークフローの取得物を、指定された実行専用ディレクトリへ保存する収集器。
class DateCollector(NetkeibaHtmlCollector):
    # 共通収集器を初期化し、この実行に限定したキャッシュ先を保持する。
    def __init__(self, logger, directory):
        super().__init__(logger)
        # この収集実行専用のキャッシュ保存先。
        self.directory = directory

    # 実行専用ディレクトリを起点に種類別の保存先を作り、年別の通常キャッシュとは分ける。
    def _cache_path(self, dataset_type, storage_key, kind, key):
        path = self.directory / kind / f"{key}.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
