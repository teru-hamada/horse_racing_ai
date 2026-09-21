"""Collect race HTML into an isolated run directory."""
from .scrapers_html_collection_netkeiba import NetkeibaHtmlCollector


class DateCollector(NetkeibaHtmlCollector):
    def __init__(self, logger, directory):
        super().__init__(logger)
        self.directory = directory

    def _cache_path(self, dataset_type, storage_key, kind, key):
        path = self.directory / kind / f"{key}.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
