"""Exercise sync actions through the actual Streamlit menu without network calls."""
from importlib import import_module
from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_database_creation_auto_sync_default_and_durable_retry(tmp_path, monkeypatch):
    import src.public_api as api
    paths = api.get_paths(tmp_path)
    monkeypatch.setattr(api, 'PATHS', paths)
    monkeypatch.setattr(import_module('src.00_common.storage'), 'PATHS', paths)
    monkeypatch.setattr(api, 'list_database_jobs', lambda *a, **k: [])
    monkeypatch.setattr(api, 'list_pending_database_syncs', lambda **k: [{'id': 'pending_batch'}])
    calls = []
    monkeypatch.setattr(api, 'retry_database_sync', lambda batch: calls.append(batch))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=60).run()
    app.sidebar.radio[0].set_value('データベース作成').run()
    assert not app.exception
    checkbox = next(item for item in app.checkbox if item.key == 'database_auto_sync')
    assert checkbox.value is True
    next(item for item in app.button if item.label == '未同期分の統合を再試行').click().run()
    assert not app.exception and calls == ['pending_batch']


def test_database_creation_can_pull_without_html_or_upload_enabled(tmp_path, monkeypatch):
    import src.public_api as api
    paths = api.get_paths(tmp_path)
    monkeypatch.setattr(api, 'PATHS', paths)
    monkeypatch.setattr(import_module('src.00_common.storage'), 'PATHS', paths)
    monkeypatch.setattr(api, 'list_database_jobs', lambda *a, **k: [])
    monkeypatch.setattr(api, 'list_pending_database_syncs', lambda **k: [])
    calls = []
    monkeypatch.setattr(api, 'start_database_job', lambda **kwargs: calls.append(kwargs) or 'pull-job')
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=60).run()
    app.sidebar.radio[0].set_value('データベース作成').run()
    next(item for item in app.checkbox if item.key == 'database_auto_sync').uncheck().run()
    button = next(item for item in app.button if item.label == '非公開リポジトリから過去結果を取り込む（全期間）')
    assert not button.disabled and not list(paths.historical_html.rglob('*.html'))
    button.click().run()
    assert not app.exception and len(calls) == 1
    assert calls[0]['import_only'] is True and calls[0]['dataset_type'] == 'historical'
    assert calls[0]['sync_repository'] == 'teru-hamada/horse-racing-ai-private-data'


def test_sync_menu_connection_save_and_confirmed_restore(tmp_path, monkeypatch):
    import src.public_api as api
    paths = api.get_paths(tmp_path)
    monkeypatch.setattr(api, 'PATHS', paths)
    monkeypatch.setattr(import_module('src.00_common.storage'), 'PATHS', paths)
    calls = []
    monkeypatch.setattr(api, 'private_repository_info', lambda repository, root: {
        'repository': repository, 'branch': 'main', 'commit': 'a' * 40})

    def save(repository, base, **kwargs):
        kwargs['check_idle']()
        calls.append(('save', repository, base))
        return {'commit': 'b' * 40, 'url': 'https://github.com/example/private/tree/candidate'}

    def restore(repository, commit, **kwargs):
        kwargs['check_idle']()
        assert 'data/racing.duckdb' in kwargs['expected_hashes']
        calls.append(('restore', repository, commit))
        return {'backup': str(tmp_path / 'backup')}

    monkeypatch.setattr(api, 'save_private_snapshot', save)
    monkeypatch.setattr(api, 'restore_private_snapshot', restore)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=60).run()
    app.sidebar.radio[0].set_value('データ同期').run()
    assert not app.exception
    buttons = lambda label: next(button for button in app.button if button.label == label)
    assert buttons('現在のデータを非公開へ保存').disabled
    assert buttons('指定したデータを取得・復元').disabled
    buttons('接続確認').click().run()
    buttons('現在のデータを非公開へ保存').click().run()
    assert calls[0][0] == 'save'
    app.checkbox[0].check().run()
    buttons('指定したデータを取得・復元').click().run()
    assert not app.exception
    assert calls[-1][0] == 'restore'
