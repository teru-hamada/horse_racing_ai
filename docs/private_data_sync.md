# GUI・Actionsでの非公開データ同期

## GUI

［データ同期］を開き、GitHubの非公開リポジトリを `owner/name` 形式で指定して［接続確認］を押します。ローカルGitの認証情報を使います。認証情報を画面やログへ出力しません。Git Credential Manager等で認証済みであることが必要です。

［現在のデータを非公開へ保存］は、DBのコピーから予測バンドルを生成し、64 MiBずつの分割形式で新しい `snapshots/gui-*` ブランチに保存します。結果DB・特徴量DB・学習済みモデルが必要です。mainやActionsの本番参照は更新しません。保存先のmainが接続確認時から変わっていた場合は保存を中止します。

復元は、40文字のコミットSHAを指定し、バックアップ・置き換えのチェックを入れて［指定したデータを取得・復元］を押します。新旧形式を自動判別し、モデルの重み・前処理器・補正定義もローカルのモデル保存先に配置します。既存のHTML取得・DB登録・特徴量生成の操作はそのまま利用できます。

保存対象はDB・バンドル・設定です。HTMLそのものは同期しません。特徴量再生成・モデル再学習も自動では行いません。

## 競合・失敗

- 収集・DB登録・特徴量生成ジョブの実行中はGUIの同期を開始しません。
- DB接続と同期で同じプロセス間ロックを利用し、転送中にDBが書き換わることを防ぎます。
- データのハッシュが画面操作時から変わっていれば復元を止めます。
- 復元前に `.private-runtime/backups/<ID>/` に既存ファイルを保存し、書き込み例外時は元へ戻します。
- プロセスが強制終了した場合は中断マーカーが残ります。通常のDB操作を止め、［データ同期］の［中断前のバックアップへ戻す］で復旧します。
- 外部のDBツールや、このロックを使用しないスクリプトを併用して書き込まないでください。WALがあるDBは保存・復元を拒否します。

## Actions

日次予想は、指定SHAから従来形式または分割形式を復元します。依存ライブラリ監査も両形式のPython・requirements設定に対応しています。

日次予想は復元後にDB・モデル・ハッシュ・実行環境・CPU推論を検査します。導入時専用の保存・復元検証ワークフローは廃止しました。

新しい候補をローカルの隔離先で復元・検証してから、`PRIVATE_DATA_REF` を採用する候補SHAへ更新します。GUIで候補を作っただけでは本番予想には反映されません。

## コマンド

```powershell
# 標準は分割形式。GUIの保存とは異なり、既存の予測バンドルをそのまま梱包する。
.venv/Scripts/python.exe scripts/private_runtime.py prepare --output .private-runtime/next

# 新旧形式を自動判別して復元する。--output省略時はアプリへ適用する。
.venv/Scripts/python.exe scripts/private_runtime.py restore --source .private-runtime/next --output .private-runtime/check

# 復元先のハッシュ・DB・実行環境・保存モデルとCPU推論を検査する。
# 候補のrequirements・Python版に合う環境で実行する。
.venv/Scripts/python.exe scripts/verify_private_runtime.py --root .private-runtime/check

# 中断前のバックアップへ復旧する。
.venv/Scripts/python.exe scripts/private_runtime.py recover
```

`--format legacy` では従来の6ファイルを出力し、各ファイル100 MiB未満の検査を維持します。分割形式では保存する各partを100 MiB未満に抑え、復元後のDBは100 MiBを超えても扱えます。Git履歴の累積容量は別途監視が必要です。

分割保存・破損検出・復元の回帰テストは `tests/test_private_snapshot.py`、`tests/test_private_runtime.py`、`tests/test_runtime_transfer.py`、GUI操作は `tests/test_private_sync_gui.py` で維持します。導入時だけの検証スクリプトやコピーDBは運用に必要ありません。`.private-runtime/db-sync/` の未同期データと `backups/` の復旧用ファイルは一時検証データとは区別して保持してください。

結果の新規取得・DB登録・非公開mainへの保存を行うActionsは、[確定結果DBの更新](result_update_actions.md)を参照してください。

## DB作成後の過去結果の統合

［データベース作成］の過去結果では、［DB作成後に非公開mainへ統合する］が初期状態で有効です。保存済みHTMLからローカルDBへ登録した後、非公開mainの最新DBを取得して統合し、分割スナップショットとしてmainへ保存します。ローカルDBにもリモートの不足分を取り込みます。チェックを外すとローカル登録だけを行います。

対象日の掲載一覧をオンラインで照合し、GUIで不足していたレースは取得して一日分の整合性を確認します。既存結果と食い違う場合は結果ページを再取得して判断します。定期更新と競合した場合は最新mainで作り直し、強制pushはしません。当日の結果は翌日以降に同期してください。

ローカル登録が成功し同期だけ失敗した場合、DB作成自体は完了とし、同期失敗を別途表示します。対象データと利用したHTMLを `.private-runtime/db-sync/` に保持するため、アプリを再起動しても［未同期分の統合を再試行］から再開できます。ローカル登録前に停止した控えは公開しません。

モデル・特徴量・予測バンドル、本番予想の参照SHAは変更しません。出馬表・オッズとデモデータはこの自動統合の対象外です。［データ同期］にある全体の候補保存・指定SHA復元とは別の機能です。
