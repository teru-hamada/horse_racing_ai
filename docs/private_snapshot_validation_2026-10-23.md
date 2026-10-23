# 非公開Gitリポジトリでの分割保存・復元検証（2026-10-23）

## 結論

保存時にDB・バンドル一式を圧縮し、64 MiB以下のファイルに分割する方式で、GitHubの非公開リポジトリへの保存・別ディレクトリへの取得・復元を確認できた。実データ6ファイルの復元前後のSHA-256がすべて一致した。

ローカルで使用するDuckDBは従来どおり2ファイルのままでよい。今回は保存・転送用の形式を検証したため、GUIのクエリやDB構成は変更していない。年別DBへの論理分割は現段階では必要ない。

## サイズと増加量

| ファイル | 現在のサイズ | 100 MiBまでの余裕 |
| --- | ---: | ---: |
| racing.duckdb | 33.01 MiB | 66.99 MiB |
| features.duckdb | 76.26 MiB | 23.74 MiB |
| 既存runtime.zip | 14.57 MiB | 85.43 MiB |

前回の新規取得検証で8月8日の462行を追加した結果DBも33.01 MiBで、ファイル増加は0バイトだった。未使用領域があるため、これだけでは年間の物理サイズ増加を推定できない。特徴量の再生成は今回行っていない。

更新した結果DBのコピーから予測バンドルも生成し直し、そのバンドルとDB2つ・設定をまとめた。圧縮後は71,382,787バイト（68.08 MiB）となり、以下の2ファイルに分割した。

| 分割ファイル | サイズ |
| --- | ---: |
| snapshot-00001.part | 64.00 MiB |
| snapshot-00002.part | 4.08 MiB |

保存対象の全ファイルが100 MiB未満であることを確認した。8 MiBずつの9分割でも、実データが元どおり復元できた。

これは1ファイル上限への対応であり、Git履歴全体の増加を解消するものではない。DBが同じサイズでも更新内容は履歴に残る。保存頻度・保持方針とリポジトリ容量の監視は、定期化時に別途組み込む。

## GitHub上の検証

保存先が非公開であることをAPIで確認した後、専用検証ブランチへアップロードした。

- リポジトリ：`teru-hamada/horse-racing-ai-private-data`
- ブランチ：`validation/storage-20261023-104608`
- コミット：`0d14f84eca60b136129dc72fde39b02c1af66b1b`
- [検証用ブランチ](https://github.com/teru-hamada/horse-racing-ai-private-data/tree/validation/storage-20261023-104608)

別のディレクトリにこのブランチだけをcloneし、そこで取得した分割ファイルから復元した。確認結果は以下のとおり。

- DB・バンドル・設定の6ファイルが、保存前とバイト単位で一致。
- historicalは267,182行・19,390レース。追加した8月8日の462行・36レースが保持されている。
- 保存ファイルはすべて100 MiB未満。
- 非公開側mainのSHAと、公開側の本番参照 `PRIVATE_DATA_REF` は従来の `38dd15387a7d9c257f9398a3beee59ac73b83e20` のまま。
- 本番DB・本番特徴量DB・既存予測バンドル等の6ファイルは、操作前後のSHA-256が一致。

GitHub上の7項目の検査がすべて成功した。検証用ブランチは結果確認用に残してあり、本番用には採用していない。

## 新しい保存処理

`scripts/private_snapshot.py` を追加した。既存の `private_runtime.py` の保存・復元はデフォルトで従来の100 MiB検査を維持する。

- 保存対象は既存の許可リスト6ファイルだけ。HTMLや認証情報は含めない。
- 分割ファイルごと、圧縮アーカイブ全体、復元ファイルごとにSHA-256を検証。
- ファイルが不足・破損している場合は復元先を公開せず停止。
- 新しい出力先のみを許可し、既存の正常なデータを上書きしない。
- 全ファイルの検査終了後、作業ディレクトリを復元先へrenameする。
- WALのあるDBや、保存中に変化した入力は保存を拒否する。
- GUI・Actionsでの有効化や本番DBの置換は行わない。

実行例（いずれも出力先は未作成の場所を指定する）。

```powershell
.venv/Scripts/python.exe scripts/private_snapshot.py prepare --source . --output .private-runtime/new-snapshot
.venv/Scripts/python.exe scripts/private_snapshot.py restore --source .private-runtime/new-snapshot --output .private-runtime/new-restored
```

## 障害時の検査

関連テストは16件成功した。

```powershell
.venv/Scripts/python.exe -m pytest tests/test_private_snapshot.py tests/test_private_runtime.py -q
```

分割ファイルの欠落・破損、誤ったハッシュ、順序変更、不正パス、既存出力先、WAL、保存中の元DB変更を検査した。また100 MiBを超えるテスト用ファイルを、Gitに保存する各ファイルは上限未満のまま復元できることを確認した。この大容量テストはローカルだけで実施した。

## 成果物と残る作業

ローカルの検証成果物はGit管理対象外の `.private-runtime/storage_validation_20261023/` にある。`report.json` がローカル検証結果、`remote_report.json` がGitHub経由の検証結果。

今回の候補は結果DBの保存テストであり、追加分の特徴量再生成・モデル再学習は行っていない。更新した結果DBはコピーのみで、本番予想には使用していない。

現行Actionsの `scripts/private_runtime.py restore` は従来形式を読む。新形式の採用には、ActionsとGUI側の共通保存・復元処理、世代の選択・切り替え、競合防止、失敗後の再開を組み込んでから、本番参照を更新する必要がある。
