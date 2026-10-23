# 新規HTML取得から結果DB登録までの検証（2026-10-23）

## 結果

2026年8月8日の中央競馬36レース・462頭を新規取得した結果HTMLから解析し、本番DBのコピーに登録できた。最終結果は `status: ok`、12項目の検査がすべて成功。実行時間は128.56秒（約2分9秒）。

本番 `data/racing.duckdb` と `data/features.duckdb` は実行前後のSHA-256が一致し、変更していない。今回のコピーは過去の検証コピーを引き継いだものではなく、本番DBから新たに作成した。

## 新規取得の確認

- `--fresh-download` を追加し、一覧と結果を `force=True` でネットワークから取得した。既存キャッシュへの検索・フォールバックは使用していない。
- GUIと同じ収集器のHTTPヘッダー・文字コード処理・取得間隔を使用した。
- 新しく取得した掲載一覧から中央競馬の36レースを抽出し、すべての結果HTMLを取得・解析した。
- 今回の取得物は実行専用 `fetch/` に保存し、チェックポイントへSHA-256を記録。解析前に結果HTMLのハッシュを再確認した。
- ヘッダーなしの単純なrequests接続はHTTP 400となったが、既存収集器と同じ設定では取得できた。

## 失敗・再試行

実サイトへ障害を起こす代わりに、最初の1レースをHTTP要求前の模擬タイムアウトとして未取得にした。

1. 他の35レースを新規取得。
2. 1レース不足の `incomplete` 状態をチェックポイントへ保存。この時点のコピーDBのハッシュは変更なし。
3. 保存したチェックポイントを読み直し、失敗した1レースだけ新規取得。
4. 全36レースが揃って解析に成功した後、DBへの登録を実行。

これは同じ実行内でチェックポイントを読み直す再試行の検証であり、別のActions実行から自動再開する仕組みはまだ実装していない。

通常の通信例外には最大3回の再試行と待機を設けた。再試行しても取得できなければ未完了を記録して登録を止める。通信失敗が続く場合、古いHTMLへ切り替えて成功扱いにはしない。

関連テストは21件成功（`python -m pytest tests/test_historical_import_validation.py tests/test_database_validation.py -q`）。チェックポイントから失敗分だけを再取得する動作、通信失敗の継続、既存キャッシュを使わない動作、一時的なタイムアウトからの回復を検査した。

## 登録後の確認

| 項目 | 結果 |
| --- | --- |
| historical行数 | 266,720 → 267,182 |
| 追加レース・頭数 | 36レース・462頭 |
| 既存historicalとのキー重複 | 0 |
| 同じ結果の2回登録 | 件数・値が不変、重複増加なし |
| 既存historical・upcoming・他テーブルの既存値 | 不変 |
| 有効な着順がある馬の走破時間欠損 | 0 |
| 着順・走破時間の欠損8頭 | 元HTMLで競走中止6頭・除外2頭と確認 |

既存の登録関数によるrace_odds管理列4列の追加は、前回同様、既存値を保持した追加移行として検査した。

## 成果物と再実行

Git管理対象外の `data/analysis/fresh_result_import_20261023/` に以下を保存した。

- `report.json`：最終合否、件数、検査、元DBのハッシュ、処理時間。
- `fetch/race_list_db/`、`fetch/result/`：新規取得HTML。
- `download_checkpoint_20260808.json`：模擬失敗を含む未完了状態。
- `download_resumed_20260808.json`：失敗分の再取得結果。
- `parsed_results.parquet`：解析した462頭の結果。
- `missing_finish_details.json`：欠損8頭の元HTML行と状態。
- `racing_validation.duckdb`：更新した検証用DB（34,615,296バイト）。

実行コマンド（再実行には新しい出力先を指定する）。

```powershell
.venv/Scripts/python.exe scripts/validate_historical_import.py --output data/analysis/fresh_result_import_next --dates 2026-08-08 --fresh-download --skip-features --simulate-failure
```

## 今回の範囲

今回は新規取得・DB登録・再試行を確認した。特徴量の再生成・再学習・本番反映・GitHubへのpushは行っていない。8月3日以降の全開催を取り込んだものではない。

100 MiBに向けた分割、共有先への保存、GUIとの同期、Actionsの別実行からの再開は、後続の実装対象。
