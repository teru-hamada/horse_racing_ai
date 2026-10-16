# 確率補正の共通処理

既定は `none`。通常のアプリ・Actions実行で勝手に補正方式は変わりません。
1レース・開催日一括・過去レース予想は共通の補正関数を通り、以下を保存します。

- `raw_top3_probability`: sigmoid後のモデル生出力
- `top3_probability`: 選択方式の補正後確率（買い目計算と順位付けはこちらを使用）
- `calibration_method` / `calibration_id`: 方式と定義の内容ID

公開方式が`none`でも、利用可能な3方式を同じ生出力から並行計算します。
`calibration_none_probability`、`calibration_class_weight_probability`、
`calibration_sigmoid_probability`へそれぞれ保存し、各方式の`_status`・`_id`・`_reason`も残します。
定義未作成は`unavailable`と欠損値、校正期間より前は`not_applicable`とし、生確率を補正値に偽装しません。
公開方式として指定された方式が使えない場合や、定義の破損・ID不一致は従来どおり停止します。

## 方式別の予想・おすすめ買い目の保存

各予想出力ディレクトリには次の比較用ファイルを追加します。

```text
calibration/
  status.json
  odds.csv                        同時点の比較用オッズ（レース実行・ローカル予想）
  none/
    predictions.csv
    predictions.parquet
    bets.csv                      条件を満たす全候補
    recommended_bets.csv          公開HTMLと同じスコア順で各レース上位3点
  class_weight/                   同じファイル構成
  sigmoid/                        同じファイル構成
```

3方式それぞれの確率で買い目を最初から計算し直します。無補正で残った候補だけを再利用するものではありません。
同じ取得済みオッズを使い、比較のための追加通信は行いません。オッズなしは`missing`として記録します。
DBには引き続き実行履歴と主予想ファイルへの参照を保存し、確率・買い目の詳細はCSV／Parquetへ保存します。

日別ワークフローでは方式ごとのCSVを集約し、一部レース欠落時は`incomplete`を記録します。
日次公開では`docs/predictions/calibration/日付/方式/`へ比較CSVを保存します。
HTMLに表示するのは従来の選択方式だけです。全オッズとParquetは公開先へ重複コピーせず実行成果物に残します。
翌日の結果照合では、保存済みの`recommended_bets.csv`を各方式それぞれ精算します。
公開HTMLの通常の回収率に加え、「補正方式の回収率（検証中）」として重み付け補正・sigmoid補正を表示します。
各レース上位3点を各100円購入した場合の購入額・的中点数・払戻額・回収率で、返還は払戻へ含めますが的中点数には数えません。
JRA結果は主方式と共通で、比較方式のための追加取得や、結果確認後の買い目再選定は行いません。
精算結果は`docs/predictions/calibration/日付/returns.json`と方式別`bets_results.csv`へ保存します。
結果不足は「集計中」、対象買い目ゼロは「購入対象なし（見送り）」、旧日付など保存なしは「比較データなし」とします。
これらを0%とは表示しません。入力の破損・不整合は「検証未完了」と表示します。
照合開始時に比較入力のハッシュも固定し、公開直前に変更・追加・削除があれば再実行を要求します。

## 方式

| 方式 | 入力条件 | 計算 |
|---|---|---|
| `none` | 旧モデルでも利用可 | 生確率をそのまま利用 |
| `class_weight` | 学習時に保存した`pos_weight`、または同モデル専用の補正JSON | `sigmoid(logit(p) - log(pos_weight))` |
| `sigmoid` | 同モデル専用の学習済み補正JSON | `sigmoid(slope * logit(p) + intercept)` |

新規学習は`metrics.json`へ正例重みを保存し、`calibration.json`へ補正なしと重み補正の両定義を保存します。
sigmoidはモデル更新ごとに学習外予測から作成する必要があります。別モデルの定義は流用しません。
新しい`test_predictions.parquet`には生確率・開催日・モデルIDも保存します。
旧モデルに正例重みの記録がない場合、現在のDBから推測して自動補完することはありません。

## 補正器の作成

まず別の出力先へ作成・評価します。予想実行中に補正器を学習しません。

```powershell
.venv\Scripts\python.exe -m src.40_ai_modeling.calibration_service --model-dir models/top3/MODEL_ID --method class_weight --output data/analysis/calibration/class_weight.json
```

sigmoidには、モデルの学習・検証期間より後、評価期間より前の予測を渡します。
CSVまたはParquetの必須列は`race_id, horse_id, race_date, model_run_id, raw_top3_probability, actual_top3`です。
`actual_top3`は0/1、入力は同じモデルの学習外予測だけとします。元のモデルテスト期間を
校正へ転用した場合、その期間を独立した最終評価データとしては再利用しません。

```powershell
.venv\Scripts\python.exe -m src.40_ai_modeling.calibration_service --model-dir models/top3/MODEL_ID --method sigmoid --predictions data/analysis/calibration_input.parquet --evaluation-start 2026-09-01 --output data/analysis/calibration/sigmoid.json
```

出力先が存在する場合は停止します。出力JSONには期間・件数・入力の指紋・作成日時と、
**校正学習データ上の指標**を記録します。この指標は独立した評価の成績ではありません。

複数定義を格納するJSON形式2を利用できます。`--include-existing`を指定すれば、
既存ファイル内の他方式を保持して新しい出力へ統合します。旧単一定義のJSON形式1も利用可能です。
利用を決めた補正定義を対象モデルの`calibration.json`へ配置します。
`none`を指定している限り、sigmoidのJSONがあっても補正は有効になりません。
モデルID・タスク・目的変数・内容IDが合わないファイルは、無補正へ戻さず拒否します。

## 予想での明示指定

Pythonの`predict_race`・`predict_race_date`・`predict_historical_race`は
キーワード引数`calibration_method="class_weight"`または`"sigmoid"`で切り替えます。
`src.public_api`経由でも同じ指定を利用できます。

CLIの1レース・日別・日次ワークフローも`--calibration-method`を受け取ります。

```powershell
.venv\Scripts\python.exe -m src.10_workflows.race_prediction --bundle prediction_bundle --race-date 2026-10-18 --race-id 202605040611 --output data/calibration_trial --calibration-method class_weight
```

上記の日付・IDは書式例です。実行時は対象レースと新しい出力先を指定します。
Actionsの定期ワークフローは変更しておらず、既定の`none`を維持します。

## バンドルと互換性

`calibration.json`があるモデルは形式2のバンドルとして収録し、SHA-256を検証します。
既存の形式1も読み込めます。旧形式は補正定義を含まず、既定の`none`で利用します。
新規学習モデルで必須と記録された補正ファイルが欠落している場合は、生成・読込を停止します。
既存の追跡済みバンドルや公開済み予想は、この実装だけでは再生成・変更しません。

既存モデル`model_20260829_214945_ef2111`には、今回、過去の検証に使った39,817頭の
学習外予測からsigmoidを作成し、3方式の定義を追加しています。重みは保存済み学習期間の
元データから復元し、学習行数と保存予測の全正解を照合した来歴を記録しています。
`scripts/prepare_calibration_comparison.py`が準備処理です。実行用バンドルは形式2へ再生成済みで、
公開方式は引き続き`none`です。過去の公開HTMLは再生成していません。

NaN・無限大・範囲外の生確率、不正係数、二重適用、校正期間と予想対象日の重複を拒否します。
補正は確率の過大評価に対処する機能です。券種近似や選定後の誤差・回収率の改善は別に検証します。

## オッズと全候補の監査記録

予想の保存時に、同じ入力オッズで3方式の採否を記録します。公開する補正方式や
買い目の閾値は変更しません。以下は各レースの`calibration/`配下に保存します。

| ファイル | 内容 |
|---|---|
| `odds.csv` | 実際に予想へ渡したオッズ全行。上下限・取得情報・DB登録情報を含む |
| `audit_odds.csv` | 上記に、取得時には存在したがDB登録前に外された行も追加した監査対象 |
| `snapshot.json` | 入力オッズのSHA-256とID、監査対象のSHA-256、判断日時、選定設定と設定ID |
| `{method}/decisions.csv` | 全候補の確率・使用オッズ・期待回収率・閾値・順位・採否・理由・モデルID・校正ID |
| `archives/{ID}.zip` | 3方式の予測・買い目・採否・オッズ・設定をまとめた圧縮保存。日次取得HTMLと登録前オッズも含む |

`candidate_id`と`odds_row_index`から、`audit_odds.csv`の元行へ戻れます。
`odds_snapshot_id`は予想入力CSVのバイト列から作成するため、3方式で共通です。
設定IDには閾値・券種別信頼度・おすすめ上限・順位規則・計算コードのハッシュを含めます。
ZIP内の`manifest.json`は、収録した各ファイルのSHA-256を記録します。
当時の買い目計算コードも`implementation/betting.py`として収録します。

主な除外理由は以下のとおりです。複数の閾値に達しないときは`|`で連結します。

- `low_probability`：券種別の最低確率に未達
- `low_expected_return`：最低期待回収率に未達
- `outside_top3`：閾値は通過したが、レースごとのおすすめ上位3点から外れた
- `missing_prediction` / `missing_frame_prediction`：馬番・枠番に必要な予測情報がない
- `invalid_odds` / `invalid_selection`：オッズまたは選択番号が不正
- `unsupported_bet_type`：計算対象外の券種
- `excluded_from_prediction_inputs`：取消等によりDB登録前に予想入力から外れた

未計算の確率・期待回収率は欠損値とし、0%と区別します。数値として解析できない
取消等の表示はCSV候補にならない場合もあるため、元HTMLもZIPに残します。

JRAオッズのHTTP取得時に、HTMLと同名の`.metadata.json`へUTC・タイムゾーン付きの
`fetched_at`、取得URL、リクエスト識別子、保存HTMLのハッシュを記録します。
キャッシュ再利用では更新せず、解析時にハッシュを照合します。
DBの`registered_at`は登録時刻として別に保存します。既存DBには取得情報の列を追加します。
取得情報がない古いHTMLの日時は推測せず、欠損値として保存します。

日次集約時はZIPも引き継ぎ、公開処理で
`docs/predictions/calibration/{開催日}/archives/{ID}.zip`へコピーします。
同日の再実行でも別IDの既存ZIPは保持し、既存のActionsのコミット対象に含めます。
これにより成功して公開された予想の監査資料はGit履歴へ保存されます。
容量の大きい`decisions.csv`とオッズCSVは公開先ではZIP内だけに保存します。
実行全体の診断Artifactの保持期間も3日から90日に延長しています。
過去の予想へ欠落情報を補完する処理は行いません。

## 購入対象・表示・精算の共通化

`src/00_common/purchase_plan.py`が購入対象の確定と検証を担当します。
予想時に、条件通過候補からおすすめスコア・期待回収率の順で各レース上位3点を選び、
既定では1点100円の購入対象を`recommended_bets.csv`へ保存します。
同順位では候補表の順序を維持するため、従来HTMLで表示していた選定と一致します。
`purchase_plan_version`・`purchase_rank`・`stake_yen`を付けて、選定済みであることを明示します。
金額は正の100円単位に限定し、重複買い目・順位の重複・未対応の版番号は拒否します。

通常予想と3方式の比較は、この共通の選定処理を使います。モデルAPI経由でも
公開方式の購入対象を予想ファイルと同じフォルダに保存します。
HTMLとアプリは保存済みの対象・金額を使い、スコアが変わっても並べ直して選定し直しません。
アプリ再表示時も現在のDBオッズで買い目を再計算しません。

日次公開ファイルの役割は次のとおりです。

| ファイル | 内容 |
|---|---|
| `{開催日}_candidates.csv` | 閾値を通過した候補一覧。回収率の購入対象には使わない |
| `{開催日}_bets.csv` | 予想時に確定した購入対象・購入額。HTMLと翌日の精算で共用 |
| `{開催日}_bets_results.csv` | 上記と同じ買い目の精算結果。`payout_per_100`は公式単価、`payout_yen`は購入額に応じた実払戻額 |

新形式の成果物で購入対象ファイルが欠けている場合は、公開を停止します。
精算では`公式100円当たり払戻金 × stake_yen / 100`を使い、返還も実際の購入額を戻します。
回収率は`実払戻合計 / 購入額合計 × 100`です。通常予想・補正比較・HTMLは
同じ精算関数・集計関数を使用します。返還を的中件数へ含めず、未確認があれば
確定回収率を出しません。見送りは0%と区別して「対象なし」と表示します。

従来の公開`_bets.csv`は候補全体だったため、旧形式だけ共通の上位3点ルールで読み込みます。
旧補正比較の`recommended_bets.csv`は既に選定済みなので、そのまま各100円として読み込みます。
前日照合開始時に解決した購入対象もファイル化してハッシュを保存し、公開前に変更を検出します。
アプリで古い予想に購入記録も候補ファイルもない場合は、その旨を表示して照合を保留します。
過去に公開したCSV・HTMLをまとめて書き換える処理は行いません。
