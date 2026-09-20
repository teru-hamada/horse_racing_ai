# 競馬予想AIシステム（私的利用向けMVP）

指定日のレース情報収集、個別保存、ニューラルネットワーク学習、過学習グラフ、週末レース予想、処理ログ表示をStreamlitのGUIから実行するサンプルです。

## GitHub Actionsで1レースのHTML取得を確認する

定期運用の第1段階として、手動実行専用の `Test one race HTML fetch`
（`.github/workflows/html-fetch-smoke.yml`）を用意しています。
DB登録・モデル推論・Gitへのpush・Pages公開は行いません。

1. この変更をGitHubのデフォルトブランチへ反映します。
2. リポジトリの **Actions → Test one race HTML fetch → Run workflow** を開きます。
3. `race_date` に開催日（例: `2026-09-21`）、`race_id` にnetkeiba出馬表URLの
   `race_id=`に続く12桁（例: `202609040701`）を入力します。
   例の日付に固定せず、実行時点で出馬表とJRAオッズが公開されているレースを選んでください。
4. 実行後、SummaryのJSONと **Artifacts → html-fetch-smoke-…** を確認します。
   HTML・解析CSV・ログ・`report.json`・`summary.md`は3日間保存します。
   失敗した場合も、取得できたファイルは保存します。

出馬表はキャッシュを使わず取得し、既存処理で単勝オッズ補完と解析を行います。
JRA側は開催一覧を経由し、指定した1レースの取得可能な券種を収集します。
血統HTMLや他レースの出馬表は取得しません。通信間隔は既存の2秒です。

- `ok`（終了コード0）: 全頭の馬番・枠番が有効で馬番の重複がなく、JRAオッズを1件以上解析できた。
- `incomplete`（終了コード2）: 馬番・枠番が不足、または有効なJRAオッズを解析できなかった。
  未発売・公開期間外・日付の指定違い・HTML構造変更などを保存HTMLで確認してください。
- `error`（終了コード1）: 取得または解析で例外が発生。ログでHTTPエラー等を確認してください。

`card_odds`は出馬表側の単勝オッズ件数、`jra_odds.rows`はJRA独立オッズの件数です。
今回の合格条件は接続・基本項目の確認までで、全券種・全頭分のオッズ充足や、
入力日と出馬表の実開催日の一致は保証しません。出走取消馬などで項目が空の場合も
要確認として扱います。赤い実行結果だけでアクセス拒否と判断しないでください。

Ubuntu / Python 3.11で、HTML用依存関係のみをインストールします。
ワークフローの権限は `contents: read`、最大実行時間は15分です。
ローカルで同じ確認を行う場合は、リポジトリ直下で以下を実行します。
出力先には新しいフォルダを指定してください。

```powershell
python -m pip install -r requirements-html-smoke.txt
python -m src.html_fetch_smoke --race-date 2026-09-21 --race-id 202609040701 --output data/html_fetch_smoke_trial1
```

## GitHub ActionsでDB登録・照合を確認する（第2段階）

HTML取得テストが成功したら、**Actions → Test one race database registration → Run workflow**
から開催日とレースIDを指定してください。ワークフローを表示するには今回の変更を
デフォルトブランチへ反映します。第1段階のHTML取得テストも引き続き利用できます。

取得・解析した1レースを `data/database_smoke/smoke.duckdb` に登録し、読み戻した後に
同じデータを別の実行IDで再登録します。アプリと同じ保存・読込関数へ専用DBパスを明示して
使用するため、既存の `data/racing.duckdb` は更新しません。

Summaryの `database.checks` がすべて `true`、全体の `status` が `ok` なら合格です。

- 出馬表の登録頭数、券種別オッズ件数が解析結果と一致する。
- 馬番・枠番が有効で、馬番とオッズの買い目が重複しない。
- レースID・日付が指定値と一致し、レースIDによるオッズ読込ができる。
- レースID・馬番で全頭に有効な単勝オッズを照合できる。
- 2回目の登録後も件数・内容が変わらず、登録実行IDが更新される。

不合格時は `database.failed_checks` と `db_win_join.csv` を確認します。
`left_only` は単勝オッズが見つからない出走馬、`right_only` は出馬表に対応しないオッズです。
出走取消などで単勝オッズがない馬も要確認（`incomplete`、終了コード2）として扱います。
取得・DB登録時の例外は `error`（終了コード1）です。
入力日と実際の開催日の一致、全券種の網羅性は今回の検証対象外です。

Artifactsの `database-smoke-…` にHTML・ログ・診断JSON・専用DB・
`db_card.csv`・`db_odds.csv`・`db_win_join.csv`を3日間保存します。
モデル学習・予想作成・Gitへのpush・Pages公開は行いません。

ローカル実行例（出力先は毎回新しいフォルダを指定）:

```powershell
python -m pip install -r requirements-database-smoke.txt
python -m src.html_fetch_smoke --race-date 2026-09-21 --race-id 202609040701 --output data/database_smoke_trial1 --verify-db
```

## GitHub Actionsで1レースの予想を確認する（第3段階）

`prediction_bundle/` に最新モデル本体・前処理器・設定・過去データの
実行用パッケージを用意しています。更新手順は
[パッケージの説明](prediction_bundle/README.md)を参照してください。
過去レースに加え、予想特徴量に必要な速度指数も含めています。

1. コード・`.github/workflows/prediction-smoke.yml`・`prediction_bundle/`一式を
   デフォルトブランチへ反映します。
2. **Actions → Test one race prediction → Run workflow** を開きます。
3. 実行時点で出馬表とJRAオッズが公開されている開催日・レースIDを指定します。
4. Summaryの最後にある「1レース予想テスト」の `status: ok` を確認します。
   途中に表示されるHTML・DB検証の成功だけでは、予想テスト全体の成功ではありません。

実行開始時の登録DBにある学習成功済み最新モデルをパッケージと照合します。
入力不足・モデル欠落時に古いモデルへ切り替えることはありません。
モデルファイルのハッシュ・ID・実行ライブラリのバージョンも検証します。
GitHubからローカルPCの未反映モデルを参照することはできません。

予想は既存アプリと共通の推論処理を使用します。DBは展開した作業用コピーを使い、
対象日より前の過去レースから特徴量を作ります。頭数・対象馬・予測確率・単勝オッズとの
照合・買い目の数値を検査します。推奨条件を満たす買い目がない場合は、異常ではなく
「推奨条件を満たす買い目なし」と記録します。

Artifactsの `prediction-smoke-…` を3日間保存します。

- `prediction_report.json`: 使用モデルID、入力のハッシュ、環境、所要時間、検証結果
- `predictions.csv`: 全頭の予想とモデルID
- `prediction_odds.csv`: 予想と単勝オッズの照合結果
- `bets.csv`: 推奨条件を満たした買い目
- `preview/index.html`、`preview/predictions/<開催日>.html`: 確認用ページ
- `prediction.log`、`fetch/`: 取得ログ・保存HTML・解析結果・テスト用DB

失敗時にも保存できた診断ファイルは残します。大きな作業用モデル・過去DBはArtifactsに
含めず、同じコミットのパッケージから復元します。予想結果のGit push・Pages公開・
スケジュール起動はまだ行いません。

### 同一入力でローカル比較する

Actionsと同じコミットのコードとパッケージを使い、Artifactを
`data/actions_prediction`へ展開します（`prediction_report.json`が直下にある状態）。
モデルのライブラリ互換性を保つため、専用仮想環境を推奨します。

```powershell
python -m venv .venv-prediction
.venv-prediction/Scripts/python.exe -m pip install -r prediction_bundle/requirements.txt
.venv-prediction/Scripts/python.exe -m src.prediction_smoke --bundle prediction_bundle --replay data/actions_prediction --output data/prediction_replay1
```

再現時は通信せず、Artifact内の入力DBを読み、同じモデルで再計算します。
入力DBとパッケージのハッシュ一致を確認し、予想と買い目をCSV比較します。
`prediction_report.json` の `comparison.status: ok` が一致です。
浮動小数点の差は絶対許容誤差 `1e-6`、相対許容誤差 `1e-5` 以内を許容し、
対象馬・買い目・モデルIDなどの相違は不合格になります。
出力先には毎回新しいフォルダを指定してください。

## 最初の起動方法（Windows）

1. Python 3.11をインストールします。インストール時に「Add Python to PATH」を有効にします。
2. このフォルダを任意の場所へ展開します。
3. `setup_windows.bat` をダブルクリックします。
4. 完了後、`run_app.bat` をダブルクリックします。
5. ブラウザで `http://localhost:8501` が開きます。

手動起動は次のとおりです。

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## 最初に試す順番

1. **データベース作成** → 「デモデータを生成」
2. **モデル学習** → 既定値で学習
3. **学習評価** → Loss・AUCグラフと過学習判定を確認
4. **週末予想** → デモ週末レースを選択して予想
5. 動作確認後、**HTML収集（学習用／予想用）**でHTMLを取得し、**データベース作成**で解析・保存

## 保存場所

- `data/raw_html/historical/`: 過去レース・学習用の元HTML
- `data/raw_html/upcoming/`: 今後レース・予想用の元HTML
- `data/racing.duckdb`: 統合データベース
- `data/predictions/`: 実行単位の予想結果
- `models/`: モデル、前処理器、学習履歴、評価指標
- `logs/`: 日次ログ

各処理には一意の実行IDが付き、個々の結果を残します。

## 現在の予測内容

各出走馬が**3着以内に入る確率**を二値分類で予測します。モデルはPyTorchのMLPです。

主な特徴量：

- 競馬場、芝・ダート、距離、馬番、枠番、性齢、斤量
- 騎手ID、調教師ID、オッズ、人気、馬体重
- 過去出走数、過去勝率、過去3着内率、過去平均着順、前走からの日数

## 過学習・データリーク対策

- 開催日順に学習70%、検証15%、テスト15%へ分割
- Early Stoppingを実装
- 学習Loss／検証Loss、学習AUC／検証AUCをグラフ表示
- 過去成績特徴量は当該レースより前の結果だけを `shift()` で集計

実運用では、予想時点で確定していない結果・払戻・確定後情報を特徴量に入れないでください。オッズを使う場合も、予想を行う時刻と同じ条件で取得した値を学習側にそろえる必要があります。

## スクレイピングについて

HTML収集とデータベース作成を次のフォルダに分離しています。

- `src/10_scrapers_html_collection/`
  - `scrapers_html_collection_netkeiba.py`: netkeiba HTML収集の公開窓口
  - `html_collection_jobs.py`: HTML収集のバックグラウンド実行管理
- `src/20_scrapers_database_creation/`
  - `scrapers_database_creation_netkeiba.py`: 保存済みHTMLの解析窓口
  - `database_creation_jobs.py`: DB作成のバックグラウンド実行管理
  - `demo_data.py`: 動作確認用デモデータ生成
- `src/30_ai_modeling/`
  - `registry.py`: 利用可能な予測タスクの登録
  - `service.py`: 学習・予想をタスク名で呼び分ける統一窓口
  - `common/`: 特徴量、時系列分割、前処理、評価、保存形式、学習設定
  - `estimators/mlp.py`: MLPアルゴリズム
  - `tasks/top3/`: 3着以内確率の目的変数、学習、評価、予想
- `src/00_common/`
  - `config.py`: データ、HTML、DB、モデル、予想結果、ログの保存先定義
  - `netkeiba_common.py`: URL定義、保存先規則、HTML解析補助などの共通実装
  - `logging_utils.py`: ファイルと画面表示で共用するログ出力
  - `storage.py`: DuckDBのテーブル定義、保存、読込、実行履歴、集計
- `src/public_api.py`: アプリから用途別パッケージを通常のimport文で利用する公開窓口

学習済みモデルは予測タスクごとに
`models/<task_name>/<model_run_id>/` へ保存します。
各モデルの `manifest.json` にタスク名、アルゴリズム名、
モデル・特徴量バージョン、目的変数、学習期間を記録します。

- 開催日ページから12桁のレースIDを検出
- 過去結果は結果ページ、週末データは出馬表ページから取得
- 取得間隔は内部定数で2秒に固定
- HTML収集はバックグラウンドで実行され、メニュー移動後も継続
- 過去レースHTMLは `data/raw_html/historical/<年>/` に年単位でキャッシュ
- 競走馬の血統HTMLは `data/raw_html/historical/horse/` に `<horse_id>_<競走馬名>.html` 形式で保存し、父・母・母父をデータベースへ登録
- 予想用HTMLは `data/raw_html/upcoming/<年>/` に年単位でキャッシュ
- 予想用HTML収集では、取得可能なJRA券種別オッズHTMLも `data/raw_html/upcoming/<年>/odds/` に保存（取得できない場合は従来のレース情報のみ保存）
- 予想用のデータベース作成時にJRAオッズを独立した `race_odds` テーブルへ登録し、モデル推論後の買い目別期待値計算にだけ使用
- 予想と確定結果の比較ではJRA公式の結果HTMLを `data/raw_html/jra/historical/<年>/result/` に保存し、確定着順と公式払戻金を使用
- 予想対象馬の血統HTMLは `data/raw_html/upcoming/horse/` に保存し、学習用に同じ競走馬HTMLがあれば再利用
- データベース作成時はネットワークへアクセスせず、取得済みHTMLだけを解析
- 取得済みHTMLからのデータベース作成はバックグラウンドで実行され、画面移動後も継続
- 解析できないページはログを残し、他レースの処理を継続

WebサイトのHTML構造は変更される可能性があります。解析エラー時は `data/raw_html/historical/<年>/` または `data/raw_html/upcoming/<年>/` のHTMLとログを確認し、パーサーの列名候補やCSSセレクタを調整してください。

## AIを使って今後開発する際の進め方

1. **取得項目一覧を固める**  
   AIへ「このHTMLから、レースID、馬ID、騎手ID、着順などを抽出するテストを書いて」と依頼します。
2. **少量データでパーサーを検証する**  
   1開催日だけ取得し、欠損・重複・型を確認します。
3. **自動テストを追加する**  
   保存したHTMLをテスト用サンプルにし、サイトへアクセスせず解析を再現できるようにします。
4. **取得期間を段階的に広げる**  
   1日 → 1か月 → 1年の順で増やします。失敗URLの再実行機能を追加します。
5. **ベースラインモデルを比較する**  
   MLPだけでなく、ロジスティック回帰、LightGBM、CatBoostと比較します。
6. **評価指標を増やす**  
   AUCだけでなく、Calibration、開催月別成績、人気帯別成績、回収率を追加します。
7. **特徴量を追加する**  
   脚質、コース適性、距離適性、騎手・調教師成績、休養日数などを追加します。
8. **再現性を固定する**  
   学習データ期間、特徴量定義、乱数シード、ライブラリ版をモデルごとに保存します。
9. **週次運用を自動化する**  
   Windowsタスクスケジューラでデータ収集を実行し、GUIから学習・予想・評価を行います。

AIへ修正を依頼する際は、次の4点を同時に渡すと精度が上がります。

- 変更したいファイル
- 実行した操作
- 画面またはログに出たエラー全文
- 期待する処理結果

## 注意

このMVPは研究・学習用の土台です。馬券購入を推奨したり、利益を保証したりするものではありません。まずデモデータで全体動作を確認してから、実データのパーサー調整へ進んでください。

## 開催日一括予想とGitHub Pages

「レース予想」画面で開催日を選ぶと、その日に保存されている出馬表の全レースを一括予想できます。画面には各レースの予測上位3頭が一覧表示され、「全出走馬の予想を見る」から全頭を確認できます。

予想後に「GitHub Pages用HTMLを生成」を押すと、次の静的ファイルが作成・更新されます。

- `docs/index.html`: 開催日一覧
- `docs/predictions/YYYY-MM-DD.html`: 開催日別の予想結果

公開手順:

1. GitHubのリポジトリ設定で **Settings → Pages → Source** を **GitHub Actions** に設定します。
2. 生成された `docs/` の変更をコミットして `main` ブランチへpushします。
3. `.github/workflows/pages.yml` が静的ページを公開します。

公開先は通常 `https://<GitHubユーザー名>.github.io/horse_racing_ai/` です。予想データには個人情報や秘密情報を含めないでください。
