# 競馬予想AIシステム（私的利用向けMVP）

指定日のレース情報収集、個別保存、ニューラルネットワーク学習、過学習グラフ、週末レース予想、処理ログ表示をStreamlitのGUIから実行するサンプルです。


## 開催日全レースの手動予想・公開

Actionsは **Publish predictions for a date** に統合しました。
旧HTML取得・DB登録・1レース予想の3ワークフローは削除しています。
以前の日付単位テストは、この公開ワークフローへ置き換えました。
HTML・DB・モデルの検証関数は本番処理でも使用するため残しています。
不要になった検証専用requirementsファイル2本は削除しました。

## GitHubでの操作

1. 今回の変更（旧ワークフローの削除を含む）をデフォルトブランチへ反映します。
2. **Actions → Publish predictions for a date → Run workflow** を開き、デフォルトブランチを選択します。
3. `race_date` に開催日（YYYY-MM-DD）を指定します。レースIDの入力は不要です。
4. Summaryのレース別一覧で、出馬表・オッズ・DB・予想・照合・買い目計算を確認します。
5. Artifactsの `date-prediction-smoke-…` をダウンロード・展開し、直下の `index.html` を開きます。

開催一覧から検出した中央競馬の全レースを順番に処理します。1レースが失敗しても
残りを処理します。取得間隔は既存の2秒、Actionsの時間制限は90分です。
検出対象は取得元の開催一覧に依存するため、検出レース数・会場も確認してください。
一覧が0件の場合は非開催日か取得不良かを区別できないため、成功にはしません。

## 結果の読み方

- `ok`: 検証成功。全体がokなら検出した全レースが成功（終了コード0）。
- `incomplete`: 馬番欠損・オッズ未発売等の不足あり（終了コード2）。
- `error`: 取得・登録・予想等の例外あり（終了コード1）。
- `skipped`: 前段が未合格のため、その工程を未実施。
- `pending`: 未処理。時間制限やキャンセル時の途中一覧に残る場合があります。

Artifactsの予想CSV・予想ページは**成功したレースのみ**を掲載します。不足・失敗のある日に
完成済みの全レース予想と誤認しないよう、最初に直下の `index.html` または
`race_status.csv` を確認してください。発売前・終了済みレースはオッズ不足になる場合があります。
買い目が推奨条件を満たさず0件でも、計算が正常に完了すれば成功です。

## Artifactsの内容（3日間保存）

- `index.html`: 全体結果、レース別一覧、成功分の予想ページへのリンク
- `date_report.json` / `race_status.csv`: 工程別結果、件数、不足・失敗理由、使用モデルID
- `predictions.csv` / `bets.csv`: 成功分を結合した予想・買い目（成功レースがある場合）
- `preview/`: 成功分をまとめた確認用HTML
- `races/<レースID>/`: レース別CSV・診断JSON・ログ・取得HTML・再現用DB
- `discovery/` / `date.log`: 開催一覧の保存HTML・全体ログ

一覧は各レースの処理後に保存します。展開した作業用モデル・過去DBはArtifactsから除外します。
既存DB・モデルは変更しません。定期実行はまだ設定していません。

## 公開条件と更新対象

検出した全レースが全工程に成功した場合だけ、公開用CSVと検証結果の対象日・レース・頭数・
モデル・予測確率を再確認し、以下をデフォルトブランチへcommit・pushします。

- `docs/predictions/<開催日>.html`: 予想ページ
- `docs/predictions/<開催日>.csv`: 全頭予想（モデルIDを含む）
- `docs/predictions/<開催日>_bets.csv`: 推奨買い目
- `docs/index.html` と `docs/.nojekyll`

他の開催日のページは保持します。生成日時だけが変わった場合は既存HTMLを保持し、
ファイルに差分がなければコミットしません。途中の不足・失敗では公開ファイルの更新・
push・Pages公開をスキップし、Artifactsに診断結果を残します。

push後は既存の `pages.yml` を再利用して、確定したコミットのPages公開を直接実行します。
GITHUB_TOKENによるpushだけに公開起動を依存させません。差分なしでもPages公開は実行し、
前回push後の公開失敗から再実行で復旧できるようにしています。
公開処理前にリモート更新を取り込み、競合やpush拒否の場合は強制pushせず停止します。

リポジトリのSettings → PagesでSourceを **GitHub Actions** に設定してください。
デフォルトブランチへのActionsからのpushがブランチ保護等で禁止されている場合は、
リポジトリ側の許可設定が必要です。ワークフローから保護ルールは変更しません。
Pagesのデプロイに失敗した場合、Gitへのコミットは残ります。Actionsのdeployジョブを
再実行して公開を再試行できます。全体の成功はdeployジョブまで確認してください。

## モデル更新とローカル比較

Actionsが参照する登録DBの学習成功済み最新モデルを使用します。モデル本体・前処理器・
過去レース・速度指数を `prediction_bundle/` から展開し、ID・ハッシュ・実行環境を照合します。
欠落時に古いモデルへ切り替えません。ローカルの未反映モデルは参照できません。
更新時は[パッケージ手順](prediction_bundle/README.md)に従って再生成し、登録DBと一緒に反映します。
Pythonは `prediction_bundle/.python-version`（現在3.12）を使用します。

ローカルで日付単位に実行する例:

```powershell
.venv/Scripts/python.exe -m src.date_prediction_smoke --race-date 2026-09-21 --output data/date_prediction_trial1
```

Actionsと同じコミット・実行パッケージを使い、Artifactを `data/actions_date` へ展開すれば、
成功したレースを通信なしで再現・比較できます。以下のレースIDは実際の成功レースに置換してください。

```powershell
py -3.12 -m venv .venv-prediction
.venv-prediction/Scripts/python.exe -m pip install -r prediction_bundle/requirements.txt
.venv-prediction/Scripts/python.exe -m src.prediction_smoke --bundle prediction_bundle --replay data/actions_date/races/202609040701 --output data/prediction_replay1
```

入力DB・パッケージのハッシュを確認後、予想・買い目を比較します。
`prediction_report.json` の `comparison.status: ok` が一致です。
浮動小数点の比較は絶対許容誤差1e-6・相対許容誤差1e-5を使用します。
出力先は毎回新しいフォルダを指定してください。
1レース予想・HTML取得・DB登録のPythonコマンドは診断用として残しています。

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
