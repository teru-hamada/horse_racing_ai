# 競馬予想AIシステム

レース情報の収集・DB登録・特徴量生成・ニューラルネットワーク学習・予想・確定結果との比較を、StreamlitのGUIから実行するプログラムです。学習曲線と過学習の確認、開催日単位の予想、買い目の確認、処理ログの表示も行えます。

GitHub Actionsでは毎日日本時間3:00に当日分の全レース予想・公開と、前日が開催日だった場合の結果照合を実行します。公開サイトから予想・照合結果とジョブ実行状況を確認できます。

## フォルダ構成

主なプログラムと保存先です。データ用フォルダ・ファイルは該当処理の実行時に作成されます。

```text
horse_racing_ai/
├─ app.py                          Streamlit GUI
├─ setup_windows.bat               Windowsの初期セットアップ
├─ run_app.bat                     GUI起動
├─ requirements.txt               ローカルGUI用の依存ライブラリ
├─ .github/workflows/
│  ├─ publish-predictions.yml      定期・手動の予想、前日照合、公開ファイルの登録
│  └─ pages.yml                    Pages配信とジョブ実行状況の生成
├─ src/
│  ├─ 00_common/                  設定、DB操作、ログ、HTML共通処理
│  ├─ 10_scrapers_html_collection/ HTML・JRAオッズ収集とジョブ管理
│  ├─ 20_scrapers_database_creation/ 保存HTMLの解析、DB登録、ジョブ管理
│  ├─ 30_ai_modeling/
│  │  ├─ common/                  前処理、評価、モデル保存
│  │  ├─ estimators/              ニューラルネットワーク（MLP）
│  │  ├─ feature_engineering/     特徴量生成・保存・鮮度管理
│  │  ├─ tasks/top3/              3着以内確率の学習・予想
│  │  ├─ betting.py               買い目と期待値計算
│  │  └─ registry.py・service.py  予測タスクの登録と呼び出し
│  ├─ public_api.py               GUIから各処理を呼ぶ窓口
│  ├─ daily_racing.py             当日予想・前日照合・公開準備の統括
│  ├─ race_calendar.py            開催日・非開催日の判定
│  ├─ date_prediction_smoke.py    全レース処理と工程別検証
│  ├─ prediction_smoke.py         1レースの予想・照合・再現検証
│  ├─ html_fetch_smoke.py         HTML取得と検証
│  ├─ database_smoke.py           隔離DBへの登録・再登録検証
│  ├─ prediction_bundle.py        Actions用モデル・データのパッケージ作成
│  ├─ publish_predictions.py      全レース検証と公開用ファイル更新
│  ├─ prediction_comparison.py    予想・買い目と確定結果の比較
│  ├─ jra_results.py              JRA確定結果・公式払戻金の取得
│  ├─ weather_forecast.py         GUIの馬場状態設定に使う天気予報
│  ├─ static_site.py              予想・比較HTMLの生成
│  └─ job_status.py               ジョブ実行状況HTMLの生成
├─ data/
│  ├─ racing.duckdb               レース、オッズ、モデル登録情報など
│  ├─ features.duckdb             特徴量DB
│  ├─ raw_html/
│  │  ├─ historical/             学習用HTML（年別）、血統（horse/）
│  │  ├─ upcoming/               予想用HTML（年別）、オッズ（年/odds/）、血統（horse/）
│  │  └─ jra/historical/         JRA結果HTML（年/result/）
│  ├─ predictions/               ローカルの実行単位の予想・比較結果
│  └─ daily_racing/              日次ジョブの予想・照合・診断
├─ models/<task_name>/<model_run_id>/ モデル、前処理器、学習履歴、評価指標
├─ prediction_bundle/             Actions用スナップショット・固定実行環境
│  ├─ runtime.zip                モデル、前処理器、登録DB、過去レース、速度指数
│  ├─ manifest.json              モデルID、ファイルハッシュ、実行環境
│  ├─ requirements.txt           固定ライブラリ版
│  └─ .python-version            Python版
├─ docs/
│  ├─ index.html                 開催日一覧
│  ├─ predictions/               開催日別HTML・公開用CSV
│  └─ job-status.html            Pages配信時に生成する実行状況（Git登録対象外）
├─ logs/                          ローカルの日次ログ
└─ tests/                         自動テスト
```

`data/racing.duckdb`・`data/features.duckdb`・`prediction_bundle/`・公開用の`docs/`はGit管理対象です。元の`models/`、収集HTML、ローカルの予想結果、ログはGit管理対象外です。各モデルの`manifest.json`にはタスク名・アルゴリズム・特徴量バージョン・学習期間などを記録します。

## ローカルでのプログラム実行方法（Windows）

1. Python **3.12**をインストールします。
2. リポジトリを取得し、`setup_windows.bat`を実行します。仮想環境`.venv`と必要なライブラリを準備します。
3. `run_app.bat`を実行し、ブラウザで`http://localhost:8501`を開きます。次回以降はこのバッチから起動できます。

PowerShellから操作する場合は、リポジトリ直下で実行します。

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

GUIのメニューは「ダッシュボード」「HTML収集（学習用）」「HTML収集（予想用）」「データベース作成」「前準備（特徴量エンジニアリング）」「モデル学習」「レース予想」「ログ・保存結果」「メンテナンス」です。

## 操作フロー

### 学習用データの準備とモデル学習

1. **HTML収集（学習用）**で過去レースの対象期間を指定し、結果・血統HTMLを取得します。
2. **データベース作成 → 取得済みHTMLから作成**で学習用HTMLを解析・登録します。「登録内容確認」で対象日・レース・出走馬を確認します。
3. **前準備（特徴量エンジニアリング）**で基礎近走成績、1走単位スピード指数、近走スピード成績、出馬表基本条件・馬場状態を生成します。近走スピード成績は1走単位スピード指数の生成後に作成します。
4. **モデル学習**で特徴量が最新であることを確認し、条件を設定して学習します。元データ更新後は先に特徴量を再生成してください。学習時には自動生成されません。
5. 同じ画面の**学習結果・過学習チェック**でLoss・AUCの推移、評価指標、過学習判定を確認し、必要に応じて再学習します。

### 開催日の予想と結果確認

1. **HTML収集（予想用）**で対象日の出馬表・血統・取得可能なJRAオッズを収集します。
2. **データベース作成 → 取得済みHTMLから作成**で予想用データを登録します。券種別オッズは`race_odds`テーブルへ保存します。
3. **レース予想 → 今後レースを予想**で使用モデルと開催日を選び、競馬場ごとの馬場状態を確認・設定します。
4. **この日の全レースを予想**を実行し、上位3頭・全頭の予想・おすすめ買い目を確認します。予想CSVと公開用HTMLが生成されます。
5. 結果確定後に**この日の全レースを確定結果と比較**を実行します。JRAの着順・公式払戻金で予想と買い目を比較し、HTMLにも反映します。

**過去レースで予想を検証**では保存済みの過去レースを対象に予想を確認できます。履歴とログは**ログ・保存結果**、データ削除は**メンテナンス**で扱います。HTML収集・DB作成・特徴量生成はバックグラウンドで実行され、GUIの画面移動後も継続します。

## 開催日一括予想とGitHub Pages

GUIの一括予想は、選択日にDBへ登録されている出馬表を対象にします。予想時に`docs/index.html`と`docs/predictions/<開催日>.html`が自動生成され、確定結果との比較後は同じHTMLが更新されます。

ローカルで生成したHTMLを公開する場合は、`docs/`の変更をコミットして`main`へpushします。`Deploy prediction pages`が配信します。日次ジョブでは全レース検証後に公開用CSVも生成します。GUIでHTMLを生成するだけでは、日次ジョブの前日照合に必要な公開用CSVは揃いません。

配信元は**Settings → Pages → Source → GitHub Actions**に設定します。日次ジョブには、Actionsからデフォルトブランチへの書き込み権限も必要です。公開先は通常`https://<GitHubユーザー名>.github.io/horse_racing_ai/`です。

配信時に開催日一覧・予想ページへ**ジョブ実行状況**リンクを追加します。実行状況ページは配信時に生成するため、ローカルで予想HTMLを生成しただけでは作成されません。

## 開催日全レースの予想・公開と前日結果照合

`Publish predictions for a date`が、開催確認、全レースのHTML取得、隔離DBへの登録、学習済みモデルによる予想、オッズ照合、買い目計算を行います。前日が開催日なら公開済みの予想・買い目を確定結果と照合します。ジョブ内でモデル学習は行いません。

当日予想と前日照合の必要な処理がすべて成功した場合だけ、公開ファイルを更新・commit・pushし、Pagesへ配信します。不足・失敗時は診断を残して今回の予想・照合結果の公開を止めます。実行状況は別途、公開済みの予想とともに配信します。

## GitHubでの手動実行

毎日日本時間3:00のスケジュールが登録されているため、通常は手動実行不要です。指定日の再実行や動作確認時に使用します。

1. **Actions → Publish predictions for a date → Run workflow**を開きます。
2. デフォルトブランチを選択します。
3. `race_date`へ対象開催日（`YYYY-MM-DD`）を入力します。レースIDは不要です。
4. 前日照合も行う場合は`compare_previous`をオンにします（手動実行の既定はオフ）。
5. **Run workflow**を押します。完了後、実行画面のSummaryとArtifacts、公開サイトを確認します。

## ジョブスケジュールの詳細

### 実行時刻と開催日の判定

`0 18 * * *`（UTC）で毎日、日本時間3:00に起動します。処理開始時の日本時間で当日・前日を決め、
当日予想と、前日が開催日だった場合の結果照合を行います。月曜開催・年/月をまたぐ日付にも対応します。
GitHub Actionsのスケジュールには遅延があり、厳密な3:00開始は保証されません。
設定をデフォルトブランチへ反映するとスケジュールが有効になります。

月間カレンダーの年・月・全日付・開催場の整合性を検証して、明示的に開催のない日だけを
`no_races`と判定します。HTTPエラー・カレンダー欠落・対象月不一致は非開催扱いにしません。
開催日と判定したのにレース一覧が0件の場合も不合格です。
両日とも非開催なら正常終了し、予想ファイルのcommit・pushは行いません。終了後の実行状況ページは更新対象です。

前日照合では公開済み予想・買い目CSVを使用し、モデルによる予想の作り直しは行いません。
前日の全レースと予想対象の一致、JRA確定着順・公式払戻金を確認し、前日ページへ照合結果を追加します。
前日が開催日なのに公開済みCSVがない場合もデータ不足として公開を止めます。
取消・中止等で数字の着順がない馬は現段階では手動確認扱いとして公開を止めます。
返還などを推測して払戻額を補いません。

当日予想と前日照合のどちらかに不足・失敗がある場合、もう一方の診断も行ったうえで、
両方の公開を止めます。3:00時点で必要なオッズが未取得の場合も、この条件を緩めません。

開催一覧から検出した中央競馬の全レースを順番に処理します。1レースが失敗しても残りを処理し、診断を残します。取得間隔は2秒、予想ジョブの時間制限は120分です。検出レース数・会場も確認してください。

### 結果の読み方と実行状況

- `ok`: 検証成功。全体がokなら検出した全レースが成功（終了コード0）。
- `no_races`: 完全な開催カレンダーで非開催と確認。正常終了の対象です。
- `incomplete`: 馬番欠損・オッズ未発売等の不足あり（終了コード2）。
- `error`: 取得・登録・予想等の例外あり（終了コード1）。
- `skipped`: 前段が未合格のため、その工程を未実施。
- `pending`: 未処理。時間制限やキャンセル時の途中一覧に残る場合があります。

Artifactsの予想CSV・予想ページは**成功したレースのみ**を掲載します。不足・失敗のある日に
完成済みの全レース予想と誤認しないよう、最初に `prediction/index.html` または
`prediction/race_status.csv` を確認してください。発売前・終了済みレースはオッズ不足になる場合があります。
買い目が推奨条件を満たさず0件でも、計算が正常に完了すれば成功です。

公開サイトの**ジョブ実行状況**には、最新100件までの実行日時（日本時間）・定期／手動・試行回数・成功／失敗／中止等を表示し、エラー詳細は掲載しません。成功には非開催日の正常終了を含みます。記録がない日は一覧上で実行を確認できない日であり、未実行とは断定しません。

表示はPages配信時点の記録です。予想ワークフロー終了後は失敗・中止・非開催日でも`Deploy prediction pages`が更新を試みます。API取得や配信自体に失敗した場合は公開済みの表示が残るため、ページの更新日時とActionsの結果を確認してください。

### Artifactsの内容（3日間保存）

Actionsの実行画面の**Artifacts → daily-racing-<実行ID>-<試行回数>**をダウンロード・展開します。処理の進行状況により一部のファイルは生成されない場合があります。

直下に `daily_report.json`（当日・前日の状態）、`daily.log`、公開準備時は `publication_report.json` を保存します。
`calendar/` に開催判定に用いたHTML、`comparison/` に前日の比較CSV・買い目的中/払戻CSV・JRA結果HTML・診断を保存します。
以下の当日予想ファイルは `prediction/` 内です。当日非開催時は生成しません。

- `index.html`: 全体結果、レース別一覧、成功分の予想ページへのリンク
- `date_report.json` / `race_status.csv`: 工程別結果、件数、不足・失敗理由、使用モデルID
- `predictions.csv` / `bets.csv`: 成功分を結合した予想・買い目（成功レースがある場合）
- `preview/`: 成功分をまとめた確認用HTML
- `races/<レースID>/`: レース別CSV・診断JSON・ログ・取得HTML・再現用DB
- `discovery/` / `date.log`: 開催一覧の保存HTML・全体ログ

一覧は各レースの処理後に保存します。展開した作業用モデル・過去DBはArtifactsから除外します。
既存DB・モデルは変更しません。

### 公開条件と更新対象

当日予想と前日照合の必要な処理がすべて成功した場合だけ、公開用CSVと検証結果の対象日・レース・頭数・
モデル・予測確率を再確認し、以下をデフォルトブランチへcommit・pushします。

- `docs/predictions/<開催日>.html`: 予想ページ
- `docs/predictions/<開催日>.csv`: 全頭予想（モデルIDを含む）
- `docs/predictions/<開催日>_bets.csv`: 推奨買い目
- `docs/index.html` と `docs/.nojekyll`

前日照合時は `docs/predictions/<前日>.html` と `<前日>_comparison.csv`、
`<前日>_bets_results.csv` を更新します。元の予想CSV・買い目CSVは保持します。

他の開催日のページは保持します。生成日時だけが変わった場合は既存HTMLを保持し、
ファイルに差分がなければコミットしません。途中の不足・失敗では公開ファイルの更新・
pushをスキップし、Artifactsに診断結果を残します。実行状況の更新には、既にGitへ登録された公開用ファイルだけを使います。

push後は`pages.yml` を呼び出して、確定したコミットのPages公開を直接実行します。
GITHUB_TOKENによるpushだけに公開起動を依存させません。差分なしでもPages公開は実行し、
前回push後の公開失敗から再実行で復旧できるようにしています。
公開処理前にリモート更新を取り込み、競合やpush拒否の場合は強制pushせず停止します。

リポジトリのSettings → PagesでSourceを **GitHub Actions** に設定してください。
デフォルトブランチへのActionsからのpushがブランチ保護等で禁止されている場合は、
リポジトリ側の許可設定が必要です。ワークフローから保護ルールは変更しません。
Pagesのデプロイに失敗した場合、Gitへのコミットは残ります。Actionsのdeployジョブを
再実行して公開を再試行できます。全体の成功はdeployジョブまで確認してください。

`docs/job-status.html`はPages配信時に生成し、Gitへはcommitしません。

### モデル更新とローカル比較

Actionsが参照する登録DBの学習成功済み最新モデルを使用します。モデル本体・前処理器・
過去レース・速度指数を `prediction_bundle/` から展開し、ID・ハッシュ・実行環境を照合します。
欠落時に古いモデルへ切り替えません。ローカルの未反映モデルは参照できません。
更新時は学習・DB更新を完了し、DBへ書き込む処理を止めてから再生成します。

```powershell
.venv/Scripts/python.exe -m src.prediction_bundle
```

生成した`runtime.zip`・`manifest.json`・`requirements.txt`・`.python-version`と更新した`data/racing.duckdb`を同じコミットで反映します。詳細は[パッケージ手順](prediction_bundle/README.md)を参照してください。
Pythonは `prediction_bundle/.python-version`（現在3.12）を使用します。

Actionsと同じ依存ライブラリを専用環境へ入れ、ローカルで日付単位に実行する例:

```powershell
py -3.12 -m venv .venv-prediction
.venv-prediction/Scripts/python.exe -m pip install -r prediction_bundle/requirements.txt
.venv-prediction/Scripts/python.exe -m src.daily_racing --race-date 2026-09-21 --with-previous --output data/daily_trial1
```

日付は対象開催日に置き換えてください。このコマンドは指定フォルダへ結果を保存し、自動push・Pages配信は行いません。

Actionsと同じコミット・実行パッケージを使い、Artifactを `data/actions_date` へ展開すれば、
成功したレースを通信なしで再現・比較できます。以下のレースIDは実際の成功レースに置換してください。

```powershell
.venv-prediction/Scripts/python.exe -m src.prediction_smoke --bundle prediction_bundle --replay data/actions_date/prediction/races/202609040701 --output data/prediction_replay1
```

入力DB・パッケージのハッシュを確認後、予想・買い目を比較します。
`prediction_report.json` の `comparison.status: ok` が一致です。
浮動小数点の比較は絶対許容誤差1e-6・相対許容誤差1e-5を使用します。
出力先は毎回新しいフォルダを指定してください。

## 予測内容とデータの扱い

各出走馬が**3着以内に入る確率**をPyTorchのMLPで予測します。出馬表の基本条件・馬場状態、近走成績、スピード成績などから特徴量を作ります。JRA券種別オッズは独立して保存し、推論後の買い目別期待値計算に使用します。

開催日順に学習・検証・テストへ分割し、Early StoppingとLoss／AUCの推移で過学習を確認します。予想対象レースより前の履歴を使って特徴量を生成します。

HTML収集とDB作成は分離しており、DB作成では保存済みHTMLだけを解析します。取得元のHTML構造変更やオッズ公開状況による不足・解析エラーは、保存HTMLと処理ログで確認してください。

本プログラムは研究・学習用です。予想の的中や利益を保証するものではありません。
