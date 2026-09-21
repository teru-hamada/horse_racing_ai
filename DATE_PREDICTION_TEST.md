# 開催日全レースの予想・公開と前日結果照合

公開サイトの「ジョブ実行状況」リンクから、日本時間の実行日時・定期／手動・試行回数・成功／失敗などを確認できます。エラー詳細は掲載しません。最新100件までの記録をページ公開時点で表示します（自動更新画面ではありません）。記録がない日は一覧上で実行を確認できない日であり、未実行とは断定しません。成功には非開催日の正常終了も含みます。

予想ワークフローの終了時には、失敗・中止・非開催日でも **Deploy prediction pages** が実行状況を更新します。予想は既にGitへ公開済みのファイルを使い、失敗した今回の予想は公開しません。Actions APIの取得失敗時には、空の履歴を公開せず更新を失敗させます。この仕組みはワークフローをデフォルトブランチへ反映した後に有効になります。

Actionsは **Publish predictions for a date** に統合しました。
旧HTML取得・DB登録・1レース予想の3ワークフローは削除しています。
以前の日付単位テストは、この公開ワークフローへ置き換えました。
HTML・DB・モデルの検証関数は本番処理でも使用するため残しています。
不要になった検証専用requirementsファイル2本は削除しました。

## 毎日3:00 JSTの自動実行

`0 18 * * *`（UTC）で毎日、日本時間3:00に起動します。処理開始時の日本時間で当日・前日を決め、
当日予想と、前日が開催日だった場合の結果照合を行います。月曜開催・年/月をまたぐ日付にも対応します。
GitHub Actionsのスケジュールには遅延があり、厳密な3:00開始は保証されません。
設定をデフォルトブランチへ反映するとスケジュールが有効になります。

月間カレンダーの年・月・全日付・開催場の整合性を検証して、明示的に開催のない日だけを
`no_races`と判定します。HTTPエラー・カレンダー欠落・対象月不一致は非開催扱いにしません。
開催日と判定したのにレース一覧が0件の場合も不合格です。
両日とも非開催なら正常終了し、commit・push・Pages公開はスキップします。

前日照合では公開済み予想・買い目CSVを使用し、モデルによる予想の作り直しは行いません。
前日の全レースと予想対象の一致、JRA確定着順・公式払戻金を確認し、前日ページへ照合結果を追加します。
前日が開催日なのに公開済みCSVがない場合もデータ不足として公開を止めます。
取消・中止等で数字の着順がない馬は現段階では手動確認扱いとして公開を止めます。
返還などを推測して払戻額を補いません。

当日予想と前日照合のどちらかに不足・失敗がある場合、もう一方の診断も行ったうえで、
両方の公開を止めます。3:00時点で必要なオッズが未取得の場合も、この条件を緩めません。

## GitHubでの操作

1. 今回の変更（旧ワークフローの削除を含む）をデフォルトブランチへ反映します。
2. **Actions → Publish predictions for a date → Run workflow** を開き、デフォルトブランチを選択します。
3. `race_date` に開催日（YYYY-MM-DD）を指定します。レースIDの入力は不要です。
   前日照合も行う場合は `compare_previous` をオンにします（既定はオフ）。
4. Summaryのレース別一覧で、出馬表・オッズ・DB・予想・照合・買い目計算を確認します。
5. Artifactsの `daily-racing-…` をダウンロード・展開し、`daily_report.json` と `prediction/index.html` を確認します。

開催一覧から検出した中央競馬の全レースを順番に処理します。1レースが失敗しても
残りを処理します。取得間隔は既存の2秒、Actionsの時間制限は120分です。
検出対象は取得元の開催一覧に依存するため、検出レース数・会場も確認してください。
開催カレンダーで開催日と確認した後の一覧0件は、取得不良として扱います。

## 結果の読み方

- `ok`: 検証成功。全体がokなら検出した全レースが成功（終了コード0）。
- `no_races`: 完全な開催カレンダーで非開催と確認。正常終了の対象です。
- `incomplete`: 馬番欠損・オッズ未発売等の不足あり（終了コード2）。
- `error`: 取得・登録・予想等の例外あり（終了コード1）。
- `skipped`: 前段が未合格のため、その工程を未実施。
- `pending`: 未処理。時間制限やキャンセル時の途中一覧に残る場合があります。

Artifactsの予想CSV・予想ページは**成功したレースのみ**を掲載します。不足・失敗のある日に
完成済みの全レース予想と誤認しないよう、最初に `prediction/index.html` または
`race_status.csv` を確認してください。発売前・終了済みレースはオッズ不足になる場合があります。
買い目が推奨条件を満たさず0件でも、計算が正常に完了すれば成功です。

## Artifactsの内容（3日間保存）

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

## 公開条件と更新対象

検出した全レースが全工程に成功した場合だけ、公開用CSVと検証結果の対象日・レース・頭数・
モデル・予測確率を再確認し、以下をデフォルトブランチへcommit・pushします。

- `docs/predictions/<開催日>.html`: 予想ページ
- `docs/predictions/<開催日>.csv`: 全頭予想（モデルIDを含む）
- `docs/predictions/<開催日>_bets.csv`: 推奨買い目
- `docs/index.html` と `docs/.nojekyll`

前日照合時は `docs/predictions/<前日>.html` と `<前日>_comparison.csv`、
`<前日>_bets_results.csv` を更新します。元の予想CSV・買い目CSVは保持します。

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
.venv/Scripts/python.exe -m src.10_workflows.daily_racing --race-date 2026-09-21 --with-previous --output data/daily_trial1
```

Actionsと同じコミット・実行パッケージを使い、Artifactを `data/actions_date` へ展開すれば、
成功したレースを通信なしで再現・比較できます。以下のレースIDは実際の成功レースに置換してください。

```powershell
py -3.12 -m venv .venv-prediction
.venv-prediction/Scripts/python.exe -m pip install -r prediction_bundle/requirements.txt
.venv-prediction/Scripts/python.exe -m src.10_workflows.race_prediction --bundle prediction_bundle --replay data/actions_date/prediction/races/202609040701 --output data/prediction_replay1
```

入力DB・パッケージのハッシュを確認後、予想・買い目を比較します。
`prediction_report.json` の `comparison.status: ok` が一致です。
浮動小数点の比較は絶対許容誤差1e-6・相対許容誤差1e-5を使用します。
出力先は毎回新しいフォルダを指定してください。
1レース予想・HTML取得・DB登録のPythonコマンドは診断用として残しています。
