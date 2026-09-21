# 開催日全レースの手動予想・公開

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
