# DB・モデルをGitHubの非公開リポジトリへ移す

GitHub Freeで利用する構成です。予想用ActionsとPagesは公開リポジトリ
`teru-hamada/horse_racing_ai` に残し、非公開リポジトリはファイル保管だけに使います。
非公開側でActions・Pages・Git LFSは使用しません。

## 現在の準備状況

非公開側への初回アップロードと、読み取り専用Deploy key・Actions設定が完了しています。
初回コミットは `38dd15387a7d9c257f9398a3beee59ac73b83e20` です。
GitHub Actionsで非公開checkout・復元・実DB・CPU推論・Pages配信の成功を確認しました。
検証実行: https://github.com/teru-hamada/horse_racing_ai/actions/runs/39754792780
公開側の最新コミットでは生成6ファイルの追跡を解除し、ローカルの元ファイルは保持します。
過去の公開履歴からの除去と既存の診断Artifactの確認・削除は別途必要です。

## 1. 非公開側を作る

GitHubで `teru-hamada/horse-racing-ai-private-data` を **Private** として作成します。
READMEなどは自動生成しない空のリポジトリにしてください。
以下の移行用コピーは `.gitignore` で除外された `.private-runtime/` に作ります。
元のDB・モデルは変更しません。DBを書き込むアプリを止めてから実行します。

```powershell
.venv/Scripts/python.exe scripts/private_runtime.py prepare --format legacy
git -C .private-runtime init -b main
git -C .private-runtime add -- .gitattributes README.md data prediction_bundle
git -C .private-runtime commit -m "Store private prediction runtime"
git -C .private-runtime remote add origin https://github.com/teru-hamada/horse-racing-ai-private-data.git
git -C .private-runtime push -u origin main
git -C .private-runtime rev-parse HEAD
```

GitHubへの認証が必要です。パスワードの代わりにGit Credential Managerなどの
認証画面を使用してください。最後に表示される40文字のコミットSHAを次の設定で使います。
初回コピーをすでに作成済みなら `prepare` の再実行は不要です。

## 2. 公開側のActionsを設定する

認証には非公開リポジトリ専用の読み取り専用SSH Deploy keyを使用しています。
個人用トークンは使いません。非公開側のSettings → Deploy keysに公開鍵を登録し、
Allow write accessはオフにします。秘密鍵は公開側の
Settings → Secrets and variables → ActionsへSecretとして登録します。
初回設定は実施済みです。秘密鍵をチャットやコードに貼らないでください。

| 種類 | 名前 | 値 |
| --- | --- | --- |
| Secret | `PRIVATE_DATA_SSH_KEY` | 読み取り専用Deploy keyの秘密鍵 |
| Variable | `PRIVATE_DATA_REPOSITORY` | `teru-hamada/horse-racing-ai-private-data` |
| Variable | `PRIVATE_DATA_REF` | 非公開側にpush済みの40文字コミットSHA |

ブランチ名ではなくSHAに固定します。非公開側の更新時には、確認したSHAへ変数も更新します。
SSHキーは非公開checkoutに使用し、後続のGit認証として保持しません。
公開リポジトリのコード・Actionsを変更できる人は非公開データにもアクセス可能になるため、
書き込み権限を持つ利用者を限定してください。

## 3. 新構成を確認する

設定後に公開側のActions変更をコミット・pushし、オッズが公開されている開催日で
「日次予想・結果照合・公開」を手動実行します。
非公開データのハッシュ・DB・実行環境・保存モデル・CPU推論の検査は毎回の予想前に実行します。
非公開checkout・復元・予想・結果照合・Pages配信を確認してください。
設定不足は停止させ、公開済みバンドルへの自動フォールバックは行いません。

予想結果と買い目・比較結果は今までどおり公開されます。
DBを含み得る `data/daily_racing/` 全体の診断Artifactはアップロードしません。
公開するArtifactはレース件数の小さいJSONだけです。過去の診断Artifactは別途確認・削除が必要です。
ジョブ状況ページは公開APIで取得できる情報だけをブラウザへ返します。

## 4. 動作確認後に公開側の追跡を解除する

次は作業ツリーのファイルを消さず、Gitの追跡だけを外す操作です。
この追跡解除は初回移行で実施済みです。コマンドは再移行時の参考として記載します。

```powershell
git rm --cached -- data/racing.duckdb data/features.duckdb prediction_bundle/runtime.zip prediction_bundle/manifest.json prediction_bundle/requirements.txt prediction_bundle/.python-version
```

その削除をコミット・pushした後、もう一度Actionsを実行し、公開リポジトリ内に
DBがない状態でも動くことを確認します。

過去のコミットからの除去は別の作業です。全ブランチ・タグと過去のファイル名を調査し、
バックアップを取ってから履歴書き換えを実施します。強制pushが必要で、ほかのクローンも
再取得などの対応が必要になるため、対象と影響を確認してから行います。
第三者が既に取得したコピーやforkは回収できません。

## モデル・DBを更新するとき

ローカルで学習・DB更新後、従来どおりバンドルを生成します。

```powershell
.venv/Scripts/python.exe -m src.40_ai_modeling.prediction_bundle
.venv/Scripts/python.exe scripts/private_runtime.py prepare --output .private-runtime/next
```

新しい分割スナップショットを非公開リポジトリに反映し、復元・推論の確認後に
SHAを `PRIVATE_DATA_REF` に登録します。公開側にDBやバンドルを再追加しないでください。
モデルの読み込みには信頼できる作成元のファイルだけを使用してください。

## 100 MiB制限に対応する分割保存

DBをローカルで使用するときの構成を維持したまま、保存時に圧縮して64 MiBずつに分割します。
`snapshot.json` に元ファイルと各partのハッシュを保存し、復元時に照合します。
Gitへ保存する各partは100 MiB未満に収まり、復元後のDBは100 MiBを超えても扱えます。
Actionsは新旧形式を自動判別します。本番参照SHAは従来形式のままです。
GUIからの保存・復元とActionsへの適用手順は[データ同期](private_data_sync.md)を参照してください。
