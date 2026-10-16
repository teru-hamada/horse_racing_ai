# JRA result regression fixtures

<!--
このフォルダーは、公式結果HTMLの解析を通信なしで再現するための固定データ。
202609040704.htmlは除外馬と返還、202609040706.htmlは競走中止を扱う回帰テストに使う。
馬番返還と「同じ枠同士の買い目だけ」の返還を区別できることが重要な検証点となる。
HTMLは処理コードではなくパーサーへの入力なので、説明用コメントはこのREADMEにまとめる。
フィクスチャを変更すると検証するサイト構造や公式情報も変わるため、実装変更とは分けて確認する。
-->

Extracted from the official desktop result HTML fetched on 2026-09-22 with
`JraResultFetcher`. Only result-table columns used by the parser and the official
payout/refund area are retained. Navigation and unrelated columns are omitted.

- `202609040704.html`: Hanshin, 2026-09-21, race 4. Horse 7 excluded;
  official refunds list horse 7 and the same-frame ticket 4-4 (not every pair with 4).
- `202609040706.html`: Hanshin, 2026-09-21, race 6. Horse 15 did not finish;
  no refund notice.

Official incident report: https://jra.jp/news/202609/092103.html
Refund rules: https://jra.jp/dento/soku/instructions.html
