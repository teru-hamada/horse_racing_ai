# 学習タスク名と実装モジュール・目的変数・推定器の対応を管理する。
# 未知のタスク名は明示的に拒否し、呼出側で実装パスを組み立てる必要をなくす。
#
from __future__ import annotations

from dataclasses import dataclass


# タスク名、画面ラベル、推定器名、正解列の対応を保持する変更不可の登録情報。
@dataclass(frozen=True)
class TaskDefinition:
    # 登録・保存に使う内部名。画面ラベルとは別に扱う。
    name: str
    # 画面に表示するタスク名。内部の登録キーとは区別する。
    label: str
    # このタスクが利用する学習器の識別名。
    estimator_name: str
    # 正解ラベルの列名。入力特徴量Xへ含めず、目的変数yとして取り出す。
    target_column: str


# タスク名から実装メタデータへの対応。ここへの登録とサービス側の実装分岐を合わせる。
MODEL_TASKS = {
    "top3": TaskDefinition(
        name="top3",
        label="3着以内確率",
        estimator_name="mlp",
        target_column="target_top3",
    ),
}


# 登録済みのタスク定義を返す。未知の名前は対応タスク一覧付きの例外へ変換する。
def get_task(task_name: str) -> TaskDefinition:
    try:
        return MODEL_TASKS[task_name]
    except KeyError as exc:
        supported = ", ".join(MODEL_TASKS)
        raise ValueError(
            f"未対応の予測タスクです: {task_name} "
            f"(対応: {supported})"
        ) from exc
