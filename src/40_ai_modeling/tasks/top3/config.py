# Top3タスクの識別名、推定器名、モデル・特徴量の版、目的変数列を定義する。
# 成果物と登録DBで同じ識別情報を利用し、異なる構成のモデルを取り違えないようにする。
#
TASK_NAME = "top3"
TASK_LABEL = "3着以内確率"
# 3着内の正解ラベルを格納する列名。推論時の入力には含めない。
TARGET_COLUMN = "target_top3"
# 学習器の種類を表す識別名。
ESTIMATOR_NAME = "mlp"
# モデル実装・構成の版。保存成果物の来歴を識別する。
MODEL_VERSION = "3"
# タスクの特徴量構成を記録する版情報。
FEATURE_VERSION = "feature-db-1.1.0"
