# 検証対象：Top3タスクの登録情報と、未知のタスク名の明示的な拒否を確認する。
#
from src.public_api import MODEL_TASKS, get_model_task


# Top3の目的変数と推定器が登録定義どおりに解決されることを確認する。
def test_top3_task_is_registered():
    task = get_model_task("top3")
    assert "top3" in MODEL_TASKS
    assert task.target_column == "target_top3"
    assert task.estimator_name == "mlp"


# 未登録のタスク名をデフォルトへ置き換えず、理由を持つ例外として拒否する。
def test_unknown_task_is_rejected():
    try:
        get_model_task("unknown")
    except ValueError as exc:
        assert "unknown" in str(exc)
    else:
        raise AssertionError("未対応タスクが受理されました。")
