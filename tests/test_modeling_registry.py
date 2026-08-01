from src.public_api import MODEL_TASKS, get_model_task


def test_top3_task_is_registered():
    task = get_model_task("top3")
    assert "top3" in MODEL_TASKS
    assert task.target_column == "target_top3"
    assert task.estimator_name == "mlp"


def test_unknown_task_is_rejected():
    try:
        get_model_task("unknown")
    except ValueError as exc:
        assert "unknown" in str(exc)
    else:
        raise AssertionError("未対応タスクが受理されました。")
