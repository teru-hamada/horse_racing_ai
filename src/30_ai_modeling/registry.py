from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskDefinition:
    name: str
    label: str
    estimator_name: str
    target_column: str


MODEL_TASKS = {
    "top3": TaskDefinition(
        name="top3",
        label="3着以内確率",
        estimator_name="mlp",
        target_column="target_top3",
    ),
}


def get_task(task_name: str) -> TaskDefinition:
    try:
        return MODEL_TASKS[task_name]
    except KeyError as exc:
        supported = ", ".join(MODEL_TASKS)
        raise ValueError(
            f"未対応の予測タスクです: {task_name} "
            f"(対応: {supported})"
        ) from exc
