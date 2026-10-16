# Top3タスクの学習関数と設定型を公開する入口。
# ネットワークや保存処理の実装はmodelモジュールを参照し、ここでは重複定義しない。
#
from .model import load_model_bundle, train_model
from ...common.types import TrainConfig

# このモジュールから明示的に公開する名前。実装内部の補助名と区別する。
__all__ = ["TrainConfig", "train_model", "load_model_bundle"]
