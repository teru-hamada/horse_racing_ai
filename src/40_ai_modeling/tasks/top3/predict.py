# Top3タスクの予想関数を外部へ公開する入口。
# 実装はmodelモジュールへ集約し、呼出側が学習処理の配置に依存しないようにする。
#
from .model import predict_historical_race, predict_race

# このモジュールから明示的に公開する名前。実装内部の補助名と区別する。
__all__ = ["predict_race", "predict_historical_race"]
