# 学習時のハイパーパラメーターをまとめた設定型。
# エポック数、バッチサイズ、学習率、ネットワーク規模、乱数シードなどを呼出側から渡す。
#
from dataclasses import dataclass


# MLPの学習条件。既定値を使う場合も、このオブジェクトの値を成果物へ記録する。
@dataclass
class TrainConfig:
    # 全学習データを繰り返し処理する最大回数。早期終了によりこれより短くなることがある。
    epochs: int = 80
    # 勾配更新一回にまとめて使う学習行数。大きさはメモリ使用量にも影響する。
    batch_size: int = 128
    # AdamWによるパラメーター更新の学習率。
    learning_rate: float = 0.001
    # 隠れ層の幅を決めるユニット数。モデル保存時にも構造の再現用として記録する。
    hidden_dim: int = 64
    # 学習中に無効化するユニットの割合。推論時はevalモードで無効化を止める。
    dropout: float = 0.25
    # 検証損失が十分に改善しないエポックを何回まで許容するか。
    patience: int = 12
    # NumPy・PyTorchの乱数初期化に使う値。学習の再現条件として保存する。
    random_seed: int = 42
