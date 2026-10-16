# 表形式の特徴量から二値分類のlogitを出力するPyTorchの多層パーセプトロン。
# forwardの出力は確率そのものではなく、学習損失や推論側のsigmoidと組み合わせて使う。
#
import torch
from torch import nn


# 入力次元数に応じた全結合ネットワーク。出力logitは損失関数またはsigmoidへ渡す。
class MLP(nn.Module):
    # 入力次元数に合わせて線形層・活性化・Dropout・二値出力の層を構成する。
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        second_dim = max(16, hidden_dim // 2)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, second_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(second_dim, 1),
        )

    # 入力テンソルをネットワークへ通し、一行一つの二値分類logitへ整形して返す。
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(1)
