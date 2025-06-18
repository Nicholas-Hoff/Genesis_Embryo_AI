# ─── world_model.py ─────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler
try:
    # PyTorch >= 2.0 provides a device agnostic autocast under torch.amp which
    # requires a device_type argument.
    from torch.amp import autocast as _autocast  # type: ignore
    _AUTOCAST_NEEDS_DEVICE = True
except Exception:  # pragma: no cover - older PyTorch
    # Fallback for older versions where autocast lives under torch.cuda.amp and
    # does not accept a device_type parameter.
    from torch.cuda.amp import autocast as _autocast  # type: ignore
    _AUTOCAST_NEEDS_DEVICE = False
from torch.nn.utils import clip_grad_norm_

# 1) Device setup (if not already there)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WorldModel(nn.Module):
    """
    A small MLP that takes [state || choice_embedding] as input and
    predicts the next-state delta (state_dim).
    Input shape:  (B, state_dim + choice_emb_dim)
    Output shape: (B, state_dim)
    """
    def __init__(self, state_dim: int, choice_emb_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1   = nn.Linear(state_dim + choice_emb_dim, hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2   = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3   = nn.Linear(hidden_dim, state_dim)

        # Enable CuDNN autotuner when input sizes are fixed
        torch.backends.cudnn.benchmark = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)

    def compile(self):
        """
        If desired, call once at startup to JIT‐compile your model.
        """
        self.scripted = torch.jit.script(self)
        return self.scripted


class WorldModelTrainer:
    """
    Wraps an existing WorldModel instance + optimizer + (optional) AMP scaler.
    Exposes train_step(...) that takes exactly three arguments:
       1) state_batch:        Tensor[B, state_dim]
       2) choice_emb_batch:   Tensor[B, choice_emb_dim]
       3) actual_delta_batch: Tensor[B, state_dim]
    """

    def __init__(
        self,
        model: WorldModel,
        lr: float = 1e-3,
        grad_clip: float | None = None
    ):
        """
        Args:
          model (WorldModel): a pre-initialized WorldModel instance, already
                              sized for (state_dim, choice_emb_dim, hidden_dim).
          lr    (float):       learning rate for Adam.
        """

        # 1) We assume `model` is already on CPU or CUDA as desired.
        self.model = model.to(device)
        self.device = device

        # 2) Create an optimizer on the model’s parameters
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.grad_clip = grad_clip

        # 3) Mixed‐precision (AMP) setup if running on CUDA
        use_amp = (self.device.type == "cuda")
        self.scaler = GradScaler(
            init_scale=2**16,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000,
            enabled=use_amp
        )

        # 4) A context manager for autocast with version compatibility
        if _AUTOCAST_NEEDS_DEVICE:
            self.autocast_ctx = lambda: _autocast(device_type=self.device.type, enabled=use_amp)
        else:  # older torch
            self.autocast_ctx = lambda: _autocast(enabled=use_amp)

    def train_step(
        self,
        state_batch: torch.Tensor,
        choice_emb_batch: torch.Tensor,
        actual_delta_batch: torch.Tensor
    ) -> float:
        """
        Performs exactly one optimization step on the shared world model.

        Args:
          state_batch        (Tensor[B, state_dim]):       before‐state
          choice_emb_batch   (Tensor[B, choice_emb_dim]):  embedding of chosen action
          actual_delta_batch (Tensor[B, state_dim]):       actual next_state - state

        Returns:
          float: the MSE loss (as a Python float)
        """
        # 1) Move all inputs to self.device
        state_batch        = state_batch.to(self.device)
        choice_emb_batch   = choice_emb_batch.to(self.device)
        actual_delta_batch = actual_delta_batch.to(self.device)

        # 2) Concatenate along last dimension → shape (B, state_dim + choice_emb_dim)
        model_input = torch.cat([state_batch, choice_emb_batch], dim=-1)

        # 3) Mixed‐precision forward/backward if AMP is enabled
        with self.autocast_ctx():
            predicted_delta = self.model(model_input)
            loss = F.mse_loss(predicted_delta, actual_delta_batch)

        # 4) Zero grads, backward (AMP aware), optimizer step
        self.optimizer.zero_grad()
        if self.scaler.is_enabled():  # AMP path
            self.scaler.scale(loss).backward()
            # Unscale before clipping so grads are in fp32
            self.scaler.unscale_(self.optimizer)
            if self.grad_clip is not None:
                clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:  # regular fp32
            loss.backward()
            if self.grad_clip is not None:
                clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

        return loss.item()
