# ─── world_model.py ─────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler
try:
    # PyTorch >= 2.0 provides a device agnostic autocast under torch.amp
    from torch.amp import autocast as _autocast
    _AUTOCAST_NEEDS_DEVICE = True
except ImportError:
    # Fallback for older PyTorch versions
    from torch.cuda.amp import autocast as _autocast
    _AUTOCAST_NEEDS_DEVICE = False
from torch.nn.utils import clip_grad_norm_
from typing import List

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _autocast_ctx(device_type: str, use_amp: bool):
    """
    Return an autocast context manager compatible with the installed PyTorch.
    """
    if _AUTOCAST_NEEDS_DEVICE:
        return _autocast(device_type=device_type, enabled=use_amp)
    return _autocast(enabled=use_amp)

class WorldModel(nn.Module):
    """
    A small MLP that takes [state || choice_emb] as input and
    predicts the next-state delta.
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

    def compile(self) -> nn.Module:
        """
        Optionally compile the model with TorchScript for performance.
        """
        self.scripted = torch.jit.script(self)
        return self.scripted

    def prepare_input(self, state: List[float], choice_emb: torch.Tensor) -> torch.Tensor:
        """
        Convert a Python state list and a precomputed choice embedding
        into a single input tensor for the world model.
        """
        # state: List[float] -> tensor [1, state_dim]
        st = torch.tensor(state, device=device).unsqueeze(0)
        # choice_emb: Tensor [1, choice_emb_dim]
        return torch.cat([st, choice_emb.to(device)], dim=1)

    def post_process(self, state: List[float], delta: torch.Tensor) -> List[float]:
        """
        Apply the predicted delta to the old state and return new state list.
        """
        new_state = torch.tensor(state, device=device) + delta.squeeze(0)
        return new_state.tolist()

    def estimate_reward(self, state: List[float]) -> float:
        """
        Estimate a reward for a given state using the survival composite score.
        """
        from health import SystemHealth, Survival
        raw = SystemHealth.check()
        return Survival.score(raw)['composite']

class WorldModelTrainer:
    """
    Wraps a WorldModel instance with an optimizer and optional AMP scaler.
    Provides a single `train_step` method.
    """
    def __init__(
        self,
        model: WorldModel,
        lr: float = 1e-3,
        grad_clip: float | None = None
    ):
        # Move model to device
        self.model = model.to(device)
        self.device = device

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.grad_clip = grad_clip

        # AMP setup
        use_amp = (self.device.type == "cuda")
        self.scaler = GradScaler(
            init_scale=2**16,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000,
            enabled=use_amp
        )
        self.autocast_ctx = lambda: _autocast_ctx(self.device.type, use_amp)

    def train_step(
        self,
        state_batch: torch.Tensor,
        choice_emb_batch: torch.Tensor,
        actual_delta_batch: torch.Tensor
    ) -> float:
        """
        Perform one optimization step on the world model.
        Args:
          state_batch        Tensor[B, state_dim]
          choice_emb_batch   Tensor[B, choice_emb_dim]
          actual_delta_batch Tensor[B, state_dim]
        Returns:
          MSE loss as Python float
        """
        # Move data to device
        state_batch      = state_batch.to(self.device)
        choice_emb_batch = choice_emb_batch.to(self.device)
        actual_delta_batch = actual_delta_batch.to(self.device)

        # Concatenate inputs
        model_input = torch.cat([state_batch, choice_emb_batch], dim=-1)

        # Forward + loss
        with self.autocast_ctx():
            pred_delta = self.model(model_input)
            loss = F.mse_loss(pred_delta, actual_delta_batch)

        # Backward + optimizer step
        self.optimizer.zero_grad()
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.grad_clip is not None:
                clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.grad_clip is not None:
                clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

        return loss.item()