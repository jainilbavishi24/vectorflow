import torch
import torch.nn as nn


class CustomDecoder(nn.Module):
    """
    Trajectory decoder stub — a single linear layer that maps noised trajectory tokens to
    predicted tokens, ignoring scene conditioning entirely. Replace with the real DiT.

    ── INPUT CONTRACT ──

        x:  (B, action_num, action_len, state_dim)
            Noised trajectory split into overlapping action chunks.
            action_num  = (future_len - action_overlap) / (action_len - action_overlap) = 7
            action_len  = 20  (2 seconds @ 10 Hz)
            state_dim   = 4   (x, y, cos_heading, sin_heading)

            NOTE: during CFG inference the ODE solver internally doubles the batch before
            calling the decoder, so x may arrive as (2B, action_num, action_len, state_dim).
            A plain nn.Linear handles both batch sizes transparently.

        t:  (B,) during training | scalar during ODE inference
            Flow-matching timestep ∈ [0, 1].  t≈0 is pure noise, t≈1 is clean data.

        **model_extra:
            Everything returned by CustomEncoder.forward() plus 'cfg_flags' injected by
            VectorFlowPlanner:
                agent_tokens:  (B, 37, encoder_hidden_dim)
                lane_tokens:   (B, 70, encoder_hidden_dim)
                cfg_flags:     (B,)  — 1 = conditioned, 0 = unconditioned (for CFG)

    ── OUTPUT CONTRACT ──

        Same shape as x: (B, action_num, action_len, state_dim)
        The flow-matching ODE solver uses this as a velocity/target prediction.
    """

    def __init__(self, action_len: int, state_dim: int, **kwargs):
        super().__init__()
        flat_dim = action_len * state_dim  # 20 * 4 = 80
        self.action_len = action_len
        self.state_dim = state_dim

        # TODO: replace with real DiT
        # Stub: single linear layer — adds learnable noise to the input so loss is non-trivial
        self.linear = nn.Linear(flat_dim, flat_dim)

        # kwargs contains all extra YAML params passed from custom_model.yaml
        # (encoder_hidden_dim, neighbor_num, static_num, lane_num, future_len, action_overlap, etc.)
        # Store anything the real DiT constructor will need:
        # e.g. self.hidden_dim = kwargs.get('hidden_dim', 256)

    def forward(self, x, t, **model_extra):
        """
        x:           (B, action_num, action_len, state_dim)
        t:           (B,) or scalar
        model_extra: scene conditioning from encoder + cfg_flags

        TODO: use t and model_extra in the real DiT forward pass.
        """
        B, P, action_len, state_dim = x.shape

        # Stub: linear projection on flattened action tokens
        out = self.linear(x.reshape(B, P, -1))          # (B, P, action_len * state_dim)
        return out.reshape(B, P, action_len, state_dim)  # (B, P, action_len, state_dim)
