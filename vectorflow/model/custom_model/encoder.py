import torch
import torch.nn as nn


class CustomEncoder(nn.Module):
    """
    Scene encoder stub — outputs zero-filled tokens so the full pipeline runs before
    real architecture is written. Replace the forward() body with the real implementation.

    ── INPUT CONTRACT (fixed by VectorFlow's data pipeline — do NOT rename these args) ──

        neighbors:             (B, 32, 21, 11)  — 32 agents × 21 past timesteps × 11 features
                                                  features: [x, y, cos_h, sin_h, vx, vy, ax, ay, w, l, type]
        static:                (B,  5, 10)       — static obstacles
        lanes:                 (B, 70, 20, 12)   — 70 lane polylines × 20 points × 12 features
        lanes_speed_limit:     (B, 70)           — speed limit per lane (float)
        lanes_has_speed_limit: (B, 70)           — validity mask (bool)
        routes:                (B, 25, 20, 12)   — ego route lanes (subset of the 70 lanes above)

    ── OUTPUT CONTRACT (you define the keys; CustomDecoder must expect the same ones) ──

        Returns a dict that will be passed as **model_extra to CustomDecoder.forward().
        VectorFlowPlanner also injects 'cfg_flags' into this dict automatically.

        Current stub keys:
            agent_tokens:  (B, neighbor_num + static_num, encoder_hidden_dim)  = (B, 37, 192)
            lane_tokens:   (B, lane_num, encoder_hidden_dim)                   = (B, 70, 192)

        Add any other keys your DiT needs (e.g., route_cond, token_dist, masks).
    """

    def __init__(
        self,
        encoder_hidden_dim: int,
        neighbor_num: int,
        static_num: int,
        lane_num: int,
    ):
        super().__init__()
        self.encoder_hidden_dim = encoder_hidden_dim
        self.neighbor_num = neighbor_num
        self.static_num = static_num
        self.lane_num = lane_num

        # TODO: add real encoding layers here
        # e.g.:
        #   self.agent_encoder  = AgentEncoder(...)
        #   self.lane_encoder   = LaneEncoder(...)
        #   self.route_encoder  = RouteEncoder(...)

    def forward(self, neighbors, static, lanes, lanes_speed_limit, lanes_has_speed_limit, routes):
        B = neighbors.shape[0]
        device = neighbors.device

        # TODO: replace these zero tensors with real encoded scene tokens
        agent_tokens = torch.zeros(B, self.neighbor_num + self.static_num, self.encoder_hidden_dim, device=device)
        lane_tokens  = torch.zeros(B, self.lane_num, self.encoder_hidden_dim, device=device)

        return dict(
            agent_tokens=agent_tokens,   # (B, 37, 192)
            lane_tokens=lane_tokens,     # (B, 70, 192)
        )
