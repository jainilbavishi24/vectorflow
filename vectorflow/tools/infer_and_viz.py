"""
Run inference on the custom_model with synthetic data and save a visualization.

Usage (from vectorflow_repo root):
    PROJECT_ROOT=$(pwd) SAVE_DIR=/tmp/vf TENSORBOARD_LOG_PATH=/tmp/vf_tb \
    python vectorflow/tools/infer_and_viz.py model=custom_model

Output: inference_viz.png in the current directory.
"""

import os
from pathlib import Path

import hydra
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from vectorflow.data.utils.collect import collect_batch


# ── synthetic data (same as preflight_shapes.py) ─────────────────────────────

def _write_synthetic_dataset(out_dir: Path, num_samples: int = 4) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    past_len, future_len = 21, 80
    neighbor_num, lane_num, lane_points = 32, 70, 20
    route_num, static_num = 25, 5

    data_list = []
    for idx in range(num_samples):
        speed = 5.0 + rng.uniform(-1, 1)
        t_past = np.linspace(-2.0, 0.0, past_len, dtype=np.float32)

        # ego: straight road with small lateral noise
        x_past = speed * (t_past + 2.0)
        y_past = rng.normal(0, 0.05, past_len).astype(np.float32)
        heading = np.zeros(past_len, dtype=np.float32)

        ego_agent_past = np.zeros((past_len, 14), dtype=np.float32)
        ego_agent_past[:, 0] = x_past
        ego_agent_past[:, 1] = y_past
        ego_agent_past[:, 2] = np.cos(heading)
        ego_agent_past[:, 3] = np.sin(heading)
        ego_agent_past[:, 4] = speed
        ego_agent_past[:, 8:10] = [2.0, 4.8]
        ego_agent_past[:, 10] = 1.0

        ego_current_state = np.zeros(16, dtype=np.float32)
        ego_current_state[:5] = [x_past[-1], y_past[-1], 1.0, 0.0, speed]
        ego_current_state[10:13] = [2.0, 4.8, 1.0]

        t_fut = np.linspace(0.1, 8.0, future_len, dtype=np.float32)
        x_fut = ego_current_state[0] + speed * t_fut
        y_fut = rng.normal(0, 0.1, future_len).astype(np.float32)
        ego_agent_future = np.stack([x_fut, y_fut, np.zeros_like(x_fut)], -1).astype(np.float32)

        # two neighbors: one alongside, one coming from ahead
        neighbor_agents_past = np.zeros((neighbor_num, past_len, 11), dtype=np.float32)
        # neighbor 0: parallel, slightly to the left
        neighbor_agents_past[0, :, 0] = x_past
        neighbor_agents_past[0, :, 1] = y_past + 3.5
        neighbor_agents_past[0, :, 2] = 1.0
        neighbor_agents_past[0, :, 4] = speed * 0.9
        neighbor_agents_past[0, :, 6:8] = [4.5, 2.0]
        neighbor_agents_past[0, :, 8] = 1.0
        # neighbor 1: ahead, same lane
        neighbor_agents_past[1, :, 0] = x_past + 15.0
        neighbor_agents_past[1, :, 1] = y_past
        neighbor_agents_past[1, :, 2] = 1.0
        neighbor_agents_past[1, :, 4] = speed * 1.1
        neighbor_agents_past[1, :, 6:8] = [4.5, 2.0]
        neighbor_agents_past[1, :, 8] = 1.0

        neighbor_agents_future = np.zeros((neighbor_num, future_len, 3), dtype=np.float32)
        neighbor_agents_future[0, :, 0] = x_fut
        neighbor_agents_future[0, :, 1] = y_fut + 3.5
        neighbor_agents_future[1, :, 0] = x_fut + 15.0
        neighbor_agents_future[1, :, 1] = y_fut

        # lanes: two parallel lanes
        lanes = np.zeros((lane_num, lane_points, 12), dtype=np.float32)
        x_lane = np.linspace(0, 50, lane_points)
        # lane 0: ego lane centre
        lanes[0, :, 0] = x_lane
        lanes[0, :, 1] = 0.0
        # lane 1: left lane
        lanes[1, :, 0] = x_lane
        lanes[1, :, 1] = 3.5
        # lane 2: right lane
        lanes[2, :, 0] = x_lane
        lanes[2, :, 1] = -3.5

        lanes_speed_limit = np.full((lane_num, 1), 13.9, dtype=np.float32)
        lanes_has_speed_limit = np.ones((lane_num, 1), dtype=np.bool_)

        route_lanes = np.zeros((route_num, lane_points, 12), dtype=np.float32)
        route_lanes[0, :, 0] = x_lane
        route_lanes[0, :, 1] = 0.0
        route_lanes_speed_limit = np.full((route_num, 1), 13.9, dtype=np.float32)
        route_lanes_has_speed_limit = np.ones((route_num, 1), dtype=np.bool_)

        static_objects = np.zeros((static_num, 10), dtype=np.float32)

        np.savez(
            out_dir / f"synthetic_{idx:04d}.npz",
            ego_agent_past=ego_agent_past,
            ego_current_state=ego_current_state,
            ego_agent_future=ego_agent_future,
            neighbor_agents_past=neighbor_agents_past,
            neighbor_agents_future=neighbor_agents_future,
            static_objects=static_objects,
            lanes=lanes,
            lanes_speed_limit=lanes_speed_limit,
            lanes_has_speed_limit=lanes_has_speed_limit,
            route_lanes=route_lanes,
            route_lanes_speed_limit=route_lanes_speed_limit,
            route_lanes_has_speed_limit=route_lanes_has_speed_limit,
        )
        data_list.append(f"synthetic_{idx:04d}.npz")

    (out_dir / "data_list.json").write_text(
        "[\n  " + ",\n  ".join(f'"{n}"' for n in data_list) + "\n]\n"
    )


# ── visualization ─────────────────────────────────────────────────────────────

def viz_inference(data, predicted, ego_future_raw=None, save_path="inference_viz.png"):
    """
    Plot one scene sample:
      grey lines  = lanes
      blue lines  = neighbor past trajectories
      red line    = ego past trajectory
      green line  = ground-truth future (ego_agent_future, raw before normalization)
      orange line = model predicted future
    """
    ego_past      = data.ego_past[0].cpu().float().numpy()        # (21, F)
    neighbor_past = data.neighbor_past[0].cpu().float().numpy()   # (32, 21, F)
    lanes         = data.lanes[0].cpu().float().numpy()           # (70, 20, 12)
    pred_xy       = predicted[0, 0].cpu().float().numpy()         # (80, 4)  x,y,cos_h,sin_h

    # prefer raw (pre-normalization) ego_future so coords match prediction scale
    if ego_future_raw is not None:
        ego_future = ego_future_raw[0].cpu().float().numpy()      # (80, 3)  x,y,heading
    else:
        ego_future = data.ego_future[0].cpu().float().numpy()

    fig, ax = plt.subplots(figsize=(14, 6))

    # ── lanes ──
    for lane in lanes:
        valid = np.any(lane != 0, axis=-1)
        if valid.sum() > 1:
            ax.plot(lane[valid, 0], lane[valid, 1],
                    color='#cccccc', linewidth=1.2, alpha=0.7, zorder=1)

    # ── neighbors ──
    for i, agent in enumerate(neighbor_past):
        valid = np.any(agent != 0, axis=-1)
        if valid.sum() > 1:
            ax.plot(agent[valid, 0], agent[valid, 1],
                    color='steelblue', linewidth=1.2, alpha=0.7, zorder=2)
            ax.scatter(agent[valid, 0][-1], agent[valid, 1][-1],
                       color='steelblue', s=25, zorder=3)

    # ── ego past ──
    ax.plot(ego_past[:, 0], ego_past[:, 1],
            color='red', linewidth=2.0, label='ego past', zorder=4)
    ax.scatter(ego_past[-1, 0], ego_past[-1, 1],
               color='red', s=80, marker='*', zorder=5)

    # ── model prediction ──
    ax.plot(pred_xy[:, 0], pred_xy[:, 1],
            color='orange', linewidth=2.5,
            label='custom_model prediction', zorder=6)
    ax.scatter(pred_xy[-1, 0], pred_xy[-1, 1],
               color='orange', s=60, marker='D', zorder=7)

    # ── ground-truth future (drawn last so always on top) ──
    ax.plot(ego_future[:, 0], ego_future[:, 1],
            color='green', linewidth=2.5, linestyle='--',
            label='GT future', zorder=8)
    ax.scatter(ego_future[-1, 0], ego_future[-1, 1],
               color='green', s=60, marker='D', zorder=9)

    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_title('custom_model inference  (stub encoder=zeros, stub decoder=linear)\n'
                 'Orange = predicted path   Green dashed = ground truth')
    ax.set_xlabel('x (m, ego-centric)')
    ax.set_ylabel('y (m, ego-centric)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved → {save_path}")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="../script", config_name="vectorflow_standard")
def main(cfg: DictConfig) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    os.environ.setdefault("PROJECT_ROOT", str(repo_root))
    os.environ.setdefault("SAVE_DIR",             str(repo_root / "tmp" / "vf_runs"))
    os.environ.setdefault("TENSORBOARD_LOG_PATH", str(repo_root / "tmp" / "vf_tb"))
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK",       "0")

    # use synthetic data if real data paths not set
    if not os.environ.get("TRAINING_DATA") or not os.environ.get("TRAINING_JSON"):
        synth_dir = repo_root / "tmp" / "vf_synth"
        _write_synthetic_dataset(synth_dir)
        os.environ["TRAINING_DATA"] = str(synth_dir)
        os.environ["TRAINING_JSON"] = str(synth_dir / "data_list.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.device = device

    # ── load one batch ──
    dataset = instantiate(cfg.data.dataset.train)
    loader  = DataLoader(dataset, batch_size=2, shuffle=False,
                         num_workers=0, collate_fn=collect_batch)
    batch = next(iter(loader)).to(device)

    # ── augment (ego-centric transform) ──
    core = instantiate(cfg.core)
    # save raw ego_future BEFORE normalization so viz coords match prediction scale
    ego_future_raw = batch.ego_future.clone()
    if hasattr(core, "input_aug") and core.input_aug is not None:
        batch = core.input_aug(batch)

    # ── build model ──
    model = instantiate(cfg.model).to(device)
    model.eval()

    print(f"\nRunning inference on device: {device}")
    print(f"Model: {cfg.model._target_}")

    with torch.no_grad():
        predicted = model(batch, mode='inference', use_cfg=False, cfg_weight=1.0)

    print(f"predicted shape: {tuple(predicted.shape)}")   # (B, 1, 80, 4)
    print(f"predicted x range: [{predicted[0,0,:,0].min():.3f}, {predicted[0,0,:,0].max():.3f}]")
    print(f"predicted y range: [{predicted[0,0,:,1].min():.3f}, {predicted[0,0,:,1].max():.3f}]")

    # ── visualize ──
    save_path = str(Path.cwd() / "inference_viz.png")
    viz_inference(batch, predicted, ego_future_raw=ego_future_raw, save_path=save_path)


if __name__ == "__main__":
    main()
