"""
BEV scene + VectorFlow inference overlay.
Uses the same npz data the model trains on — no FlowPlanner dependency.

Usage (from vectorflow_repo root):
    PROJECT_ROOT=$(pwd) SAVE_DIR=/tmp/vf TENSORBOARD_LOG_PATH=/tmp/vf_tb \
    TRAINING_DATA=/scratch3/jainil.bavishi/flowplanner_npz/trainval \
    TRAINING_JSON=/scratch3/jainil.bavishi/flowplanner_npz/trainval/data_list.json \
    python vectorflow/tools/bev_inference.py model=custom_model

Output: bev_inference.png
"""

import os
from pathlib import Path

import hydra
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from vectorflow.data.utils.collect import collect_batch


@hydra.main(version_base=None, config_path="../script", config_name="vectorflow_standard")
def main(cfg: DictConfig) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    os.environ.setdefault("PROJECT_ROOT",          str(repo_root))
    os.environ.setdefault("SAVE_DIR",              str(repo_root / "tmp" / "vf_runs"))
    os.environ.setdefault("TENSORBOARD_LOG_PATH",  str(repo_root / "tmp" / "vf_tb"))
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK",       "0")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.device = device

    # ── load one batch from real npz data ────────────────────────────────────
    dataset = instantiate(cfg.data.dataset.train)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False,
                         num_workers=0, collate_fn=collect_batch)
    batch_raw = next(iter(loader))          # keep raw (pre-norm) for GT coords

    # ego-centric normalization (same as training)
    core = instantiate(cfg.core)
    batch = batch_raw.to(device)
    ego_future_raw    = batch.ego_future[0].cpu().float().numpy()     # (80,3) raw GT
    ego_past_raw      = batch.ego_past[0].cpu().float().numpy()       # (21,F)
    neighbor_past_raw = batch.neighbor_past[0].cpu().float().numpy()  # (32,21,F)
    lanes_raw         = batch.lanes[0].cpu().float().numpy()          # (70,20,12)

    if hasattr(core, "input_aug") and core.input_aug is not None:
        batch = core.input_aug(batch)

    # ── build model and run inference ─────────────────────────────────────────
    model = instantiate(cfg.model).to(device)
    model.eval()
    print(f"Model: {cfg.model._target_}  |  device: {device}")

    with torch.no_grad():
        predicted = model(batch, mode='inference', use_cfg=False, cfg_weight=1.0)

    pred_xy = predicted[0, 0].cpu().float().numpy()   # (80,4) x,y,cos_h,sin_h

    print(f"pred x range: [{pred_xy[:,0].min():.1f}, {pred_xy[:,0].max():.1f}]")
    print(f"pred y range: [{pred_xy[:,1].min():.1f}, {pred_xy[:,1].max():.1f}]")

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 11))
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')

    # lanes
    for lane in lanes_raw:
        valid = np.any(lane != 0, axis=-1)
        if valid.sum() > 1:
            ax.plot(lane[valid, 0], lane[valid, 1],
                    color='#555577', lw=1.2, alpha=0.7, zorder=1)

    # neighbor agents — draw each as a small trail + endpoint dot
    for agent in neighbor_past_raw:
        valid = np.any(agent != 0, axis=-1)
        if valid.sum() < 2:
            continue
        ax.plot(agent[valid, 0], agent[valid, 1],
                color='#4fc3f7', lw=1.0, alpha=0.6, zorder=2)
        ax.scatter(agent[valid, 0][-1], agent[valid, 1][-1],
                   color='#4fc3f7', s=18, zorder=3)

    # ego past
    ax.plot(ego_past_raw[:, 0], ego_past_raw[:, 1],
            color='#ef5350', lw=2.2, label='Ego past', zorder=4)
    ax.scatter(ego_past_raw[-1, 0], ego_past_raw[-1, 1],
               color='#ef5350', s=100, marker='*', zorder=5)

    # GT future
    ax.plot(ego_future_raw[:, 0], ego_future_raw[:, 1],
            color='#69f0ae', lw=2.5, linestyle='--',
            label='GT future', zorder=8)
    ax.scatter(ego_future_raw[-1, 0], ego_future_raw[-1, 1],
               color='#69f0ae', s=70, marker='D', zorder=9)

    # VectorFlow prediction
    ax.plot(pred_xy[:, 0], pred_xy[:, 1],
            color='#ffab40', lw=2.5,
            label='VectorFlow prediction (stub)', zorder=6)
    ax.scatter(pred_xy[-1, 0], pred_xy[-1, 1],
               color='#ffab40', s=70, marker='D', zorder=7)

    ax.set_aspect('equal')
    ax.set_xlabel('x (m, ego-centric)', color='white', fontsize=11)
    ax.set_ylabel('y (m, ego-centric)', color='white', fontsize=11)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

    legend_handles = [
        mpatches.Patch(color='#ef5350',  label='Ego past'),
        mpatches.Patch(color='#4fc3f7',  label='Neighbor agents'),
        mpatches.Patch(color='#555577',  label='Lane centerlines'),
        plt.Line2D([0],[0], color='#69f0ae', lw=2.5, linestyle='--', label='GT future (8 s)'),
        plt.Line2D([0],[0], color='#ffab40', lw=2.5, label='VectorFlow pred (stub)'),
    ]
    ax.legend(handles=legend_handles, loc='upper left',
              facecolor='#2a2a3e', edgecolor='#555', labelcolor='white', fontsize=10)
    ax.set_title(
        'VectorFlow inference on real NuPlan data\n'
        'Orange = model prediction   Green dashed = ground truth   '
        '(stub encoder=zeros → chaotic until trained)',
        color='white', fontsize=11,
    )

    out = str(Path.cwd() / "bev_inference.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
