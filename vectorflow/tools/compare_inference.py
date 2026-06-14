"""
Side-by-side comparison: FlowPlanner (trained) vs VectorFlow custom_model (stub)
on the exact same NuPlan scene.

Usage (from vectorflow_repo root):
    PROJECT_ROOT=$(pwd) SAVE_DIR=/tmp/vf TENSORBOARD_LOG_PATH=/tmp/vf_tb \
    TRAINING_DATA=/scratch3/jainil.bavishi/flowplanner_npz/trainval \
    TRAINING_JSON=/scratch3/jainil.bavishi/flowplanner_npz/trainval/data_list.json \
    python vectorflow/tools/compare_inference.py model=custom_model

Output: compare_inference.png
"""

import os, sys
from pathlib import Path

import hydra
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from vectorflow.data.utils.collect import collect_batch

FLOWPLANNER_CKPT   = '/scratch3/jainil.bavishi/flowplanner_ckpt/model_wrapped.pth'
FLOWPLANNER_CONFIG = '/scratch3/jainil.bavishi/flowplanner_ckpt/model_config.yaml'
FLOWPLANNER_ROOT   = '/home/AutoDP/jainil.bavishi/Flow-Planner'


def run_flowplanner(batch, device):
    """Load the trained FlowPlanner and run inference on the same batch."""
    sys.path.insert(0, FLOWPLANNER_ROOT)
    import omegaconf
    fp_cfg = omegaconf.OmegaConf.load(FLOWPLANNER_CONFIG)

    from hydra.utils import instantiate as inst
    model = inst(fp_cfg.model)

    # load EMA weights
    ckpt = torch.load(FLOWPLANNER_CKPT, map_location=device, weights_only=True)
    state = {k[len("module."):]: v for k, v in ckpt['ema_state_dict'].items()}
    model.load_state_dict(state)
    model = model.to(device).eval()

    core = inst(fp_cfg.core)

    # FlowPlanner uses its own normalizer — apply it
    if hasattr(core, "input_aug") and core.input_aug is not None:
        batch_fp = core.input_aug(batch)
    else:
        batch_fp = batch

    with torch.no_grad():
        pred = core.inference(model, batch_fp, use_cfg=True, cfg_weight=fp_cfg.model.cfg_weight)

    return pred[0, 0].cpu().float().numpy()   # (80, 4)


def plot_bev(ax, lanes, neighbor_past, ego_past, ego_future_raw, pred_xy, pred_color, pred_label, title):
    ax.set_facecolor('#1a1a2e')

    # lanes
    for lane in lanes:
        valid = np.any(lane != 0, axis=-1)
        if valid.sum() > 1:
            ax.plot(lane[valid, 0], lane[valid, 1],
                    color='#555577', lw=1.1, alpha=0.7, zorder=1)

    # neighbors
    for agent in neighbor_past:
        valid = np.any(agent != 0, axis=-1)
        if valid.sum() < 2:
            continue
        ax.plot(agent[valid, 0], agent[valid, 1],
                color='#4fc3f7', lw=0.9, alpha=0.55, zorder=2)
        ax.scatter(agent[valid, 0][-1], agent[valid, 1][-1],
                   color='#4fc3f7', s=16, zorder=3)

    # ego past
    ax.plot(ego_past[:, 0], ego_past[:, 1],
            color='#ef5350', lw=2.0, zorder=4)
    ax.scatter(ego_past[-1, 0], ego_past[-1, 1],
               color='#ef5350', s=90, marker='*', zorder=5)

    # GT future
    ax.plot(ego_future_raw[:, 0], ego_future_raw[:, 1],
            color='#69f0ae', lw=2.2, linestyle='--', zorder=8, label='GT future')
    ax.scatter(ego_future_raw[-1, 0], ego_future_raw[-1, 1],
               color='#69f0ae', s=55, marker='D', zorder=9)

    # prediction
    ax.plot(pred_xy[:, 0], pred_xy[:, 1],
            color=pred_color, lw=2.5, zorder=6, label=pred_label)
    ax.scatter(pred_xy[-1, 0], pred_xy[-1, 1],
               color=pred_color, s=55, marker='D', zorder=7)

    ax.set_xlim(-60, 80)
    ax.set_ylim(-20, 30)
    ax.set_aspect('equal')
    ax.set_title(title, color='white', fontsize=11)
    ax.set_xlabel('x (m, ego-centric)', color='white', fontsize=9)
    ax.set_ylabel('y (m, ego-centric)', color='white', fontsize=9)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

    legend_handles = [
        mpatches.Patch(color='#ef5350', label='Ego past'),
        mpatches.Patch(color='#4fc3f7', label='Neighbor agents'),
        mpatches.Patch(color='#555577', label='Lane centerlines'),
        plt.Line2D([0],[0], color='#69f0ae', lw=2, linestyle='--', label='GT future (8 s)'),
        plt.Line2D([0],[0], color=pred_color, lw=2.5, label=pred_label),
    ]
    ax.legend(handles=legend_handles, loc='upper left',
              facecolor='#2a2a3e', edgecolor='#555', labelcolor='white', fontsize=8)


@hydra.main(version_base=None, config_path="../script", config_name="vectorflow_standard")
def main(cfg: DictConfig) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    os.environ.setdefault("PROJECT_ROOT",         str(repo_root))
    os.environ.setdefault("SAVE_DIR",             str(repo_root / "tmp" / "vf_runs"))
    os.environ.setdefault("TENSORBOARD_LOG_PATH", str(repo_root / "tmp" / "vf_tb"))
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK",       "0")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.device = device

    # ── load one batch (same scene for both models) ───────────────────────────
    dataset = instantiate(cfg.data.dataset.train)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False,
                         num_workers=0, collate_fn=collect_batch)
    batch_raw = next(iter(loader)).to(device)

    # extract raw (pre-norm) arrays for visualization
    lanes_raw         = batch_raw.lanes[0].cpu().float().numpy()
    neighbor_past_raw = batch_raw.neighbor_past[0].cpu().float().numpy()
    ego_past_raw      = batch_raw.ego_past[0].cpu().float().numpy()
    ego_future_raw    = batch_raw.ego_future[0].cpu().float().numpy()

    # ── VectorFlow custom_model (stub) ────────────────────────────────────────
    core_vf = instantiate(cfg.core)
    batch_vf = core_vf.input_aug(batch_raw) if hasattr(core_vf, "input_aug") and core_vf.input_aug else batch_raw
    model_vf = instantiate(cfg.model).to(device)
    model_vf.eval()
    with torch.no_grad():
        pred_vf = core_vf.inference(model_vf, batch_vf, use_cfg=False, cfg_weight=1.0)
    pred_vf_xy = pred_vf[0, 0].cpu().float().numpy()[:, :2]
    print(f"VectorFlow pred x range: [{pred_vf_xy[:,0].min():.1f}, {pred_vf_xy[:,0].max():.1f}]")

    # ── FlowPlanner (trained) ─────────────────────────────────────────────────
    print("Loading FlowPlanner trained model...")
    try:
        pred_fp = run_flowplanner(batch_raw, device)
        pred_fp_xy = pred_fp[:, :2]
        fp_ok = True
        print(f"FlowPlanner pred x range: [{pred_fp_xy[:,0].min():.1f}, {pred_fp_xy[:,0].max():.1f}]")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"FlowPlanner inference failed: {e}")
        pred_fp_xy = None
        fp_ok = False

    # ── plot side by side ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    fig.patch.set_facecolor('#0f0f1a')

    plot_bev(
        axes[0], lanes_raw, neighbor_past_raw, ego_past_raw, ego_future_raw,
        pred_fp_xy if fp_ok else np.zeros((80,2)),
        pred_color='#ff6b6b',
        pred_label='FlowPlanner (trained ✓)',
        title='FlowPlanner — trained checkpoint\n(real DiT encoder + decoder)',
    )

    plot_bev(
        axes[1], lanes_raw, neighbor_past_raw, ego_past_raw, ego_future_raw,
        pred_vf_xy,
        pred_color='#ffab40',
        pred_label='VectorFlow custom_model (stub)',
        title='VectorFlow custom_model — stub\n(encoder=zeros, decoder=random linear)',
    )

    fig.suptitle(
        'Same NuPlan scene — FlowPlanner (trained) vs VectorFlow custom_model (stub)\n'
        'Green dashed = ground truth  |  Once trained, orange should match red',
        color='white', fontsize=13, y=1.01,
    )

    out = str(Path.cwd() / "compare_inference.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"\nSaved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
