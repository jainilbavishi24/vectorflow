"""
Preflight shape check — run before first training epoch.
Verifies every tensor shape matches the expected contract derived from custom_model.yaml.

Usage (from vectorflow_repo root):
    PROJECT_ROOT=$(pwd) SAVE_DIR=/tmp/vf TENSORBOARD_LOG_PATH=/tmp/vf_tb \
    TRAINING_DATA=/scratch3/jainil.bavishi/flowplanner_npz/trainval \
    TRAINING_JSON=/scratch3/jainil.bavishi/flowplanner_npz/trainval/data_list.json \
    python vectorflow/tools/preflight_shapes.py model=custom_model
"""

import os
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from vectorflow.data.utils.collect import collect_batch

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
HEAD = "\033[1;34m"
END  = "\033[0m"


def check(label, actual, expected):
    ok = tuple(actual) == tuple(expected)
    status = PASS if ok else FAIL
    print(f"  {status}  {label:<40s}  actual={list(actual)}  expected={list(expected)}")
    return ok


def section(title):
    print(f"\n{HEAD}{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}{END}")


def _write_synthetic_dataset(out_dir: Path, num_samples: int = 8) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3402)
    past_len, future_len = 21, 80
    neighbor_num, lane_num, lane_points = 32, 70, 20
    route_num, static_num = 25, 5
    data_list = []
    for idx in range(num_samples):
        speed = 1.0
        t_past = np.linspace(-2.0, 0.0, past_len, dtype=np.float32)
        x_past = speed * (t_past + 2.0)
        y_past = rng.normal(0.0, 0.05, size=past_len).astype(np.float32)
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
        ego_agent_future = np.stack(
            [ego_current_state[0] + speed * t_fut,
             rng.normal(0.0, 0.1, future_len).astype(np.float32),
             np.zeros(future_len, dtype=np.float32)], axis=-1)
        neighbor_agents_past = np.zeros((neighbor_num, past_len, 11), dtype=np.float32)
        neighbor_agents_past[0, :, :5] = np.stack(
            [x_past, y_past + 2.0, np.ones(past_len), np.zeros(past_len),
             np.full(past_len, speed)], axis=-1)
        neighbor_agents_past[0, :, 6:9] = [4.5, 2.0, 1.0]
        neighbor_agents_future = np.zeros((neighbor_num, future_len, 3), dtype=np.float32)
        fname = f"synthetic_{idx:04d}.npz"
        np.savez(out_dir / fname,
            ego_agent_past=ego_agent_past, ego_current_state=ego_current_state,
            ego_agent_future=ego_agent_future,
            neighbor_agents_past=neighbor_agents_past,
            neighbor_agents_future=neighbor_agents_future,
            static_objects=np.zeros((static_num, 10), dtype=np.float32),
            lanes=np.zeros((lane_num, lane_points, 12), dtype=np.float32),
            lanes_speed_limit=np.zeros((lane_num, 1), dtype=np.float32),
            lanes_has_speed_limit=np.zeros((lane_num, 1), dtype=np.bool_),
            route_lanes=np.zeros((route_num, lane_points, 12), dtype=np.float32),
            route_lanes_speed_limit=np.zeros((route_num, 1), dtype=np.float32),
            route_lanes_has_speed_limit=np.zeros((route_num, 1), dtype=np.bool_),
        )
        data_list.append(fname)
    (out_dir / "data_list.json").write_text(
        "[\n  " + ",\n  ".join(f'"{n}"' for n in data_list) + "\n]\n")


@hydra.main(version_base=None, config_path="../script", config_name="vectorflow_standard")
def main(cfg: DictConfig) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    os.environ.setdefault("PROJECT_ROOT",         str(repo_root))
    os.environ.setdefault("SAVE_DIR",             str(repo_root / "tmp" / "vf_runs"))
    os.environ.setdefault("TENSORBOARD_LOG_PATH", str(repo_root / "tmp" / "vf_tb"))
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK",       "0")

    if not os.environ.get("TRAINING_DATA"):
        synth_dir = repo_root / "tmp" / "vf_synth"
        _write_synthetic_dataset(synth_dir)
        os.environ["TRAINING_DATA"] = str(synth_dir)
        os.environ["TRAINING_JSON"] = str(synth_dir / "data_list.json")

    device = cfg.device if torch.cuda.is_available() else "cpu"
    cfg.device = device

    # derive constants from config
    B          = 2
    past_len   = 21
    future_len = cfg.model.future_len        # 80
    action_len = cfg.model.action_len        # 20
    overlap    = cfg.model.action_overlap    # 10
    action_num = (future_len - overlap) // (action_len - overlap)   # 7
    state_dim  = cfg.model.state_dim         # 4
    N          = cfg.model.neighbor_num      # 32
    L          = cfg.model.lane_num          # 70
    S          = cfg.model.static_num        # 5

    passes = []

    # ── 1. Raw batch ──────────────────────────────────────────────────────────
    section("1 · Raw batch  (DataLoader output, CPU)")
    print(f"  batch_size={B}  |  data: {os.environ['TRAINING_DATA']}")

    dataset = instantiate(cfg.data.dataset.train)
    loader  = DataLoader(dataset, batch_size=B, shuffle=False,
                         num_workers=0, collate_fn=collect_batch)
    batch = next(iter(loader))

    passes += [
        check("ego_past",                  batch.ego_past.shape,              [B, past_len, 14]),
        check("ego_current",               batch.ego_current.shape,           [B, 16]),
        check("ego_future  (GT raw)",      batch.ego_future.shape,            [B, future_len, 3]),
        check("neighbor_past",             batch.neighbor_past.shape,         [B, N, past_len, 11]),
        check("neighbor_future_observed",  batch.neighbor_future_observed.shape, [B, N, future_len, 3]),
        check("lanes",                     batch.lanes.shape,                 [B, L, 20, 12]),
        check("lanes_speedlimit",          batch.lanes_speedlimit.shape,      [B, L, 1]),
        check("routes",                    batch.routes.shape,                [B, 25, 20, 12]),
        check("map_objects  (static)",     batch.map_objects.shape,           [B, S, 10]),
    ]

    # ── 2. After ego-centric normalisation ───────────────────────────────────
    section("2 · After ego-centric normalisation  (same shapes, moved to GPU)")
    batch = batch.to(device)
    core  = instantiate(cfg.core)
    if hasattr(core, "input_aug") and core.input_aug is not None:
        batch = core.input_aug(batch)

    passes += [
        check("ego_past        device=cuda", batch.ego_past.shape,      [B, past_len, 14]),
        check("neighbor_past   device=cuda", batch.neighbor_past.shape, [B, N, past_len, 11]),
        check("lanes           device=cuda", batch.lanes.shape,         [B, L, 20, 12]),
    ]
    gpu_ok = str(batch.ego_past.device).startswith("cuda")
    status = PASS if gpu_ok else FAIL
    print(f"  {status}  {'tensors on GPU':<40s}  device={batch.ego_past.device}")
    passes.append(gpu_ok)

    # ── 3. Model inputs (what encoder receives) ───────────────────────────────
    section("3 · Encoder inputs  (after ModelInputProcessor)")
    model = instantiate(cfg.model).to(device)
    cfg_flags = torch.ones((B, 1), dtype=torch.int32, device=device)
    model_inputs, gt = model.prepare_model_input(cfg_flags, batch, use_cfg=False, is_training=True)

    passes += [
        check("neighbors → encoder",  model_inputs["neighbor_past"].shape, [B, N, past_len, 11]),
        check("lanes     → encoder",  model_inputs["lanes"].shape,         [B, L, 20, 12]),
        check("static    → encoder",  model_inputs["map_objects"].shape,   [B, S, 10]),
        check("ego_past  → encoder",  model_inputs["ego_past"].shape,      [B, past_len, 14]),
        check("cfg_flags → decoder",  model_inputs["cfg_flags"].shape,     [B, 1]),
    ]

    # ── 4. Ground truth trajectory ────────────────────────────────────────────
    section("4 · Ground truth trajectory  (chunked for flow matching)")
    # gt: (B, 1, future_len+1, state_dim)  — +1 because current state is prepended
    passes += [
        check("gt  (B, 1, future_len+1, state_dim)", gt.shape,
              [B, 1, future_len + 1, state_dim]),
    ]

    # ── 5. Forward pass ───────────────────────────────────────────────────────
    section("5 · Forward pass  (train mode)")
    model.eval()
    prediction, loss_dict = model(batch, mode="train")

    passes += [
        check("prediction  (B, action_num, action_len, state_dim)",
              prediction.shape, [B, action_num, action_len, state_dim]),
        check("loss.batch_loss  (per-element MSE, unreduced)",
              loss_dict["batch_loss"].shape, [B, action_num, action_len, state_dim]),
        check("loss.ego_planning_loss  (scalar)",
              loss_dict["ego_planning_loss"].shape, []),
        check("loss.consistency_loss   (scalar)",
              loss_dict["consistency_loss"].shape,  []),
    ]

    # ── 6. Inference pass ────────────────────────────────────────────────────
    section("6 · Inference pass  (ODE sampler output)")
    model.eval()
    with torch.no_grad():
        pred_inf = core.inference(model, batch, use_cfg=False, cfg_weight=1.0)
    # after assemble_actions + postprocess: (B, 1, future_len, state_dim)
    passes += [
        check("inference output  (B, 1, future_len, state_dim)",
              pred_inf.shape, [B, 1, future_len, state_dim]),
    ]

    # ── Summary ──────────────────────────────────────────────────────────────
    total  = len(passes)
    passed = sum(passes)
    failed = total - passed

    print(f"\n{HEAD}{'═'*70}{END}")
    if failed == 0:
        print(f"\033[92m  ALL {total} SHAPE CHECKS PASSED — ready for training\033[0m")
    else:
        print(f"\033[91m  {failed}/{total} CHECKS FAILED — fix shapes before training\033[0m")
    print(f"\n  model          : {cfg.model._target_}")
    print(f"  encoder stub   : {cfg.model.model_encoder._target_}")
    print(f"  decoder stub   : {cfg.model.model_decoder._target_}")
    print(f"  future_len     : {future_len}  |  action_num={action_num}  action_len={action_len}  overlap={overlap}")
    print(f"  state_dim      : {state_dim}  (x, y, cos_heading, sin_heading)")
    print(f"  device         : {device}")
    print(f"{HEAD}{'═'*70}{END}\n")


if __name__ == "__main__":
    main()
