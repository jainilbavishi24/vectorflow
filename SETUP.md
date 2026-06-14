# VectorFlow — Setup & Development Guide

VectorFlow is built on top of FlowPlanner (NeurIPS 2025). The core training/data/simulation infrastructure is identical — only the model directory is new. We have to implement `CustomEncoder` and `CustomDecoder` in `vectorflow/model/custom_model/`.

---

## 1. Environment

```bash
# Python environment (already set up on spectre)
/scratch3/jainil.bavishi/flowplanner_env/bin/python


source /scratch3/jainil.bavishi/flowplanner_env/bin/activate
```

---

## 2. Data

Preprocessed NuPlan NPZ files (ready to use — no further preprocessing needed):

```
/scratch3/jainil.bavishi/flowplanner_npz/trainval/
    data_list.json          ← index of all training files
    *.npz                   ← one file per scenario
```

Each NPZ contains:

| Key | Shape | Description |
|---|---|---|
| `ego_agent_past` | `(21, 14)` | Ego past 2.1s |
| `ego_current_state` | `(16,)` | Ego at t=0 |
| `ego_agent_future` | `(80, 3)` | GT future 8s (x, y, heading) |
| `neighbor_agents_past` | `(32, 21, 11)` | 32 nearest agents, past 2.1s |
| `lanes` | `(70, 20, 12)` | HD map lane polylines |
| `route_lanes` | `(25, 20, 12)` | Ego route lanes |
| `static_objects` | `(5, 10)` | Static obstacles |

---

## 3. Repository structure

```
vectorflow_repo/
├── vectorflow/
│   ├── model/
│   │   ├── custom_model/         ← Our Model goes here
│   │   │   ├── encoder.py        ← implement CustomEncoder.forward()
│   │   │   └── decoder.py        ← implement CustomDecoder.forward()
│   │   └── vectorflow_model/     ← DO NOT TOUCH (FlowPlanner's DiT, unchanged)
│   ├── core/                     ← DO NOT TOUCH (flow matching ODE, loss)
│   ├── data/                     ← DO NOT TOUCH (NuPlan data pipeline)
│   ├── train_utils/              ← DO NOT TOUCH (DDP, EMA, checkpointing)
│   ├── nuplan_simulation/        ← DO NOT TOUCH (NuPlan planner interface)
│   ├── script/
│   │   └── model/
│   │       └── custom_model.yaml ← model hyperparameters (edit as needed)
│   └── tools/
│       ├── preflight_shapes.py   ← validate shapes before training
│       ├── bev_inference.py      ← visualize inference on a real scene
│       └── compare_inference.py  ← side-by-side vs FlowPlanner
└── launch_nuboard.py             ← launch NuBoard sim viewer
```

---

## 4. Implement the custom model

### What to implement

**`vectorflow/model/custom_model/encoder.py`** — `CustomEncoder.forward()`

Input:
```
neighbors:             (B, 32, 21, 11)   — agent histories
static:                (B,  5, 10)       — static obstacles
lanes:                 (B, 70, 20, 12)   — lane polylines
lanes_speed_limit:     (B, 70, 1)
lanes_has_speed_limit: (B, 70, 1)
routes:                (B, 25, 20, 12)   — ego route lanes
```

Output (dict passed to decoder as `**model_extra`):
```python
return dict(
    agent_tokens=...,   # (B, 37, encoder_hidden_dim)
    lane_tokens=...,    # (B, 70, encoder_hidden_dim)
    # add any other keys your DiT needs
)
```

**`vectorflow/model/custom_model/decoder.py`** — `CustomDecoder.forward()`

```python
def forward(self, x, t, **model_extra):
    # x:           (B, 7, 20, 4)  — noised trajectory chunks
    # t:           (B,) or scalar — flow timestep in [0, 1]
    # model_extra: agent_tokens, lane_tokens, cfg_flags, ...
    # return:      same shape as x — predicted clean trajectory
```

### Register new params in the config

Edit `vectorflow/script/model/custom_model.yaml` under `model_encoder:` and `model_decoder:` to add your new constructor arguments.

---

## 5. Verify shapes before training

```bash
cd /home/AutoDP/jainil.bavishi/vectorflow_repo

PROJECT_ROOT=$(pwd) \
SAVE_DIR=/tmp/vf \
TENSORBOARD_LOG_PATH=/tmp/vf_tb \
TRAINING_DATA=/scratch3/jainil.bavishi/flowplanner_npz/trainval \
TRAINING_JSON=/scratch3/jainil.bavishi/flowplanner_npz/trainval/data_list.json \
/scratch3/jainil.bavishi/flowplanner_env/bin/python \
  vectorflow/tools/preflight_shapes.py model=custom_model
```

Expected output — all 24 checks green:
```
ALL 24 SHAPE CHECKS PASSED — ready for training
```

---

## 6. Train

```bash
cd /home/AutoDP/jainil.bavishi/vectorflow_repo

PROJECT_ROOT=$(pwd) \
SAVE_DIR=/scratch3/jainil.bavishi/vectorflow_exp \
TENSORBOARD_LOG_PATH=/scratch3/jainil.bavishi/vectorflow_tb \
TRAINING_DATA=/scratch3/jainil.bavishi/flowplanner_npz/trainval \
TRAINING_JSON=/scratch3/jainil.bavishi/flowplanner_npz/trainval/data_list.json \
WORLD_SIZE=1 LOCAL_RANK=0 RANK=0 \
/scratch3/jainil.bavishi/flowplanner_env/bin/python \
  vectorflow/trainer.py model=custom_model
```

Checkpoints save to `$SAVE_DIR/outputs/VectorFlowTraining/custom_model_standard/<timestamp>/`.

For multi-GPU training (e.g., 2 GPUs):
```bash
torchrun --nproc_per_node=2 vectorflow/trainer.py model=custom_model ddp.distributed=true
```

---

## 7. Run NuPlan closed-loop simulation

```bash
cd /home/AutoDP/jainil.bavishi/vectorflow_repo

CUDA_VISIBLE_DEVICES=0 \
NUPLAN_DATA_ROOT=/fast_scratch/autodp/nuplan/dataset \
NUPLAN_MAPS_ROOT=/fast_scratch/autodp/nuplan/dataset/maps \
NUPLAN_EXP_ROOT=/scratch3/jainil.bavishi/flowplanner_exp \
/scratch3/jainil.bavishi/flowplanner_env/bin/python \
  /home/AutoDP/jainil.bavishi/nuplan-devkit/nuplan/planning/script/run_simulation.py \
  +simulation=closed_loop_nonreactive_agents \
  planner=vectorflow \
  planner.vectorflow.config_path=/scratch3/jainil.bavishi/vectorflow_ckpt/custom_model_config.yaml \
  planner.vectorflow.ckpt_path=/scratch3/jainil.bavishi/vectorflow_ckpt/custom_model_random.pth \
  +planner.vectorflow.use_cfg=false \
  scenario_builder=nuplan_mini \
  scenario_filter=one_of_each_scenario_type \
  experiment_uid="vectorflow/mini/my_run" \
  hydra.searchpath="[pkg://vectorflow.nuplan_simulation.scenario_filter, pkg://vectorflow.nuplan_simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"
```

Replace `ckpt_path` with your trained checkpoint once available.

---

## 8. View simulation in NuBoard

```bash
cd /home/AutoDP/jainil.bavishi/vectorflow_repo

NUPLAN_DATA_ROOT=/fast_scratch/autodp/nuplan/dataset \
NUPLAN_MAPS_ROOT=/fast_scratch/autodp/nuplan/dataset/maps \
NUPLAN_EXP_ROOT=/scratch3/jainil.bavishi/flowplanner_exp \
/scratch3/jainil.bavishi/flowplanner_env/bin/python launch_nuboard.py
```

Open **http://localhost:5007** → Scenarios → gear icon ⚙ → select a token → Query Scenario.

### Scenario token map (random-weights run)

| Token | Scenario |
|---|---|
| `99ca544752f255ad` | accelerating_at_traffic_light_without_lead |
| `485e78d3d4035b52` | following_lane_with_lead |
| `1f151e15c9cf5c81` | near_multiple_vehicles |
| `6bd0988fce0f548b` | stopping_at_stop_sign_with_lead |
| `6e256d585b245983` | starting_unprotected_cross_turn |
| `a3a4c3242d345082` | starting_left_turn |
| `aa8237ebd54f5a0b` | starting_protected_noncross_turn |
| `b2a5c363d1dd5abe` | changing_lane_to_left |
| `d0b68e15688c58ad` | on_pickup_dropoff |
| `e4eb6ff392715216` | waiting_for_pedestrian_to_cross |

---

## 9. Key paths on this machine

| What | Path |
|---|---|
| Python env | `/scratch3/jainil.bavishi/flowplanner_env/` |
| NuPlan devkit | `/home/AutoDP/jainil.bavishi/nuplan-devkit/` |
| NuPlan dataset | `/fast_scratch/autodp/nuplan/dataset/` |
| Training NPZ data | `/scratch3/jainil.bavishi/flowplanner_npz/trainval/` |
| FlowPlanner checkpoint | `/scratch3/jainil.bavishi/flowplanner_ckpt/model_wrapped.pth` |
| VectorFlow random ckpt | `/scratch3/jainil.bavishi/vectorflow_ckpt/custom_model_random.pth` |
| Simulation outputs | `/scratch3/jainil.bavishi/flowplanner_exp/exp/simulation/` |
