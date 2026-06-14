# VectorFlow vs FlowPlanner — What Changed

## Package Rename
`flow_planner/` → `vectorflow/` (entire package)

---

## Model Directory

| FlowPlanner | VectorFlow | What |
|---|---|---|
| `model/flow_planner_model/` | `model/vectorflow_model/` | Renamed — same encoder, decoder, DiT, FlowODE |
| _(doesn't exist)_ | `model/custom_model/` | **NEW** — stub encoder + stub decoder to implement |

---

## Configs

| FlowPlanner | VectorFlow | What |
|---|---|---|
| `script/flow_planner_standard.yaml` | `script/vectorflow_standard.yaml` | Renamed top-level config |
| `script/model/flow_planner.yaml` | `script/model/vectorflow.yaml` | Renamed model config |
| _(doesn't exist)_ | `script/model/custom_model.yaml` | **NEW** — wires CustomEncoder + CustomDecoder |
| _(doesn't exist)_ | `script/custom_model_standard.yaml` | **NEW** — top-level config for training custom_model |

---

## New Tools
All under `vectorflow/tools/` — none of these existed in FlowPlanner:

| File | Purpose |
|---|---|
| `preflight_shapes.py` | Validates all 24 input/output tensor shapes before training |
| `infer_and_viz.py` | Run inference + plot predicted vs GT trajectory |
| `bev_inference.py` | Dark BEV scene with VectorFlow prediction overlay |
| `compare_inference.py` | Side-by-side: FlowPlanner (trained) vs custom_model (stub) |
| `generate_synthetic_dataset.py` | Generates fake NPZ data for testing without real data |

---

## Unchanged (identical between both repos)

- All `data/` processing code
- All `core/` flow matching logic
- `trainer.py`
- All `train_utils/` (DDP, EMA, save/load)
- All `nuplan_simulation/` planner interface
- `normalization_stats.yaml`
- All scheduler/optimizer/recorder configs

---

## What FlowPlanner has that VectorFlow doesn't

- `run_mini_sim.sh`, `run_val14_sim.sh`, `run_interplan_sim.sh` — NuPlan closed-loop eval scripts
- `launch_nuboard.py`, `render_sim_video.py` — visualization runners
- `TRAINING_WALKTHROUGH.md`, `AGENTS.md` — documentation

---

## Bottom Line

VectorFlow is FlowPlanner with the model renamed, a new `custom_model/` scaffold added to build in, and 5 new dev tools. All training, data, and simulation infrastructure is shared.

### Preflight Check — 24/24 PASS

```
ALL 24 SHAPE CHECKS PASSED — ready for training

model        : vectorflow.model.vectorflow_model.flow_planner.VectorFlowPlanner
encoder stub : vectorflow.model.custom_model.encoder.CustomEncoder
decoder stub : vectorflow.model.custom_model.decoder.CustomDecoder
future_len   : 80  |  action_num=7  action_len=20  overlap=10
state_dim    : 4  (x, y, cos_heading, sin_heading)
device       : cuda
```
