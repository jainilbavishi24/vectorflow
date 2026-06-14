"""Generate a small synthetic dataset for VectorFlow smoke tests.

Creates linear trajectories with small Gaussian noise and writes .npz files
plus a data_list.json compatible with NuPlanDataset.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def _make_sample(rng, past_len, future_len, neighbor_num, lane_num, lane_points, route_num, static_num):
    # Ego past (past_len, 14)
    t_past = np.linspace(-2.0, 0.0, past_len, dtype=np.float32)
    speed = 1.0
    x_past = speed * (t_past + 2.0)
    y_past = rng.normal(0.0, 0.05, size=past_len).astype(np.float32)
    heading = np.zeros(past_len, dtype=np.float32)

    ego_agent_past = np.zeros((past_len, 14), dtype=np.float32)
    ego_agent_past[:, 0] = x_past
    ego_agent_past[:, 1] = y_past
    ego_agent_past[:, 2] = np.cos(heading)
    ego_agent_past[:, 3] = np.sin(heading)
    ego_agent_past[:, 4] = speed
    ego_agent_past[:, 5] = 0.0
    ego_agent_past[:, 6] = 0.0
    ego_agent_past[:, 7] = 0.0
    ego_agent_past[:, 8] = 2.0
    ego_agent_past[:, 9] = 4.8
    ego_agent_past[:, 10] = 1.0

    # Ego current state (16,)
    ego_current_state = np.zeros((16,), dtype=np.float32)
    ego_current_state[0] = x_past[-1]
    ego_current_state[1] = y_past[-1]
    ego_current_state[2] = np.cos(heading[-1])
    ego_current_state[3] = np.sin(heading[-1])
    ego_current_state[4] = speed
    ego_current_state[5] = 0.0
    ego_current_state[6] = 0.0
    ego_current_state[7] = 0.0
    ego_current_state[8] = 0.0
    ego_current_state[9] = 0.0
    ego_current_state[10] = 2.0
    ego_current_state[11] = 4.8
    ego_current_state[12] = 1.0

    # Ego future (future_len, 3)
    t_fut = np.linspace(0.1, 8.0, future_len, dtype=np.float32)
    x_fut = ego_current_state[0] + speed * t_fut
    y_fut = rng.normal(0.0, 0.1, size=future_len).astype(np.float32)
    ego_agent_future = np.stack([x_fut, y_fut, np.zeros_like(x_fut)], axis=-1).astype(np.float32)

    # Neighbor past (neighbor_num, past_len, 11)
    neighbor_agents_past = np.zeros((neighbor_num, past_len, 11), dtype=np.float32)
    neighbor_agents_past[0, :, 0] = x_past
    neighbor_agents_past[0, :, 1] = y_past + 2.0
    neighbor_agents_past[0, :, 2] = 1.0
    neighbor_agents_past[0, :, 3] = 0.0
    neighbor_agents_past[0, :, 4] = speed
    neighbor_agents_past[0, :, 5] = 0.0
    neighbor_agents_past[0, :, 6] = 4.5
    neighbor_agents_past[0, :, 7] = 2.0
    neighbor_agents_past[0, :, 8] = 1.0  # vehicle type one-hot

    # Neighbor future (neighbor_num, future_len, 3)
    neighbor_agents_future = np.zeros((neighbor_num, future_len, 3), dtype=np.float32)
    neighbor_agents_future[0, :, 0] = x_fut
    neighbor_agents_future[0, :, 1] = y_fut + 2.0
    neighbor_agents_future[0, :, 2] = 0.0

    # Static objects (static_num, 10)
    static_objects = np.zeros((static_num, 10), dtype=np.float32)

    # Lanes and routes
    lanes = np.zeros((lane_num, lane_points, 12), dtype=np.float32)
    lanes_speed_limit = np.zeros((lane_num, 1), dtype=np.float32)
    lanes_has_speed_limit = np.zeros((lane_num, 1), dtype=np.bool_)

    route_lanes = np.zeros((route_num, lane_points, 12), dtype=np.float32)
    route_lanes_speed_limit = np.zeros((route_num, 1), dtype=np.float32)
    route_lanes_has_speed_limit = np.zeros((route_num, 1), dtype=np.bool_)

    return {
        "ego_agent_past": ego_agent_past,
        "ego_current_state": ego_current_state,
        "ego_agent_future": ego_agent_future,
        "neighbor_agents_past": neighbor_agents_past,
        "neighbor_agents_future": neighbor_agents_future,
        "static_objects": static_objects,
        "lanes": lanes,
        "lanes_speed_limit": lanes_speed_limit,
        "lanes_has_speed_limit": lanes_has_speed_limit,
        "route_lanes": route_lanes,
        "route_lanes_speed_limit": route_lanes_speed_limit,
        "route_lanes_has_speed_limit": route_lanes_has_speed_limit,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3402)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    past_len = 21
    future_len = 80
    neighbor_num = 32
    lane_num = 70
    lane_points = 20
    route_num = 25
    static_num = 5

    data_list = []
    for idx in range(args.num_samples):
        sample = _make_sample(
            rng,
            past_len,
            future_len,
            neighbor_num,
            lane_num,
            lane_points,
            route_num,
            static_num,
        )
        fname = f"synthetic_{idx:04d}.npz"
        np.savez(out_dir / fname, **sample)
        data_list.append(fname)

    with open(out_dir / "data_list.json", "w") as f:
        json.dump(data_list, f, indent=2)

    print(f"Wrote {len(data_list)} samples to {out_dir}")


if __name__ == "__main__":
    main()
