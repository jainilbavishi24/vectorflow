"""Preprocess nuPlan logs into VectorFlow .npz training files.

Uses VectorFlow's DataProcessor (which saves ego_agent_past, required by NuPlanDataset).

Usage:
    # Quick smoke test (50 scenarios from mini split):
    python preprocess_nuplan.py --split mini --save_dir /scratch3/jainil.bavishi/flowplanner_npz/trainval --limit 50

    # Full trainval:
    python preprocess_nuplan.py --split trainval --save_dir /scratch3/jainil.bavishi/flowplanner_npz/trainval
"""

import sys
sys.path.insert(0, '/home/AutoDP/jainil.bavishi/nuplan-devkit')

import argparse
import json
import os
import traceback
from pathlib import Path

import numpy as np
from tqdm import tqdm

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_sequential import Sequential

from vectorflow.data.data_process.data_processor import DataProcessor
from vectorflow.data.data_process.agent_process import agent_past_process

# ── Paths ──────────────────────────────────────────────────────────────────────
NUPLAN_DATA_ROOT = '/fast_scratch/autodp/nuplan/dataset/nuplan-v1.1'
NUPLAN_MAPS_ROOT = '/scratch3/jainil.bavishi/nuplan_maps/maps'
MAP_VERSION      = 'nuplan-maps-v1.0'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--split',    default='mini',
                   help='Dataset split: trainval, test, mini')
    p.add_argument('--save_dir', default='/scratch3/jainil.bavishi/flowplanner_npz/trainval',
                   help='Directory to write .npz files and data list JSON')
    p.add_argument('--num_scenarios_per_type', type=int, default=None,
                   help='Max scenarios per scenario type (None = all)')
    p.add_argument('--limit', type=int, default=None,
                   help='Hard cap on total scenarios (useful for smoke tests)')
    p.add_argument('--shuffle', action='store_true', default=True,
                   help='Shuffle scenario order')
    p.add_argument('--db_subset', type=int, default=None,
                   help='Only use the first N .db files (for faster testing)')
    return p.parse_args()


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    data_root = os.path.join(NUPLAN_DATA_ROOT, args.split)

    # ── discover db files ─────────────────────────────────────────────────────
    split_dir = os.path.join(NUPLAN_DATA_ROOT, 'splits', args.split)
    if os.path.isdir(split_dir):
        db_files = sorted(str(p) for p in Path(split_dir).glob('*.db'))
    else:
        db_files = None  # use all in data_root

    if db_files and args.db_subset:
        db_files = db_files[:args.db_subset]
        print(f'Using {len(db_files)} .db files (subset)')
    elif db_files:
        print(f'Using {len(db_files)} .db files from splits/{args.split}/')

    # ── build scenarios ───────────────────────────────────────────────────────
    print('Building scenario builder...')
    scenario_builder = NuPlanScenarioBuilder(
        data_root=data_root,
        map_root=NUPLAN_MAPS_ROOT,
        sensor_root=None,
        db_files=db_files,
        map_version=MAP_VERSION,
    )

    scenario_filter = ScenarioFilter(
        scenario_types=None,
        scenario_tokens=None,
        log_names=None,
        map_names=None,
        num_scenarios_per_type=args.num_scenarios_per_type,
        limit_total_scenarios=args.limit,
        timestamp_threshold_s=None,
        ego_displacement_minimum_m=None,
        ego_start_speed_threshold=None,
        ego_stop_speed_threshold=None,
        speed_noise_tolerance=None,
        expand_scenarios=True,
        remove_invalid_goals=True,
        shuffle=args.shuffle,
    )

    worker = Sequential()

    print('Collecting scenarios...')
    scenarios = scenario_builder.get_scenarios(scenario_filter, worker)

    if args.limit:
        scenarios = scenarios[:args.limit]

    print(f'Total scenarios to process: {len(scenarios)}')

    # ── process + save ────────────────────────────────────────────────────────
    processor = DataProcessor(save_dir=str(save_dir))

    success_files = []
    fail_tokens   = []

    for scenario in tqdm(scenarios, desc='Processing scenarios'):
        map_name = scenario._map_name
        token    = scenario.token
        fname    = f'{map_name}_{token}.npz'

        out_path = save_dir / fname
        if out_path.exists():
            success_files.append(fname)
            continue

        try:
            processor.scenario = scenario
            processor.map_api  = scenario.map_api

            ego_agent_past, time_stamps_past = processor.get_ego_agent()

            (neighbor_agents_past, neighbor_agents_types,
             static_objects, static_objects_types) = processor.get_neighbor_agents()

            ego_agent_past, neighbor_agents_past, neighbor_indices, static_objects = \
                agent_past_process(
                    ego_agent_past, neighbor_agents_past, neighbor_agents_types,
                    processor.num_agents, static_objects, static_objects_types,
                    processor.num_static, processor.max_ped_bike)

            vector_map = processor.get_map()

            ego_agent_future       = processor.get_ego_agent_future()
            neighbor_agents_future = processor.get_neighbor_agents_future(neighbor_indices)

            ego_agent_past, ego_current_state = processor.calculate_additional_ego_states(
                ego_agent_past, time_stamps_past)

            data = {
                'map_name':               map_name,
                'token':                  token,
                'ego_agent_past':         ego_agent_past,
                'ego_current_state':      ego_current_state,
                'ego_agent_future':       ego_agent_future,
                'neighbor_agents_past':   neighbor_agents_past,
                'neighbor_agents_future': neighbor_agents_future,
                'static_objects':         static_objects,
            }
            data.update(vector_map)
            np.savez(str(out_path), **data)

            success_files.append(fname)

        except Exception as e:
            fail_tokens.append(token)
            tqdm.write(f'[SKIP] {token}: {e}')
            if os.environ.get('DEBUG_PREPROCESS'):
                traceback.print_exc()

    # ── write data list JSON ──────────────────────────────────────────────────
    list_path = save_dir / 'data_list.json'
    with open(list_path, 'w') as f:
        json.dump(success_files, f, indent=2)

    fail_path = save_dir / 'fail_tokens.json'
    with open(fail_path, 'w') as f:
        json.dump(fail_tokens, f, indent=2)

    print(f'\nDone.')
    print(f'  Saved:  {len(success_files)} .npz files → {save_dir}')
    print(f'  Failed: {len(fail_tokens)} scenarios      → {fail_path}')
    print(f'  List:   {list_path}')
    print()
    print('To train, set in launch_train.sh:')
    print(f'  TRAINING_DATA={save_dir}')
    print(f'  TRAINING_JSON={list_path}')


if __name__ == '__main__':
    main()
