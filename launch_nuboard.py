"""Launch NuBoard for the VectorFlow closed-loop simulation run (random weights)."""

import os
import sys

os.environ.setdefault('NUPLAN_EXP_ROOT',  '/scratch3/jainil.bavishi/flowplanner_exp')
os.environ.setdefault('NUPLAN_DATA_ROOT', '/fast_scratch/autodp/nuplan/dataset')
os.environ.setdefault('NUPLAN_MAPS_ROOT', '/fast_scratch/autodp/nuplan/dataset/maps')

sys.path.insert(0, '/home/AutoDP/jainil.bavishi/nuplan-devkit')

import nest_asyncio
nest_asyncio.apply()

from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.nuboard.nuboard import NuBoard
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping

NUBOARD_FILE = ('/scratch3/jainil.bavishi/flowplanner_exp/exp/simulation/'
                'closed_loop_nonreactive_agents/vectorflow/mini/random_weights/'
                'custom_model_2026-06-14-11-22-42/nuboard_1781436181.nuboard')
DATA_ROOT    = '/fast_scratch/autodp/nuplan/dataset/nuplan-v1.1/splits/mini'
MAPS_ROOT    = '/fast_scratch/autodp/nuplan/dataset/maps'
SENSOR_ROOT  = '/fast_scratch/autodp/nuplan/dataset/nuplan-v1.1/sensor_blobs'
PORT         = 5007  # different port so both can run side-by-side

print('Building scenario builder...')
DB_FILES = [
    DATA_ROOT + '/2021.06.09.14.58.55_veh-35_01894_02311.db',  # changing_lane_to_left / starting_unprotected_cross_turn
    DATA_ROOT + '/2021.05.12.23.36.44_veh-35_01133_01535.db',  # starting_protected_noncross_turn
    DATA_ROOT + '/2021.06.07.12.54.00_veh-35_01843_02314.db',  # following_lane_with_lead
    DATA_ROOT + '/2021.06.08.14.35.24_veh-26_02555_03004.db',  # near_multiple_vehicles
    DATA_ROOT + '/2021.05.25.14.16.10_veh-35_01690_02183.db',  # starting_left_turn
    DATA_ROOT + '/2021.07.16.18.06.21_veh-38_04933_05307.db',  # stopping_at_stop_sign_with_lead
]

scenario_builder = NuPlanScenarioBuilder(
    data_root=DATA_ROOT,
    map_root=MAPS_ROOT,
    sensor_root=SENSOR_ROOT,
    db_files=DB_FILES,
    map_version='nuplan-maps-v1.0',
    scenario_mapping=ScenarioMapping(scenario_map={}, subsample_ratio_override=0.5),
    vehicle_parameters=get_pacifica_parameters(),
    verbose=True,
)

print(f'Launching nuBoard at http://localhost:{PORT}')
print(f'Simulation: VectorFlow custom_model (random weights) — 10 mini scenarios')
nuboard = NuBoard(
    nuboard_paths=[NUBOARD_FILE],
    scenario_builder=scenario_builder,
    vehicle_parameters=get_pacifica_parameters(),
    port_number=PORT,
    async_scenario_rendering=True,
)
nuboard.run()
