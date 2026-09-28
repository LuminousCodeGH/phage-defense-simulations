from src.simulation_controller import SimulationController
from src.cells.running_cell import RunningCell
from src.cells.hiding_cell import HidingCell
from src.cells.cell import Cell
from data.parameters import *
import argparse


CELL_TYPES = {
    'Cell': Cell,
    'RunningCell': RunningCell,
    'HidingCell': HidingCell,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run phage simulation with specified bacterial cell types')
    parser.add_argument(
        '--cell-type',
        type=str,
        choices=list(CELL_TYPES.keys()),
        default='Cell',
        help='Type of cell to model in the simulation (default: Cell)')
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='File name of the output video and data (default: qs_baseline)')
    parser.add_argument(
        '--live',
        action='store_true',
        help='Show a live visualization of the simulation (warning, likely very slow!)'
    )
    args = parser.parse_args()

    print('-------------------------------------------------------')
    print(f'Output file name: {args.output}')
    print(f'Simulation cell type: {args.cell_type}')
    print(f'Running live: {args.live} (always live if no output!)')
    print('-------------------------------------------------------')

    # Retrieve the class based on user argument
    selected_cell_type = CELL_TYPES[args.cell_type]

    params = {
        'cell_params': cell_params, 
        'substrate_params': substrate_params,
        'phage_params': phage_params,
        'quorum_params': quorum_params
    }
    grid_length = grid_params['length']
    size_factor = grid_params['size_factor']
    dt = grid_params['dt']
    grid_sizes = {'cell': grid_length, 'substrate': grid_length*size_factor, 'phage': grid_length*size_factor, 'quorum': grid_length*size_factor}
    sim = SimulationController(
        grid_sizes, 
        dt, 
        steps=int(1 / dt), 
        initial_cells=300, 
        initial_infected=0, 
        params=params, 
        cell_type=selected_cell_type)
    sim.simulate(t_inject_phage=dt*60*5, sim_title=f'QS {args.cell_type} Simulation', file_name=args.output, live_vis=args.live)
