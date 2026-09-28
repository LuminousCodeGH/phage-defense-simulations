import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from matplotlib import colormaps
from typing import List
from src.cells.cell import Cell
from src.diffusers.substrate_diffuser import SubstrateDiffuser
from src.diffusers.phage_diffuser import PhageDiffuser
from src.diffusers.quorum_diffuser import QuorumDiffuser


class SimulationController:
    '''Simulation controller to simulate the cells move grow and divide'''
    def __init__(self, grid_sizes: dict[str, int], dt: float, steps: int, initial_cells: int, initial_infected: int, params: dict[str, dict], cell_type: type[Cell]):
        self.cell_grid_size = grid_sizes['cell']
        self.cell_grid = np.full((self.cell_grid_size, self.cell_grid_size), None)
        self.subs_grid_size = grid_sizes['substrate']
        self.subs_grid = np.zeros((self.subs_grid_size, self.subs_grid_size))
        self.phage_grid_size = grid_sizes['phage']
        self.phage_grid = np.zeros((self.phage_grid_size, self.phage_grid_size))
        self.quorum_grid_size = grid_sizes['quorum']
        self.quorum_grid = np.zeros((self.quorum_grid_size, self.quorum_grid_size))
        self.cell_type = cell_type
        self.cells: List[Cell] = []
        self.healthy_cells = 0
        self.infected_cells = 0
        self.t_inject_phage = 0
        self.substrate_diffuser: SubstrateDiffuser = None
        self.phage_diffuser: PhageDiffuser = None
        self.quorum_diffuser: QuorumDiffuser = None
        self.steps = steps
        self.dt = dt
        self.t = 0
        self.combined_cmap = None
        self.sim_data: dict[str, list] = {
            'time': [],
            'healthy_cells': [],
            'infected_cells': [],
            'mean_c_substrate': [],
            'mean_c_phage': [],
            'mean_c_quorum': []
        }

        print('-------------------------------------------------------')
        print('Starting simulation with grid sizes:')
        print(f'\t{self.cell_grid_size=}')
        print(f'\t{self.subs_grid_size=}')
        print(f'\t{self.phage_grid_size=}')
        print(f'\t{self.quorum_grid_size=}')
        print(f'Starting with {initial_cells=} and {initial_infected=}')
        print('-------------------------------------------------------')

        assert grid_sizes['substrate'] % grid_sizes['cell'] == 0, 'The substrate grid size must be an even multiple of the cell grid!'
        assert grid_sizes['phage'] % grid_sizes['cell'] == 0, 'The phage grid sizes must be an even multiple of the cell grid!'

        self._init_substrate(params)
        self._init_phage(params)
        self._init_quorum(params)
        self._init_cells(initial_cells, initial_infected, params)

    @property
    def grid(self) -> dict[str, np.ndarray]:
        return {'cell': self.cell_grid, 'substrate': self.subs_grid, 'phage': self.phage_grid, 'quorum': self.quorum_grid}

    @property
    def grid_size(self) -> dict[str, int]:
        return {'cell': self.cell_grid.shape[0], 'substrate': self.subs_grid.shape[0], 'phage': self.phage_grid.shape[0], 'quorum': self.quorum_grid.shape[0]}

    def _step_cells(self, dt: float) -> None:
        '''Updates the cells every step'''
        new_cells = []
        alive_cells = []
        self.healthy_cells = 0
        self.infected_cells = 0

        for cell in self.cells:
            self.cell_grid[cell.row, cell.col] = None
            if cell.dead:
                continue  # Skip dead cells
            
            # Otherwise, process live cell
            if cell.infected:
                self.infected_cells += 1
            else:
                self.healthy_cells += 1
            new_cell = cell.grow(dt, self.grid)
            cell.do_random_walk(self.grid)

            self.cell_grid[cell.row, cell.col] = cell
            alive_cells.append(cell)

            if new_cell is not None:
                self.cell_grid[new_cell.row, new_cell.col] = new_cell
                new_cells.append(new_cell)

        self.cells = alive_cells + new_cells

    def _step_diffusion(self, dt: float) -> None:
        self.subs_grid = self.substrate_diffuser.diffuse(dt, self.grid)
        self.phage_grid = self.phage_diffuser.diffuse(dt, self.grid)
        self.quorum_grid = self.quorum_diffuser.diffuse(dt, self.grid)

    def _init_cells(self, num_cells: int, num_infected: int, params: dict[str, dict]):
        placed = 0
        while placed < num_cells:
            row = random.randint(0, self.grid_size['cell'] - 1)
            col = random.randint(0, self.grid_size['cell'] - 1)
            if self.grid['cell'][row, col] is None:
                cell = self.cell_type(
                    row=row,
                    col=col,
                    adsorption_constant=0.5,
                    cell_params=params['cell_params'],
                    substrate_diffuser=self.substrate_diffuser,
                    phage_diffuser=self.phage_diffuser,
                    quorum_diffuser=self.quorum_diffuser
                )
                # Randomize their starting mass
                cell.mass = np.random.uniform(cell.mass_init, 2 * cell.mass_init)

                self.cell_grid[row, col] = cell
                self.cells.append(cell)
                placed += 1

        for cell in random.choices(self.cells, k=num_infected):
            cell.infected = True
        
        inferno = colormaps['inferno'](np.linspace(0, 1, 128))
        viridis = colormaps['viridis'](np.linspace(0, 1, 128))
        combined_colors = np.vstack((inferno, viridis))
        self.combined_cmap = ListedColormap(combined_colors)
    
    def _init_substrate(self, params: dict[str, dict]) -> None:
        self.substrate_diffuser = SubstrateDiffuser(self.grid, params['substrate_params'], params['cell_params'])

    def _init_phage(self, params: dict[str, dict]) -> None:
        self.phage_diffuser = PhageDiffuser(self.grid, params['phage_params'], params['cell_params'])

    def _init_quorum(self, params: dict[str, dict]) -> None:
        self.quorum_diffuser = QuorumDiffuser(self.grid, params['quorum_params'], params['cell_params'])

    def _inject_phage(self) -> None:
        self.phage_grid = self.phage_diffuser.release_phage(25, 25, self.cells[0].mass_init, self.cells[0].mass_init, self.grid)

    def _save_data(self, file_name: str) -> None:
        pd.DataFrame(self.sim_data).to_csv(f'results/{file_name}.csv')

    def step(self, dt: float):
        self.sim_data['time'].append(self.t)
        self.sim_data['healthy_cells'].append(self.healthy_cells)
        self.sim_data['infected_cells'].append(self.infected_cells)
        self.sim_data['mean_c_substrate'].append(np.mean(self.grid['substrate']))
        self.sim_data['mean_c_phage'].append(np.mean(self.grid['phage']))
        self.sim_data['mean_c_quorum'].append(np.mean(self.grid['quorum']))
        if self.t >= self.t_inject_phage:
            self._inject_phage()
            self.t_inject_phage = np.inf
            print('Injecting phage')
        self._step_cells(dt)
        self._step_diffusion(dt)
        self.substrate_diffuser.grid = self.grid
        self.phage_diffuser.grid = self.grid
        self.quorum_diffuser.grid = self.grid
        self.t += dt

    def simulate(self, t_inject_phage: float = 0, do_animation: bool = True, interval: int=1, sim_title: str='Simulation', file_name: str=None, live_vis: bool=False):
        dt = self.dt
        self.t_inject_phage = t_inject_phage
        if do_animation:
            self.animate_simulation(dt, interval=interval, sim_title=sim_title, file_name=file_name, live_vis=live_vis)
        else:
            raise NotImplementedError()

    def animate_simulation(self, dt: float, interval: int = 1, sim_title: str='Simulation', file_name: str=None, live_vis: bool=False):
        '''Animate the simulation that is being run'''
        fig, ax = plt.subplots(2, 2, figsize=(12, 12))
        suptitle = fig.suptitle('', fontsize=16)
        cell_grid_size = self.grid_size['cell']
    
        # Create a visual grid: 0 = empty, 1 = cell
        cell_display = np.zeros((cell_grid_size, cell_grid_size))
        im_X = ax[0][0].imshow(cell_display, cmap=self.combined_cmap, vmin=-1, vmax=1)
        im_S = ax[0][1].imshow(self.grid['substrate'], cmap='inferno', vmin=0, vmax=self.substrate_diffuser.max_concentration)
        im_P = ax[1][0].imshow(self.grid['phage'], cmap='inferno', vmin=0, vmax=1)
        im_Q = ax[1][1].imshow(self.grid['quorum'], cmap='inferno', vmin=0, vmax=self.quorum_diffuser.max_concentration)

        # Add colorbar
        cbar = plt.colorbar(im_X, ax=ax[0][0])

        # Custom tick labels
        cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
        cbar.set_ticklabels(["High $M_{Y}$", "Low $M_{Y}$", "0", "Low $M_{X}$", "High $M_{X}$"])

        def update(frame):
            self.step(dt)
            
            # Update display grid
            cell_display[:] = 0  # Clear the grid
            for cell in self.cells:
                infection_factor = -1 if cell.infected else 1
                cell_display[cell.row, cell.col] = min(cell.mass, cell.mass_init * 2) / (cell.mass_init * 2) * infection_factor

            im_X.set_array(cell_display)
            im_S.set_array(self.grid['substrate'])
            im_P.set_array(self.grid['phage'])
            im_Q.set_array(self.grid['quorum'])
            suptitle.set_text(f'{sim_title} (t={(frame+1)*dt*60:0.3f}min)')
            ax[0][0].set_title(f"Cells, Total={len(self.cells)}")
            ax[0][1].set_title(f"[S], Mean={np.mean(self.grid['substrate'])*10**6:0.3f} g/mL")
            ax[1][0].set_title(f"[P], Mean={np.mean(self.grid['phage']):0.3f} PFU/um^3")
            ax[1][1].set_title(f"[Q], Mean={np.mean(self.grid['quorum'])*10**9:0.3f} mg/mL")
            return [im_X, im_S, im_P, im_Q]

        ani = FuncAnimation(fig, update, frames=self.steps, interval=interval, blit=False)
        if file_name is None or live_vis == True: 
            plt.show()
        elif file_name is not None:
            ani.save(f"./results/{file_name}.mp4", fps=60, dpi=200, bitrate=1800, extra_args=['-vcodec', 'libx264'])
            self._save_data(file_name)
