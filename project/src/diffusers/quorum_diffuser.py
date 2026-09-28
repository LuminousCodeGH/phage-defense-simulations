import numpy as np
from fipy import CellVariable, Grid2D, DiffusionTerm, TransientTerm


class QuorumDiffuser():
    def __init__(self, grid: dict[str, np.ndarray], quorum_params: dict[str, float], cell_params: dict[str, float]):
        self.quorum_params = quorum_params
        self.cell_params = cell_params

        grid_size = self._get_grid_size(grid)
        self._quorum_dx = self.cell_dx / (self._get_grid_size(grid, subgrid='cell') / grid_size)
        self.mesh = Grid2D(dx=self.quorum_dx, dy=self.quorum_dx, nx=grid_size, ny=grid_size)
        self.quorum = CellVariable(name='quorum', mesh=self.mesh, value=quorum_params.get('c_init', 0.0))
    
    @property
    def diffusion_coefficient(self) -> float:
        return self.quorum_params['D']

    @property
    def production_rate(self) -> float:
        return self.quorum_params['r_production']
    
    @property
    def t_production_factor(self) -> float:
        return self.quorum_params['t_production_factor']
    
    @property
    def max_concentration(self) -> float:
        return self.quorum_params['c_max']
    
    @property
    def movement_threshold(self) -> float:
        return self.quorum_params['c_thresh']
    
    @property
    def cell_dx(self) -> float:
        return self.cell_params['dx']
    
    @property
    def quorum_dx(self) -> float:
        return self._quorum_dx
    
    @staticmethod
    def _get_grid_size(grid: dict[str, np.ndarray], subgrid: str='quorum') -> int:
        return grid[subgrid].shape[0]

    def _generate_infection_map(self, grid: dict[str, np.ndarray]) -> np.ndarray:
        '''Map cell mass from cell grid to quorum grid'''
        cell_grid = grid['cell']
        quorum_grid = grid['quorum']
        scale = quorum_grid.shape[0] // cell_grid.shape[0]
        infection_map = np.zeros_like(quorum_grid)

        for row in range(cell_grid.shape[0]):
            for col in range(cell_grid.shape[1]):
                cell = cell_grid[row, col]
                if cell is not None:
                    if not cell.infected or cell.infection_progression > cell.t_burst*self.t_production_factor:
                        continue  # Healthy cells do not produce quorum and infected cells stop producing quorum after some time
                    r0, r1 = row * scale, (row + 1) * scale
                    c0, c1 = col * scale, (col + 1) * scale
                    infection_map[r0:r1, c0:c1] = 1 / scale**2  # Equalize the mass across the new resolution grid

        return infection_map

    def diffuse(self, dt: float, grid: dict[str, np.ndarray]) -> np.ndarray:
        '''Diffuse the quorum across the grid space'''
        infection_map = self._generate_infection_map(grid)

        biomass_term = CellVariable(name='biomass', mesh=self.mesh, value=infection_map.ravel())
        production_term = self.production_rate * biomass_term

        diffusion_equation = TransientTerm() == DiffusionTerm(coeff=self.diffusion_coefficient) + production_term

        diffusion_equation.solve(var=self.quorum, dt=dt)

        return self.mesh_to_numpy(grid)

    def mesh_to_numpy(self, grid: dict[str, np.ndarray]) -> np.ndarray:
        return np.array(self.quorum.value, dtype=np.float64).reshape((self._get_grid_size(grid), self._get_grid_size(grid)))
