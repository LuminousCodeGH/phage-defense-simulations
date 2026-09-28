import numpy as np
from fipy import CellVariable, Grid2D, DiffusionTerm, TransientTerm


class PhageDiffuser():
    def __init__(self, grid: np.ndarray, phage_params: dict[str, float], cell_params: dict[str, float]):
        self.phage_params = phage_params
        self.cell_params = cell_params

        grid_size = self._get_grid_size(grid)
        self._phage_dx = self.cell_dx / (self._get_grid_size(grid, subgrid='cell') / grid_size)
        self.mesh = Grid2D(dx=self.phage_dx, dy=self.phage_dx, nx=grid_size, ny=grid_size)
        self.phage = CellVariable(name='phage', mesh=self.mesh, value=phage_params.get('c_init', 0.0))
    
    @property
    def diffusion_coefficient(self) -> float:
        return self.phage_params['D']

    @property
    def decay_rate(self) -> float:
        return self.phage_params.get('r_decay', 0)
    
    @property
    def burst_amount(self) -> float:
        return self.phage_params['n_burst']
    
    @property
    def max_concentration(self) -> float:
        return self.phage_params['c_max']
    
    @property
    def cell_dx(self) -> float:
        return self.cell_params['dx']
    
    @property
    def phage_dx(self) -> int:
        return self._phage_dx
    
    @staticmethod
    def _get_grid_size(grid: dict[str, np.ndarray], subgrid: str='phage') -> int:
        return grid[subgrid].shape[0]

    def diffuse(self, dt: float, grid: dict[str, np.ndarray]) -> np.ndarray:
        '''Diffuse the phage across the grid space'''
        decay_term = self.decay_rate * self.phage

        diffusion_equation = TransientTerm() == DiffusionTerm(coeff=self.diffusion_coefficient) - decay_term

        diffusion_equation.solve(var=self.phage, dt=dt)

        return self.mesh_to_numpy(grid)

    def release_phage(self, cell_row: int, cell_col: int, mass_init: float, mass: float, grid: dict[str, np.ndarray]):
        '''Inject phages at the cell location corresponding to a lysed cell. Called from Cell'''
        # Determine scaling from cell grid to phage grid
        phage_grid = grid['phage']
        cell_grid = grid['cell']
        scale = phage_grid.shape[0] // cell_grid.shape[0]

        # Get phage grid indices
        r0, r1 = cell_row * scale, (cell_row + 1) * scale
        c0, c1 = cell_col * scale, (cell_col + 1) * scale

        # Distribute the burst phage evenly across the corresponding phage grid region
        patch_volume = scale**2
        delta_value = (self.burst_amount / patch_volume) * (mass / mass_init)
        print(f'Releasing {delta_value} new phages per square')

        # Inject phages into the internal FiPy CellVariable (flattened 1D)
        phage_2d = self.mesh_to_numpy(grid)
        phage_2d[r0:r1, c0:c1] += delta_value
        self.phage.setValue(phage_2d.ravel())  # Update the CellVariable with new values
        return self.mesh_to_numpy(grid)

    def mesh_to_numpy(self, grid: dict[np.ndarray]) -> np.ndarray:
        return np.array(self.phage.value, dtype=np.float64).reshape((self._get_grid_size(grid), self._get_grid_size(grid)))
