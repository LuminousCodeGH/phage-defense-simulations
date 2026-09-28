import numpy as np
from fipy import CellVariable, Grid2D, DiffusionTerm, TransientTerm


class SubstrateDiffuser():
    def __init__(self, grid: dict[str, np.ndarray], substrate_params: dict[str, float], cell_params: dict[str, float]):
        self.substrate_params = substrate_params
        self.cell_params = cell_params

        grid_size = self._get_grid_size(grid)
        self._subs_dx = self.cell_dx / (self._get_grid_size(grid, subgrid='cell') / grid_size)
        self.mesh = Grid2D(dx=self.subs_dx, dy=self.subs_dx, nx=grid_size, ny=grid_size)
        self.substrate = CellVariable(name='substrate', mesh=self.mesh, value=substrate_params['c_init'])
    
    @property
    def diffusion_coefficient(self) -> float:
        return self.substrate_params['D']

    @property
    def decay_rate(self) -> float:
        return self.substrate_params.get('r_decay', 0)
    
    @property
    def yield_xs(self) -> float:
        return self.substrate_params['Y_xs']
    
    @property
    def inflow_rate(self) -> float:
        return self.substrate_params['r_in']
    
    @property
    def max_concentration(self) -> float:
        return self.substrate_params['c_max']
    
    @property
    def mu_max(self) -> float:
        return self.cell_params['mu_max']
    
    @property
    def substrate_saturation_constant(self) -> float:
        return self.cell_params['Ks']
    
    @property
    def cell_dx(self) -> float:
        return self.cell_params['dx']
    
    @property
    def subs_dx(self) -> float:
        return self._subs_dx
    
    @staticmethod
    def _get_grid_size(grid: dict[str, np.ndarray], subgrid: str='substrate') -> int:
        return grid[subgrid].shape[0]
    
    def _generate_cell_mass_map(self, grid: dict[str, np.ndarray]) -> np.ndarray:
        '''Map cell mass from cell grid to substrate grid'''
        cell_grid = grid['cell']
        subs_grid = grid['substrate']
        scale = subs_grid.shape[0] // cell_grid.shape[0]
        cell_mass_map = np.zeros_like(subs_grid)

        for row in range(cell_grid.shape[0]):
            for col in range(cell_grid.shape[1]):
                cell = cell_grid[row, col]
                if cell is not None:
                    if cell.infected:
                        continue  # Infected cells do not eat according to our model
                    mass = cell.mass
                    r0, r1 = row * scale, (row + 1) * scale
                    c0, c1 = col * scale, (col + 1) * scale
                    cell_mass_map[r0:r1, c0:c1] = mass / scale**2  # Equalize the mass across the new resolution grid

        return cell_mass_map

    def diffuse(self, dt: float, grid: dict[str, np.ndarray]) -> np.ndarray:
        '''Diffuse the substrate across the grid space'''
        cell_mass_map = self._generate_cell_mass_map(grid)

        biomass_term = CellVariable(name='biomass', mesh=self.mesh, value=cell_mass_map.ravel())
        monod_term = (self.substrate / (self.substrate_saturation_constant + self.substrate)) * biomass_term
        consumption_term = (self.mu_max / self.yield_xs) * monod_term
        inflow_term = self.inflow_rate

        diffusion_equation = TransientTerm() == DiffusionTerm(coeff=self.diffusion_coefficient) - consumption_term + inflow_term

        diffusion_equation.solve(var=self.substrate, dt=dt)

        self.substrate.setValue(np.minimum(self.substrate.value, self.max_concentration))
        return self.mesh_to_numpy(grid)

    def mesh_to_numpy(self, grid: dict[str, np.ndarray]) -> np.ndarray:
        return np.array(self.substrate.value).reshape((self._get_grid_size(grid), self._get_grid_size(grid)))
