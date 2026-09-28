import numpy as np
import random
from src.diffusers.substrate_diffuser import SubstrateDiffuser
from src.diffusers.phage_diffuser import PhageDiffuser
from src.diffusers.quorum_diffuser import QuorumDiffuser


class Cell():
    def __init__(self, row: int, col: int, 
                 adsorption_constant: float,
                 cell_params: dict[str, float],
                 substrate_diffuser: SubstrateDiffuser,
                 phage_diffuser: PhageDiffuser,
                 quorum_diffuser: QuorumDiffuser):
        self.row = row
        self.col = col
        self._adsorption_constant = adsorption_constant
        self.adsorption_mod = 1.0
        self._mass_init = cell_params['m_init']
        self.mass = self._mass_init
        self.infection_progression = 0
        self.cell_params = cell_params
        self.weights = {'random_walk': np.array([2**-0.5, 1, 2**-0.5, 1, 1, 1, 2**-0.5, 1, 2**-0.5])}
        self.infected = False
        self.moved_last_turn = False
        self.dead = False
        self.available_spots = []
        self.available_spot_w = self.weights['random_walk']
        self.substrate_diffuser = substrate_diffuser
        self.phage_diffuser = phage_diffuser
        self.quorum_diffuser = quorum_diffuser

    @property
    def adsorption_constant(self) -> float:
        return self._adsorption_constant

    @property
    def adsoption_net(self) -> float:
        return self._adsorption_constant * self.adsorption_mod
    
    @property
    def coord(self) -> tuple[int, int]:
        return self.row, self.col
    
    @property
    def mass_init(self) -> float:
        return self._mass_init
    
    @property
    def mu_max(self) -> float:
        return self.cell_params['mu_max']
    
    @property
    def substrate_saturation_constant(self) -> float:
        return self.cell_params['Ks']
    
    @property
    def t_burst(self) -> float:
        return self.cell_params['t_burst']
    
    def _update_free_spots(self, grid: dict[str, np.ndarray]) -> None:
        '''Check every spot in order of TL TM TR, CL CM CR, BL BM BR'''
        self.available_spots = []
        self.available_spot_w = np.zeros(9)
        wi = 0
        for r in range(self.row-1, self.row+2):
            for c in range(self.col-1, self.col+2):
                coord = (r, c)
                self.available_spots.append(coord)
                if 0 <= r < grid['cell'].shape[0] and 0 <= c < grid['cell'].shape[1]:  # Check boundries
                    if grid['cell'][r ,c] is None or grid['cell'][r, c] is self:  # Check grid for free spots (including self)
                        self.available_spot_w[wi] = self.weights['random_walk'][wi]
                wi += 1
    
    def _get_average_at_pos(self, grid: dict[str, np.ndarray], coord_check: tuple[int, int]=None, compound: str='substrate') -> float:
        '''Get the average substrate concentration for this cellular grid square'''
        size_factor = int(grid[compound].shape[0] / grid['cell'].shape[0])
        if coord_check is None:
            coord_check = self.coord
        row, col = coord_check
        
        # Get top-left corner in substrate grid
        start_row = row * size_factor
        start_col = col * size_factor
        
        if compound == 'substrate':
            # Get the corresponding patch of substrate
            patch = self.substrate_diffuser.mesh_to_numpy(grid)[start_row:start_row + size_factor,
                                                                start_col:start_col + size_factor]
        elif compound == 'phage':
            patch = self.phage_diffuser.mesh_to_numpy(grid)[start_row:start_row + size_factor,
                                                            start_col:start_col + size_factor]
        elif compound == 'quorum':
            patch = self.quorum_diffuser.mesh_to_numpy(grid)[start_row:start_row + size_factor,
                                                            start_col:start_col + size_factor]
            
        else:
            raise ValueError(f'"compound" does not match any compound found in the model! ({compound})')
        
        average_concentration = np.mean(patch)
        return average_concentration
    
    def _check_division(self, grid: dict[str, np.ndarray]) -> object | None:
        if self.mass >= 2 * self.mass_init:
            new_cell = self._divide(grid)
            if new_cell is None:
                return  # Try again next time
            self.mass = self.mass_init
            return new_cell

    def _divide(self, grid: dict[str, np.ndarray]) -> 'Cell':
        self._update_free_spots(grid)
        # Filter out zero-weighted spots
        filtered_spots = []
        filtered_weights = []
        for spot, weight in zip(self.available_spots, self.available_spot_w):
            if weight > 0 and spot != self.coord:
                filtered_spots.append(spot)
                filtered_weights.append(weight)

        if not filtered_spots:
            print(f'Division failed at {self.coord}: No space')
            return  # No spaces for division
        
        new_spot = random.choices(filtered_spots, weights=filtered_weights)[0]
        new_cell = Cell(new_spot[0], new_spot[1], 
                self.adsorption_constant,
                self.cell_params,
                self.substrate_diffuser,
                self.phage_diffuser,
                self.quorum_diffuser)
        return new_cell

    def _roll_infection_probability(self, grid: dict[str, np.ndarray]) -> bool:
        p_infection = 1 - np.exp(-self.adsoption_net * self._get_average_at_pos(grid, compound='phage'))
        if random.random() < p_infection and p_infection > 0.00001:
            print(f'Infected cell at position {self.coord}')
            self.infected = True

    def do_random_walk(self, grid: dict[str, np.ndarray]) -> None:
        '''Calculate which free space the cell will move to given all equal probability'''
        if self.infected:
            return  # Do not move
        
        self._roll_infection_probability(grid)
        self._update_free_spots(grid)

        # Filter out zero-weighted spots
        filtered_spots = []
        filtered_weights = []
        for spot, weight in zip(self.available_spots, self.available_spot_w):
            if weight > 0:
                filtered_spots.append(spot)
                filtered_weights.append(weight)

        if filtered_spots:
            new_spot = random.choices(filtered_spots, weights=filtered_weights)[0]
            if new_spot == self.coord:
                self.moved_last_turn = False
                self.adsorption_mod = 1.0
            else:
                self.moved_last_turn = True
                self.adsorption_mod = 1.5
            self.row, self.col = new_spot

    def grow(self, dt: float, grid: dict[str, np.ndarray]) -> object | None:
        '''Calculate how much the cells will grow by in this time step. Also check if they are ready for division'''
        if self.infected:
            if self.infection_progression >= self.t_burst:
                print(f'Cell burst at position {self.coord}')
                self.phage_diffuser.release_phage(self.row, self.col, self.mass_init, self.mass, grid)
                self.dead = True
            self.infection_progression += dt
            return  # Do not grow in mass
        average_substrate_conc = self._get_average_at_pos(grid)
        delta_mass = self.mass * self.mu_max * (average_substrate_conc / (self.substrate_saturation_constant + average_substrate_conc))
        self.mass += delta_mass * dt
        new_cell = self._check_division(grid)
        return new_cell