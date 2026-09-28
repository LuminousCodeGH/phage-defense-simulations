import numpy as np
import random
from src.diffusers.substrate_diffuser import SubstrateDiffuser
from src.diffusers.phage_diffuser import PhageDiffuser
from src.diffusers.quorum_diffuser import QuorumDiffuser
from src.cells.cell import Cell


class HidingCell(Cell):
    def do_random_walk(self, grid: dict[str, np.ndarray]) -> None:
        '''Move with a bias away from high quorum concentration'''
        if self.infected:
            return  # Infected cells do not move

        self._roll_infection_probability(grid)
        self._update_free_spots(grid)
        grid_rows, grid_cols = grid['cell'].shape

        # Get quorum concentrations at all 9 neighboring positions
        quorum_values = []
        for spot in self.available_spots:
            quorum = self._get_average_at_pos(grid, coord_check=spot, compound='quorum')
            quorum_values.append(quorum)

        # Invert quorum values to bias movement away from quorum
        max_quorum = max(quorum_values)
        stealth_coefficient = 10*(2.5 + np.clip(max_quorum / self.quorum_diffuser.max_concentration, 0, 1)*7.5)

        # Filter out zero-weighted and out-of-bound spots
        filtered_spots = []
        filtered_weights = []
        for spot, weight in zip(self.available_spots, self.weights['random_walk']):
            r, c = spot
            if 0 <= r < grid_rows and 0 <= c < grid_cols and weight > 0:
                filtered_spots.append(spot)
                filtered_weights.append(weight)

        if filtered_spots:
            new_spot = random.choices(filtered_spots, weights=filtered_weights)[0]
            if new_spot == self.coord:
                self.moved_last_turn = False
                self.adsorption_mod = 1.0 / stealth_coefficient
            else:
                self.moved_last_turn = True
                self.adsorption_mod = 1.5 / stealth_coefficient
            self.row, self.col = new_spot
