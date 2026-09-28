# Phage-Bacteria Spatial Dynamics Simulation

An agent-based computational biology project in Python modeling spatial interactions between bacterial populations, bacteriophages (phages), substrate dynamics, and quorum sensing signals. Part of the last two week project in the Computational Biology course in University of Amsterdam.

## Overview

This project simulates the spatio-temporal dynamics of bacterial growth, infection, movement, and chemical diffusion on a 2D spatial grid. The framework models four interacting components:

1. **Bacterial Cells (`Cell`):** Individual agents undergoing substrate-dependent growth, random walks, division, and phage infection.
2. **Substrate Diffusion (`SubstrateDiffuser`):** Nutrient grid supporting cell growth and mass accumulation.
3. **Phage Diffusion (`PhageDiffuser`):** Viral particles that diffuse through the domain, adsorb onto susceptible cells, and propagate infections.
4. **Quorum Sensing (`QuorumDiffuser`):** Autoinducer molecule diffusion enabling density-dependent behaviors across cell variants.

## Features

- **Multiple Bacterial Phenotypes:** Support for different behavioral traits (e.g., standard `Cell`, `RunningCell`, `HidingCell`).
- **Multi-Grid Diffusion Model:** Multi-scale spatial grids allowing fine or coarse resolution for diffusing chemical species relative to cellular grids.
- **Dynamic Visualization:** Live animation rendering standard/infected cells alongside 2D field heatmaps for Substrate $[S]$, Phage $[P]$, and Quorum sensing $[Q]$ concentrations.
- **Data Export:** Automatic saving of spatial simulation videos (`.mp4`) and time-series metrics (`.csv`).

## Installation
1. Clone the repository to your local device
2. Create a virtual environment: 
    - `python -m venv venv` 
3. Activate the environment: 
    - `source venv/bin/activate` (Unix) 
    - `venv\Scripts\activate` (Windows)
3. Install the required packages for Python 3.9+ (Python 3.12 was used by us): 
    - `pip install fipy==4.0.3 pandas==3.0.6 numpy==2.5.3 matplotlib==3.11.2`

 NOTE: For the saving of the simulation videos `ffmpeg` is required.

 ## Usage
 Run the `main.py` script from inside the project folder. You can optionally use the following arguments:
 - `--cell-type`: specifies the type of cell to simulate (Choose: `Cell`, `HidingCell`, `RunningCell`).
 - `--live`: tells the simulation to show a live visualization of the simulation.
 - `-o` / `--output`: tells the simulation to save the results using the specified name.

 Example: `python3 main.py --cell-type RunningCell --live -o running_cell_sim`

 ## Output
 The metrics tracked by the simulation are the following:
 - `time`: Elapsed simulation time
 - `healthy_cells`: Number of healthy cells
 - `infected_cells`: Number of infected cells
 - `mean_c_substrate`: Mean substrate concentration
 - `mean_c_phage`: Average phage density
 - `mean_c_quorum`: Mean quorum sensing inducer concentration

## Credits and Attribution
This project was largely inspired by the following paper: 

"A Quorum-Sensing-Induced Bacteriophage Defense Mechanism", [Høyland-Kroghsbo et. al (2013)](https://doi.org/10.1128/mbio.00362-12).

## Known Issues
- Performance was less of a goal in this two week project, so improvements on this front can be made by removing loops and moving to numpy vectorizations and element-wise operations.
- Simulation will crash if no cells are present at the time of the phage injection.
- Directory `project/results/` needs to be created manually if it is missing.
