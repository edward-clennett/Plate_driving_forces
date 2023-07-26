# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PLATEFO
# Algorithm to calculate plate forces from tectonic reconstructions
# Thomas Schouten and Edward Clennett, 2023
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Import packages
import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import newton
import matplotlib.pyplot as plt
from gplately import pygplates

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to find optimal slab pull coefficient and mantle drag
# plate_torques

def optimise_coefficients(plate_torques, options, plates_of_interest=None, grid_size=500):
    # Generate grid
    viscs = np.linspace(4.33e19,3.19e+20,500)
    sp_consts = np.linspace(0.05,1,500)
    visc_grid = np.repeat(viscs[:, np.newaxis], 500, axis=1)
    sp_const_grid = np.repeat(sp_consts[np.newaxis, :], 500, axis=0)
    ones_grid = np.ones_like(visc_grid)

    plt.imshow(visc_grid)
    plt.show()
    plt.imshow(sp_const_grid)
    plt.show()
    # Filter torques
    torques = plate_torques.copy()
    if plates_of_interest:
        torques.plateID = torques.plateID.astype(int)
        torques = torques[torques.plateID.isin(plates_of_interest)]

    residual_magnitude = np.zeros_like(sp_const_grid)

    # Get torques
    for k, _ in enumerate(plates_of_interest):
        residual_x = np.zeros_like(sp_const_grid); residual_y = np.zeros_like(sp_const_grid); residual_z = np.zeros_like(sp_const_grid)

        if options["Slab pull torque"] and "slab_pull_torque_x" in torques.columns:
            residual_x += torques.slab_pull_torque_x.iloc[k] * sp_const_grid
            residual_y += torques.slab_pull_torque_y.iloc[k] * sp_const_grid
            residual_z += torques.slab_pull_torque_z.iloc[k] * sp_const_grid

        # Add GPE torque
        if options["GPE torque"] and "GPE_torque_x" in torques.columns:
            residual_x += torques.GPE_torque_x.iloc[k] * ones_grid
            residual_y += torques.GPE_torque_y.iloc[k] * ones_grid
            residual_z += torques.GPE_torque_z.iloc[k] * ones_grid

        # Compute magnitude of driving torque
        driving_magnitude += np.sqrt(residual_x**2 + residual_y**2 + residual_z**2) / torques.area.iloc[k]

        # Add Bending torque
        if options["Bending torque"] and "bending_torque_x" in torques.columns:
            residual_x += torques.bending_torque_x.iloc[k] * ones_grid
            residual_y += torques.bending_torque_y.iloc[k] * ones_grid
            residual_z += torques.bending_torque_z.iloc[k] * ones_grid

        # Add mantle drag torque
        if options["Mantle drag torque"] and "mantle_drag_torque_x" in torques.columns:
            residual_x += torques.mantle_drag_torque_x.iloc[k] * visc_grid /150e3
            residual_y += torques.mantle_drag_torque_y.iloc[k] * visc_grid /150e3
            residual_z += torques.mantle_drag_torque_z.iloc[k] * visc_grid /150e3

        # Compute magnitude of residual
        residual_magnitude += np.sqrt(residual_x**2 + residual_y**2 + residual_z**2) / torques.area.iloc[k]

        # Divide residual by driving torque
        residual_normalised = residual_magnitude / driving_magnitude

    p = plt.imshow(residual_normalised)
    plt.colorbar(p)

    opt_val = np.amin(residual_normalised)
    print(opt_val)
    
    # Find the indices of the minimum value directly using np.argmin
    opt_i, opt_j = np.unravel_index(np.argmin(residual_magnitude), residual_magnitude.shape)
    opt_visc = viscs[opt_i]
    opt_sp_const = sp_const[opt_j]
    
    print("Optimum viscosity [Pa s]:", opt_visc)
    print("Optimum Drag Coefficient [Pa s/m]:", opt_visc / 150e3)
    print("Optimum Slab Pull constant:", opt_sp_const)

    return opt_visc, opt_sp_const