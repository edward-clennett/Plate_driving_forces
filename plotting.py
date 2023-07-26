# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PLATEFO
# Algorithm to calculate plate forces from tectonic reconstructions
# Thomas Schouten, 2023
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Import packages
import xarray as xr
import numpy as np
import gplately
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cmcrameri as cmc

# def plot_(reconstruction_time, slabs, plate_torques, raster, reconstruction_files):
#     fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": ccrs.Robinson(central_longitude=30)},
#                                         figsize=(16, 8), dpi=300)

#     gplot = set_gplot(reconstruction_time, reconstruction_files)

#     ax1, slab_pull = plot_vectors(ax1, slabs.lon, slabs.lat, slabs.effective_slab_pull_force_lon, slabs.effective_slab_pull_force_lat, "force", gplot)
#     ax3 = plot_centroid_forces(ax3, plate_torques, ["slab_pull"])

#     ax2, velocities = plot_vectors(ax2, slabs.lon, slabs.lat, slabs.v_theoretical_lon, slabs.v_theoretical_lat, "velocity", gplot)

#     plt.tight_layout()
#     plt.show()

#     return fig

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to plot forces at centroid
# ax:                   axes object.
# plate_torques:        pd.DataFrame with plate_torques
# variable:             variable of interest.

def plot_torques(reconstruction_time, slabs, plate_torques, raster, reconstruction_files):
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson(central_longitude=30)},
                                        figsize=(8, 4), dpi=300)

    # Set the gplot object to plot reconstructed coastlines
    gplot = set_gplot(reconstruction_time, reconstruction_files)

    # Normalise the vectors
    # scale = np.sqrt(plate_torques.slab_pull_force_lon.max()**2 + plate_torques.slab_pull_force_lat.max()**2)
    scale = 4e18
    
    ax, sed_grid = plot_raster_data(ax, raster, "age", gplot)
    ax, slab_pull = plot_vectors(ax, slabs.lon, slabs.lat, slabs.effective_slab_pull_force_lon, slabs.effective_slab_pull_force_lat, "force")
    ax = plot_centroid_forces(ax, plate_torques, ["slab_pull", "GPE"])

    plt.tight_layout()
    plt.show()

    return fig
    
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to plot forces at centroid
# ax:                   axes object.
# plate_torques:        pd.DataFrame with plate_torques
# variable:             variable of interest.

# Set gplot object
def set_gplot(reconstruction_time, reconstruction_files):

    # Plot reconstructed features
    if reconstruction_files and reconstruction_time:
        
        # Unpack files
        rotation_file, topology_features, coastlines = reconstruction_files

        # Set plate reconstruction
        plate_reconstruction = gplately.PlateReconstruction(rotation_file, topology_features, coastlines)

        # Set gplot object
        gplot = gplately.PlotTopologies(plate_reconstruction, coastlines=coastlines, time=reconstruction_time)
        
    return gplot


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to plot forces at centroid
# ax:                   axes object.
# plate_torques:        pd.DataFrame with plate_torques
# variable:             variable of interest.
    
def plot_centroid_forces(ax, plate_torques, forces):
    for force in forces:
        vectors = ax.quiver(
            x=plate_torques.centroid_lon,
            y=plate_torques.centroid_lat,
            u=plate_torques[force + "_force_lon"],
            v=plate_torques[force + "_force_lat"],
            transform=ccrs.PlateCarree(),
            label=force.capitalize(),
            width=3e-3,
            # scale=scale,
            zorder=2
        )
    
    return ax

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to plot raster
# ax:                   axes object.
# slabs:                pd.DataFrame with slab data.
# variable:             variable of interest.

def plot_raster_data(ax, raster, variable, gplot=None):
    # Plot basemap
    if gplot:
        ax = plot_basemap(ax, gplot, plot_trenches=True)

    # Plot variables with standardised colourmaps
    if "age" in variable:
        cmap = mpl.colormaps["cmc.lajolla"]
        vmin = 0
        vmax = 250
    elif "sed" in variable:
        cmap = mpl.colormaps["cmc.imola"]
        vmin = 0
        vmax = 700

    # Plot grid
    grid = ax.imshow(
        raster[variable], 
        cmap=cmap, 
        transform=ccrs.PlateCarree(), 
        zorder=1, 
        vmin=vmin, 
        vmax=vmax, 
        origin='lower'
    )

    return ax, grid

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to plot slab data
# ax:                   axes object.
# slabs:                pd.DataFrame with slab data.
# variable:             variable of interest.

def plot_point_data(ax, data, variable, gplot=None):
    # Plot basemap
    if gplot:
        ax = plot_basemap(ax, gplot)

    # Plot variables with standardised colourmaps
    if "age" in variable:
        cmap = mpl.colormaps["cmc.lajolla"]
        vmin = 0
        vmax = 150
    elif "sed" in variable:
        cmap = mpl.colormaps["cmc.imola"]
        vmin = 0
        vmax = 700
    elif "force" in variable:
        cmap = mpl.colormaps["cmc.batlow"]
        vmin = 12.6
        vmax = 14.2
    if "viscosity" in variable:
        cmap = mpl.colormaps["cmc.davos_r"]
        vmin = 19
        vmax = 21
    points = ax.scatter(
        data.lon, 
        data.lat, 
        s=10, 
        c=data[variable], 
        cmap=cmap, 
        transform=ccrs.PlateCarree(), 
        zorder=3, 
        vmin=vmin, 
        vmax=vmax
    )

    return ax, points

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to plot vectors on map
# ax:                   axes object.
# slabs:                pd.DataFrame with slab data.
# vel_lon:              longitudinal component of velocity.
# vel_lat:              latitudinal component of velocity.

def plot_vectors(ax, lon, lat, v_lon, v_lat, type, gplot=None):
    # Plot basemap
    if gplot:
        ax = plot_basemap(ax, gplot)

    # Plot vectors
    # if type == "force" and scale is None:
    #     scale = 2.5e15
    # if type == "velocity" and scale is None:
    #     scale = 2.5e-8
    vectors = ax.quiver(
        lon, 
        lat, 
        v_lon, 
        v_lat, 
        transform=ccrs.PlateCarree(), 
        # scale=scale, 
        width=1e-3
    )
    
    return ax, vectors

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to get correct reconstruction name
# identifier:           Reconstruction identifier string for the GPlately DataServer.

def reconstruction_name(identifier):
    if identifier == "Seton2012":
        reconstruction_name1 = "Seton et al. (2012)"
        reconstruction_name2 = "(Seton et al., 2012)"
    elif identifier == "Muller2016":
        reconstruction_name1 = "Müller et al. (2016)"
        reconstruction_name2 = "(Müller et al., 2016)"
    elif identifier == "Muller2019":
        reconstruction_name1 = "Müller et al. (2019)"
        reconstruction_name2 = "(Müller et al., 2019)"
    elif identifier == "Clennett2020":
        reconstruction_name1 = "Clennett et al. (2020)"
        reconstruction_name2 = "(Clennett et al., 2020)"
    
    return reconstruction_name1, reconstruction_name2

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to plot basemap.
# ax:                       axes object.
# reconstruction_files:     tuple of rotation_file, topology_features and coastlines 

def plot_basemap(ax, gplot=None, plot_trenches=True, plot_subduction_teeth=False, plot_coastlines=True):
    # Plot trenches and coastlines (optional)
    if gplot:
        # Plot trenches
        if plot_trenches:
            # Plot subduction teeth
            if plot_subduction_teeth:
                gplot.plot_trenches(ax, zorder=2, lw=0.75, color="k")
                gplot.plot_subduction_teeth(ax, zorder=2, color="k")
            else:
                gplot.plot_trenches(ax, zorder=2, lw=1, color="k")
                
        # Plot coastlines
        if plot_coastlines:
            gplot.plot_coastlines(ax, color="lightgrey")

    # Set labels
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Set global extent
    ax.set_global()

    # Set gridlines
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(), 
        draw_labels=True, 
        linewidth=0.5, 
        color='gray', 
        alpha=0.5, 
        linestyle='--', 
        zorder=5
    )

    gl.bottom_labels = False
    gl.right_labels = False  

    return ax