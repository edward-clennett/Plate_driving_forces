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
# Class containing mechanical parameters used in calculations.

class set_mech_params:
    def __init__(self):

        # Mechanical and rheological parameters:
        self.g = 9.81                                       # gravity [m/s2]
        self.dT = 1200                                      # mantle-surface T contrast [K]
        self.rho0 = 3300                                    # reference mantle density  [kg/m3]
        self.rho_w = 1000                                   # water density [kg/m3]
        self.rho_sw = 1020                                  # water density for plate model
        self.rho_s = 2650                                   # density of sediments (quartz sand)
        self.rho_c = 2868                                   # density of continental crust
        self.rho_l = 3412                                   # lithosphere density
        self.rho_a = 3350                                   # asthenosphere density 
        self.alpha = 3e-5                                   # thermal expansivity [K-1]
        self.kappa = 1e-6                                   # thermal diffusivity [m2/s]
        self.depth = 700e3                                  # slab depth [m]
        self.rad_curv = 390e3                               # slab curvature [m]
        self.L = 130e3                                      # compensation depth [m]
        self.L0 = 100e3                                     # lithospheric shell thickness [m]
        self.La = 200e3                                     # asthenospheric thickness [m]
        self.visc_a = 1e20                                  # reference astheospheric viscosity [Pa s]
        self.lith_visc = 500e20                             # lithospheric viscosity [Pa s]
        self.lith_age_RP = 60                               # age of oldest sea-floor in approximate ridge push calculation  [Ma]
        self.yield_stress = 1050e6                          # Byerlee yield strength at 40km, i.e. 60e6 + 0.6*(3300*10.0*40e3) [Pa]
        self.cont_lith_thick = 100e3                        # continental lithospheric thickness (where there is no age) [m]
        self.cont_crust_thick = 33e3                        # continental cruustal thickness (where there is no age) [m]
        self.ocean_crust_thick = 8e3                        # oceanic crustal thickness [m]

        # Derived parameters
        self.drho_slab = self.rho0 * self.alpha * self.dT   # Density contrast between slab and surrounding mantle [kg/m3]
        self.drhio_sed = self.rho_s - self.rho0             # Density contrast between sediments (quartz sand) and surrounding mantle [kg/m3]

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Class containing constants and conversions used calculations.

class set_constants:
    def __init__(self):

        # Constants
        self.mean_Earth_radius_km = 6371                # mean Earth radius [km]
        self.mean_Earth_radius_m = 6371e3               # mean Earth radius [m]
        self.equatorial_Earth_radius_m = 6378.1e3       # Earth radius at equator
        self.equatorial_Earth_circumference = 40075e3   # Earth circumference at equator [m]
        
        # Conversions
        self.ma2s = 1e6 * 365.25 * 24 * 60 * 60         # Ma to s
        self.s2ma = 1 / self.ma2s                       # s to Ma
        self.ms2cma = 1e2 / (365.25 * 24 * 60 * 60)    # m/s to cm/a 
        self.cma2ms = 1 / self.ms2cma                   # cm/a to m/s
        self.rada2cma = 1e-5                            # rad/a to cm/a
        self.cma2rada = 1 / self.rada2cma               # cm/a to rad/a

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# COMPUTE TORQUES
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to compute torques on tectonic plates
# plates:               pd.DataFrame containing plate data (generated during preprocessing).
# slabs:                pd.DataFrame containing processed data of tesselated subduction zones (generated during preprocessing).
# seafloor:             xr.DataSet containing seafloor of seafloor ages and (optionally) sediment thicknesses.
# age_variable:         Name of the xr.DataSet variable containing oceanic ages.
# coords_name:          Name of the xr.DataSet variable names for latitude and longitude, respectively. 

def compute_torques(plates, subduction_zones, grid, seafloor, cases, options):

    # Initialise dictionaries to store torques, slabs and points
    torques = {}
    slabs = {}
    points = {}

    # Loop through cases
    for case in cases:

        # Copy the plates DataFrame to store torques
        torques[case] = plates.copy()
        slabs[case] = subduction_zones.copy()
        points[case] = grid.copy()

        #---------------------#
        #   DRIVING TORQUES   #
        #---------------------#

        # Calculate slab pull torque
        if options[case]["Slab pull torque"]:
            slabs[case] = subduction_zones.copy
            slabs[case] = slab_pull_force(subduction_zones, seafloor, options[case])
            torques[case] = torque_on_plates(
                torques[case], 
                slabs[case].lat, 
                slabs[case].lon, 
                slabs[case].lower_plateID, 
                slabs[case].effective_slab_pull_force_lat, 
                slabs[case].effective_slab_pull_force_lon,
                segment_length=slabs[case].trench_segment_length,
                torque_variable="slab_pull_torque"
            )

        # Calculate GPE torque
        if options[case]["GPE torque"]:
            points[case] = gpe_force(points[case], seafloor, options[case])
            torques[case] = torque_on_plates(
                torques[case], 
                points[case].lat, 
                points[case].lon, 
                points[case].plateID, 
                points[case].GPE_force_lat, 
                points[case].GPE_force_lon,
                torque_variable="GPE_torque"
            )
            
        #-----------------------#
        #   RESISTIVE TORQUES   #
        #-----------------------#

        # Calculate slab bending torque
        if options[case]["Bending torque"]:
            if options[case]["Slab pull torque"]:
                slabs[case] = slab_bending_force(slabs, options)
            else:
                slabs[case] = slab_bending_force(slabs, options, seafloor)
            
            torques[case] = torque_on_plates(
                torques[case], 
                slabs[case].lat, 
                slabs[case].lon, 
                slabs[case].lower_plateID, 
                slabs[case].bending_force_lat, 
                slabs[case].bending_force_lon,
                segment_length=slabs[case].trench_segment_length,
                torque_variable="slab_bending_torque"
            )

        # Calculate Mantle drag torque
        if options[case]["Mantle drag torque"]:
            torques[case], points[case] = mantle_drag_force(torques[case], points[case], options[case])
            torques[case] = torque_on_plates(
                torques[case], 
                points[case].lat, 
                points[case].lon, 
                points[case].plateID, 
                points[case].mantle_drag_force_lat, 
                points[case].mantle_drag_force_lon,
                torque_variable="mantle_drag_torque"
            )

    # Return all DataFrames
    return torques, slabs, points

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SUBDUCTION ZONES
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to calculate i) slab pull force and theoretical subduction velocity at subduction zones
# slabs:                pd.DataFrame with subduction zone data
# seafloor:               seafloor containing ocean age and optionally sediment thickness distribution
# options:              dictionary containing options for calculation

def slab_pull_force(slabs, seafloor, options):

    # Set mechanical parameters and constants
    mech = set_mech_params()
    constants = set_constants()
    
    # Sample age and arc type of upper plate from seafloor
    slabs["upper_plate_age"] = np.nan; slabs["continental_arc"] = False
    slabs["upper_plate_age"], slabs["continental_arc"] = sample_slabs_from_seafloor(
        slabs.lat, 
        slabs.lon,
        slabs.trench_normal_azimuth,  
        seafloor,
        options,
        "upper plate"
    )

    # Calculate upper plate thickness
    slabs["upper_plate_thickness"] = np.nan
    slabs["upper_plate_thickness"], crust_thickness, water_depth = calculate_thicknesses(
        slabs.upper_plate_age,
        options,
        crust=False,
        water=False
    )

    # Sample age and sediment thickness of lower plate from seafloor
    slabs["lower_plate_age"] = np.nan; slabs["sediment_thickness"] = np.nan
    slabs["lower_plate_age"], slabs["sediment_thickness"] = sample_slabs_from_seafloor(
        slabs.lat, 
        slabs.lon,
        slabs.trench_normal_azimuth,
        seafloor, 
        options,
        "lower plate"
    )

    # Calculate lower plate thickness
    slabs["lower_plate_thickness"] = np.nan
    slabs["lower_plate_thickness"], crust_thickness, water_depth = calculate_thicknesses(
        slabs.lower_plate_age,
        options,
        crust = False, 
        water = False
    )

    # Calculate slab pull force acting on point along subduction zone
    slab_pull_force = np.where(np.isnan(slabs.lower_plate_age), 0, slabs["lower_plate_thickness"] * mech.depth * mech.drho_slab * mech.g)
    
    # Add a layer of sediment on top (optionally):
    if options["Sediment subduction"]:
        slab_pull_force_ref = slab_pull_force
        slab_pull_force += np.where(np.isnan(slabs.sediment_thickness), 0, slabs["sediment_thickness"] * mech.depth * mech.drho_sed * mech.g)

    # Calculate theoretical subduction velocity and effective slab pull force
    v_plate_guess = 1 * constants.cma2ms
    v_plate = []
    if options["Sediment subduction"] == True:

        # Calculate sediment fraction
        sed_frac = slabs["sediment_thickness"] / options["Shear zone width"]
        sed_frac[sed_frac > 1] = 1

        # Initialise empty list:
        slab_pull_efficiency = []

        # Calculate theoretical subduction velocity with and without sediments
        for i in range(len(slabs.lat)):
            v_plate_sed = newton(conrad_hager, v_plate_guess, args=(sed_frac[i], 
                                                                slabs.lower_plate_thickness[i], 
                                                                slab_pull_force[i], 
                                                                slabs.upper_plate_thickness[i], 
                                                                mech, 
                                                                options))
            v_plate_ref = newton(conrad_hager, v_plate_guess, args=(np.zeros(len(slabs.lat))[i], 
                                                                slabs.lower_plate_thickness[i], 
                                                                slab_pull_force_ref[i], 
                                                                slabs.upper_plate_thickness[i], 
                                                                mech, 
                                                                options))

            # Append values to list and calculate slab_pull_efficiency ratio
            v_plate.append(v_plate_sed * constants.ms2cma)
            slab_pull_efficiency.append(v_plate / v_plate_ref)

        # Calculate effective slab pull force
        slabs["effective_slab_pull_force"] = slab_pull_force * slab_pull_efficiency

    else:
        # Initialise empty list:
        slab_pull_efficiency = []        # Calculate theoretical subduction velocity

        # Calculate theoretical subduction velocity with and without sediments
        for i in range(len(slabs.lat)):
            v_plate_ref = newton(conrad_hager, v_plate_guess, args=(np.zeros(len(slabs.lat))[i], 
                                                            slabs.lower_plate_thickness[i], 
                                                            slab_pull_force[i], 
                                                            slabs.upper_plate_thickness[i], 
                                                            mech, 
                                                            options))
            # Append values to list and calculate slab_pull_efficiency ratio
            v_plate.append(v_plate_ref * constants.ms2cma)

        # Effective slab pull force is equal to slab pull force
        slabs["effective_slab_pull_force"] = slab_pull_force
    
    # Store theoretical subduction velocities
    slabs["v_theoretical"] = np.array(v_plate) / constants.ms2cma
    slabs["v_theoretical_lat"], slabs["v_theoretical_lon"] = mag_azi2lat_lon(slabs.v_theoretical, slabs.trench_normal_azimuth)

    # Decompose into latitudinal and longitudinal components
    slabs["effective_slab_pull_force_lat"], slabs["effective_slab_pull_force_lon"] = mag_azi2lat_lon(slabs["effective_slab_pull_force"], slabs["trench_normal_azimuth"])

    return slabs

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to calculate the slab bending force acting on 
# slabs:                pd.DataFrame containing processed data of tesselated subduction zones.
# seafloor:             xr.DataSet containing seafloor of age of the downgoing plate.
# age_variable:         Name of the xr.DataSet variable containing oceanic ages.
# coords_name:          Name of the xr.DataSet variable names for latitude and longitude, respectively. 

def slab_bending_force(slabs, options, seafloor=None):
    # Set mechanical parameters
    mech = set_mech_params()

    # Check if "lower_plate_thickness is not in columns"
    if not options["Slab pull torque"] and seafloor:
        # Sample age and arc type of upper plate from seafloor
        slabs["upper_plate_age"] = np.nan; slabs["continental_arc"] = False
        slabs["upper_plate_age"], slabs["continental_arc"] = sample_slabs_from_seafloor(
            slabs.lat, 
            slabs.lon,
            slabs.trench_normal_azimuth,  
            seafloor,
            options,
            "upper plate"
        )

        # Calculate upper plate thickness
        slabs["upper_plate_thickness"] = np.nan
        slabs["upper_plate_thickness"], crust_thickness, water_depth = calculate_thicknesses(
            slabs.upper_plate_age,
            options,
            crust=False,
            water=False
        )
    elif not options["Slab pull torque"] and not seafloor:
        print("No slab bending torque calculated! Please provide a seafloor age grid.")

    # Calculate slab bending torque
    if options["Bending mechanism"] == "viscous":
        bending_force = (-2. / 3.) * ((slabs.lower_plate_thickness) / (mech.rad_curv)) ** 3 * mech.lith_visc * slabs.v_convergence  # [n-s , e-w], [N/m]
    elif options["Bending mechanism"] == "plastic":
        bending_force = (-1. / 6.) * ((slabs.lower_plate_thickness ** 2) / mech.rad_curv) * mech.yield_stress * np.array(
            (np.cos(slabs.trench_normal_vector + slabs.obliquity_convergence), np.sin(slabs.trench_normal_vector + slabs.obliquity_convergence)))  # [n-s, e-w], [N/m]
        
    slabs["bending_force_lat"] = np.nan; slabs["bending_force_lon"] = np.nan
    slabs["bending_force_lat"], slabs["bending_force_lon"] = mag_azi2lat_lon(bending_force, slabs.trench_normal_vector + slabs.obliquity_convergence)
    
    return slabs

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to obtain relevant upper or lower plate data from tesselated subduction zones.
# slabs:                pd.DataFrame containing processed data of tesselated subduction zones.
# seafloor:               xr.DataSet containing seafloor of age of the downgoing plate.
# age_variable_name:    Name of the xr.DataSet variable containing oceanic ages.
# coords_name:          Name of the xr.DataSet variable names for latitude and longitude, respectively. 

def sample_slabs_from_seafloor(lat, lon, trench_normal_azimuth, seafloor, options, plate, age_variable="age", sediment_variable="passive_margin_sediment", coords=["lat", "lon"], continental_arc=None):
    # Load seafloor into memory to decrease computation time
    seafloor = seafloor.load()

    # Define sampling distance [km]
    if plate == "lower plate":
        initial_sampling_distance = -30
    if plate == "upper plate":
        initial_sampling_distance = 200

    # Sample lower plate
    sampling_lat, sampling_lon = project_points(lat, lon, trench_normal_azimuth, initial_sampling_distance)

    # Extract latitude and longitude values from slabs and convert to xarray DataArrays
    sampling_lat_da = xr.DataArray(sampling_lat, dims="point")
    sampling_lon_da = xr.DataArray(sampling_lon, dims="point")

    # Extract the variable values from the seafloor dataset using the indices
    variable_names = list(seafloor.data_vars.keys())  # Get all variable names from the xr.Dataset

    # Interpolate age value at point
    ages = seafloor[age_variable].interp({coords[0]: sampling_lat_da, coords[1]: sampling_lon_da}).values.tolist()

    # Find problematic indices to iteratively find age of lower plate
    initial_mask = np.isnan(ages)
    mask = initial_mask

    # Define sampling distance [km] and number of iterations
    if plate == "lower plate":
        current_sampling_distance = initial_sampling_distance - 30
        iterations = 10
    if plate == "upper plate":
        current_sampling_distance = initial_sampling_distance + 200
        iterations = 8

    for i in range(iterations):
        sampling_lat[mask], sampling_lon[mask] = project_points(lat[mask], lon[mask], trench_normal_azimuth[mask], current_sampling_distance)
        sampling_lat_da = xr.DataArray(sampling_lat, dims="point")
        sampling_lon_da = xr.DataArray(sampling_lon, dims="point")
        ages = np.where(mask, seafloor[age_variable].interp({coords[0]: sampling_lat_da, coords[1]: sampling_lon_da}).values.tolist(), ages)
        mask = np.isnan(ages)

        # Define new sampling distance
        if plate == "lower plate":
            if i <= 2:
                current_sampling_distance -= 30
            elif i > 2:
                current_sampling_distance -= 60
            elif i > 4:
                current_sampling_distance -= 120
            elif i < 6:
                current_sampling_distance -= 240
            elif i < 8:
                current_sampling_distance -= 480

        if plate == "upper plate":
                current_sampling_distance += 100

    # Check whether arc is continental or not
    if plate == "upper plate":
        continental_arc = np.isnan(ages)

        # Close the seafloor to free memory space
        seafloor.close()

        return ages, continental_arc
    
    if plate == "lower plate":

        # Sample sediment thickness grids (optional)
        sediment_thickness = np.zeros(len(lat))
        if options["Sediment subduction"]:

            if options["Sample sediment grid"]:
                sediment_thickness += np.array(seafloor["passive_margin_sediment"].interp({coords[0]: sampling_lat, coords[1]: sampling_lon}).values.tolist())

            # if options["Equatorial bulge sediments"]
            #     if "equatorial_bulge_sediment" in variable_names:
            #         sediment_thickness += np.array(seafloor["equatorial_bulge_sediment"].interp({coords[0]: sampling_lat, coords[1]: sampling_lon}).values.tolist())
            #         print

            # if options["Active margin sediments"] and continental_arc is not None:
            #     sediment_thickness += options["continental_weathering_rate"] * 

        # Close the seafloor to free memory space
        seafloor.close()

        return ages, sediment_thickness

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to calculate coordinates of sampling points
# lon:      column of pd.DataFrame containing longitudes.
# lat:      column of pd.DataFrame containing latitudes.
# azimuth:  column of pd.DataFrame containing trench normal azimuth.

def project_points(lat, lon, azimuth, distance):
    # Set constants
    constants = set_constants()

    # Convert to radians
    lon_radians = np.deg2rad(lon)
    lat_radians = np.deg2rad(lat)
    azimuth_radians = np.deg2rad(azimuth)

    # Angular distance in km
    angular_distance = distance / constants.mean_Earth_radius_km

    # Calculate sample points
    new_lat_radians = np.arcsin(np.sin(lat_radians) * np.cos(angular_distance) + np.cos(lat_radians) * np.sin(angular_distance) * np.cos(azimuth_radians))
    new_lon_radians = lon_radians + np.arctan2(np.sin(azimuth_radians) * np.sin(angular_distance) * np.cos(lat_radians), np.cos(angular_distance) - np.sin(lat_radians) * np.sin(new_lat_radians))
    new_lon = np.degrees(new_lon_radians)
    new_lat = np.degrees(new_lat_radians)

    return new_lat, new_lon

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to calculate subduction velocity according to semi-empirical formulation from Conrad & Haeger (1999)
# sediment_fraction:        np.array containing latitude of subduction zone segment.
# lower_plate_thickness:    np.array containing latitude of subduction zone segment.
# lower_plate_thickness:    np.array containing latitude of subduction zone segment.

def conrad_hager(v_plate, sediment_fraction, lower_plate_thickness, slab_pull, upper_plate_thickness, mech, options):
    # Define empirical constants
    Cs = 1.2
    Cf = 1
    Cm = 2.5
    Cl = 2.5
    A = 1.25

    lf = upper_plate_thickness / np.sin(np.deg2rad(30))

    # Calculate interface viscosity as a function of sediment fraction
    if options["Interface mixing"] == "linear":
        log_interface_viscosity = 19 + 2 * (1 - sediment_fraction)
    if options["Interface mixing"] == "threshold":
        if sediment_fraction > 0.7:
            log_interface_viscosity = 19
        else:
            log_interface_viscosity = 19 + 2 * (1 - sediment_fraction / 0.7)
    
    interface_viscosity = 10 ** log_interface_viscosity
    
    residual = (Cs * slab_pull -
                    Cf * lf * 2 * interface_viscosity * (v_plate / options["Shear zone width"])) / (
                    3 * mech.visc_a * (A + Cm) + Cl * mech.lith_visc * (lower_plate_thickness / mech.rad_curv) ** 3) - v_plate

    return residual

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# GRAVITATIONAL POTENTIAL ENERGY
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to calculate GPE force at points 
# torques:                  pd.DataFrame containing 
# points:                   pd.DataFrame containing data of points including columns with latitude, longitude and plateID
# options:                  dictionary with options
# age_variable:             name of variable in xr.dataset containing seafloor ages

def gpe_force(points, seafloor, options, age_variable="age"):
    # Set mechanical parameters and constants
    mech = set_mech_params()
    constants = set_constants()

    # Get grid spacing
    grid_spacing_deg = points.lon[1] - points.lon[0]

    # Get nearby points
    # Longitude
    dx_lon = points.lon + 0.5 * grid_spacing_deg
    minus_dx_lon = points.lon - 0.5 * grid_spacing_deg

    # Adjust for dateline
    dx_lon = np.where(dx_lon > 180, dx_lon - 360, dx_lon)
    minus_dx_lon = np.where(minus_dx_lon < -180, minus_dx_lon + 360, minus_dx_lon)

    # Latitude
    dy_lat = points.lat + 0.5 * grid_spacing_deg
    minus_dy_lat = points.lat - 0.5 * grid_spacing_deg

    # Adjust for poles
    dy_lat = np.where(dy_lat > 90, 90 - 2 * grid_spacing_deg, dy_lat)
    dy_lon = np.where(dy_lat > 90, points.lon + 180, points.lon)
    dy_lon = np.where(dy_lon > 180, dy_lon - 360, dy_lon)
    minus_dy_lat = np.where(minus_dy_lat < -90, -90 + 2 * grid_spacing_deg, minus_dy_lat)
    minus_dy_lon = np.where(minus_dy_lat < -90, points.lon + 180, points.lon)
    minus_dy_lon = np.where(minus_dy_lon > 180, minus_dy_lon - 360, minus_dy_lon)

    # Sample ages and compute crustal thicknesses at points
    points["age"] = np.nan; points["lithospheric_thickness"] = np.nan; points["crustal_thickness"] = np.nan; points["water_height"] = np.nan
    points["age"] = sample_ages(points.lat, points.lon, seafloor[age_variable])
    points["lithospheric_mantle_thickness"], points["crustal_thickness"], points["water_depth"] = calculate_thicknesses(
                points["age"],
                options
    )

    # Height of layers for integration
    zw = mech.L - points.water_depth
    zc = mech.L - (points.crustal_thickness + points.crustal_thickness)
    zl = mech.L - (points.crustal_thickness + points.crustal_thickness + points.lithospheric_mantle_thickness)

    # Calculate U
    points["U"] = 0.5 * mech.g * (
        mech.rho_a * (zl) ** 2 +
        mech.rho_l * (zc) ** 2 -
        mech.rho_l * (zl) ** 2 +
        mech.rho_c * (zw) ** 2 -
        mech.rho_c * (zc) ** 2 +
        mech.rho_sw * (mech.L) ** 2 -
        mech.rho_sw * (zw) ** 2
    )
    
    # Sample ages and compute crustal thicknesses at nearby points
    # dx_lon
    for i in range(0,4):
        if i == 0:
            sampling_lat = points.lat; sampling_lon = dx_lon
        if i == 1:
            sampling_lat = points.lat; sampling_lon = minus_dx_lon
        if i == 2:
            sampling_lat = dy_lat; sampling_lon = dy_lon
        if i == 3:
            sampling_lat = minus_dy_lat; sampling_lon = minus_dy_lon

        ages = sample_ages(sampling_lat, sampling_lon, seafloor[age_variable])
        lithospheric_mantle_thickness, crustal_thickness, water_depth = calculate_thicknesses(
                    ages,
                    options
        )

        # Height of layers for integration
        zw = mech.L - water_depth
        zc = mech.L - (crustal_thickness + crustal_thickness)
        zl = mech.L - (crustal_thickness + crustal_thickness + lithospheric_mantle_thickness)

        # Calculate U
        U = 0.5 * mech.g * (
            mech.rho_a * (zl) ** 2 +
            mech.rho_l * (zc) ** 2 -
            mech.rho_l * (zl) ** 2 +
            mech.rho_c * (zw) ** 2 -
            mech.rho_c * (zc) ** 2 +
            mech.rho_sw * (mech.L) ** 2 -
            mech.rho_sw * (zw) ** 2
        )

        if i == 0:
            dx_U = U
        if i == 1:
            minus_dx_U = U
        if i == 2:
            dy_U = U
        if i == 3:
            minus_dy_U = U

    # Convert degree spacing to metre spacing
    lat_grid_spacing_m = constants.mean_Earth_radius_m * (np.pi/180) / grid_spacing_deg
    lon_grid_spacing_m = constants.mean_Earth_radius_m * (np.pi/180) / (np.cos(np.deg2rad(points.lat)) * grid_spacing_deg)

    # Calculate force
    points["GPE_force_lat"] = (-mech.L0 / mech.L) * (dy_U - minus_dy_U) / lat_grid_spacing_m
    points["GPE_force_lon"] = (-mech.L0 / mech.L) * (dx_U - minus_dx_U) / lon_grid_spacing_m

    return points

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# BASAL TRACTIONS
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to calculate mantle drag force at points 
# torques:                  pd.DataFrame containing 
# points:                   pd.DataFrame containing data of points including columns with latitude, longitude and plateID
# options:                  dictionary with options

def mantle_drag_force(torques, points, options):
    # Import mechanical parameters and constants
    mech = set_mech_params()
    constants = set_constants()

    # Calculate drag coefficient
    drag_coeff = mech.visc_a / mech.La

    # Get grid spacing
    grid_spacing_deg = points.lon[1] - points.lon[0]

    # Convert degree spacing to metre spacing
    lat_grid_spacing_m = constants.mean_Earth_radius_m * (np.pi/180) / grid_spacing_deg
    lon_grid_spacing_m = constants.mean_Earth_radius_m * (np.pi/180) / (np.cos(np.deg2rad(points.lat)) * grid_spacing_deg)

    # Get velocities at points
    if options["Reconstructed motions"]:
        points = reconstructed_velocities_at_points(torques, points)
        
        # Calculate mantle drag force
        points["mantle_drag_force_lat"] = -1 * points.v_lat / (lat_grid_spacing_m * lon_grid_spacing_m)
        points["mantle_drag_force_lon"] = -1 * points.v_lon / (lat_grid_spacing_m * lon_grid_spacing_m)

    else:
        # Calculate residual torque
        axes = ["_x", "_y", "_z"]
        for axis in axes:
            columns_to_sum = [col for col in torques.columns if col.endswith(axis)]
            torques["mantle_drag_torque" + axis] = torques[columns_to_sum].sum(axis=1) * -1

        # Convert to centroid
        centroid_position = points_to_cartesian(torques.centroid_lat, torques.centroid_lon)
        centroid_unit_position = centroid_position / constants.mean_Earth_radius_m
        
        summed_torques_cartesian = np.array([torques["mantle_drag_torque_x"], torques["mantle_drag_torque_y"], torques["mantle_drag_torque_z"]])

        force_at_centroid = np.cross(summed_torques_cartesian, centroid_position, axis=0)
        # Normalise centroid for pygplates.PointOnSphere

        centroid_unit_position = centroid_position / np.sqrt(centroid_position[0]**2 + centroid_position[1]**2 + centroid_position[2]**2)

        force_magnitude = np.zeros(len(force_at_centroid[0])); force_azimuth = np.zeros(len(force_at_centroid[0])); force_inclination = np.zeros(len(force_at_centroid[0]))
        for i in range(len(force_at_centroid[0])):    
            centroid_point = pygplates.PointOnSphere(centroid_unit_position[0][i], centroid_unit_position[1][i], centroid_unit_position[2][i])
            force_magnitude[i], force_azimuth[i], force_inclination[i] = pygplates.LocalCartesian.convert_from_geocentric_to_magnitude_azimuth_inclination(
                centroid_point, 
                (force_at_centroid[0][i], force_at_centroid[1][i], force_at_centroid[2][i])
            )

        # Convert to latitudinal and longitudinal components of torque at centroid
        torques["mantle_drag_force_lat"] = np.nan; torques["mantle_drag_force_lon"] = np.nan
        torques["mantle_drag_force_lat"], torques["mantle_drag_force_lon"] = mag_azi2lat_lon(force_magnitude, np.rad2deg(force_azimuth))

        # Convert to velocity at centroid
        velocity_x, velocity_y, velocity_z = -1 * force_at_centroid/drag_coeff
        v_magnitude = np.zeros_like(velocity_x)
        v_azimuth = np.zeros_like(velocity_x)
        v_inclination = np.zeros_like(velocity_x)

        unit_position = points_to_cartesian(points.lat, points.lon)
        for i in range(len(points.lat[0])):
            unit_position = pygplates.PointOnSphere(unit_position[:,0], unit_position[:,1], unit_position[:,2])
            v_magnitude[i], v_azimuth[i], v_inclination[i] = pygplates.LocalCartesian.convert_from_geocentric_to_magnitude_azimuth_inclination(
                unit_position, 
                (velocity_x[i], velocity_y[i], velocity_z[i])
            )

    return torques, points

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# GENERAL FUNCTIONS
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to sample ages of a seafloor at a set of points
# lat, lon:         np.arrays of equal length containing latitude and longitude values of points at which forces are acting.
# seafloor:           xr.DataSet containing lithospheric ages

def sample_ages(lat, lon, seafloor, coords=["lat", "lon"]):
    # Extract latitude and longitude values from points and convert to xarray DataArrays
    lat_da = xr.DataArray(lat, dims="point")
    lon_da = xr.DataArray(lon, dims="point")

    # Interpolate age value at point
    ages = np.array(seafloor.interp({coords[0]: lat_da, coords[1]: lon_da}).values.tolist())

    return ages

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to calculate thickness of lithospheric mantle and crust, and water depth at points.
# ages:             np.array containing seafloor ages.
# options:          dictionary with options

def calculate_thicknesses(ages, options, crust=True, water=True):
    # Set mechanical parameters and constants
    mech = set_mech_params()
    constants = set_constants()

    # Thickness of oceanic lithosphere from half space cooling and water depth from isostasy
    if options["Seafloor age profile"] == "half space cooling":
        lithospheric_mantle_thickness = np.where(np.isnan(ages), 
                                                mech.cont_lith_thick, 
                                                2.32 * np.sqrt(mech.kappa * ages * constants.ma2s))
        
        if crust:
            crustal_thickness = np.where(np.isnan(ages), 
                                        mech.cont_crust_thick, 
                                        mech.ocean_crust_thick)
        else:
            crustal_thickness = None
            
        if water:
            water_depth = np.where(np.isnan(ages), 
                                0,
                                (lithospheric_mantle_thickness * ((mech.rho_a - mech.rho_l) / (mech.rho_sw - mech.rho_a))) + 2600)
        else:
            water_depth = None
        
    # Water depth from half space cooling and lithospheric thickness from isostasy
    elif options["Seafloor age profile"] == "plate model":
        hw = np.where(ages > 81, 6586 - 3200 * np.exp((-ages / 62.8)), ages)
        hw = np.where(hw <= 81, 2600 + 345 * np.sqrt(hw), hw)
        lithospheric_mantle_thickness = (hw - 2600) * ((mech.rho_sw - mech.rho_a) / (mech.rho_a - mech.rho_l))

        if crust:
            crustal_thickness = np.where(np.isnan(ages), 
                                        mech.cont_crust_thick, 
                                        mech.ocean_crust_thick)
        else:
            crustal_thickness = None
        
        if water:
            water_depth = hw
        else:
            water_depth = None

    return lithospheric_mantle_thickness, crustal_thickness, water_depth

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to calculate torque acting on plate.
# torques:                  pd.DataFrame containing torques on plates.
# lat, lon:                 np.arrays of equal length containing latitude and longitude values of points at which forces are acting.
# force_lat, force_lon:     Latitudinal and longitudinal components of force.

def torque_on_plates(torques, lat, lon, plateID, force_lat, force_lon, segment_length=1, torque_variable="torque"):
    # Initialize dataframes and sort plateIDs
    data = pd.DataFrame({"plateID": plateID})

    # Convert points to Cartesian coordinates
    position = points_to_cartesian(lat, lon)
    
    # Calculate torques in Cartesian coordinates
    torques_cartesian = torques_in_cartesian(position, lat, lon, force_lat, force_lon, segment_length)

    # Assign the calculated torques to the new torque_variable columns
    data[torque_variable + "_x"] = torques_cartesian[0]
    data[torque_variable + "_y"] = torques_cartesian[1]
    data[torque_variable + "_z"] = torques_cartesian[2]

    # Sum components of plates based on plateID
    summed_data = data.groupby("plateID", as_index=False).sum()

    # Merge with unique plateIDs to ensure all plateIDs are present
    torques = pd.merge(torques, summed_data, left_on="plateID", right_on="plateID", how="left").fillna(0)
    
    # Calculate the position vector of the centroid of the plate in Cartesian coordinates
    centroid_position = points_to_cartesian(torques.centroid_lat, torques.centroid_lon)
    
    # Calculate the torque vector as the cross product of the Cartesian torque vector (x, y, z) with the position vector of the centroid
    summed_torques_cartesian = np.array([torques[torque_variable + "_x"], torques[torque_variable + "_y"], torques[torque_variable + "_z"]])
    force_at_centroid = np.cross(summed_torques_cartesian, centroid_position, axis=0) 

    # Normalise centroid for pygplates.PointOnSphere
    centroid_unit_position = centroid_position / np.sqrt(centroid_position[0]**2 + centroid_position[1]**2 + centroid_position[2]**2)

    force_magnitude = np.zeros(len(force_at_centroid[0])); force_azimuth = np.zeros(len(force_at_centroid[0])); force_inclination = np.zeros(len(force_at_centroid[0]))
    for i in range(len(force_at_centroid[0])):    
        centroid_point = pygplates.PointOnSphere(centroid_unit_position[0][i], centroid_unit_position[1][i], centroid_unit_position[2][i])
        force_magnitude[i], force_azimuth[i], force_inclination[i] = pygplates.LocalCartesian.convert_from_geocentric_to_magnitude_azimuth_inclination(
            centroid_point, 
            (force_at_centroid[0][i], force_at_centroid[1][i], force_at_centroid[2][i])
        )
        
    # Convert to latitudinal and longitudinal components of torque at centroid
    force_variable = torque_variable.replace("torque", "force")
    torques[force_variable + "_lat"] = np.nan; torques[force_variable + "_lon"] = np.nan
    torques[force_variable + "_lat"], torques[force_variable + "_lon"] = mag_azi2lat_lon(force_magnitude, np.rad2deg(force_azimuth))
    
    return torques

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to convert a set of points in spherical coordinates to Cartesian coordinates.
# lat, lon:                                             np.arrays of equal length containing latitude and longitude values of points.

def points_to_cartesian(lat, lon):
    # Set constants
    constants = set_constants()

    # Convert to radians
    lat_rads = np.deg2rad(lat)
    lon_rads = np.deg2rad(lon)

    # Calculate position vectors
    position = constants.mean_Earth_radius_m * np.array([np.cos(lat_rads) * np.cos(lon_rads), np.cos(lat_rads) * np.sin(lon_rads), np.sin(lat_rads)])

    return position

#  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to calculate Euler rotation rate [cm/yr] and azimuth [degrees from North, clockwise] at points
# position:                                             position vector in Cartesian coordinates
# euler_pole_lat, euler_pole_lon, euler_pole_angle:     floats values of latitude, longitude and finite rotation angle of Euler rotation

def torques_in_cartesian(position, lat, lon, force_lat, force_lon, segment_length):
    # Convert lon, lat to radian
    lon_rads = np.deg2rad(lon)
    lat_rads = np.deg2rad(lat)

    force_lat *= 1/np.sqrt(np.pi)
    force_lon *= 1/np.sqrt(np.pi)

    # Calculate force_magnitude
    force_magnitude = np.sqrt((force_lat*segment_length)**2 + (force_lon*segment_length)**2)

    theta = np.where(
        (force_lon >= 0) & (force_lat >= 0),                     # Condition 1
        np.arctan(force_lat/force_lon),                          # Value when Condition 1 is True
        np.where(
            (force_lon < 0) & (force_lat >= 0) | (force_lon < 0) & (force_lat < 0),    # Condition 2
            np.pi + np.arctan(force_lat/force_lon),              # Value when Condition 2 is True
            (2*np.pi) + np.arctan(force_lat/force_lon)           # Value when Condition 3 is True
        )
    )

    np.cos(theta) * (-1.0*np.sin(lon_rads)),  np.cos(theta) * np.cos(lon_rads), np.sin(theta) * np.cos(lat_rads)

    force_x = force_magnitude * np.cos(theta) * (-1.0 * np.sin(lon_rads))
    force_y = force_magnitude * np.cos(theta) * np.cos(lon_rads)
    force_z = force_magnitude * np.sin(theta) * np.cos(lat_rads)

    force = np.array([force_x, force_y, force_z])

    # Calculate torque
    torque = np.cross(position, force, axis=0)

    return torque

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to decompose a vector into latitudinal and longitudinal components
# magnitude:        magnitude of vector
# azimuth:          azimuth of vector [degrees from north]

def mag_azi2lat_lon(magnitude, azimuth):
    # Convert azimuth from degrees to radians
    azimuth_rad = np.deg2rad(azimuth)

    # Calculate components
    component_lat = np.cos(azimuth_rad) * magnitude
    component_lon = np.sin(azimuth_rad) * magnitude

    return component_lat, component_lon

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to convert a 2D vector into magnitude and azimuth [degrees from north]
# component_lat:        latitudinal component of vector
# component_lon:        latitudinal component of vector

def lat_lon2mag_azi(component_lat, component_lon):
    # Calculate magnitude
    magnitude = np.sqrt(component_lat**2 + component_lon**2)

    # Calculate azimuth in radians
    azimuth_rad = np.arctan2(component_lon, component_lat)

    # Convert azimuth from radians to degrees
    azimuth_deg = np.rad2deg(azimuth_rad)

    return magnitude, azimuth_deg

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to calculate Euler rotation rate [cm/yr] and azimuth [degrees from North, clockwise] at points
# plates:                  pd.DataFrame containing plateID, latitude, longitude and finite rotation angle of each plate.                               
# lat, lon, plateID:       np.arrays of equal length containing latitude and longitude values and plateIDs of points.

def reconstructed_velocities_at_points(plates, points):
    # Iterate over each row in the 'plates' DataFrame
    for _, plate_row in plates.iterrows():
        # Get the Euler pole latitude, longitude, and angle for the current plate
        euler_pole_lat = plate_row['pole_lat']
        euler_pole_lon = plate_row['pole_lon']
        euler_angle = plate_row['pole_angle']

        # Find the indices of points that match the current plateID in the 'points' DataFrame
        matching_indices = points.index[points['plateID'] == plate_row['plateID']]

        # Calculate the rotation rate and azimuth for each matching point
        rotation_rate_cm_per_yr, azimuth_deg = euler_rotation_at_point(
            points.loc[matching_indices, 'lat'],
            points.loc[matching_indices, 'lon'],
            euler_pole_lat,
            euler_pole_lon,
            euler_angle
        )

        # Update the 'rotation_rate_cm_per_yr' and 'azimuth_deg' columns in the matching rows
        points.loc[matching_indices, 'v_magnitude'] = rotation_rate_cm_per_yr
        points.loc[matching_indices, 'v_azimuth'] = np.degrees(azimuth_deg)

    # Decompose into latitudinal and longitudinal components
    points["v_lat"], points["v_lon"] = mag_azi2lat_lon(points["v_magnitude"], points["v_azimuth"])

    return points

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to calculate Euler rotation rate [cm/yr] and azimuth [degrees from North, clockwise] at points
# lat_points, lon_points:                               np.arrays of equal length containing latitude and longitude values of points.
# euler_pole_lat, euler_pole_lon, euler_pole_angle:     floats values of latitude, longitude and finite rotation angle of Euler rotation

def euler_rotation_at_point(lat_points, lon_points, euler_pole_lat, euler_pole_lon, euler_angle):
    # Set constants
    constants = set_constants()

    # Convert Euler pole latitude and longitude to radians
    euler_pole_lat_rad = np.radians(euler_pole_lat)
    euler_pole_lon_rad = np.radians(euler_pole_lon)

    # Convert latitude and longitude of points to radians
    lat_rad = np.radians(lat_points)
    lon_rad = np.radians(lon_points)

    # Calculate the rotation rate at each point using the Euler pole and angle
    rotation_rate_rad_per_yr = constants.mean_Earth_radius_m * euler_angle / 1e6
    
    # Calculate the azimuth of rotation at each point
    azimuth_rad = np.arctan2(np.cos(euler_pole_lat_rad) * np.sin(lon_rad - euler_pole_lon_rad),
                             np.cos(lat_rad) * np.sin(euler_pole_lat_rad) - np.sin(lat_rad) * np.cos(euler_pole_lat_rad) * np.cos(lon_rad - euler_pole_lon_rad))

    # Calculate the distance from each point to the Euler pole
    distance = constants.mean_Earth_radius_m * np.arccos(np.sin(euler_pole_lat_rad) * np.sin(lat_rad) + np.cos(euler_pole_lat_rad) * np.cos(lat_rad) * np.cos(lon_rad - euler_pole_lon_rad))

    # Convert the rotation rate from radians/year to cm/year
    rotation_rate_cm_per_yr = rotation_rate_rad_per_yr * distance * constants.rada2cma

    return rotation_rate_cm_per_yr, azimuth_rad