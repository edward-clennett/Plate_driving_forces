# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PLATEFO
# Algorithm to calculate plate forces from tectonic reconstructions
# Thomas Schouten, 2023
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Import packages
import numpy as np
import xarray as xr
import pandas as pd
import os
import gplately
import warnings
from gplately import pygplates
from collections import defaultdict
from plate_force_calculation import set_constants

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to get data on plates in reconstruction
# topology_features     topology features to be reconstructed (.gpml)
# rotation_file         rotation file (.rot)
# time:                 reconstruction time (integer)
# timestep:             reconstruction timestep
# anchor_plate_id:      anchor plate ID

def get_plates(topology_features, rotation_file, reconstruction_time, output_directory=None, reconstruction_name=None, timestep=1, anchor_plate_id=0):
    # Set constants
    constants = set_constants()

    # Set rotations
    rotations = pygplates.RotationModel(rotation_file)

    # Resolve topologies
    resolved_topologies = []
    pygplates.resolve_topologies(topology_features, rotation_file, resolved_topologies, reconstruction_time, anchor_plate_id=anchor_plate_id)

    # Make pd.df with all plates
    # Initialise list
    plates = np.zeros([len(resolved_topologies),9])
    
    # Loop through plates
    for n, topology in enumerate(resolved_topologies):

        # Get plate ID and area
        plates[n,0] = topology.get_resolved_feature().get_reconstruction_plate_id()
        plates[n,1] = topology.get_resolved_geometry().get_area() * constants.mean_Earth_radius_m * constants.mean_Earth_radius_m

        # Get Euler rotations
        stage_rotation = rotations.get_rotation(to_time=reconstruction_time,
                                                moving_plate_id=int(plates[n,0]),
                                                from_time=reconstruction_time + timestep,
                                                anchor_plate_id=anchor_plate_id)
        pole_lat, pole_lon, pole_angle = stage_rotation.get_lat_lon_euler_pole_and_angle_degrees()
        plates[n,2] = pole_lat
        plates[n,3] = pole_lon
        plates[n,4] = pole_angle

        # Get plate centroid
        centroid = topology.get_resolved_geometry().get_interior_centroid()
        centroid_lat, centroid_lon = centroid.to_lat_lon_array()[0]
        plates[n,5] = centroid_lon
        plates[n,6] = centroid_lat

        # Get velocity at centroid
        centroid_vectors = pygplates.calculate_velocities(centroid,
                                                          stage_rotation,
                                                          timestep,
                                                          velocity_units = pygplates.VelocityUnits.cms_per_yr)
        centroid_velocity = np.asarray(pygplates.LocalCartesian.convert_from_geocentric_to_magnitude_azimuth_inclination(centroid,centroid_vectors))[0]
        plates[n,7] = centroid_velocity[1]
        plates[n,8] = centroid_velocity[0]
        
    plates = pd.DataFrame(plates)
    plates.columns = ["plateID", "area", "pole_lat", "pole_lon", "pole_angle", "centroid_lon", "centroid_lat", "centroid_vel_lon", "centroid_vel_lat"]
    plates["name"] = np.nan; plates.name = get_plate_names(plates.plateID)
    plates = plates.sort_values(by="plateID")
    plates = plates.reset_index(drop=True)

    # Save for next time
    if reconstruction_name and output_directory:
        plates.to_csv(output_directory + "/Plates_" + reconstruction_name + "_" + str(reconstruction_time) + "Ma.csv")

    return plates, resolved_topologies

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to get data on plates in reconstruction
# topology_features     topology features to be reconstructed (.gpml)
# rotation_file         rotation file (.rot)
# time:                 reconstruction time (integer)
# timestep:             reconstruction timestep
# anchor_plate_id:      anchor plate ID

def get_points_with_plate_ids(resolved_topologies, rotation_file, reconstruction_time, grid_spacing=1, output_directory=None, reconstruction_name=None):
    # Define grid spacing and 
    lat = np.arange(-90,91,grid_spacing)
    lon = np.arange(-180,181,grid_spacing)

    # Create a meshgrid of latitudes and longitudes
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Flatten the grids to get 1D arrays of all points
    lon_points = lon_grid.flatten()
    lat_points = lat_grid.flatten()

    # Initialise array to store plate_ids
    plate_ids = np.zeros(len(lat_points), dtype=int)

    # BUG: this section raises an error that pygplates should be imported from gplately, which it is. We will therefore ignore the error.         
    with warnings.catch_warnings():
        # Set the warning action to "ignore"
        warnings.simplefilter("ignore")
        partitioner = pygplates.PlatePartitioner(resolved_topologies, rotation_file)

        for i in range(len(lat_points)):
            point = pygplates.PointOnSphere(pygplates.LatLonPoint(lat_points[i], lon_points[i]))
            point_feature = pygplates.Feature(pygplates.FeatureType.gpml_unclassified_feature)
            point_feature.set_geometry(point)
            partitioned_points = partitioner.partition_features(point_feature)
            plate_ids[i] = partitioned_points[0].get_reconstruction_plate_id()

    points = pd.DataFrame({"lat": lat_points, "lon": lon_points, "plateID": plate_ids})

    # Save for next time
    if reconstruction_name and output_directory:
        points.to_csv(output_directory + "/Points_" + reconstruction_name + "_" + str(reconstruction_time) + "Ma.csv")

    return points

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to get data on plates in reconstruction
# topology_features     topology features to be reconstructed (.gpml)
# rotation_file         rotation file (.rot)
# time:                 reconstruction time (integer)
# timestep:             reconstruction timestep
# anchor_plate_id:      anchor plate ID

def get_plate_names(plate_id_list):
    plate_name_dict = {
        101: "North America",
        201: "South America",
        301: "Eurasia",
        302: "Baltica",
        501: "India",
        503: "Arabia",
        701: "South Africa",
        702: "Madagascar",
        709: "Somalia",
        801: "Australia",
        802: "Antarctica",
        901: "Pacific",
        902: "Farallon",
        904: "Aluk",
        926: "Izanagi",
        5400: "Burma",
        5599: "Tethyan Himalaya",
        7520: "Argoland",
        9002: "Farallon",
        9006: "Izanami",
        9009: "Izanagi",
        9010: "Pontus"
    } 

    # Create a defaultdict with the default value as the plate ID
    default_plate_name = defaultdict(lambda: "Unknown", plate_name_dict)

    # Retrieve the plate names based on the plate IDs
    plate_names = [default_plate_name[plate_id] for plate_id in plate_id_list]

    return plate_names

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to download age from gplately and save to local directory
# reconstruction:                            String containing the model identifier
# reconstruction_time:                             reconstruction time
# output_directory:                        Name of the directory to save file
# overwrite:                        True/False statement whether to overwrite previous files or not 

def get_subduction_zones(rotation_file, topology_features, reconstruction_time, tesselation_spacing=250, output_directory=None, reconstruction_name=None):
    # Set constants
    constants = set_constants()

    # Set up plate reconstruction
    reconstruction = gplately.PlateReconstruction(rotation_file, topology_features)

    # Tesselate subduction zones and get slab pull and bending torques along subduction zones
    subduction_zones = reconstruction.tessellate_subduction_zones(reconstruction_time, ignore_warnings=True, tessellation_threshold_radians=(tesselation_spacing/constants.mean_Earth_radius_km))

    # Convert to pd.DataFrame
    subduction_zones = pd.DataFrame(subduction_zones) 
    subduction_zones.columns = ["lon", "lat", "v_convergence", "obliquity_convergence", "v_absolute", "obliquity_absolute", "trench_segment_length", "trench_normal_azimuth", "lower_plateID", "trench_plateID"]

    # Set trench obliquity to 0, 360 range
    subduction_zones['obliquity_absolute'] = subduction_zones['obliquity_absolute'].apply(lambda x: x + 360 if x < 0 else x)
    subduction_zones['obliquity_convergence'] = subduction_zones['obliquity_convergence'].apply(lambda x: x + 360 if x < 0 else x)

    # Convert trench segment length from degree to m
    subduction_zones.trench_segment_length *= constants.equatorial_Earth_circumference / 360

    # Set negative convergence rates to zero
    subduction_zones.loc[subduction_zones.v_convergence < 0, "v_convergence"] = 0

    # Save for next time
    if reconstruction_name and output_directory:
        subduction_zones.to_csv(output_directory + "/Subduction_zones_" + reconstruction_name + "_" + str(reconstruction_time) + "Ma.csv")

    return subduction_zones

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to download age from gplately and save to local directory
# reconstruction:                            String containing the model identifier
# reconstruction_time:                             reconstruction time
# output_directory:                        Name of the directory to save file
# overwrite:                        True/False statement whether to overwrite previous files or not 

def get_options(file_name, sheet_name="Sheet1"):
    # Read file
    case_options = pd.read_excel(file_name, sheet_name=sheet_name, comment="#")

    # Initialise list of cases
    cases = []

    # Initialise options dictionary
    options = {}

    all_options = ["Slab pull torque",
                   "GPE torque",
                   "Mantle drag torque",
                   "Bending torque",
                   "Bending mechanism",
                   "Reconstructed motions",
                   "Continental crust", 
                   "Seafloor age profile",
                   "Shear zone width",
                   "Sediment subduction", 
                   "Interface mixing",
                   "Passive margin sediments", 
                   "Active margin sediments", 
                   "Equatorial bulge sediments",
                   "Generalised sediments",
                   "Accretionary wedge",
                   "Present-day sediments"]
    
    default_values = [True,
                      True,
                      True,
                      False,
                      "viscous",
                      True,
                      False,
                      "half space cooling",
                      1e3,
                      False,
                      "linear",
                      False,
                      0,
                      0,
                      0,
                      0,
                      False
                      ]

    # Adjust TRUE/FALSE values in excel file to boolean
    boolean_options = ["Reconstructed motions", "Continental crust", "Bending torque", "Sediment subduction", "Passive margin sediments", "Present-day sediments"]

    # Loop over rows to obtain options from excel file
    for _, row in case_options.iterrows():
        case = row["Name"]
        cases.append(case)
        options[case] = {}
        for i, option in enumerate(all_options):
            if option in case_options:
                if option in boolean_options and row[option] == 1:
                    row[option] = True
                elif option in boolean_options and row[option] == 0:
                    row[option] = False
                options[case][option] = row[option]
            else:
                options[case][option] = default_values[i]

    return cases, options

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # Function to download age from gplately and save to local directory
# reconstruction:                            String containing the model identifier
# reconstruction_time:                             reconstruction time
# output_directory:                        Name of the directory to save file
# overwrite:                        True/False statement whether to overwrite previous files or not 

def get_age_grid(reconstruction, reconstruction_time, output_directory, overwrite=False):
    # Check if the chosen model is valid
    if reconstruction not in ["Seton2012", "Muller2016", "Muller2019", "Clennett2020"]:
        return print("Please choose one of the following models: Seton2012, Muller2016, Muller2019, or Clennett2020")
    
    # Round the time to a whole number
    if reconstruction_time != math.ceil(reconstruction_time):
        print(str(reconstruction_time) + " rounded to " + str(round(reconstruction_time)))
        reconstruction_time = round(reconstruction_time)

    # Check if output directories exist
    if not os.path.exists(os.path.join(output_directory, "Seafloor_age_grids")):
        os.makedirs(os.path.join(output_directory, "Seafloor_age_grids"))

    # Generate the file path for the sediment grid
    age_gridFile = os.path.join(output_directory, "Seafloor_age_grids", f"Seafloor_age_grids_{reconstruction}_{round(reconstruction_time)}Ma.nc")

    # Check if the sediment grid file exists and overwrite is False
    if os.path.exists(age_gridFile) and not overwrite:
        # Open the file
        age_grid = xr.open_dataset(age_gridFile)

        # Close the file
        age_grid.close()

        return age_grid

    else:
        # Before 170 Ma, the Clennett2020 model is identical to the Muller2019 model and does not have 
        if reconstruction == "Clennett2020" and reconstruction_time > 170:
            # Set DataServer to correct model
            gdownload = gplately.download.DataServer("Muller2019")

        else:
            # Set DataServer to correct model
            gdownload = gplately.download.DataServer(reconstruction)

        # Download age grid
        age_grid = gdownload.get_age_grid(time=reconstruction_time)

        # Save the age grid to a local directory
        age_grid.save_to_netcdf4(age_gridFile)

        # Open the file
        age_grid = xr.open_dataset(age_gridFile)

        # Close the file
        age_grid.close()

        return age_grid

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to open or generate and save sediment grids from age grid and terrigenous sediment grid
# reconstruction:                                String containing the model identifier
# reconstruction_time:                                 reconstruction time
# output_directory:                            Name of the directory to save file
# AgrGrd:                               Age grid
# age_variable:                               Name of the age variable in age_grid
# biogenic_sediments:                   Allow computation of biogenic sediment thickness
# passive_margin_sediment_directory:    Directory that has terrigenous sediment thickness stored
# overwrite:                            True/False statement whether to overwrite previous files or not

def get_sed_grid(reconstruction, reconstruction_time, output_directory, age_grid=None, age_variable="z", biogenic_sediments=True, passive_margin_sediment_directory=None, overwrite=False):
    # Check if the chosen model is valid
    if reconstruction not in ["Muller2016", "Muller2019", "Clennett2020"]:
        return print("Please choose one of the following models: Muller2016, Muller2019 or Clennett2020")
    
    # Check if age grid is provided
    if age_grid is None:
        return "No file generated. Please provide an age grid."

    # Check if computation of biogenic sediments is disabled and terrigenous sediment grid is provided
    if not biogenic_sediments and passive_margin_sediment_directory is None:
        return "No file generated. Please allow computation of the biogenic sediment thickness or provide a grid for the terrigenous sediment thickness."
    
    # Round time if not a whole number
    if reconstruction_time != math.ceil(reconstruction_time):
        print(str(reconstruction_time) + " rounded to " + str(round(reconstruction_time)))
        reconstruction_time = round(reconstruction_time)

    # Make output directory if it doesn't exist
    if not os.path.exists(os.path.join(output_directory, "Sediment_thickness_grids")):
        os.makedirs(os.path.join(output_directory, "Sediment_thickness_grids"))

    # Generate the file path for the sediment grid
    sed_grid_file = os.path.join(output_directory, "Sediment_thickness_grids", f"Sediment_thickness_grid_{reconstruction}_{round(reconstruction_time)}Ma.nc")

    # Check if the sediment grid file exists and overwrite is False
    if os.path.exists(sed_grid_file) and not overwrite:
        # Open the file
        sed_grid = xr.open_dataset(sed_grid_file)

        # Close the file
        sed_grid.close()

        return sed_grid

    else:
        # Only the Muller2016 model has terrigenous sediment grids available
        if reconstruction == "Muller2016" and passive_margin_sediment_directory:
            PMsed_grid = xr.open_dataset(os.path.join(passive_margin_sediment_directory, f"sed_thick_0.2d_{reconstruction_time}.nc"))
            sed_grid = sediment_gridder(age_grid, biogenic_sediments=biogenic_sediments, PMsed_grid=PMsed_grid)
            
        else:
            sed_grid = sediment_gridder(age_grid, biogenic_sediments=biogenic_sediments)
        
        if reconstruction == "Seton2012":
            publication = "Seton et al. (2012)"
        elif reconstruction == "Muller2016":
            publication = "Muller et al. (2016)"
        elif reconstruction == "Muller2019":
            publication = "Muller et al. (2019)"
        elif reconstruction == "Clennett2020":
            publication = "Clennett et al. (2020)"

        sed_grid.attrs["description"] = "Age and sediment thickness grid at " + str(reconstruction_time) + " Ma for the " + publication + " Reconstruction"
        # Save the file to netcdf
        sed_grid.to_netcdf(sed_grid_file)

        # Open the file
        sed_grid = xr.open_dataset(sed_grid_file)

        # Close the file
        sed_grid.close()

        return sed_grid

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to generate sediment grids from age grid and terrigenous sediment grid
# reconstruction:                            String containing the model identifier
# reconstruction_time:                             reconstruction time
# output_directory:                        Name of the directory to save file
# AgrGrd:                           Age grid
# age_variable:                           Name of the age variable in age_grid
# biogenic_sediments:               Allow computation of biogenic sediment thickness
# passive_margin_sediment_directory:   Directory that has terrigenous sediment thickness stored
# overwrite:                        True/False statement whether to overwrite previous files or not

def sediment_gridder(age_grid=None, age_variable="z", biogenic_sediments=False, PMsed_grid=None, PMSedVar="z"):
    # Extract variables and coordinates from raster
    Ages = age_grid[age_variable].values
    Lat = age_grid["lat"].values
    Lon = age_grid["lon"].values

    # Interpolate terrigenous sediment thickness grid to 0.1 resolution of corresponding age grid
    # NOTE: ONLY linear interpolation works!
    if PMsed_grid is not None:
        PMsed_grid = PMsed_grid.interp(coords={"lat": Lat, "lon": Lon}, method="linear")
        
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # TO DO: Interpolation of NaN values raises KeyError!
        # Newage_grid = xr.Dataset({"z": Ages}, coords={"lat": Lat, "lon": Lon})
        # print("PMsed_grid:")
        # print(PMsed_grid.dims)
        # print(PMsed_grid.coords)
        # print("age_grid:")
        # print(Newage_grid.dims)
        # print(Newage_grid.coords)
        # PMsed_grid = PMsed_grid.where(PMsed_grid.isnull() & ~Newage_grid.isnull(), PMsed_grid.interpolate_na({"lat": Lat}, method="cubic").interpolate_na({"lon": Lon}, method="cubic")["z"])
        # Newage_grid = Newage_grid.where(Newage_grid.isnull() & ~PMsed_grid.isnull(), Newage_grid.interpolate_na({"lat": Lat}, method="cubic").interpolate_na({"lon": Lon}, method="cubic")[age_variable])
        # Ages = Newage_grid.values
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        PMSeds = PMsed_grid[PMSedVar].values

    else:
        PMSeds = None

    if biogenic_sediments:
        BioSeds = bio_sed_grid(Ages, Lat)

    else:
        BioSeds = None

    data_vars = {"age": Ages}

    if biogenic_sediments:
        data_vars["bio_sed"] = BioSeds

    if PMSeds is not None:
        data_vars["passive_margin_sed"] = PMSeds

    if biogenic_sediments and PMSeds is not None:
        data_vars["all_sed"] = PMSeds + BioSeds

    sed_grid = xr.Dataset(
        data_vars={k: (("lat", "lon"), v) for k, v in data_vars.items()},
        coords={"lat": Lat, "lon": Lon}
        # Add an attribute to the grid
    )  

    return sed_grid

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Function to calculate equatorial sediment thickness for age grid.
# Ages:                             Array of ages
# reconstruction_time:                             reconstruction time
# output_directory:                        Name of the directory to save file


def bio_sed_grid(Ages, Lat):
    if Ages is None:
        return "No file generated. Please provide seafloor ages."

    # if Lon is None:
    #     Lon = [1]

    C1 = 52
    C2 = -2.46
    C3 = 0.045

    # Set up vectors
    LatArray = np.repeat(Lat[:, np.newaxis], Ages.shape[1], axis=1)
    Ones = np.ones((Ages.shape[0], Ages.shape[1]))

    # Compute sediment thickness as function for latitude for plate age = 1
    EqSed = C1 * Ones + C2 * np.abs(LatArray) + C3 * np.abs(LatArray) ** 2
    NonEqSed = C1 * Ones + C2 * Ones * 25 + C3 * Ones * 25 ** 2 #* ( (np.abs(LatArray) - 90) / (25 - 90) )
    
    # Create a mask to exclude the first X rows
    Mask = np.ones_like(Ages, dtype=bool)
    Mask[:651, :] = False
    Mask[-651:, :] = False
    BioSeds = np.squeeze(np.where(Mask, EqSed * np.sqrt(Ages), NonEqSed * np.sqrt(Ages)))

    return BioSeds