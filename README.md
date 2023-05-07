## Code from "Assessing Plate Reconstruction Models Using Plate Driving Forces"
### Edward J. Clennett, Adam F. Holt, Michael G. Tetley, Thorsten W. Becker & Claudio Faccenna

This folder contains a number of files:  
  
#### Plate_driving_forces_analysis.ipynb  
  
This is a jupyter notebook with code that loads the appropriate files and runs the plate driving forces caluclation, as well as performs analysis and plotting tasks. The plate driving forces calculation calls two python scripts, **compute_torques.py** and **functions_main.py**. The following python packages are required to run these scripts:   
#### environment.yml

This is a conda environment that includes the following packages, compatible with python 3.8. For more information on anaconda, see the [conda user guide](https://conda.io/projects/conda/en/latest/user-guide/index.html).
- numpy  
- netCDF4  
- gpxpy  
- geographiclib  
- matplotlib  
- cartopy  
- cmcrameri  

To create the environment, execute the following line in the conda terminal: <code>conda env create -f environment.yml</code>.

To activate the environment, type: <code>conda activate plate_forces</code>. 

Finally, to open the notebook, type: <code>jupyter notebook</code> and click on the Plate_driving_forces_analysis.ipynb file. 


#### Requirements  
  
This folder contains the pygplates code required to run the scripts. The folder contains the pre-compiled code for both windows and macOS, and users of these systems to do need to do anything except make sure that the correct folder is being called in the first cell of Plate_driving_forces_analysis.ipynb.  


To run this code on a different operating system, see the [pygplates documentation](https://www.gplates.org/docs/pygplates/index.html) for installation instructions.  

#### Plate_model  
  
This folder contains the model files required to run the plate forces caluclation. An example of the files from the Muller et al. (2016) plate reconstruction model are:  
1. A rotation file, which describes the movement of the plates (required for forces calculation):  
    - Global_EarthByte_230-0Ma_GK07_AREPS_fixed_crossovers.rot  
2. A topology file, which describes the geometries and types of plate boundaries (required for forces calculation):  
    - Global_EarthByte_230-0Ma_GK07_AREPS_PlateBoundaries.gpml  
    - Global_EarthByte_230-0Ma_GK07_AREPS_Topology_BuildingBlocks.gpml  
3. A coastlines shapefile (only required for plotting):  
    - Global_EarthByte_230-0Ma_GK07_AREPS_Coastlines.gpml  
4. A continents shapefile (required to generate seafloor agegrids and crustal thickness rasters):  
    - Global_EarthByte_230-0Ma_GK07_AREPS_COB_Terranes.gpml
  
#### Seafloor_age_data
  
This folder contains the seafloor age grids which are required for the plate driving forces calculation. Seafloor age grids are often available in the supplementary data of published plate models. If age grids are not provided, or you are creating your own plate model, then you can generate the seafloor age grids using the algorithm from Williams et al. (2019).

#### Crustal_thickness  
  
This folder contains the crustal thickness rasters which are only required for the plate driving forces calculation when the option to calculate continental GPE tractions is set to true. Crustal thickness rasters can be created by loading in a present-day crustal thickness raster into GPlates and "cookie cutting" it to the continents shapefile. Information on cookie cutting can be found in the GPlates tutorials [here](https://docs.google.com/document/d/1BohvVbw0n3w8EW7asEIo72dCyRHY_aaC4BTP9Y8zSig/pub#id.nl8kz7s4totv>).  

#### References  
  
Clennett, E. J., Holt, A. F., Tetley, M. G., Becker, T. W., & Faccenna, C. (2023). Assessing plate reconstruction models using plate driving forces.  
  
Müller, R. D., Seton, M., Zahirovic, S., Williams, S. E., Matthews, K. J., Wright, N. M., et al. (2016). Ocean Basin Evolution and Global-Scale Plate Reorganization Events Since Pangea Breakup. *Annual Review of Earth and Planetary Sciences*, 44(1), 107–138. https://doi.org/10.1146/annurev-earth-060115-012211  
  
Williams, S., Wright, N. M., Cannon, J., Flament, N., & Müller, R. D. (2021). Reconstructing seafloor age distributions in lost ocean basins. *Geoscience Frontiers*, 12(2), 769–780. https://doi.org/10.1016/j.gsf.2020.06.004  
