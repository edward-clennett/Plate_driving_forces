## Code from "Assessing Plate Reconstruction Models Using Plate Driving Forces"
### Edward J. Clennett, Adam F. Holt, Michael G. Tetley, Thorsten W. Becker & Claudio Faccenna

This folder contains a number of files:  
  
#### Plate_driving_forces_analysis.ipynb  
  
This is a jupyter notebook with code that loads the appropriate files and run the plate driving forces caluclation, as well as performs analysis tasks. The plate driving forces calculation calls two python scripts, **compute_torques.py** and **functions_main.py**. The following python packages are required to run these scripts:  
-- pygplates (can be found in the requirements folder)  
-- numpy  
-- sys  
-- statistics  
-- math  
-- netCDF4  
-- gpxpy  
-- geographiclib  
-- matplotlib  
-- cartopy  

These packages, as well as python and jupyter notebook, can be installed via [anaconda](https://www.anaconda.com/).  

#### Requirements  
  
This is the pygplates code required to run the scripts. Other dependencies are listed above. For more information on pygplates, see the [GPlates website](https://www.gplates.org/docs/pygplates/index.html).  
  
#### Plate_model  
  
This folder contains the Muller et al. (2016) model as an example. The files required to run the plate forces caluclation are:  
-- A rotation file, which describes the movement of the plates (required for forces calculation):  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Global_EarthByte_230-0Ma_GK07_AREPS_fixed_crossovers.rot  
-- A topology file, which describes the geometries and types of plate boundaries (required for forces calculation):  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Global_EarthByte_230-0Ma_GK07_AREPS_PlateBoundaries.gpml  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Global_EarthByte_230-0Ma_GK07_AREPS_Topology_BuildingBlocks.gpml  
-- A coastlines shapefile (only required for plotting).  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Global_EarthByte_230-0Ma_GK07_AREPS_Coastlines.gpml  
-- A continents shapefile (required to generate seafloor agegrids and crustal thickness rasters)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Global_EarthByte_230-0Ma_GK07_AREPS_COB_Terranes.gpml
  
#### Seafloor_age_data
  
This folder contains the seafloor age grids which are required for the plate driving forces calculation. Seafloor age grids are often available in the supplementary data of published plate models. If age grids are not provided, or you are creating your own plate model, then you can generate the seafloor age grids using the algorithm from Williams et al. (2019).

#### Crustal_thickness  
  
This folder contains the crustal thickness rasters which are only required for the plate driving forces calculation when the option to calculate continental GPE tractions is set to true. The crustal thickness rasters were created by loading in a present-day crustal thickness raster into GPlates and "cookie cutting" it to the continents shapefile. Information on cookie cutting can be found in the GPlates tutorials [here](https://docs.google.com/document/d/1BohvVbw0n3w8EW7asEIo72dCyRHY_aaC4BTP9Y8zSig/pub#id.nl8kz7s4totv>).  

#### References  
  
Clennett, E. J., Holt, A. F., Tetley, M. G., Becker, T. W., & Faccenna, C. (2023). Assessing plate reconstruction models using plate driving forces.  
  
Müller, R. D., Seton, M., Zahirovic, S., Williams, S. E., Matthews, K. J., Wright, N. M., Shephard, G. E., Maloney, K. T., Barnett-Moore, N., Hosseinpour, M., Bower, D. J., & Cannon, J. (2016). Ocean Basin Evolution and Global-Scale Plate Reorganization Events Since Pangea Breakup. *Annual Review of Earth and Planetary Sciences*, 44(1), 107–138. https://doi.org/10.1146/annurev-earth-060115-012211  
  
Williams, S., Wright, N. M., Cannon, J., Flament, N., & Müller, R. D. (2021). Reconstructing seafloor age distributions in lost ocean basins. *Geoscience Frontiers*, 12(2), 769–780. https://doi.org/10.1016/j.gsf.2020.06.004  
