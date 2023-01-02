#!/usr/bin/python
import numpy as np
import math, pygplates
import gpxpy.geo
from geographiclib.geodesic import Geodesic
import netCDF4 as nc

class set_mech_params:
    def __init__(self):

        # mechanical and rheological parameters:
        self.g = 9.81               # gravity [m/s2]
        self.dT = 1200.             # mantle-surface T contrast [K]
        self.rho0 = 3300.           # reference/mantle density  [kg/m3]
        self.rhoW = 1000.           # water density [kg/m3]
        self.rho_w = 1020           # water density for plate model
        self.rho_l = 3412           # lithosphere density
        self.rho_a = 3350           # asthenosphere density 
        self.alpha = 3e-5           # thermal expansivity [K-1]
        self.kappa = 1e-6           # thermal diffusivity [m2/s]
        self.depth = 700.e3         # slab depth [m]
        self.rad_curv = 390.e3      # slab curvature [m]
        self.L = 130000             # compensation depth
        self.L0 = 100000            # lithospheric shell thickness
        self.lith_visc = 500e20     # lithospheric viscosity [Pa s]
        self.lith_age_RP = 60       # age of oldest sea-floor in approximate ridge push calculation  [Ma]
        self.yield_stress = 1050e6  # Byerlee yield strength at 40km, i.e. 60e6 + 0.6*(3300*10.0*40e3) [Pa]
        self.cont_thick = 100e3     # continental thickness (where there is no age) [m]

class global_plates:

    def __init__(self,features_name,rotation_model,reconstruction_t,anch_ID):
        #print "loading global features..."

        self.topology_features = pygplates.FeatureCollection(features_name)  # plate boundaries
        self.poles = rotation_model
        self.resolved_topologies = []
        pygplates.resolve_topologies(self.topology_features, self.poles, self.resolved_topologies, reconstruction_t, anchor_plate_id=anch_ID)
        self.num_resolved_topologies = len(self.resolved_topologies);

        self.plate_list = np.zeros([len(self.resolved_topologies),5])
        n = 0;
        # record plate IDs and plate areas
        for topology in self.resolved_topologies:
            self.plate_list[n,0] = topology.get_resolved_feature().get_reconstruction_plate_id();
            self.plate_list[n,1] = topology.get_resolved_geometry().get_area()*pygplates.Earth.mean_radius_in_kms*pygplates.Earth.mean_radius_in_kms;
            self.plate_list[n,2] = topology.get_resolved_geometry().get_arc_length()*pygplates.Earth.mean_radius_in_kms
            sub_length = 0
            ridge_length = 0
            segments = topology.get_boundary_sub_segments()
            for segment in segments:
                if segment.get_resolved_feature().get_feature_type() == pygplates.FeatureType.gpml_subduction_zone \
                    or segment.get_resolved_feature().get_feature_type() == pygplates.FeatureType.gpml_mid_ocean_ridge:
                    
                    if segment.get_resolved_feature().get_feature_type() == pygplates.FeatureType.gpml_subduction_zone:
                        sub_length += segment.get_resolved_geometry().get_arc_length()*pygplates.Earth.mean_radius_in_kms
                    else:
                        ridge_length += segment.get_resolved_geometry().get_arc_length()*pygplates.Earth.mean_radius_in_kms
            self.plate_list[n,3] = sub_length
            self.plate_list[n,4] = ridge_length
            n += 1;
        
class grids:
    def __init__(self, options, mech, globe, seafloor_ages,crustal_thickness_file):
            
        #Calculating GPE from seafloor ages
        #params
        rho_w = 1020
        rho_c = 2868
        rho_l = 3412
        rho_a = 3350
        hc = 8000
        ma_to_s = 1e6 * 365.25 * 24 * 60 * 60
        
        #thickness of lithosphere from half space cooling and water depth from isostasy
        if options['Seafloor age profile'] == 'half space cooling':
            hl = 2.32 * np.sqrt(mech.kappa * seafloor_ages * ma_to_s)
            hw = (hl * ((rho_a - rho_l)/(rho_w - rho_a))) + 2600
        
        #water depth from half space cooling and lithospheric thickness from isostasy
        if options['Seafloor age profile'] == 'plate model':            
            hw = np.where(seafloor_ages>81, 6586 - 3200 * np.exp((-seafloor_ages/62.8)), seafloor_ages)
            hw = np.where(hw<=81, 2600 + 345 * np.sqrt(hw),hw)
            hl = (hw - 2600) * ((rho_w - rho_a)/(rho_a - rho_l))
        
        #height of layers for integration
        zw = mech.L - hw
        zc = mech.L - (hw+hc)
        zl = mech.L - (hw+hc+hl)
        
        #calculate U
        self.U = 0.5 * mech.g * (rho_a*(zl)**2 + rho_l*(zc)**2 - rho_l*(zl)**2 + rho_c*(zw)**2 - rho_c*(zc)**2 + rho_w*(mech.L)**2 - rho_w*(zw)**2)
        
        if options['Continental crust']:
            cthick = nc.Dataset(crustal_thickness_file)
            crustal_thickness = cthick['Band1'][:]
            
            #Calculating GPE from crustal thickness
            rho_cc = 2861
            rho_cm = 3380
            continental_margin_transition = (2600*rho_w + hc*rho_c + rho_a*(mech.L-hc-2600) - mech.L*rho_cm)/(rho_cc - rho_cm)
            
    
            hcc = crustal_thickness*1000
            zma = (2600*rho_w + hc*rho_c + rho_a*(mech.L-hc-2600) - hcc*rho_cc)/rho_cm
            Uc = 0.5*mech.g*(rho_cm*(zma)**2 + rho_cc*(zma+hcc)**2 - rho_cc*(zma)**2)
            
            hwmarg = (2600*rho_w + hc*rho_c + rho_a*(mech.L-hc-2600) + hcc*(rho_cm - rho_cc) - mech.L*rho_cm)/(rho_w - rho_cm)
            zmarg = mech.L - hwmarg - hcc
            Ucmarg = 0.5 * mech.g * (rho_cm*(zmarg)**2 + rho_cc*(zmarg+hcc)**2 - rho_cc*(zmarg)**2 + rho_w*(mech.L)**2 - rho_w*(zmarg+hcc)**2)
    
            lats = np.arange(-90,90.5,0.5)
            lons = np.arange(-180,180.5,0.5)
            self.globe_gpe = np.zeros([len(lats),len(lons)])
            
            # Run this for just seafloor age and unchanging continental GPE
            for i in range(0,len(lats)):
                for j in range(0,len(lons)):
                    U_at_this_point = grid_to_point(lats[i],lons[j],self.U)
                    if str(U_at_this_point) != '--' and str(U_at_this_point) != 'nan':
                        self.globe_gpe[i,j] = U_at_this_point
                    else:
                        point_cthick = grid_to_point(lats[i],lons[j],crustal_thickness)*1000
                        if point_cthick > continental_margin_transition:
                            self.globe_gpe[i,j] = grid_to_point(lats[i],lons[j],Uc)
                        else:
                            self.globe_gpe[i,j] = grid_to_point(lats[i],lons[j],Ucmarg)
        
        ## If you have a crustal thickness file from a deforming mesh region, this can be incorporated into the GPE calculation here
        deforming_regions = False
        if  deforming_regions == True:
            mesh_thick = nc.Dataset('deforming_thickness_file')
            modelled_thickness = mesh_thick['z'][:]
            
            # modelled crustal thickness GPE
            hmesh = modelled_thickness*1000
            zma_mesh = (2600*rho_w + hc*rho_c + rho_a*(mech.L-hc-2600) - hmesh*rho_cc)/rho_cm
            Umesh = 0.5 * mech.g * (rho_cm*(zma_mesh)**2 + rho_cc*(zma_mesh+hmesh)**2 - rho_cc*(zma_mesh)**2)
            for i in range(0,len(lats)):
                for j in range(0,len(lons)):
                    U_at_this_point = grid_to_point(lats[i],lons[j],self.U)
                    if str(U_at_this_point) != '--':
                        self.globe_gpe[i,j] = U_at_this_point
                    else:
                        U_modelled = get_modelled_GPE(lats[i],lons[j],Umesh, mesh_thick)
                        if str(U_modelled) != 'None':
                            self.globe_gpe[i,j] = U_modelled
                        else:
                            self.globe_gpe[i,j] = grid_to_point(lats[i],lons[j],Uc)
        
        self.grid_spacing_degrees = 1
        num_latitudes = int(math.floor(180.0 / self.grid_spacing_degrees))
        num_longitudes = int(math.floor(360.0 / self.grid_spacing_degrees))
            
        # Create the sample points.
        points = []
        for lat_index in range(num_latitudes+1):
            lat = -90 + (lat_index) * self.grid_spacing_degrees
        
            for lon_index in range(num_longitudes+1):
                lon = -180 + (lon_index) * self.grid_spacing_degrees
                points.append(pygplates.PointOnSphere(lat, lon))
        
        self.points_in_resolved_topologies = {resolved_topology : [] for resolved_topology in globe.resolved_topologies}
        
        # Find the resolved topology plate that each point is contained within.
        # This is quite slow. PlateTectonicTools has 'points_in_polygons.find_polygons()' which speeds this up a lot.
        # And a future version of pyGPlates will have a similar approach.
        for point in points:
            for resolved_topology in globe.resolved_topologies:
                if resolved_topology.get_resolved_boundary().is_point_in_polygon(point):
                    self.points_in_resolved_topologies[resolved_topology].append(point)
                    break
                
    

class whole_plate:
    
    def __init__(self,features_name,globe,rotation_model,reconstruction_t,plate_ID,plate_index,anch_ID,delta_time):

        self.topology = globe.resolved_topologies[plate_index]
        self.num_segments = len(self.topology.get_boundary_sub_segments());
        self.topology_type = str(self.topology.get_feature().get_feature_type())
        self.plate_geometry = self.topology.get_resolved_geometry();
        self.stage_rotation = globe.poles.get_rotation(reconstruction_t, plate_ID, reconstruction_t + delta_time, anchor_plate_id=0)
        self.pole_lat, self.pole_lon, self.pole_angle = self.stage_rotation.get_lat_lon_euler_pole_and_angle_degrees()
        self.rotation_rate = self.pole_angle/delta_time              # [deg / Ma]
        self.plate_area =  globe.plate_list[plate_index,1] * 1e6;     # [m^2]
        self.plate_boundary_length = globe.plate_list[plate_index,2] * 1e3 # [m]
        self.plate_subducting_length = globe.plate_list[plate_index,3] * 1e3 # [m]
        self.plate_spreading_length = globe.plate_list[plate_index,4] * 1e3 # [m]
        self.plate_interior_centroid = self.plate_geometry.get_interior_centroid().to_lat_lon_array()
        centroid_vectors = pygplates.calculate_velocities(self.plate_interior_centroid,self.stage_rotation,delta_time,velocity_units = pygplates.VelocityUnits.cms_per_yr)
        self.centroid_velocity = np.asarray(pygplates.LocalCartesian.convert_from_geocentric_to_magnitude_azimuth_inclination(self.plate_interior_centroid,centroid_vectors))

class plate_bound:

    def __init__(self,globe,plate,reconstruction_t,delta_time,anch_ID,plate_ID,segment_index,segment_properties_ind,seafloor_ages,mech, segment_length, options):

        
        boundary_sub_segment = plate.topology.get_boundary_sub_segments()[segment_index]
        points = boundary_sub_segment.get_resolved_geometry().to_lat_lon_array()
        self.bound_name = boundary_sub_segment.get_resolved_feature().get_name()
        # segment splitting parameters
        point_interval = segment_length; # [km] (the small segments are combined into sections with this length)
        split_interval = 5;   # [km] (segments are initially split into small pieces with this length)
        point_chunk    = int(point_interval/split_interval) # i.e. how many little (temporary) chunks per big chunk
        
        
        # geometry stuff: split subduction and ridge boundary into segments
        if boundary_sub_segment.get_resolved_feature().get_feature_type() == pygplates.FeatureType.gpml_subduction_zone \
            or boundary_sub_segment.get_resolved_feature().get_feature_type() == pygplates.FeatureType.gpml_mid_ocean_ridge:
            
            # ignore if there's only one point
            if (len(points)) == 2 and (points[0,0]-points[1,0]) < 0.00001 and (points[0,1]-points[1,1]) < 0.00001:
                self.polarity = 'OnePoint'
                #print('Segment %.0f (%.0f plate ID) has 2 points, which are the same...') % (segment_index,plate_ID)
            # else, split!
            else:
                self.polarity = 'MultiplePoints' # placeholder (gets written over)

                # if the two start points (or end points) are two close, delete one of them
                start_point_distance = 1e-3 * abs(gpxpy.geo.haversine_distance(points[0,0], points[0,1], points[1,0], points[1,1]));  # [km]
                end_point_distance = 1e-3 * abs(gpxpy.geo.haversine_distance(points[len(points)-2,0], points[len(points)-2,1], points[len(points)-1,0], points[len(points)-1,1]));  # [km]
                if start_point_distance < 10.0 and len(points) > 2:
                    points = np.delete(points, (1), axis=0)
                if end_point_distance < 10.0 and len(points) > 2:
                    points = np.delete(points, (len(points)-2), axis=0)

                # split the segment into equal length sections (length = point_interval)
                for j in range(0,len(points)-1):
                    point_distance = 1e-3 * abs(gpxpy.geo.haversine_distance(points[j,0], points[j,1], points[j+1,0], points[j+1,1]));  # [km]
                    npoints = max(2,int(point_distance/split_interval))
                    gd = Geodesic.WGS84.Inverse(points[j,0], points[j,1], points[j+1,0], points[j+1,1])
                    line = Geodesic.WGS84.Line(gd['lat1'], gd['lon1'], gd['azi1'])
                    for i in range(npoints + 1):
                        point = line.Position(gd['s12'] / npoints * i)
                        if j == 0 and i == 0: # initiate new array
                            new_points=np.array([point['lat2'],point['lon2']])
                        elif i != 0: # avoid repetition at start and end points
                            new_points=np.vstack((new_points,np.array([point['lat2'],point['lon2']])))
                if point_chunk >= len(new_points):
                    point_chunk = len(new_points)-1;
                chunk_remainder = (float(len(new_points))/float(point_chunk)) - (len(new_points)/point_chunk)
                self.points=new_points[int(0.5*chunk_remainder*point_chunk)::point_chunk,:]
                del new_points;

                # get and save midpoints
                self.midpoints     = np.zeros((len(self.points)-1,2)) # lat, lon
                for p in range(0,len(self.points)-1):
                    lat_mid, lon_mid = midpoint(self.points[p,0],self.points[p,1],self.points[p+1,0],self.points[p+1,1])
                    self.midpoints[p,0:2] = lat_mid, lon_mid

        # subduction segments
        if  boundary_sub_segment.get_resolved_feature().get_feature_type() == pygplates.FeatureType.gpml_subduction_zone and self.polarity != 'OnePoint':
            
              # find subduction zone polarity
              self.type = 'subduction';
              subduction_polarity = boundary_sub_segment.get_feature().get_enumeration(pygplates.PropertyName.gpml_subduction_polarity)
              if subduction_polarity == 'Left':
                  self.polarity = 'Left'; polarity_ind = 1;
              elif subduction_polarity == 'Right':
                  self.polarity = 'Right'; polarity_ind = 2;
              else:
                  self.polarity = 'None'; polarity_ind = 0;
                  #print('Subduction segment %.0f (%.0f plate ID) has no subduction polarity, segment ignored in force calc..') % (segment_index,plate_ID)
              self.polarity_col    = polarity_ind*np.ones([len(self.points),1])
              self.boundary_points = np.concatenate((self.points, self.polarity_col), 1);
      
              # get plate IDs of subducting and overriding plates
              self.compute_plate_IDs(globe,plate,plate_ID,boundary_sub_segment,reconstruction_t)
      
              # (ignore if can't find subduction zone polarity or plate considered is not the subducting plate)
              if self.polarity != 'None' and self.plate_ids[0,0] == plate_ID:
      
                  # get trench-normal vectors and coordinates to extract ages (SP and OP)
                  self.compute_normal_vectors_and_age_coords(segment_index,plate_ID,seafloor_ages);
      
                  # calculate subduction and trench velocities
                  self.vc_vt      = np.zeros([len(self.points)-1,6]) # vc_mag, vc_azi, vc_mag_trench-norm, vt_mag, vt_azi, vt_mag_trench-norm,
                  self.av_vc      = np.zeros([len(self.points)-1])
                  subduction_initiation = boundary_sub_segment.get_feature().get_valid_time()[0]
    # DOESN'T WORK FOR "DISTANT PAST"
                  for p in range(0,len(self.points)-1):
                      # midpoints
                      # plate_bound_point = pygplates.PointOnSphere(pygplates.LatLonPoint(lat_mid,lon_mid))
                      plate_bound_point = pygplates.PointOnSphere(pygplates.LatLonPoint(self.midpoints[p,0],self.midpoints[p,1]))
                      # convergence rate
                      conv_velocity_rotation = globe.poles.get_rotation(reconstruction_t, int(self.plate_ids[p,0]), reconstruction_t + delta_time, int(self.plate_ids[p,1]))
                      conv_velocity_vector = pygplates.calculate_velocities(plate_bound_point,conv_velocity_rotation,delta_time,velocity_units = pygplates.VelocityUnits.cms_per_yr)
                      conv_velocity = pygplates.LocalCartesian.convert_from_geocentric_to_magnitude_azimuth_inclination(plate_bound_point,conv_velocity_vector)
                      conv_velocity_nsew = np.asarray(conv_velocity)[0,0] * np.array([np.cos(np.asarray(conv_velocity)[0,1]), np.sin(np.asarray(conv_velocity)[0,1])])
                      # average conv rate
                      try:
                          av_conv_velocity_rotation = globe.poles.get_rotation(reconstruction_t, int(self.plate_ids[p,0]), subduction_initiation - reconstruction_t, int(self.plate_ids[p,1]))
                          av_conv_velocity_vector = pygplates.calculate_velocities(plate_bound_point,av_conv_velocity_rotation,delta_time,velocity_units = pygplates.VelocityUnits.cms_per_yr)
                          av_conv_velocity = pygplates.LocalCartesian.convert_from_geocentric_to_magnitude_azimuth_inclination(plate_bound_point,av_conv_velocity_vector)
                          self.av_vc[p] = np.asarray(av_conv_velocity)[0,0]
                      except:
                          pass
                      # trench motion rate
                      abs_trench_stage_rotation = globe.poles.get_rotation(reconstruction_t, int(self.plate_ids[p,1]), reconstruction_t + delta_time, anchor_plate_id=anch_ID)
                      trench_velocity_vector = pygplates.calculate_velocities(plate_bound_point,abs_trench_stage_rotation,delta_time,velocity_units = pygplates.VelocityUnits.cms_per_yr)
                      trench_velocity = pygplates.LocalCartesian.convert_from_geocentric_to_magnitude_azimuth_inclination(plate_bound_point,trench_velocity_vector)
                      trench_velocity_nsew = np.asarray(trench_velocity)[0,0] * np.array([np.cos(np.asarray(trench_velocity)[0,1]), np.sin(np.asarray(trench_velocity)[0,1])])
                      #save mid-points and velocities
                      self.vc_vt[p,0:2] = np.asarray(conv_velocity)[0,0], np.asarray(conv_velocity)[0,1];
                      self.vc_vt[p,2]   = abs(np.dot(self.normal_vectors[p,2:4],conv_velocity_nsew))
                      self.vc_vt[p,3:5] = np.asarray(trench_velocity)[0,0], np.asarray(trench_velocity)[0,1];
                      self.vc_vt[p,5]   = np.dot(self.normal_vectors[p,2:4],trench_velocity_nsew)
                  
                      
                  # Get age of trench
                  self.subduction_duration = subduction_initiation - reconstruction_t
                    # compute subduction-related forces for each segment that has a polarity
                  self.compute_forces(mech,reconstruction_t,seafloor_ages,segment_index,segment_properties_ind,options);
    

        # ridge segments
        elif boundary_sub_segment.get_resolved_feature().get_feature_type() == pygplates.FeatureType.gpml_mid_ocean_ridge and self.polarity != 'OnePoint' :
            self.type = 'ridge';
            self.boundary_points = self.points;

            # get ridge-normal vectors (pointing towards plate with ID = "plate_ID")
            self.compute_normal_vectors_ridge(globe,segment_index,plate_ID,seafloor_ages);

            # compute ridge push for each segment that has a polarity
            if self.polarity != 'None':
                self.compute_forces(mech,reconstruction_t,seafloor_ages,segment_index,segment_properties_ind,options);

        else:
            self.type = 'other';
            self.points = points;
            self.boundary_points = points;

    def compute_plate_IDs(self,globe,plate,plate_ID,boundary_sub_segments,reconstruction_t):

        self.plate_ids = np.zeros((len(self.points)-1,2))  # plate id's of upper and subducting plates
        for j in range(len(self.points)-1):

            # azimuth
            lat1 = np.deg2rad(self.points[j,0]); lat2 = np.deg2rad(self.points[j+1,0]);
            lon1 = np.deg2rad(self.points[j,1]); lon2 = np.deg2rad(self.points[j+1,1]);
            trench_azim = math.atan2( np.sin(lon2-lon1)*np.cos(lat2),np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(lon2-lon1));
            if self.polarity_col[j,0] == 2 or self.polarity_col[j,0] == 0: # note ad-hoc fix for segments w/o polarities (i.e. = 0)
                SP_azim = trench_azim - (np.pi/2);
                OP_azim = trench_azim + (np.pi/2);
            else:
                SP_azim = trench_azim + (np.pi/2);
                OP_azim = trench_azim - (np.pi/2);

            # mid-points
            trench_lat, trench_lon = self.midpoints[j,0], self.midpoints[j,1]

            # project points onto OP and SP sides
            R = 6378.1; d = 30; # (radius and distance to project point distance, both km)
            SP_lat_pt, SP_lon_pt = project_point(trench_lat,trench_lon,SP_azim,d,R);
            OP_lat_pt, OP_lon_pt = project_point(trench_lat,trench_lon,OP_azim,d,R);

            SPpoint = pygplates.PointOnSphere(pygplates.LatLonPoint(SP_lat_pt,SP_lon_pt))
            SPpoint_feature = pygplates.Feature(); SPpoint_feature.set_geometry(SPpoint)
            OPpoint = pygplates.PointOnSphere(pygplates.LatLonPoint(OP_lat_pt,OP_lon_pt))
            OPpoint_feature = pygplates.Feature(); OPpoint_feature.set_geometry(OPpoint)

            # partition projected points into topologies, to extract ID's
            for plate_ind in range(0,len(globe.resolved_topologies)-1):
                partitioner = pygplates.PlatePartitioner(globe.resolved_topologies, globe.poles)
                partitioned_SPpoints = partitioner.partition_features(SPpoint_feature)
                if partitioned_SPpoints[0].get_reconstruction_plate_id() != 0:
                    self.plate_ids[j,0] = partitioned_SPpoints[0].get_reconstruction_plate_id()

                partitioned_OPpoints = partitioner.partition_features(OPpoint_feature)
                if partitioned_OPpoints[0].get_reconstruction_plate_id() != 0:
                      self.plate_ids[j,1] = partitioned_OPpoints[0].get_reconstruction_plate_id()

    def compute_normal_vectors_and_age_coords(self,segment_index,plate_ID,seafloor_ages):

        self.slab_ages = np.zeros((len(self.boundary_points)-1,1))
        self.slab_age_coords = np.zeros((len(self.boundary_points)-1,2))
        self.op_age_coords = np.zeros((len(self.boundary_points)-1,2))
        self.normal_vectors = np.zeros((len(self.boundary_points)-1,6))


        for j in range(len(self.normal_vectors)):

            lat1 = np.deg2rad(self.boundary_points[j,0]); lat2 = np.deg2rad(self.boundary_points[j+1,0]);
            lon1 = np.deg2rad(self.boundary_points[j,1]); lon2 = np.deg2rad(self.boundary_points[j+1,1]);
            trench_azimA = math.atan2( np.sin(lon2-lon1)*np.cos(lat2),np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(lon2-lon1)); # -pi to pi
            trench_azimB = (trench_azimA + 2*np.pi) % (2*np.pi); # 0 to 2*pi
            if self.boundary_points[j,2] == 2:
                trench_norm_SP = trench_azimA - (np.pi/2);
                trench_norm_OP = trench_azimA + (np.pi/2);
                trench_normVect = trench_azimA + (np.pi/2);
            elif self.boundary_points[j,2] == 1:
                trench_norm_SP = trench_azimA + (np.pi/2);
                trench_norm_OP = trench_azimA - (np.pi/2);
                trench_normVect = trench_azimA - (np.pi/2);

            if trench_normVect > 2*np.pi:
                trench_normVect -= 2*np.pi;
            elif trench_normVect < 0:
                trench_normVect += 2*np.pi;

            # NORMAL VECTORS
            self.normal_vectors[j,0:2]  = self.midpoints[j,0], self.midpoints[j,1]
            self.normal_vectors[j,2:4] = np.cos(trench_normVect), np.sin(trench_normVect);  # N-S (N positive), # E-W (E positive)
            self.normal_vectors[j,4] = trench_azimB;             # 0 to 2*pi, from north
            self.normal_vectors[j,5] = trench_normVect;          # 0 to 2*pi, from north

            # GET AGE EXTRACTION POINTS BY PROJECTING
            trench_lat, trench_lon  = self.midpoints[j,0], self.midpoints[j,1]
            #trench_lat = np.deg2rad(trench_lat); trench_lon = np.deg2rad(trench_lon)
            R = 6378.1;
            for d in range(50,550,50):
                lat_pt_SP, lon_pt_SP = project_point(trench_lat,trench_lon,trench_norm_SP,d,R);
                ocean_age = grid_to_point(lat_pt_SP,lon_pt_SP,seafloor_ages)
                if str(ocean_age) != '--':
                    continue
            self.slab_ages[j] = ocean_age
            self.slab_age_coords[j,0], self.slab_age_coords[j,1] = (lat_pt_SP,lon_pt_SP)
            lat_pt_OP, lon_pt_OP = project_point(trench_lat,trench_lon,trench_norm_OP,d,R);
            self.op_age_coords[j,0], self.op_age_coords[j,1] = (lat_pt_OP,lon_pt_OP)


    def compute_normal_vectors_ridge(self,globe,segment_index,plate_ID,seafloor_ages):

        self.normal_vectors = np.zeros((len(self.boundary_points)-1,6))
        self.ocean_age_coords = np.zeros((len(self.boundary_points)-1,3))

        self.polarity = 'None';
        for j in range(len(self.normal_vectors)):

            lat1 = np.deg2rad(self.boundary_points[j,0]); lat2 = np.deg2rad(self.boundary_points[j+1,0]);
            lon1 = np.deg2rad(self.boundary_points[j,1]); lon2 = np.deg2rad(self.boundary_points[j+1,1]);

            ridge_azim  = math.atan2( np.sin(lon2-lon1)*np.cos(lat2),np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(lon2-lon1)); # -pi to pi
            ridge_lat, ridge_lon  = self.midpoints[j,0], self.midpoints[j,1]

            R = 6378.1; d = 20; # (radius and distance to project point distance, both km)

            # In order to find which direction to apply the ridge push,..
            # partition points to find azimuth pointing towards plate of interest.
            norm_azimA = ridge_azim + (np.pi/2);
            lat_ptA, lon_ptA = project_point(ridge_lat,ridge_lon,norm_azimA,d,R);
            pointA = pygplates.PointOnSphere(pygplates.LatLonPoint(lat_ptA,lon_ptA))
            pointA_feature = pygplates.Feature(); pointA_feature.set_geometry(pointA)

            norm_azimB = ridge_azim - (np.pi/2);
            lat_ptB, lon_ptB = project_point(ridge_lat,ridge_lon,norm_azimB,d,R);
            pointB = pygplates.PointOnSphere(pygplates.LatLonPoint(lat_ptB,lon_ptB))
            pointB_feature = pygplates.Feature(); pointB_feature.set_geometry(pointB)

            # partition into plates and figure out which is correct azimuth
            partitioner = pygplates.PlatePartitioner(globe.resolved_topologies, globe.poles)
            partitioned_pointA = partitioner.partition_features(pointA_feature)
            partitioned_pointB = partitioner.partition_features(pointB_feature)
            if partitioned_pointA[0].get_reconstruction_plate_id() == plate_ID:
                ridge_norm_azim = norm_azimA; self.polarity = 'Right';
            elif partitioned_pointB[0].get_reconstruction_plate_id() == plate_ID:
                ridge_norm_azim = norm_azimB; self.polarity = 'Left';
            else:
                ridge_norm_azim = norm_azimA; # arbritary place-holder

            # output for subsequent force calculation
            if ridge_norm_azim > 2*np.pi:
                ridge_norm_azim -= 2*np.pi;
            elif ridge_norm_azim < 0:
                ridge_norm_azim += 2*np.pi;
            self.normal_vectors[j,0], self.normal_vectors[j,1]  = self.midpoints[j,0], self.midpoints[j,1]
            self.normal_vectors[j,2] = np.cos(ridge_norm_azim);  # N-S (N positive)
            self.normal_vectors[j,3] = np.sin(ridge_norm_azim);  # E-W (E positive)
            
            # get age profile of seafloor
            for age_profile in np.arange(100,9100,500):
                ocean_lat, ocean_lon = project_point(self.midpoints[j,0],self.midpoints[j,1],ridge_norm_azim,age_profile,R)
                if ocean_lat > 90:
                    ocean_lat = 90 - (ocean_lat-90)
                if ocean_lat < -90:
                    ocean_lat = -90 + (ocean_lat + 90)
                if ocean_lon > 180:
                    ocean_lon = ocean_lon - 360
                if ocean_lon < -180:
                    ocean_lon = ocean_lon + 360
                ocean_age = grid_to_point(ocean_lat,ocean_lon, seafloor_ages)
                ocean_point = pygplates.PointOnSphere(pygplates.LatLonPoint(ocean_lat,ocean_lon))
                ocean_point_feature = pygplates.Feature(); ocean_point_feature.set_geometry(ocean_point)
                partitioned_oceanp = partitioner.partition_features(ocean_point_feature)
                if partitioned_oceanp[0].get_reconstruction_plate_id() == plate_ID and str(ocean_age) != '--':
                    continue
                else:
                    age_profile = age_profile - 500
                    ocean_lat, ocean_lon = project_point(self.midpoints[j,0],self.midpoints[j,1],ridge_norm_azim,age_profile,R)
                    if ocean_lat > 90:
                        ocean_lat = 90 - (ocean_lat-90)
                    if ocean_lat < -90:
                        ocean_lat = -90 + (ocean_lat + 90)
                    if ocean_lon > 180:
                        ocean_lon = ocean_lon - 360
                    if ocean_lon < -180:
                        ocean_lon = ocean_lon + 360
                    ocean_age = grid_to_point(ocean_lat,ocean_lon, seafloor_ages)
                    ocean_point = pygplates.PointOnSphere(pygplates.LatLonPoint(ocean_lat,ocean_lon))
                    ocean_point_feature = pygplates.Feature(); ocean_point_feature.set_geometry(ocean_point)
                    partitioned_oceanp = partitioner.partition_features(ocean_point_feature)
                    break
            self.ocean_age_coords[j,0], self.ocean_age_coords[j,1] = ocean_lat, ocean_lon
            self.ocean_age_coords[j,2] = ocean_age
                

        if self.polarity == 'None':
            pass
            #print('Cant find ridge push direction for segment %.0f (%.0f plate ID), segment ignored in force calc..') % (segment_index,plate_ID)


    def compute_forces(self,mech,reconstruction_t,seafloor_ages,segment_index,segment_properties_ind,options):
        ma_to_s = 1e6 * 365.25 * 24 * 60 * 60;
        yr_to_s = 365.25 * 24 * 60 * 60;
        radius = 6378.1e3;
        
        # if self.type == 'subduction':
        #     self.slab_ages = seafloor_ages[segment_properties_ind:segment_properties_ind+len(self.normal_vectors[:,0]),:]

        # compute forces and torques---------------------
        self.plate_torques = np.zeros((len(self.normal_vectors),9)) # columns 3-5: slab pull torque, 5-7: bending torque, 7-9: ridge push torque
        self.plate_forces_for_plotting = np.zeros((len(self.normal_vectors),4))
        self.ridge_forces_for_plotting = np.zeros((len(self.normal_vectors),4))
        self.ages_for_plotting = np.zeros((len(self.normal_vectors),3))
        self.ocean_ages_for_plotting = np.zeros((len(self.normal_vectors),3))
        for j in range(len(self.normal_vectors)):

            try:

                segment_length = abs(gpxpy.geo.haversine_distance(self.boundary_points[j,0], self.boundary_points[j,1], self.boundary_points[j+1,0], self.boundary_points[j+1,1]));  # [m]
               
                if self.type == 'subduction':
                    #age of subducting slab
                    if str(self.slab_ages[j]) == '--':
                        continue
                    # segment velocities
                    vc_mag = self.vc_vt[j,0]; vc_norm = self.vc_vt[j,2]; vc_azi = self.vc_vt[j,1]; # 0 to 2*pi
                    vc = (1e-2/yr_to_s) * np.array((vc_mag * np.cos(vc_azi),vc_mag * np.sin(vc_azi)));  # [n-s , e-w]
                    vt_mag = self.vc_vt[j,3]; vt_norm = self.vc_vt[j,5]; vt_azi = self.vc_vt[j,4]; # 0 to 2*pi
                    vt = (1e-2/yr_to_s) * np.array((vt_mag * np.cos(vt_azi),vt_mag * np.sin(vt_azi)));  # [n-s , e-w]
                    
                    # segment forces
                    drho = mech.rho0 * mech.alpha * mech.dT;
                    if options['Seafloor age profile'] == 'half space cooling':
                        thick = 2.32 * math.sqrt(mech.kappa * self.slab_ages[j] * ma_to_s);  # from faccenna et al., 2011
                    elif options['Seafloor age profile'] == 'plate model':
                        if self.slab_ages[j] < 81:
                            hw = 2600 + 345 * np.sqrt(self.slab_ages[j])
                        else:
                            hw = 6586 - 3200 * np.exp((-self.slab_ages[j]/62.8))
                        thick = (hw - 2600) * ((mech.rho_w - mech.rho_a)/(mech.rho_a - mech.rho_l))

                    slab_pull = thick * mech.depth * drho * mech.g * self.normal_vectors[j,2:4] * (1./math.sqrt(math.pi)); # [n-s , e-w], [N/m]

                    # Slab bending
                    if options['bending mechanism'] == 'viscous':
                        bending_force = (-2./3.) * ((thick)/(mech.rad_curv))**3 * mech.lith_visc * vc;   # [n-s , e-w], [N/m]
                    elif options['bending mechanism'] == 'plastic':
                        bending_force = (-1./6.) * ((thick**2)/mech.rad_curv) * mech.yield_stress * np.array((np.cos(vc_azi),np.sin(vc_azi))) # [n-s, e-w], [N/m]
                    lat_rads = np.deg2rad(self.midpoints[j,0]); lon_rads = np.deg2rad(self.midpoints[j,1]);
                    # convert lon,lat to cartesian position vector (e.g. https://vvvv.org/blog/polar-spherical-and-geographic-coordinates)
                    position = radius * np.array([np.cos(lat_rads) * np.cos(lon_rads), np.cos(lat_rads) * np.sin(lon_rads), np.sin(lat_rads)]); # cartesian: x, y, z
                    # calc. segment torques
                    self.plate_torques[j,0:3] = torque_in_cartesian(lat_rads,lon_rads,position,slab_pull[0],slab_pull[1],segment_length) # [Nm]
                    self.plate_torques[j,3:6] = torque_in_cartesian(lat_rads,lon_rads,position,bending_force[0],bending_force[1],segment_length)
                    # just for plotting on map:
                    self.plate_forces_for_plotting[j,0:2]   = self.midpoints[j,0], self.midpoints[j,1]
                    self.plate_forces_for_plotting[j,2:4]   = slab_pull # + bending_force
                    self.ages_for_plotting[j,0:2] = self.slab_age_coords[j,1], self.slab_age_coords[j,0]
                    self.ages_for_plotting[j,2] = self.slab_ages[j]

                elif self.type == 'ridge':
                    
                    # ridge push (expression from Turcotte and Schubert 2nd ed., p 283, eq. 6-405)
                    ridge_push = mech.g * mech.rho0 * mech.alpha * mech.dT * ( 1.0 + ((2.0 * mech.rho0 * mech.alpha * mech.dT)/(np.pi * (mech.rho0 - mech.rhoW))) ) \
                        * mech.kappa * (self.ocean_age_coords[j,2] * ma_to_s) * self.normal_vectors[j,2:4]; # [n-s , e-w], [N/m]
                        #* mech.kappa * (mech.lith_age_RP * ma_to_s) * self.normal_vectors[j,2:4]; # [n-s , e-w], [N/m]     <----- uncomment for constant age RP force
                         
                    lat_rads = np.deg2rad(self.midpoints[j,0]); lon_rads = np.deg2rad(self.midpoints[j,1]);

                    # convert lon,lat to cartesian position vector (e.g. https://vvvv.org/blog/polar-spherical-and-geographic-coordinates)
                    position = radius * np.array([np.cos(lat_rads) * np.cos(lon_rads), np.cos(lat_rads) * np.sin(lon_rads), np.sin(lat_rads)]); # cartesian: x, y, z
                    # calc. segment torque
                    self.plate_torques[j,6:9] = torque_in_cartesian(lat_rads,lon_rads,position,ridge_push[0],ridge_push[1],segment_length) # [Nm]
                    
                    #just for plotting on map
                    self.ridge_forces_for_plotting[j,0:2] = self.midpoints[j,0], self.midpoints[j,1]
                    self.ridge_forces_for_plotting[j,2:4] = ridge_push
                    self.ocean_ages_for_plotting[j,0:2] = self.ocean_age_coords[j,1], self.ocean_age_coords[j,0]
                    self.ocean_ages_for_plotting[j,2] = self.ocean_age_coords[j,2]

                else:
                    pass

            except:

                pass
            
def basal_tractions(mech, time, globe, grids, plate, plate_ID, options):
    
    ma_to_s = 1e6 * 365.25 * 24 * 60 * 60
    points_in_resolved_topology = grids.points_in_resolved_topologies[plate.topology]
    bs_torques_ocean = []
    bs_torques_cont = []
    gpe_torques_list = []
    
    for point in range(len(points_in_resolved_topology)):
         
        # Get position vector of point
        lat, lon = points_in_resolved_topology[point].to_lat_lon()
        lat = round(lat,2)
        lon = round(lon,2)
        p = pygplates.PointOnSphere(lat,lon)
        lat_rads = np.deg2rad(lat); lon_rads = np.deg2rad(lon)
        position = 6378.1e3 * np.array([np.cos(lat_rads) * np.cos(lon_rads), np.cos(lat_rads) * np.sin(lon_rads), np.sin(lat_rads)]); # cartesian: x, y, z
        dx_length = grids.grid_spacing_degrees*(np.pi/180)*6378.1e3*np.cos(lat_rads)
        dy_length = grids.grid_spacing_degrees*(np.pi/180)*6378.1e3
        
        # Calculate velocity at point
        velocity_rotation = globe.poles.get_rotation(time, plate_ID, time + 1)
        vel = pygplates.calculate_velocities(p,velocity_rotation,1,velocity_units = pygplates.VelocityUnits.kms_per_my)
        velocity = np.asarray(pygplates.LocalCartesian.convert_from_geocentric_to_magnitude_azimuth_inclination(p, vel))
        velx = -1 * velocity[0][0] * (1000/ma_to_s) * np.sin(velocity[0][1])
        vely = -1 * velocity[0][0] * (1000/ma_to_s) * np.cos(velocity[0][1])
        
                
        # # Calculate basal shear traction
        Fco_mag = np.sqrt((vely*dx_length*dy_length)**2 + (velx*dx_length*dy_length)**2)
        
        alpha = 0
        
        if velx > 0:
            alpha = np.arctan(vely/velx)
        elif velx < 0 and vely >= 0:
            alpha = np.pi + np.arctan(vely/velx)
        elif velx < 0 and vely < 0:
            alpha = np.arctan(vely/velx) - np.pi
        elif velx == 0 and vely > 0:
            alpha = np.pi
        elif velx == 0 and vely < 0:
            alpha = - np.pi
        
        F_co = Fco_mag * np.array([np.cos(alpha) * (-1.0 * np.sin(lon_rads)), np.cos(alpha) * np.cos(lon_rads), np.sin(alpha) * np.cos(lat_rads)])            
        
        # Determine whether oceanic or continental
        U_at_this_point = grid_to_point(lat,lon,grids.U)
        if str(U_at_this_point) == '--' or str(U_at_this_point) == 'nan':
            is_continental = True
        else:
            is_continental = False

        
        # get torque
        temp_bs_torque = np.cross(position, F_co)
        if str(temp_bs_torque[0]) != 'nan' and is_continental == False:
            bs_torques_ocean.append(temp_bs_torque)
        elif str(temp_bs_torque[0]) != 'nan' and is_continental ==True:
            bs_torques_cont.append(temp_bs_torque)
        
        # Get nearby points
        
        dx_lon = lon+0.5 * grids.grid_spacing_degrees
        minus_dx_lon = lon-0.5 * grids.grid_spacing_degrees
        if dx_lon>180:
            dx_lon = dx_lon-360
        if minus_dx_lon<-180:
            minus_dx_lon = minus_dx_lon+360
        dy_lat = lat+0.5 * grids.grid_spacing_degrees
        dy_lon = lon
        if dy_lat>90:
            dy_lat = 90 - 2 * grids.grid_spacing_degrees
            dy_lon = lon + 180
            if dy_lon>180:
                dy_lon = dy_lon - 360
        minus_dy_lat = lat-0.5 * grids.grid_spacing_degrees
        minus_dy_lon = lon
        if minus_dy_lat<-90:
            minus_dy_lat = -90 + 2 * grids.grid_spacing_degrees
            minus_dy_lon = lon + 180
            if minus_dy_lon>180:
                minus_dy_lon = minus_dy_lon - 360
                dx_lon = round(dx_lon,2)
        minus_dx_lon = round(minus_dx_lon,2)
        dy_lat = round(dy_lat,2)
        minus_dy_lat = round(minus_dy_lat,2)
        dy_lon = round(dy_lon,2)
        minus_dy_lon = round(minus_dy_lon,2)
        if plate.topology.get_resolved_boundary().is_point_in_polygon(pygplates.PointOnSphere(lat,dx_lon)) and plate.topology.get_resolved_boundary().is_point_in_polygon(pygplates.PointOnSphere(lat,minus_dx_lon)) and plate.topology.get_resolved_boundary().is_point_in_polygon(pygplates.PointOnSphere(dy_lat,dy_lon)) and plate.topology.get_resolved_boundary().is_point_in_polygon(pygplates.PointOnSphere(minus_dy_lat,minus_dy_lon)):
            if options['Continental crust']:
                Fx = (-mech.L0/mech.L)* (grid_to_point(lat,dx_lon,grids.globe_gpe) - grid_to_point(lat,minus_dx_lon,grids.globe_gpe))/(grids.grid_spacing_degrees*(np.pi/180)*6378.1e3*np.cos(lat*(np.pi)/180))     
                Fy = (-mech.L0/mech.L)*(grid_to_point(dy_lat,dy_lon,grids.globe_gpe) - grid_to_point(minus_dy_lat,minus_dy_lon,grids.globe_gpe))/(grids.grid_spacing_degrees*(np.pi/180)*6378.1e3)
            else:
                Fx = (-mech.L0/mech.L)* (grid_to_point(lat,dx_lon,grids.U) - grid_to_point(lat,minus_dx_lon,grids.U))/(grids.grid_spacing_degrees*(np.pi/180)*6378.1e3*np.cos(lat*(np.pi)/180))     
                Fy = (-mech.L0/mech.L)*(grid_to_point(dy_lat,dy_lon,grids.U) - grid_to_point(minus_dy_lat,minus_dy_lon,grids.U))/(grids.grid_spacing_degrees*(np.pi/180)*6378.1e3)
            
            
            
            #calculate torque
            force_mag = np.sqrt((Fy*dx_length*dy_length)**2 + (Fx*dx_length*dy_length)**2) # [N]
            
            theta = 0
            
            if str(force_mag) != '--' and force_mag != 0:
                
                # gives angle anti-clockwise from E-W
                if Fx >= 0 and Fy >= 0:
                    theta = np.arctan(Fy/Fx)
                elif (Fx <= 0 and Fy >= 0) or (Fx <= 0 and Fy <= 0):
                    theta = np.pi + np.arctan(Fy/Fx)
                elif Fx >= 0 and Fy < 0:
                    theta = (2*np.pi) + np.arctan(Fy/Fx)

                # get force in cartesian coordinates
                force = force_mag * np.array([ np.cos(theta) * (-1.0*np.sin(lon_rads)),  np.cos(theta) * np.cos(lon_rads), np.sin(theta) * np.cos(lat_rads)]); # cartesian: x, y, z
                
                # get torque
                temp_torque = np.cross(position,force)
                if str(temp_torque[0]) != 'nan':
                    gpe_torques_list.append(temp_torque)

        else:
            continue
        
    gpe_torques = np.array(gpe_torques_list)
    bs_ocean_torques = np.array(bs_torques_ocean)
    bs_cont_torques = np.array(bs_torques_cont)
    try:
        net_bs_ocean_torque = np.array([np.sum(bs_ocean_torques[:,0]),np.sum(bs_ocean_torques[:,1]),np.sum(bs_ocean_torques[:,2])])
    except:
        net_bs_ocean_torque = np.zeros([1,3])[0]
    try:
        net_bs_cont_torque = np.array([np.sum(bs_cont_torques[:,0]),np.sum(bs_cont_torques[:,1]),np.sum(bs_cont_torques[:,2])])
    except:
        net_bs_cont_torque = np.zeros([1,3])[0]
    try:
        net_gpe_torque = np.array([np.sum(gpe_torques[:,0]),np.sum(gpe_torques[:,1]),np.sum(gpe_torques[:,2])])
    except:
        net_gpe_torque = np.zeros([1,3])[0]
    return net_bs_ocean_torque, net_bs_cont_torque, net_gpe_torque

#--------------------------------------------------
# put non-PyGPlates-related functions below:
#--------------------------------------------------

def torque_in_cartesian(lat_rads,lon_rads,position,Fns,Fes,segment_length):

    force_mag = np.sqrt((Fns*segment_length)**2 + (Fes*segment_length)**2) # [N]
    if str(force_mag) != '--' and force_mag != 0:
        # gives angle anti-clockwise from E-W
        if Fes >= 0 and Fns >= 0:
            theta = np.arctan(Fns/Fes)
        elif (Fes < 0 and Fns >= 0) or (Fes < 0 and Fns < 0):
            theta = np.pi + np.arctan(Fns/Fes)
        elif Fes >= 0 and Fns < 0:
            theta = (2*np.pi) + np.arctan(Fns/Fes)
    
        # get force in cartesian coordinates
        force = force_mag * np.array([ np.cos(theta) * (-1.0*np.sin(lon_rads)),  np.cos(theta) * np.cos(lon_rads), np.sin(theta) * np.cos(lat_rads)]); # cartesian: x, y, z
    
        # get torque
        return np.cross(position,force)  # torque [Nm] in cartesian
    else:
        return np.zeros([3])

def midpoint(latA, lonA, latB, lonB):
    lonA = math.radians(lonA); lonB = math.radians(lonB)
    latA = math.radians(latA); latB = math.radians(latB)
    dLon = lonB - lonA
    Bx = math.cos(latB) * math.cos(dLon)
    By = math.cos(latB) * math.sin(dLon)
    latC = math.atan2(math.sin(latA) + math.sin(latB),math.sqrt((math.cos(latA) + Bx) * (math.cos(latA) + Bx) + By * By))
    lonC = lonA + math.atan2(By, math.cos(latA) + Bx)
    lonC = (lonC + 3 * math.pi) % (2 * math.pi) - math.pi
    return math.degrees(latC), math.degrees(lonC)


def project_point(lat1,lon1,azim,d,R):
    #projects a point from lat1,lon1 to lat2,lon2 along an azimuth of "azim" and distance "d"
    #takes in degrees, spits out degrees.

    lon1 = np.deg2rad(lon1); lat1 = np.deg2rad(lat1)
    lat2 = math.asin( math.sin(lat1)*math.cos(d/R) + math.cos(lat1)*math.sin(d/R)*math.cos(azim))
    lon2 = lon1 + math.atan2(math.sin(azim)*math.sin(d/R)*math.cos(lat1),math.cos(d/R)-math.sin(lat1)*math.sin(lat2))
    lat2 = math.degrees(lat2); lon2 = math.degrees(lon2)
    
    if lat2>90:
        lat2 = 90 - (lat2-90)
        lon2 = lon2+180
    
    if lon2>180:
            lon2 = lon2-360
    if lon2<-180:
        lon2 = lon2+360

    return lat2, lon2

def grid_to_point(lat,lon,file):
    lat_spacing = 180/(len(file[:,0])-1)
    lon_spacing = 360/(len(file[0,:])-1)
    rounded_lat = lat_spacing*round(lat/lat_spacing)
    index_lat = (rounded_lat+90)*(1/lat_spacing)
    rounded_lon = lon_spacing*round(lon/lon_spacing)
    index_lon = (rounded_lon+180)*(1/lon_spacing)
    value = file[int(index_lat),int(index_lon)]
    return value

def get_modelled_GPE(lat,lon,Umesh,mesh_thick):
    numlat = len(mesh_thick['y'])
    numlon = len(mesh_thick['x'])
    lat_spacing = (mesh_thick['y'][numlat-1].data - mesh_thick['y'][0].data)/(numlat-1)
    lon_spacing = (mesh_thick['x'][numlon-1].data - mesh_thick['x'][0].data)/(numlon-1)
    index_lat = (lat-mesh_thick['y'][0].data)*(1/lat_spacing)
    index_lon = (lon-mesh_thick['x'][0].data)*(1/lon_spacing)
    if index_lat < 0 or index_lat > numlat-1 or index_lon < 0 or index_lon > numlon-1:
        GPE_model = None
    else:
        GPE_model = Umesh[int(index_lat),int(index_lon)]
    return GPE_model
