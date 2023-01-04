#!/usr/bin/python

def computeForces(times, features_name, rotation_model, oceanic_ages, crustal_thickness_file, plateIDs_of_interest,options):

    import numpy as np
    import pygplates
    import netCDF4 as nc

    from functions_main import plate_bound, whole_plate, global_plates, grids, basal_tractions, set_mech_params

    topology_features = pygplates.FeatureCollection()
    for feature in features_name:
        #topology_feature = pygplates.FeatureCollection(filename)
        topology_features.add(feature)
    # ---------------------------------------

    # Set some options/parameters: --------
    delta_time = 1.0;
    anch_ID = 0;
    segment_length = 250
    mech=set_mech_params() # modify this class (in functions_main.py) to modify mechanical parameters
    # ---------------------------------------

    for reconstruction_t in times: # all times specified

        agegrid = nc.Dataset(oceanic_ages)
        seafloor_ages = agegrid['z'][:]

        # get global properties
        globe=global_plates(topology_features,rotation_model,reconstruction_t,anch_ID)
        if options['calculate tractions'] == True:
            grids=grids(options, mech, globe, seafloor_ages,crustal_thickness_file)
        
        # store plate boundary coordinates for plotting
        platebound_coords = np.empty((0,5), float)

        centroid_vels = np.zeros([0,4])
        # store individual segment forces for plotting
        segment_fullsubd_force = np.zeros([0,4]) # force due to: full subduction (slab pull + bending)
        segment_ridge_force = np.zeros([0,4]) # force due to ridge push
        slab_ages = np.zeros([0,3])
        ocean_ages = np.zeros([0,3])
        segment_properties_ind = 0;
        plate_id_list = []
        
        # store torques for each of the plates
        sp_torque_list = np.zeros([0,3])
        rp_torque_list = np.zeros([0,3])
        bending_torque_list = np.zeros([0,3])
        gpe_torque_list = np.zeros([0,3])
        bs_ocean_torque_list = np.zeros([0,3])
        bs_cont_torque_list = np.zeros([0,3])
        plate_data = np.zeros([0,6])



        # loop through every plate
        for plate_index in range(0,len(globe.plate_list)): # all plate boundaries

            # get whole plate properties
            plate_ID = int(globe.plate_list[plate_index,0])

            if plate_ID in plateIDs_of_interest:

                plate = whole_plate(topology_features,globe,rotation_model,reconstruction_t,plate_ID,plate_index,anch_ID,delta_time)
                
                # Ignore deforming networks for now
                if plate.topology_type != 'gpml:TopologicalClosedPlateBoundary':
                    continue
                
                # for summing up individual plate torques
                net_sp_torque = np.zeros([1,3]); net_bending_torque = np.zeros([1,3]); net_rp_torque = np.zeros([1,3]);
                
                # store centroid velocities
                centroid_vels = np.append(centroid_vels, np.array([[plate.plate_interior_centroid[0,1], plate.plate_interior_centroid[0,0], plate.centroid_velocity[0,0], plate.centroid_velocity[0,1]]]), axis=0)

                if options['calculate boundary forces']:
                    # loop through all segments
                    bound=[]
                    for segment_ind in range(0,plate.num_segments): # all plate boundary segments
    
    
                        bound.append(plate_bound(globe,plate,reconstruction_t,delta_time,anch_ID,plate_ID,segment_ind,segment_properties_ind,seafloor_ages,mech, segment_length,options));
    
                        # compute torque due to each segment
                        if bound[segment_ind].type == 'subduction':
    
                            # save plate boundary coordinates for plotting
                            formatted_coords = np.concatenate(( bound[segment_ind].points[0:len(bound[segment_ind].points)-1,:]  , \
                                bound[segment_ind].points[1:len(bound[segment_ind].points),:], np.ones((len(bound[segment_ind].points)-1,1))  ), axis=1)
                            platebound_coords = np.append(platebound_coords, formatted_coords, axis=0)
                            
                            # only a force acting if the plate is on the subduction side of the trench
                            if int(bound[segment_ind].plate_ids[0,0]) == plate_ID:
                            
    
                                ## only do force calculation if we have the subduction polarity
                                if bound[segment_ind].polarity != 'OnePoint'  and bound[segment_ind].polarity != 'None':
  
                                    ### sum torque acting on the plate
                                    net_sp_torque += np.array([np.sum(bound[segment_ind].plate_torques[:,0]),np.sum(bound[segment_ind].plate_torques[:,1]),np.sum(bound[segment_ind].plate_torques[:,2])])
                                    net_bending_torque += np.array([np.sum(bound[segment_ind].plate_torques[:,3]),np.sum(bound[segment_ind].plate_torques[:,4]),np.sum(bound[segment_ind].plate_torques[:,5])])

                                    # store segment forces for plotting
                                    segment_fullsubd_force = np.append(segment_fullsubd_force, bound[segment_ind].plate_forces_for_plotting, axis=0)
                                    slab_ages = np.append(slab_ages, bound[segment_ind].ages_for_plotting, axis=0)
    
                                    segment_properties_ind += len(bound[segment_ind].normal_vectors[:,0])
                                    
                        elif bound[segment_ind].type == 'ridge':
    
                            # save plate boundary coordinates for plotting
                            formatted_coords = np.concatenate(( bound[segment_ind].points[0:len(bound[segment_ind].points)-1,:]  , \
                                bound[segment_ind].points[1:len(bound[segment_ind].points),:], 2 * np.ones((len(bound[segment_ind].points)-1,1))  ), axis=1)
                            platebound_coords = np.append(platebound_coords, formatted_coords, axis=0)
    
                            if bound[segment_ind].polarity != 'OnePoint'  and bound[segment_ind].polarity != 'None':

                                ### add ridge torque acting on plate to "full torque"
                                net_rp_torque += np.array([np.sum(bound[segment_ind].plate_torques[:,6]),np.sum(bound[segment_ind].plate_torques[:,7]),np.sum(bound[segment_ind].plate_torques[:,8])])
                                
                                # store segment forces for plotting
                                segment_ridge_force = np.append(segment_ridge_force, bound[segment_ind].ridge_forces_for_plotting, axis=0)
                                ocean_ages = np.append(ocean_ages, bound[segment_ind].ocean_ages_for_plotting, axis=0)
    
                        else: # if transform, just save plate boundary coordinates for plotting
    
                            formatted_coords = np.concatenate(( bound[segment_ind].points[0:len(bound[segment_ind].points)-1,:]  , \
                                bound[segment_ind].points[1:len(bound[segment_ind].points),:], 3 * np.ones((len(bound[segment_ind].points)-1,1))), axis=1)
                            platebound_coords = np.append(platebound_coords, formatted_coords, axis=0)


                # retrieve plate centroid (where the force due to edge-torques is calculates)
                plate_data = np.append(plate_data, np.array([[plate.plate_interior_centroid[0,1], plate.plate_interior_centroid[0,0], plate.plate_area, plate.plate_boundary_length,plate.plate_subducting_length,plate.plate_spreading_length]]), axis=0)
                
                # Iterate over points in plates
                if options['calculate tractions'] == True:
                    net_bs_ocean_torque,net_bs_cont_torque, net_gpe_torque = basal_tractions(mech, reconstruction_t, globe, grids, plate, plate_ID, options)
                    net_gpe_torque = np.array([net_gpe_torque])
                    net_bs_ocean_torque = np.array([net_bs_ocean_torque])
                    net_bs_cont_torque = np.array([net_bs_cont_torque])
                
                plate_id_list.append(plate_ID)
                
                sp_torque_list = np.append(sp_torque_list,np.array([[net_sp_torque[0][0],net_sp_torque[0][1],net_sp_torque[0][2]]]),axis=0)
                rp_torque_list = np.append(rp_torque_list,np.array([[net_rp_torque[0][0],net_rp_torque[0][1],net_rp_torque[0][2]]]),axis=0)
                bending_torque_list = np.append(bending_torque_list,np.array([[net_bending_torque[0][0],net_bending_torque[0][1],net_bending_torque[0][2]]]),axis=0)
                try:
                    gpe_torque_list = np.append(gpe_torque_list,np.array([[net_gpe_torque[0][0],net_gpe_torque[0][1],net_gpe_torque[0][2]]]),axis=0)
                    bs_ocean_torque_list = np.append(bs_ocean_torque_list,np.array([[net_bs_ocean_torque[0][0],net_bs_ocean_torque[0][1],net_bs_ocean_torque[0][2]]]),axis=0)
                    bs_cont_torque_list = np.append(bs_cont_torque_list,np.array([[net_bs_cont_torque[0][0],net_bs_cont_torque[0][1],net_bs_cont_torque[0][2]]]),axis=0)
                except:
                    gpe_torque_list = None
                    bs_ocean_torque_list = None
                    bs_cont_torque_list = None
                
                
    return sp_torque_list, rp_torque_list, bending_torque_list, gpe_torque_list, bs_ocean_torque_list,bs_cont_torque_list, centroid_vels, platebound_coords, segment_fullsubd_force, segment_ridge_force, plate_id_list, plate_data
