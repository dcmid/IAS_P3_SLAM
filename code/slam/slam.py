from slam.slam_map import SLAMMap
import slam.load_data as load_data
from slam.slam_utils import get_local_movement
import numpy as np
import pickle as pk

WHEEL_DIAM = 254/100    # wheel diameter (dm)
BOT_WIDTH = 393.7/100   # distance from center of left wheel to center of right (dm)
SKID_FACTOR = 1.85  # experimentally-determined scalar to account for effect of skid steering on rotation

WHEEL_CIRC = np.pi*WHEEL_DIAM    # wheel circumference (dm)
ENC_TICK_LEN = WHEEL_CIRC / 360  # length of one encoder tick (dm)


def slam(enc_path, lidar_path, pickle_path=None, anim_path=None, map_shape=(800,800), num_particles=30, xy_std=0.3, theta_std=0.012):
    """ executes SLAM using data at provided paths. Returns SLAMMap object. 
    
    Args:
        enc_path: path to encoder data
        lidar_path: path to lidar data
        pickle_path: path to write pickled final map. If None, map isn't written anywhere.
        anim_path: path to write animation. If None, animation isn't written anywhere.
        map_shape: shape of occupancy grid map
        num_particles: number of particles in particle filter
        xy_std: standard deviation in x and y when adding noise to particle motion
        theta_std: standard deviation in theta when adding noise to particle motion

    Returns:
        slmap: SLAMMap object containing occupancy grid map and all particles
    """

    slmap = SLAMMap(map_shape=map_shape, num_particles=num_particles, xy_std=xy_std, theta_std=theta_std)
    # Front Right, Front Left,... encoder readings, time
    FR_enc, FL_enc, RR_enc, RL_enc, enc_ts = load_data.get_encoder(enc_path)

    R_enc = (FR_enc + RR_enc) / 2  #average value of right encoders
    L_enc = (FL_enc + RL_enc) / 2  #average value of left encoders

    local_mv = get_local_movement(R_enc, L_enc,
                                enc_tick_len = ENC_TICK_LEN, 
                                bot_width = BOT_WIDTH * SKID_FACTOR)


    lidar = load_data.get_lidar(lidar_path)

    i = 0
    j = 0
    # loop through movement and lidar in time-sequential order
    while ( (i < len(local_mv)) and (j < len(lidar)) ):  
        if (enc_ts[i] < lidar[j]['t']):
            if ( i % 500 == 0):  # print once every 500 moves so we know it's working
                print('Move Num:', i)
            slmap.move_bot(local_mv[i])
            i += 1
        else:
            slmap.sense_walls(lidar[j])
            j += 1
    while (i < len(local_mv)):  # loop through any remaining movement
        slmap.move_bot(local_mv[i])
        i += 1
    while (j < len(lidar)):  # loop through any remaining lidar
        slmap.sense_walls(lidar[j])
        j += 1

    # pickle occupancy grid map
    if pickle_path is not None:
        print('pickling...')
        pk.dump(slmap.occ_grid_map, open(pickle_path, "wb"))

    if anim_path is not None:
        slmap.make_animation(anim_path,200)

    print('fin')
    return slmap