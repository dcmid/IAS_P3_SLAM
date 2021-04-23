from slam.slam_map import SLAMMap
import slam.load_data as load_data
from slam.slam_utils import get_local_movement
import numpy as np
import pickle as pk

WHEEL_DIAM = 254/100    # wheel diameter (dm)
BOT_WIDTH = 393.7/100   # distance from center of left wheel to center of right (dm)
SKID_FACTOR = 1.85  # made up scalar to account for effect of skid steering effective width for rotation

WHEEL_CIRC = np.pi*WHEEL_DIAM    # wheel circumference (mm)
ENC_TICK_LEN = WHEEL_CIRC / 360  # length of one encoder tick (mm)

def slam(enc_path, lidar_path, pickle_path=None, map_shape=(800,800), num_particles=30, xy_var=0.3, theta_var=0.001):
    slmap = SLAMMap(map_shape=map_shape, num_particles=num_particles, xy_var=xy_var, theta_var=theta_var)
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
    while ( (i < len(local_mv)) and (j < len(lidar)) ):  # loop through movement and lidar in time-sequential order
        if (enc_ts[i] < lidar[j]['t']):
            if ( i % 500 == 0):
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

    if pickle_path is not None:
        print('pickling...')
        pk.dump(slmap.occ_grid_map, open(pickle_path, "wb"))

    print('fin')
    return slmap