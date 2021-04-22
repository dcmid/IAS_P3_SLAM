import slam.load_data as load_data
import matplotlib.pyplot as plt
import numpy as np
from slam.slam_utils import get_local_movement, get_occupied_coords, map_correlation
from slam.slam_map import SLAMMap
import pickle as pk

# np.random.seed(0xdeadbeef)

# CONSANTS ----------------------------------------------------------------------------------------------
WHEEL_DIAM = 254/100    # wheel diameter (dm)
BOT_WIDTH = 393.7/100   # distance from center of left wheel to center of right (dm)
SKID_FACTOR = 1.85  # made up scalar to account for effect of skid steering effective width for rotation

WHEEL_CIRC = np.pi*WHEEL_DIAM    # wheel circumference (mm)
ENC_TICK_LEN = WHEEL_CIRC / 360  # length of one encoder tick (mm)

# SLAM --------------------------------------------------------------------------------------------------

slam20 = SLAMMap(map_shape=(800,800), num_particles=30, xy_var=0.3, theta_var=0.001)  # create new map with bot in it

# Front Right, Front Left,... encoder readings, time
FR_enc, FL_enc, RR_enc, RL_enc, enc_ts = load_data.get_encoder('../data/Encoders20')

R_enc = (FR_enc + RR_enc) / 2  #average value of right encoders
L_enc = (FL_enc + RL_enc) / 2  #average value of left encoders

local20 = get_local_movement(R_enc, L_enc,
                             enc_tick_len = ENC_TICK_LEN, 
                             bot_width = BOT_WIDTH * SKID_FACTOR)


lidar20 = load_data.get_lidar('../data/Hokuyo20')

i = 0
j = 0
# while( np.all(local20[i] == 0 )):  # don't move bot during initial rest
#     if (enc_ts[i] < lidar20[j]['t']):
#         i += 1
#     else:
#         slam20.sense_walls(lidar20[j])
#         j += 1
# while ( (i < len(local20)) and (j < len(lidar20)) ):  # loop through movement and lidar in time-sequential order
while ( (i < 1000) ):  # loop through movement and lidar in time-sequential order
    if (enc_ts[i] < lidar20[j]['t']):
        if ( i % 500 == 0):
            print('Move Num:', i)
        slam20.move_bot(local20[i])
        i += 1
    else:
        slam20.sense_walls(lidar20[j])
        j += 1
# while (i < len(local20)):  # loop through any remaining movement
#     slam20.move_bot(local20[i])
#     i += 1
# while (j < len(lidar20)):  # loop through any remaining lidar
#     slam20.sense_walls(lidar20[j])
#     j += 1

pk.dump(slam20.get_reduced_histories(factor=200), open("slam20_traj.pickle", "wb"))

print('fin')
slam20.plot()
plt.show()


# import cProfile
# import pstats
# profile = cProfile.Profile()
# profile.run('slam20.sense_walls(lidar20[j])')
# ps = pstats.Stats(profile)
# ps.print_stats()