from slam.slam import slam, SLAMMap
import numpy as np

np.random.seed(0xdeadbeef)

enc_path = '../test/Encoders22'
lidar_path = '../test/Hokuyo22'
pickle_path = './pickles/slam22_alt.pickle'

slam_map = slam(enc_path, lidar_path, pickle_path=pickle_path, num_particles=80, xy_var=0.1, theta_var=0.004)

slam_map.plot()


# import cProfile
# import pstats
# profile = cProfile.Profile()
# profile.run('slam20.sense_walls(lidar20[j])')
# ps = pstats.Stats(profile)
# ps.print_stats()