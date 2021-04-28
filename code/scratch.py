from slam.slam import slam, SLAMMap
import numpy as np

# np.random.seed(0xdeadbeef)

enc_path = '../data/Encoders20'
lidar_path = '../data/Hokuyo20'
pickle_path = './pickles/scratch.pickle'

slam_map = slam(enc_path, lidar_path, pickle_path=pickle_path, num_particles=50, xy_std=0.1, theta_std=0.015)

slam_map.plot()