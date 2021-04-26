from slam.slam import slam, SLAMMap
import numpy as np

np.random.seed(0xdeadbeef)

enc_path = '../test/Encoders22'
lidar_path = '../test/Hokuyo22'
pickle_path = './pickles/slam22.pickle'

slam_map = slam(enc_path, lidar_path, pickle_path=pickle_path, num_particles=80, xy_var=0.1, theta_var=0.009)

slam_map.plot()