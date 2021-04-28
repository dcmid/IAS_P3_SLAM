from slam.slam import slam
import numpy as np

# np.random.seed(0xdeadbeef)

# paths to data
enc_path = '../data/Encoders20'
lidar_path = '../data/Hokuyo20'
pickle_path = './pickles/scratch.pickle'

# execute SLAM on data at provided paths and pickle final map at pickle_path
slam_map = slam(enc_path, lidar_path, pickle_path=pickle_path, num_particles=50, xy_std=0.1, theta_std=0.015)

# plot map and particle trajectories
slam_map.plot()