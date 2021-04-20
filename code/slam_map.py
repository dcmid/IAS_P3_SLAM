import load_data
import slam_utils
import MapUtils
import MapUtilsCython.MapUtils_fclad as mu
import numpy as np
import matplotlib.pyplot as plt

class Bot:
    def __init__(self,pose=np.zeros(3)):
        self.pose = pose.copy().astype(np.double)  # global pose [x, y, theta] of bot
        self.trajectory = [self.pose.copy()]
    
    def move(self,local_move):
        d_i = local_move[0]  # local 'x' movement
        d_j = local_move[1]  # local 'y' movement
        d_theta = local_move[2]  # change in angle
        theta = self.pose[2]  # angle before move
        d_x = (d_i * np.cos(theta)) + (d_j * np.sin(theta/2))  # global x movement
        d_y = (d_i * np.sin(theta)) + (d_j * np.cos(theta/2))  # global y movement

        self.pose += [d_x, d_y, d_theta]
        self.trajectory.append(self.pose.copy())

    def plot_trajectory(self):
        plt.plot(np.asarray(self.trajectory)[:,0],np.asarray(self.trajectory)[:,1])


class SLAMMap:
    def __init__(self, num_particles, map_shape=(1000,1000)):
        self.occ_grid_map = np.zeros(map_shape, dtype=np.double)  # initialize occupancy grid map
        self.origin = np.asarray([map_shape[0], map_shape[1], 0], dtype=np.double) // 2  # center of map
        self.bots = np.asarray([Bot(self.origin) for p in range(num_particles)])  # initialize bot in center of map with angle 0
        self.weights = np.ones(num_particles) / num_particles  # initialize all bot (particle) weights equally

    def move_bot(self, local_move):
        for bot in self.bots:
            noisy_move = local_move + [np.random.normal(0,0.1,1)[0], np.random.normal(0,0.1,1)[0], np.random.normal(0,0.01,1)[0]]
            bot.move(noisy_move)
        # print(self.bot.pose)

    def update_weights(self, occ_coords):
        poses = np.asarray([bot.pose for bot in self.bots])  # poses of all bots
        corrs = slam_utils.map_correlation(self.occ_grid_map, occ_coords, poses)  # correlation for each possible bot position
        new_weights = self.weights * corrs
        new_weights = new_weights / np.sum(new_weights)  # normalize weights
        self.weights = new_weights


    def sense_walls(self, lidar):
        # theta = self.bots[0].pose[2]
        occ_coords = np.zeros((len(self.bots), 2), dtype=np.int16)
        poses = np.array([bot.pose for bot in self.bots])
        occ_coords = slam_utils.get_occupied_coords(poses, lidar).astype(np.int16)  # coords detected occupied
        print(occ_coords.shape)

        self.update_weights(occ_coords)  # update weight of each bot (particle)

        print(self.weights)

        for i,bot in enumerate(self.bots):
            x = bot.pose[0].astype(np.int16)
            y = bot.pose[1].astype(np.int16)
            empty_coords = mu.getMapCellsFromRay_fclad(bot.pose[0],bot.pose[1],  # current bot pose
                                                       occ_coords[i,:,0],occ_coords[i,:,1],  # ray end points
                                                       np.max(self.occ_grid_map.shape)).astype(np.int32, copy=True).T  # map max

            for c in occ_coords[i]:
                self.occ_grid_map[c[0], c[1]] += 0.1 * self.weights[i]
            for c in empty_coords:
                self.occ_grid_map[c[0], c[1]] -= 0.01 * self.weights[i]

    def plot(self):
        for bot in self.bots:
            bot.plot_trajectory()
        plt.title('Dead Reckoning Trajectory Estimate')
        plt.xlabel('x (dm)')
        plt.ylabel('y (dm)')
        plt.show()