import slam.slam_utils as slam_utils
import slam.MapUtils as MapUtils
import slam.MapUtilsCython.MapUtils_fclad as mu
import slam.MapUtilsCython.update_ogm as og
import numpy as np
import matplotlib.pyplot as plt

class Bot:
    def __init__(self,pose=np.zeros(3),trajectory=None):
        self.pose = pose.copy().astype(np.double)  # global pose [x, y, theta] of bot
        if trajectory is None:
            self.trajectory = [self.pose.copy()]
        else:
            self.trajectory = trajectory
    
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
    def __init__(self, num_particles, map_shape=(1000,1000), xy_var = 0.3, theta_var = 0.001):
        self.occ_grid_map = np.zeros(map_shape, dtype=np.float64)  # initialize occupancy grid map
        self.ogm_history = [self.occ_grid_map.copy()]
        self.origin = np.asarray([map_shape[0], map_shape[1], 0], dtype=np.double) // 2  # center of map
        self.bots = np.asarray([Bot(self.origin) for p in range(num_particles)])  # initialize bot in center of map with angle 0
        self.weights = np.ones(num_particles) / num_particles  # initialize all bot (particle) weights equally

        self.xy_var = xy_var
        self.theta_var = theta_var

    def move_bot(self, local_move):
        if(np.any(local_move > 0)):
            xy_noise = np.random.normal(0, self.xy_var, (len(self.bots),2))
            theta_noise = np.random.normal(0, self.theta_var, len(self.bots))
            pose_noise = np.asarray([xy_noise[:,0], xy_noise[:,1], theta_noise]).T

            for i,bot in enumerate(self.bots):
                noisy_move = local_move + pose_noise[i]
                bot.move(noisy_move)
        
        else:  # if bot is still, don't add noise
            xy_noise = np.zeros((len(self.bots),2))
            theta_noise = np.random.normal(0, 0, len(self.bots))
            pose_noise = np.asarray([xy_noise[:,0], xy_noise[:,1], theta_noise]).T
            for i,bot in enumerate(self.bots):
                noisy_move = local_move + pose_noise[i]
                bot.move(noisy_move)

    def update_weights(self, occ_coords):
        corrs = slam_utils.map_correlation(self.occ_grid_map, occ_coords)  # correlation for each possible bot position
        new_weights = self.weights * corrs
        if(np.any(new_weights > 0)):  # don't update weights if all new weights are 0
            new_weights = new_weights / np.sum(new_weights)  # normalize weights
            self.weights = new_weights
        else:
            print('BAD WEIGHTS')

    def eff_particles(self):  # Neff = 1/sum_i(w_i^2)
        sq_weights = np.square(self.weights)
        sq_weights_sum = np.sum(sq_weights)
        n_eff = 1/sq_weights_sum
        # print(n_eff)
        return n_eff
    
    def resample(self):
        # print(self.weights)
        is_degen = ( self.weights < 1 / (5*len(self.weights)) )  # boolean array that is true where weights are very small
        degenerate_idxs = np.where(is_degen)[0]  # indices of degenerate particles
        valid_idxs = np.where(np.invert(is_degen))[0]  # indices of valid particles
        valid_weights = self.weights[valid_idxs] / np.sum(self.weights[valid_idxs])  # normalized weights of valid particles

        resampled_idxs = np.random.choice(a=valid_idxs, size=len(self.weights), replace=True, p=valid_weights)

        self.bots = [Bot(self.bots[idx].pose, self.bots[i].trajectory) for i,idx in enumerate(resampled_idxs)]

        # best_idx = np.argmax(self.weights)  # index of bot with maximum weight
        # best_pose = self.bots[best_idx].pose
        # #poses = [best_pose + [np.random.normal(0,0.1,1)[0], np.random.normal(0,0.1,1)[0], np.random.normal(0,0.001,1)[0]] for i in range(len(self.bots))]
        # self.bots = np.array([Bot(best_pose,b.trajectory) for i,b in enumerate(self.bots)])  # initialize all particles at 'best' pose
        self.weights[:] = 1/len(self.weights)  # reset all weights

    def sense_walls(self, lidar):
        # theta = self.bots[0].pose[2]
        poses = np.array([bot.pose for bot in self.bots])
        occ_coords = slam_utils.get_occupied_coords(poses, lidar).astype(np.int16)  # coords detected occupied

        self.update_weights(occ_coords)  # update weight of each bot (particle)

        # if there are too many degenerate particles (low effective # of particles), resample and reset weights
        if ( self.eff_particles() < len(self.bots) * 0.5 ):
            self.resample()
            poses = np.array([bot.pose for bot in self.bots])
            occ_coords = slam_utils.get_occupied_coords(poses, lidar).astype(np.int16)  # coords detected occupied

        # max_idx = np.argmax(self.weights)
        # empty_coords = mu.getMapCellsFromRay_fclad(self.bots[max_idx].pose[0],self.bots[max_idx].pose[1],  # current bot pose
        #                                             occ_coords[max_idx,:,0],occ_coords[max_idx,:,1],  # ray end points
        #                                             np.max(self.occ_grid_map.shape)).astype(np.int16, copy=True).T  # map max

        # for c in occ_coords[max_idx]:
        #     self.occ_grid_map[c[0], c[1]] += 0.1
        # for c in empty_coords:
        #     self.occ_grid_map[c[0], c[1]] -= 0.01
        # self.ogm_history.append(self.occ_grid_map.copy())
        for i,bot in enumerate(self.bots):
            x = bot.pose[0].astype(np.int16)
            y = bot.pose[1].astype(np.int16)
            empty_coords = mu.getMapCellsFromRay_fclad(bot.pose[0],bot.pose[1],  # current bot pose
                                                       occ_coords[i,:,0],occ_coords[i,:,1],  # ray end points
                                                       np.max(self.occ_grid_map.shape)).astype(np.int16, copy=True).T  # map max

            self.occ_grid_map = og.update_ogm(self.occ_grid_map, occ_coords[i], empty_coords, self.weights[i])
            self.ogm_history.append(self.occ_grid_map.copy())
            # for c in occ_coords[i]:
            #     self.occ_grid_map[c[0], c[1]] += 0.1 * self.weights[i]
            # for c in empty_coords:
            #     self.occ_grid_map[c[0], c[1]] -= 0.01 * self.weights[i]

    def plot(self):
        plt.imshow(self.occ_grid_map.T, origin='lower', cmap=slam_utils.slam_cmap, vmin = -30, vmax=30)
        for bot in self.bots:
            bot.plot_trajectory()
        plt.title('Occupancy Grid Map')
        plt.xlabel('x (dm)')
        plt.ylabel('y (dm)')
        plt.show()

    def plot_step(self, n, vmin=-30, vmax=30, cmap=slam_utils.slam_cmap):
        plt.imshow(self.ogm_history[n].T, origin='lower', cmap=cmap, vmin = vmin, vmax=vmax)
        for bot in self.bots:
            plt.scatter(bot.trajectory[n][0], bot.trajectory[n][1])
        plt.show()

    def get_reduced_histories(self, factor):
        r_len = len(self.bots[0].trajectory) // factor
        r_trajectories = np.zeros(((len(self.bots), r_len , 2)))
        for i,bot in enumerate(self.bots):
            traj = np.asarray(bot.trajectory)[0::factor,0:2]
            r_trajectories[i,:,:] = traj[0:r_len]
        
        ogm_hist = np.asarray(self.ogm_history)
        r_map_hist = ogm_hist[0::factor][0:r_len]

        reduced_histories = {
            'bot_traj' : r_trajectories,
            'ogm_hist' : ogm_hist,
        }

        return reduced_histories