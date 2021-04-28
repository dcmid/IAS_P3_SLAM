import slam.slam_utils as slam_utils
import slam.MapUtils as MapUtils
import slam.MapUtilsCython.MapUtils_fclad as mu
import slam.MapUtilsCython.update_ogm as og
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

class Bot:
    def __init__(self,pose=np.zeros(3),trajectory=None):
        self.pose = np.asarray(pose).copy().astype(np.double)  # global pose [x, y, theta] of bot
        if trajectory is None:
            self.trajectory = [self.pose.copy()]
        else:
            self.trajectory = trajectory
    
    def move(self,local_move):
        """ adjust bot global position given local movement """
        d_i = local_move[0]  # local 'x' movement
        d_j = local_move[1]  # local 'y' movement
        d_theta = local_move[2]  # change in angle
        theta = self.pose[2]  # angle before move
        d_x = (d_i * np.cos(theta)) + (d_j * np.sin(theta))  # global x movement
        d_y = (d_i * np.sin(theta)) + (d_j * np.cos(theta))  # global y movement

        self.pose += [d_x, d_y, d_theta]
        self.trajectory.append(self.pose.copy())

    def plot_trajectory(self):
        """ plots entire trajectory of bot """
        plt.plot(np.asarray(self.trajectory)[:,0],np.asarray(self.trajectory)[:,1])


class SLAMMap:
    def __init__(self, num_particles, map_shape=(1000,1000), xy_std = 0.3, theta_std = 0.001):
        self.occ_grid_map = np.zeros(map_shape, dtype=np.float64)  # initialize occupancy grid map
        self.ogm_history = [self.occ_grid_map.copy()]
        self.origin = np.asarray([map_shape[0], map_shape[1], 0], dtype=np.double) // 2  # center of map
        self.bots = np.asarray([Bot(self.origin) for p in range(num_particles)])  # initialize bot in center of map with angle 0
        self.weights = np.ones(num_particles) / num_particles  # initialize all bot (particle) weights equally

        self.xy_std = xy_std
        self.theta_std = theta_std

    def move_bot(self, local_move):
        """ move all particles globally given local movement """

        if(np.any(local_move > 0)):  # if bot moves, add noise to motion
            xy_noise = np.random.normal(0, self.xy_std, (len(self.bots),2))
            theta_noise = np.random.normal(0, self.theta_std, len(self.bots))
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
        """ update weights of each particle based on which map cells it detects as occupied """
        corrs = slam_utils.map_correlation(self.occ_grid_map, occ_coords)  # correlation for each possible bot position
        new_weights = self.weights * corrs
        if(np.any(new_weights > 0)):  # don't update weights if all new weights are 0
            new_weights = new_weights / np.sum(new_weights)  # normalize weights
            self.weights = new_weights
        else:
            print('BAD WEIGHTS')

    def eff_particles(self):  # Neff = 1/sum_i(w_i^2)
        """ calculate number of effective particles """
        sq_weights = np.square(self.weights)
        sq_weights_sum = np.sum(sq_weights)
        n_eff = 1/sq_weights_sum
        # print(n_eff)
        return n_eff
    
    def resample(self):
        """ resample all particle positions and set weights equal """
        resampled_idxs = np.random.choice(a=range(len(self.weights)), size=len(self.weights), replace=True, p=self.weights)
        self.bots = [Bot(self.bots[idx].pose, self.bots[i].trajectory) for i,idx in enumerate(resampled_idxs)]
        self.weights[:] = 1/len(self.weights)  # reset all weights

    def sense_walls(self, lidar):
        """ given lidar readings, determine which cells each particle detects to be occupied and empty """
        poses = np.array([bot.pose for bot in self.bots])
        occ_coords = slam_utils.get_occupied_coords(poses, lidar).astype(np.int16)  # coords detected occupied

        self.update_weights(occ_coords)  # update weight of each bot (particle)

        # if there are too many degenerate particles (low effective # of particles), resample and reset weights
        if ( self.eff_particles() < len(self.bots) * 0.75 ):
            self.resample()
            poses = np.array([bot.pose for bot in self.bots])
            occ_coords = slam_utils.get_occupied_coords(poses, lidar).astype(np.int16)  # coords detected occupied

        # loop through bots (particles), calculating empty_coords and updating occupancy grid map
        for i,bot in enumerate(self.bots):
            empty_coords = mu.getMapCellsFromRay_fclad(bot.pose[0],bot.pose[1],  # current bot pose
                                                       occ_coords[i,:,0],occ_coords[i,:,1],  # ray end points
                                                       np.max(self.occ_grid_map.shape)).astype(np.int16, copy=True).T  # map max

            self.occ_grid_map = og.update_ogm(self.occ_grid_map, occ_coords[i], empty_coords, self.weights[i])
        self.ogm_history.append(self.occ_grid_map.copy())  # keep a history of map for generating animations


    def plot(self):
        """ plot occupancy grid map and all particle trajectories """
        plt.imshow(self.occ_grid_map.T, origin='lower', cmap=slam_utils.slam_cmap, vmin = -30, vmax=30)
        for bot in self.bots:
            bot.plot_trajectory()
        plt.title('Occupancy Grid Map')
        plt.xlabel('x (dm)')
        plt.ylabel('y (dm)')
        plt.show()

    def plot_step(self, n, vmin=-30, vmax=30, cmap=slam_utils.slam_cmap):
        """ plot occupancy grid map and all particle positions at time step n """
        plt.imshow(self.ogm_history[n].T, origin='lower', cmap=cmap, vmin = vmin, vmax=vmax)
        for bot in self.bots:
            plt.scatter(bot.trajectory[n][0], bot.trajectory[n][1])
        plt.show()
            

    def get_reduced_histories(self, factor):
        """ history/trajectories are huge. L """
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

    def make_animation(self, wr_path, resolution):
        reduced_hist = self.get_reduced_histories(resolution)
        part_trajs = reduced_hist['bot_traj']
        map_traj = reduced_hist['ogm_hist']

        fix,ax = plt.subplots()
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

        def animate(i):
            ax.cla();
            ax.imshow(self.map_traj[i].T, origin='lower', cmap=cmap, vmin = vmin, vmax=vmax);
            for bot in self.bots:
                ax.scatter(bot.trajectory[n][0], bot.trajectory[n][1])
            plt.axis('off')

        anim = animation.FuncAnimation(fig, animate, frames = len(sensors_scaled)-1, interval = 1, blit = False)
        wr = animation.FFMpegWriter(fps=10)
        anim.save(wr_path, writer=wr)