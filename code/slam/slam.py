import pickle as pk
import matplotlib.pyplot as plt
import slam.slam_utils as slam_utils

hist_dict = pk.load(open("slam20_traj.pickle", "rb"))

bot_trajs = hist_dict['bot_traj']
ogm_hist = hist_dict['ogm_hist']

plt.imshow(ogm_hist[-1].T, origin='lower', cmap=slam_utils.slam_cmap, vmin = -30, vmax=30)
for traj in bot_trajs:
    plt.scatter(traj[-1][0], traj[-1][1])
plt.show()