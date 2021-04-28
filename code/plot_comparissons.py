import pickle as pk
import matplotlib.pyplot as plt
from slam.slam_utils import slam_cmap

ogm20 = pk.load(open('./pickles/ogm20.pickle', 'rb'))
ogm21 = pk.load(open('./pickles/ogm21.pickle', 'rb'))
ogm22 = pk.load(open('./pickles/ogm22.pickle', 'rb'))
ogm23 = pk.load(open('./pickles/ogm23.pickle', 'rb'))
ogm24 = pk.load(open('./pickles/ogm24.pickle', 'rb'))

map20 = pk.load(open('./pickles/slam20.pickle', 'rb'))[300:700,300:700]
map21 = pk.load(open('./pickles/slam21.pickle', 'rb'))[300:700,300:700]
map22 = pk.load(open('./pickles/slam22.pickle', 'rb'))[150:550,250:650]
map23 = pk.load(open('./pickles/slam23.pickle', 'rb'))[280:680, 150:550]
map24 = pk.load(open('./pickles/slam24.pickle', 'rb'))[100:500,300:700]

ogm20_fubar = pk.load(open('./pickles/ogm20_fubar.pickle', 'rb'))
traj20_fubar = pk.load(open('./pickles/traj20_fubar.pickle', 'rb'))
map20_fubar = pk.load(open('./pickles/slam20_fubar.pickle', 'rb'))[300:700,300:700]

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(ogm20.T, origin='lower', cmap=slam_cmap, vmin=-30, vmax=30)
plt.axis([250,750,200,700])
plt.axis('off')
plt.title('Dead Reckoning')

plt.subplot(1,2,2)
plt.imshow(map20.T, origin='lower', cmap=slam_cmap, vmin=-30, vmax=30)
plt.axis('off')
plt.title('SLAM')

plt.show()
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(ogm21.T, origin='lower', cmap=slam_cmap, vmin=-30, vmax=30)
plt.axis([200,700,200,700])
plt.axis('off')
plt.title('Dead Reckoning')

plt.subplot(1,2,2)
plt.imshow(map21.T, origin='lower', cmap=slam_cmap, vmin=-30, vmax=30)
plt.axis('off')
plt.title('SLAM')

plt.show()
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(ogm22.T, origin='lower', cmap=slam_cmap, vmin=-30, vmax=30)
plt.axis([150,650,200,700])
plt.axis('off')
plt.title('Dead Reckoning')

plt.subplot(1,2,2)
plt.imshow(map22.T, origin='lower', cmap=slam_cmap, vmin=-30, vmax=30)
plt.axis('off')
plt.title('SLAM')

plt.show()
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(ogm23.T, origin='lower', cmap=slam_cmap, vmin=-30, vmax=30)
plt.axis([250,750,50,550])
plt.axis('off')
plt.title('Dead Reckoning')

plt.subplot(1,2,2)
plt.imshow(map23.T, origin='lower', cmap=slam_cmap, vmin=-30, vmax=30)
plt.axis('off')
plt.title('SLAM')

plt.show()
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(ogm24.T, origin='lower', cmap=slam_cmap, vmin=-30, vmax=30)
plt.axis([120,620,280,780])
plt.axis('off')
plt.title('Dead Reckoning')

plt.subplot(1,2,2)
plt.imshow(map24.T, origin='lower', cmap=slam_cmap, vmin=-30, vmax=30)
plt.axis('off')
plt.title('SLAM')

plt.show()
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(ogm20_fubar.T, origin='lower', cmap=slam_cmap, vmin=-30, vmax=30)
plt.plot(traj20_fubar[:,0], traj20_fubar[:,1])
plt.axis([100,600,200,700])
plt.axis('off')
plt.title('Dead Reckoning')

plt.subplot(1,2,2)
plt.imshow(map20_fubar.T, origin='lower', cmap=slam_cmap, vmin=-30, vmax=30)
plt.plot
plt.axis('off')
plt.title('SLAM')

plt.show()