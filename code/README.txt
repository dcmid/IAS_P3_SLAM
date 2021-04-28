The files in this top-level directory are:

- odometry_only.ipynb
  This notebook is used to generate dead-reckoning trajectory and
  occupancy grid map estimates, plot them, and save them for later use.

- slam_main.py
  This script will run SLAM on data from provided paths, plot the
  resulting map/particle trajectories, and save the final map for later use.
  To run my SLAM algorithm on a different file, just change the appropriate 
  paths at the top of this script and run it.
  It is expected to get one print of "BAD WEIGHTS" initially, as the whole map
  is initialized to 0, so all weights will be calculated as 0. This is fine.
  If it happens after that, all particles have gone degenerate simultaneously.
  Either more agents or higher noise can fix this.

- plot_comparissons.py
  This script reads in all the previously generated maps from odometry_only 
  and slam_main and generates comparisson plots for my report.

- test_mapCorrelation.py
  This is just a test file for my implementation of mapCorrelation


The slam package includes everything needed for my top-level scripts to work.
Relevant files are listed here.

- slam.py
  This module includes the slam() function, which can be used to perform the
  full SLAM algorithm on data at provided paths and returns the resulting
  SLAMMap object. An example of slam() use is found in the top-level slam_main.py.

- slam_map.py
  This module includes my Bot and SLAMMap classes. The SLAMMap contains the
  occupancy grid map and one Bot for each particle.

- slam_utils.py
  This module contains a variety of helper functions used in the other modules.
  It also contains my custom color map, slam_cmap, which is used for plotting
  the occupancy grid map.

- MapUtilsCython/update_ogm.pyx
  I implemented my own Cython function for updating my occupancy grid map.
  It substantially reduced runtime.
