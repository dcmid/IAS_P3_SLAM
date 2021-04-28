import numpy as np
from scipy.special import expit
import math
from matplotlib.colors import LinearSegmentedColormap

colors = [(0,0,1), (0.8,0.8,0.8), (0,0,0)]
slam_cmap = LinearSegmentedColormap.from_list('slam', colors, N=100)

def get_local_movement(R_enc_val, L_enc_val, enc_tick_len, bot_width):
    """Calculates movement and angular change from encoder values with bot frame of reference

    Args:
        R_enc: a vector of readings from the right encoder
        L_enc: a vector of readings from the left encoder
        Both encoder vectors must be equal length

    Returns:
        d_i: movement in the direction that the robot was facing
        d_j: movement perpendicular to the direction the robot was facing
        d_theta: change in angle that robot is facing
    """
    R_mm = enc_tick_len * R_enc_val  # convert encoder to right wheel movement in mm
    L_mm = enc_tick_len * L_enc_val  # convert encoder to left wheel movement in mm
    d_theta = (R_mm - L_mm) / bot_width  # find change in angle
    d_i = (R_mm + L_mm) / 2 * np.cos(d_theta/2)  # motion in i direction
    d_j = (R_mm + L_mm) / 2 * np.sin(d_theta/2)  # motion in j direction
    return np.asarray([d_i, d_j, d_theta]).T

def dead_rec_trajectory(R_enc, L_enc, enc_tick_len, bot_width):
    """Estimates robot trajectory from encoder readings using dead reconning

    Args:
        R_enc: a vector of readings from the right encoder
        L_enc: a vector of readings from the left encoder
        Both encoder vectors must be equal length

    Returns:
        pose: numpy array of [x,y,theta] for every time step (len(pose) == len(R_enc) == len(L_enc))
    """
    pose = np.zeros((len(R_enc)+1,3))  # robot pose [x, y, theta]; theta is angle in radians
    move = get_local_movement(R_enc, L_enc, enc_tick_len, bot_width)  # local movement
    for i, m in enumerate(move):
        theta = pose[i,2]   # robot orientation angle before movement
        
        d_i = m[0]
        d_j = m[1]
        d_theta = m[2]
        d_x = (d_i * np.cos(theta)) + (d_j * np.sin(theta/2))  # global x movement
        d_y = (d_i * np.sin(theta)) + (d_j * np.cos(theta/2))  # global y movement
        
        pose[i+1] = pose[i] + [d_x, d_y, d_theta]  # update next robot pose using calculated changes
    pose[:,2] = np.mod(pose[:,2], 2*np.pi)
    return pose

def get_ray_end_coords(length,thetas):
    """ find gridmap coords of terminal point of ray

    Args:
        length: magnitude of ray (okay technically line segment but w/e)
                length should be given in units of grid squares
        theta: angle of ray

    Returns:
        end_coords: array with dims (len(thetas), 2)
                    [[x0,y0], [x1,y1], ...]
    """
    x = np.floor(length[np.newaxis,:]*np.cos(thetas)).astype(np.int16)
    y = np.floor(length[np.newaxis,:]*np.sin(thetas)).astype(np.int16)
    end_coords = np.asarray([x,y]).astype(np.int16)
    end_coords = np.moveaxis(end_coords,0,2)  # switch from (2 nbots nlidar) to (nbots nlidar 2)
    return end_coords

def get_occupied_coords(bot_poses, lidar):
    """ find occupied coordinates in map given bot poses and lidar reading

    Args:
        bot_pose: current pose of bot [x, y, angle]
        lidar: lidar reading from HokuyoXX.mat file

    Returns:
        occ_coords: coordinates of terminal points of all lidar readings
                    array with dimensions (nbots nlidar 2)
    """
    scan_dm = 10 * lidar['scan']  # lidar ranges in decimeters (resolution of grid)
    theta = (lidar['angle'] + bot_poses[:,2]).T  # add bot angle to lidar
    theta = np.mod(theta, 2*np.pi)  # reduce to minimum equivalent angle

    occ_coords = (bot_poses[:,np.newaxis,:2] + get_ray_end_coords(scan_dm,theta)).astype(np.int32)
    return occ_coords

def map_correlation(og_map, occ_coords):
    """ calculate correlation between detected occupied coordinates and current map

    Args:
        og_map: occupancy grid map; each value is log likelihood of occupancy
        occ_coords: detected occupied coordinates realative to bot
                    array dims (nbots, nendpoints, 2)

    Returns:
        cor: correlation between the two inputs
    """
    cor = np.zeros(len(occ_coords), dtype=np.float64)
    xs = occ_coords[:,:,0]  # occupied x coords
    ys = occ_coords[:,:,1]  # occupied y coords
    hits = og_map[xs,ys]  # values in all occupied cells
    cor[:] = np.sum(hits,axis=1)  # correlation = sum of hits
    cor -= min(np.min(cor),0)  # shift so all are positive
    if np.any(cor > 0):
        cor /= np.sum(cor)  # normalize to sum to 1

    return cor  
