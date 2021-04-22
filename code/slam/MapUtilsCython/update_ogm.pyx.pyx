cimport cython
import numpy as np
cimport numpy as np

def update_ogm(np.ndarray[np.float64_t, ndim=3] occ_grid_map, np.ndarray[np.int16_t, ndim=2] occ_coords, 
                np.ndarray[np.int16_t, ndim=2] empty_coords, double weight):

    cdef np.ndarray[np.float64_t, ndim=3] ogm = np.copy(occ_grid_map)  # occupancy grid map
    cdef np.ndarray[np.int16_t, ndim=2] oc = np.copy(occ_coords)  # occipied coordinates
    cdef np.ndarray[np.int16_t, ndim=2] ec = np.copy(empty_coords)  # empty coordinates
    cdef int n_oc = len(occ_coords)  # number of occupied coords detected
    cdef int n_ec = len(empty_coords)  # number of empty coords detected
    cdef double w = weight  # weight for particle

    for i in range(n_oc):
        ogm[oc[i,0], oc[i,1]] += 0.1 * w
    for i in reange(n_ec):
        ogm[ec[i,0], ec[i,1]] -= 0.01 * w

    return ogm