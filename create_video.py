#!/home/cherr/workspace/stochastic_phase_field_model/.venv/bin/python3
#%%
from util import nparray_to_video, colored_line
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

def find_center_of_mass(phi, directions):
    center_of_mass = np.zeros((phi.shape[0], 2))
    center_of_mass[:,0] = np.sum(phi*np.arange(phi.shape[1])[None, None, :], axis=(1,2))/np.sum(phi, axis=(1,2))
    center_of_mass[:,1] = np.sum(phi*np.arange(phi.shape[2])[None, :, None], axis=(1,2))/np.sum(phi, axis=(1,2))
    center_of_mass += directions
    return center_of_mass

def create_composite_array(phi, directions):
    '''
    
    Arguments:
    ----------
    phi : np.ndarray
        4D numpy array of shape (num_cells, T, 256, 256)
    directions : np.ndarray
        3D numpy array of shape (num_cells, T, 2)
    '''
    min_x = np.min(directions[:,:,0]).astype(int)
    min_y = np.min(directions[:,:,1]).astype(int)
    max_x = np.max(directions[:,:,0]).astype(int)
    max_y = np.max(directions[:,:,1]).astype(int)
    T = phi.shape[1]
    shape_x = max_x-min_x+256
    if shape_x % 2 != 0:
        shape_x += 1

    shape_y = max_y-min_y+256
    if shape_y % 2 != 0:
        shape_y += 1
    composite_array_phi = np.zeros((T, shape_x, shape_y))
    directions = directions.astype(int)
    for t in trange(T):
        for j in range(phi.shape[0]):
            xi = int(directions[j,t,0])-min_x
            yi = int(directions[j,t,1])-min_y
            composite_array_phi[t, xi:xi+256, yi:yi+256] += phi[j,t]


    return composite_array_phi

def create_composite_array_streaming(phi_filename, A_filename, directions, numCells, N):
    min_x = np.min(directions[:,:,0]).astype(int)
    min_y = np.min(directions[:,:,1]).astype(int)
    max_x = np.max(directions[:,:,0]).astype(int)
    max_y = np.max(directions[:,:,1]).astype(int)
    T = directions.shape[1]
    shape_x = max_x-min_x+N
    if shape_x % 2 != 0:
        shape_x += 1

    shape_y = max_y-min_y+N
    if shape_y % 2 != 0:
        shape_y += 1
    composite_array_phi = np.zeros((T, shape_x, shape_y))
    composite_array_A = np.zeros((T, shape_x, shape_y)) 
    directions = directions.astype(int)
    for t in trange(T):
        phi = np.fromfile(phi_filename, dtype=np.float32, count=numCells*N*N, offset=t*numCells*N*N*4).reshape(numCells, N, N)
        A = np.fromfile(A_filename, dtype=np.float32, count=numCells*N*N, offset=t*numCells*N*N*4).reshape(numCells, N, N)
        for j in range(numCells):
            xi = int(directions[j,t,0])-min_x
            yi = int(directions[j,t,1])-min_y
            composite_array_phi[t, xi:xi+N, yi:yi+N] += phi[j]
            composite_array_A[t, xi:xi+N, yi:yi+N] += A[j]*phi[j]

    return composite_array_phi, composite_array_A

def plot_center_of_mass(phi, directions):
    com = np.array( [find_center_of_mass(phi[i], directions[i]) for i in range(numCells)] )
    # reshape the data so it is 16xTx256x256 where every 16th frame is taken
    fig, ax = plt.subplots()
    for i in range(numCells):
        line = colored_line(com[i, :, 0], com[i, :, 1], np.arange(com.shape[1]), ax, linewidth=2, cmap='plasma')

    ax.set_xlim(com[:,:,0].min(), com[:,:,0].max())
    ax.set_ylim(com[:,:,1].min(), com[:,:,1].max())

    return fig, ax

import os
if __name__ == '__main__':
    # read the numpy array from the binary file
    filename_A = 'results/A.dat'
    filename_phi = 'results/phi.dat'
    filename_directions = 'results/direction.dat'

    directions = np.fromfile(filename_directions, dtype=np.int32)
    numCells = directions[0]
    directions = directions[1:].reshape(-1, 2)
    directions = directions[:, ::-1]
    directions = np.array([directions[i::numCells] for i in range(numCells)])

    phi_all, A_all = create_composite_array_streaming(filename_phi, filename_A, directions, numCells, 256)

    # pad the composite array so the width and heigh are divisible by 2
    nparray_to_video('phi.mp4', phi_all, fps=60)
    nparray_to_video('A.mp4', A_all, fps=60)
