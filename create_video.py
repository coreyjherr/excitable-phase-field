#!/home/cherr/miniconda3/envs/opencv/bin/python
#%%
# from util import nparray_to_video, colored_line
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from pathlib import Path

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
    composite_array = np.zeros((T, shape_x, shape_y))
    directions = directions.astype(int)
    for t in trange(T):
        phi = np.fromfile(phi_filename, dtype=np.float32, count=numCells*N*N, offset=t*numCells*N*N*4).reshape(numCells, N, N)
        if A_filename is not None:
            A = np.fromfile(A_filename, dtype=np.float32, count=numCells*N*N, offset=t*numCells*N*N*4).reshape(numCells, N, N)
        for j in range(numCells):
            xi = int(directions[j,t,0])-min_x
            yi = int(directions[j,t,1])-min_y
            if A_filename is None:
                composite_array[t, xi:xi+N, yi:yi+N] += phi[j]
            else:
                composite_array[t, xi:xi+N, yi:yi+N] += A[j]*phi[j]

    return composite_array

import cv2

def create_video_from_condensed(phi_filename: Path, A_filename: Path, directions_filename: Path):
    # get size of video in bytes
    directions = np.fromfile(directions_filename, dtype=np.int32)
    numCells = directions[0]
    T = directions[1]
    directions = directions[2:].reshape(-1, 2)
    directions = directions[:, ::-1]
    directions = np.array([directions[i::numCells] for i in range(numCells)])

    file_size = os.path.getsize(phi_filename)
    max_x = np.max(directions[:, :, 0])
    max_y = np.max(directions[:, :, 1])
    min_x = np.min(directions[:, :, 0])
    min_y = np.min(directions[:, :, 1])
    size_y = max_x-min_x+256
    size_x = max_y-min_y+256

    movie_path = Path('movies')
    phi_movie_filename = movie_path/phi_filename.with_suffix('.mp4').name
    A_movie_filename = movie_path/A_filename.with_suffix('.mp4').name
    vidphi = cv2.VideoWriter(phi_movie_filename, cv2.VideoWriter_fourcc(*'avc1'), 60, (size_y, size_x), isColor=False)
    vidA = cv2.VideoWriter(A_movie_filename, cv2.VideoWriter_fourcc(*'avc1'), 60, (size_y, size_x), isColor=False)

    current_position = 0
    current_frame = 0
    for i in trange(T-1, desc=str(phi_filename), dynamic_ncols=True):
        frame = np.zeros((size_x, size_y))
        current_size = np.fromfile(phi_filename, dtype=np.int32, count=2, offset=current_position)
        current_position += 8
        phi = np.fromfile(phi_filename, dtype=np.float32, count=current_size[0]*current_size[1], offset=current_position).reshape(current_size[1], current_size[0]).T
        A = np.fromfile(A_filename, dtype=np.float32, count=current_size[0]*current_size[1], offset=current_position).reshape(current_size[1], current_size[0]).T
        current_position += current_size[0]*current_size[1]*4

        current_min_x = np.min(directions[:, current_frame, 1]) - min_y
        current_min_y = np.min(directions[:, current_frame, 0]) - min_x
        frame[current_min_x:current_size[0]+current_min_x,current_min_y:current_size[1]+current_min_y] = phi
        # print(frame.shape)
        # print(frame.max())
        # convert frame to CV 8 bit image
        vidphi.write((frame*0.5*255).astype(np.uint8))

        frame[current_min_x:current_size[0]+current_min_x,current_min_y:current_size[1]+current_min_y] = A + 1
        # frame *= (A+1)
        vidA.write(((frame)*0.25*255).astype(np.uint8))

        current_frame += 1

    vidphi.release()
    vidA.release()


def plot_center_of_mass(com_filename: Path, numCells):
    # reshape the data so it is 16xTx256x256 where every 16th frame is taken
    com = np.fromfile(com_filename, dtype=np.float32).reshape(-1, 2)
    com = np.array([com[i::numCells] for i in range(numCells)])
    fig, ax = plt.subplots()
    for i in range(numCells):
        line = colored_line(com[i, :, 0], com[i, :, 1], np.arange(com.shape[1]), ax, linewidth=2, cmap='plasma')

    ax.set_xlim(com[:,:,0].min(), com[:,:,0].max())
    ax.set_ylim(com[:,:,1].min(), com[:,:,1].max())

    return fig, ax

def get_wall_matrix(shape, radius):
    wall = np.zeros(shape)
    x = (np.arange(wall.shape[0])-wall.shape[0]//2) * 0.2
    X, Y = np.meshgrid(x, x)
    # draw the wall between r>20 r<21
    wall[np.logical_and(X**2+Y**2>radius**2, X**2+Y**2<(radius+1)**2)] = 1
    return wall

def create_video_from_binary(phi_filename: Path, A_filename: Path, directions_filename: Path):
    movie_path = Path('movies')
    directions = np.fromfile(directions_filename, dtype=np.int32)
    numCells = directions[0]
    T = directions[1]
    directions = directions[2:].reshape(-1, 2)
    directions = directions[:, ::-1]
    directions = np.array([directions[i::numCells] for i in range(numCells)])

    phi_all = create_composite_array_streaming(phi_filename, None, directions, numCells, 256)
    phi_movie_filename = movie_path/phi_filename.with_suffix('.mp4').name
    del phi_all

    A_all = create_composite_array_streaming(filename_phi, A_filename, directions, numCells, 256)
    A_movie_filename = movie_path/A_filename.with_suffix('.mp4').name


import os,sys
def cleanup(phi_filename: Path, A_filename: Path, directions_filename: Path):
    phi_filename.unlink()
    A_filename.unlink()
    directions_filename.unlink()


if __name__ == '__main__':
    # read the numpy array from the binary file
    results_dir = Path('results')
    for filename_phi in results_dir.glob('phi*.dat'):
        filename_directions = Path(str(filename_phi).replace('phi', 'grid_location'))
        filename_A = Path(str(filename_phi).replace('phi', 'A'))
        filename_com = Path(str(filename_phi).replace('phi', 'com'))
        # create_compacted_array(filename_phi, filename_A, filename_directions)
        try:
            create_video_from_condensed(filename_phi, filename_A, filename_directions)
            if len(sys.argv) > 1 and sys.argv[1] == 'delete':
                cleanup(filename_phi, filename_A, filename_directions)
        except Exception as e:
            print(e)
            continue

